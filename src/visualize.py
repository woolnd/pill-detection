"""val 예측 결과 시각화 + 실패 사례 분석

실행: uv run python src/visualize.py   (train.py 로 학습을 먼저 끝내야 한다)
결과: runs/<NAME>_analysis/
        ├── failures.png    실패가 있는 val 이미지 6장 (정답 박스 + 예측 박스)
        ├── curves.png      epoch 별 loss, mAP 곡선
        ├── class_ap.csv    클래스별 AP50, AP75-95 (낮은 순)
        └── val/            class_table() 이 val 평가할 때 만드는 폴더

예측 결과 분류 (그림에 한글을 쓰면 글자가 깨져서 약어로 표시한다):
    OK   정답         클래스 맞음 + IoU 0.75 이상
    LOC  위치 부정확  클래스 맞음 + IoU 0.75 미만
    CLS  클래스 틀림  위치는 맞는데(IoU 0.5 이상) 클래스가 다름
    DUP  중복         이미 짝지어진 알약에 또 예측함
    FP   오탐         알약이 없는 곳에 예측함
    FN   놓침         정답 알약에 예측이 없음

박스 딕셔너리 모양 (정답과 예측을 같은 모양으로 맞췄다):
    정답: {"x": 167, "y": 248, "w": 184, "h": 182, "class_id": 1900, "class_name": "..."}
    예측: {"x": 166.8, "y": 247.5, "w": 185.1, "h": 181.9, "class_id": 1900, "score": 0.83}
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from ultralytics import YOLO

from annotations import load_annotations
from make_yolo import OUT
from train import DATA_YAML, IMGSZ, NAME, RUNS, get_device

# ===== 설정값 =====
WEIGHTS = RUNS / NAME / "weights" / "best.pt"  # 분석할 모델
VAL_DIR = OUT / "images" / "val"  # val 이미지 42장
ANALYSIS = RUNS / f"{NAME}_analysis"  # 결과 저장 폴더
CONF = 0.25  # 그림용 신뢰도 기준 (제출용 0.001 이면 박스가 너무 많아 안 보인다)
MATCH_IOU = 0.5  # 이 이상 겹치면 같은 알약을 가리킨 것으로 본다
GOOD_IOU = 0.75  # 대회 지표 하한. 이 이상이어야 위치가 맞은 것


# 분류별 박스 색
COLORS = {
    "OK": "lime",
    "LOC": "orange",
    "CLS": "red",
    "DUP": "cyan",
    "FP": "magenta",
    "FN": "yellow",
}


# ---------- 1. 예측 ----------
def predict_val(model, device):
    """val 이미지 전체를 예측한다

    입력:
        model (YOLO): 학습된 모델
        device (str): get_device() 결과

    반환:
        dict: 이미지 파일명 -> 예측 박스 리스트
              예) {"K-001900-..._70_000_200.png": [{"x": 640.2, "y": 856.1, "w": 208.3, "h": 166.2,
                                                    "class_id": 1900, "score": 0.83}, ...]}

    동작:
        1. val 이미지를 정렬해서 하나씩 예측한다.
        2. 박스마다
           a. 좌표(xyxy)를 정답과 같은 x, y, w, h 로 바꾼다.
           b. YOLO 번호를 약 ID 로 되돌린다.
           c. 딕셔너리로 만들어 리스트에 넣는다.
    """

    preds = {}

    # 1. 이미지마다 예측
    for path in sorted(VAL_DIR.glob("*.png")):
        result = model.predict(
            path, imgsz=IMGSZ, conf=CONF, device=device, verbose=False
        )[0]
        boxes = result.boxes

        # 2. 박스마다 (predict.py 의 to_rows 와 같은 방식)
        rows = []
        for (x1, y1, x2, y2), cls, score in zip(
            boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
        ):
            rows.append(
                {
                    "x": x1,  # 2-a. 왼쪽 위 x
                    "y": y1,  #      왼쪽 위 y
                    "w": x2 - x1,  #      너비
                    "h": y2 - y1,  #      높이
                    "class_id": int(model.names[int(cls)]),  # 2-b. YOLO 번호 -> 약 ID
                    "score": score,
                }
            )
        preds[path.name] = rows  # 2-c.
    return preds


# ---------- 2. 예측과 정답 짝짓기 ----------
def compute_iou(a, b):
    """두 박스가 얼마나 겹치는지(IoU) 계산한다

    입력:
        a (dict): 박스 1개 (x, y, w, h 키가 있으면 정답이든 예측이든 된다)
        b (dict): 박스 1개

    반환:
        float: 0~1. 겹친 넓이 / 합친 넓이
               예) 완전히 같으면 1.0, 안 겹치면 0.0

    동작:
        1. 겹치는 영역의 네 변을 구한다.
           왼쪽 = 두 왼쪽 중 큰 값, 오른쪽 = 두 오른쪽 중 작은 값 (위, 아래도 같은 방식)
        2. 겹친 넓이 = 가로 x 세로. 안 겹치면 음수가 나오니까 0 으로 막는다.
        3. 합친 넓이 = 두 박스 넓이 합 - 겹친 넓이 (겹친 부분이 두 번 더해져서 한 번 뺀다)
        4. 겹친 넓이 / 합친 넓이
    """
    # 1. 겹치는 영역의 네 변
    left = max(a["x"], b["x"])
    top = max(a["y"], b["y"])
    right = min(a["x"] + a["w"], b["x"] + b["w"])
    bottom = min(a["y"] + a["h"], b["y"] + b["h"])

    # 2. 겹친 넓이
    inter = max(0, right - left) * max(0, bottom - top)

    # 3. 합친 넓이
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter

    # 4. IoU
    return inter / union


def match(preds, gts):
    """이미지 1장의 예측과 정답을 짝지어 분류한다

    입력:
        preds (list): predict_val() 결과 중 이미지 1장 분 (예측 박스 리스트)
        gts (list): load_annotations() 결과 중 같은 이미지의 정답 박스 리스트

    반환:
        list: 결과 딕셔너리 리스트
              예) [{"status": "OK", "pred": {...}, "gt": {...}, "iou": 0.95},
                   {"status": "FN", "pred": None, "gt": {...}, "iou": 0.0}]

    동작:
        1. 예측을 신뢰도 높은 순으로 정렬한다.
           (같은 알약에 두 번 예측하면 점수 높은 쪽이 먼저 짝을 가져가고, 낮은 쪽이 DUP 가 된다)
        2. 예측마다
           a. 아직 짝이 없는 정답 중 IoU 가 가장 큰 것을 찾는다.
           b. 그 IoU 가 0.5 미만이면 짝이 없다.
              모든 정답 중 0.5 이상 겹치는 게 있으면 DUP (이미 짝지어진 알약), 없으면 FP
           c. 짝이 있으면 그 정답을 짝 목록에 넣고
              클래스가 다르면 CLS, IoU 0.75 이상이면 OK, 아니면 LOC
        3. 끝까지 짝이 없는 정답은 FN

    정답 좌표는 check_yolo.read_label() 대신 원본 JSON 을 쓴다. (read_label 은 정수로 잘라서 1px 오차가 생김)
    """
    results = []
    used = []  # 짝이 정해진 정답 번호

    # 1. 신뢰도 높은 순
    preds = sorted(preds, key=lambda p: p["score"], reverse=True)

    for p in preds:
        # 2-a. 짝 없는 정답 중 IoU 최대
        best_iou = 0
        best_j = None
        for j, gt in enumerate(gts):
            if j in used:
                continue
            iou = compute_iou(p, gt)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        # 2-b. 짝 없음 -> DUP 또는 FP
        if best_iou < MATCH_IOU:
            max_iou = max(
                compute_iou(p, gt) for gt in gts
            )  # 짝 있는 정답까지 포함한 최대 IoU
            if max_iou >= MATCH_IOU:
                status = "DUP"
            else:
                status = "FP"
            results.append({"status": status, "pred": p, "gt": None, "iou": max_iou})
            continue

        # 2-c. 짝 있음 -> CLS / OK / LOC
        used.append(best_j)
        gt = gts[best_j]
        if p["class_id"] != gt["class_id"]:
            status = "CLS"
        elif best_iou >= GOOD_IOU:
            status = "OK"
        else:
            status = "LOC"
        results.append({"status": status, "pred": p, "gt": gt, "iou": best_iou})

    # 3. 남은 정답 -> FN
    for j, gt in enumerate(gts):
        if j not in used:
            results.append({"status": "FN", "pred": None, "gt": gt, "iou": 0.0})
    return results


# ---------- 3. 개수 세기 / 그림 ----------
def count_status(all_results):
    """분류별 개수를 센다

    입력:
        all_results (dict): 이미지 파일명 -> match() 결과

    반환:
        dict: 분류 -> 개수  예) {"OK": 93, "LOC": 0, "CLS": 24, "DUP": 17, "FP": 3, "FN": 14}

    동작:
        1. 분류마다 0 으로 시작한다.
        2. 모든 이미지의 결과를 보면서 해당 분류에 1 을 더한다.
    """
    # 1. 0 으로 시작
    counts = {status: 0 for status in COLORS}

    # 2. 하나씩 더하기
    for results in all_results.values():
        for r in results:
            counts[r["status"]] += 1
    return counts


def draw(ax, image_path, results):
    """이미지 1장에 정답 박스와 분류별 예측 박스를 그린다

    입력:
        ax (Axes): 그릴 칸
        image_path (Path): 이미지 경로
        results (list): match() 결과

    반환:
        없음. ax 에 그린다.

    동작:
        1. 이미지를 띄운다.
        2. 결과마다
           a. 정답이 있으면 흰 점선으로 그린다.
           b. 예측이 있으면 분류 색 실선 + "분류 약ID 신뢰도 IoU" 글자
              (DUP 는 짝 박스와 글자가 겹쳐서 박스 아래에 쓴다)
           c. 예측이 없으면(FN) 정답 위치에 노란 실선 + "FN 약ID"
        3. 제목에 파일명을 쓰고 축을 끈다.
    """

    # 1. 이미지
    ax.imshow(plt.imread(image_path))

    for r in results:
        color = COLORS[r["status"]]
        p = r["pred"]
        gt = r["gt"]

        # 2-a. 정답
        if gt is not None:
            ax.add_patch(
                Rectangle(
                    (gt["x"], gt["y"]),
                    gt["w"],
                    gt["h"],
                    fill=False,
                    edgecolor="white",
                    linestyle="--",
                )
            )

        # 2-b. 예측
        if p is not None:
            ax.add_patch(
                Rectangle(
                    (p["x"], p["y"]),
                    p["w"],
                    p["h"],
                    fill=False,
                    edgecolor=color,
                    linewidth=2,
                )
            )
            if r["status"] == "DUP":
                text_y = p["y"] + p["h"] + 30  # 박스 아래
            else:
                text_y = p["y"] - 5  # 박스 위
            text = f"{r['status']} {p['class_id']} {p['score']:.2f} iou{r['iou']:.2f}"
            ax.text(p["x"], text_y, text, color=color, fontsize=7)

        # 2-c. 놓침
        else:
            ax.add_patch(
                Rectangle(
                    (gt["x"], gt["y"]),
                    gt["w"],
                    gt["h"],
                    fill=False,
                    edgecolor=color,
                    linewidth=2,
                )
            )
            ax.text(
                gt["x"], gt["y"] - 5, f"FN {gt['class_id']}", color=color, fontsize=7
            )

    # 3. 제목 (파일명이 길어서 40자까지)
    ax.set_title(image_path.name[:40], fontsize=7)
    ax.axis("off")


def save_failures(all_results, n=6):
    """실패가 하나라도 있는 이미지를 모아서 그림으로 저장한다

    입력:
        all_results (dict): 이미지 파일명 -> match() 결과
        n (int): 그릴 장수 (6 이하)

    반환:
        failed (list): 실패가 있는 이미지 파일명 리스트
        path (Path): 저장한 그림 경로  예) runs/baseline_analysis/failures.png

    동작:
        1. 결과 중 OK 가 아닌 게 하나라도 있는 이미지를 고른다.
        2. 앞에서 n장을 2행 3열로 그린다.
        3. png 로 저장한다.
    """
    # 1. 실패 포함 이미지
    failed = []
    for name, results in all_results.items():
        for r in results:
            if r["status"] != "OK":
                failed.append(name)
                break  # 하나만 찾으면 이 이미지는 끝

    # 2. 2행 3열
    plt.figure(figsize=(12, 10))
    for i, name in enumerate(failed[:n]):
        ax = plt.subplot(2, 3, i + 1)  # subplot 번호는 1부터
        draw(ax, VAL_DIR / name, all_results[name])

    # 3. 저장
    plt.tight_layout()
    path = ANALYSIS / "failures.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return failed, path


# ---------- 4. 클래스별 점수 / 학습 곡선 ----------
def class_table(model, device, ann):
    """클래스별 AP 표를 만든다

    입력:
        model (YOLO): 학습된 모델
        device (str): get_device() 결과
        ann (dict): load_annotations() 결과 (한글 약 이름을 가져온다)

    반환:
        DataFrame: class_id, name, AP50, AP75-95 열. AP75-95 낮은 순
                   예) 33009  신바로정  0.000  0.000

    동작:
        1. val 평가를 한다.
        2. 약 ID -> 한글 이름 표를 만든다.
        3. all_ap 행마다
           a. ap_class_index 로 이 행이 몇 번 클래스인지 찾고 약 ID 로 바꾼다.
           b. AP50 = 0번 열, AP75-95 = 5번 열부터 평균 (train.py 의 evaluate 와 같은 방식)
        4. AP75-95 낮은 순으로 정렬한다.
    """
    # 1. val 평가
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=IMGSZ,
        batch=16,
        device=device,
        verbose=False,
        plots=False,
        project=str(ANALYSIS),
        name="val",
        exist_ok=True,
    )

    # 2. 약 ID -> 이름
    names = {}
    for boxes in ann.values():
        for b in boxes:
            names[b["class_id"]] = b["class_name"]

    # 3. 클래스마다 한 행
    ap = metrics.box.all_ap  # (클래스 수, IoU 10개)
    rows = []
    for i, idx in enumerate(metrics.box.ap_class_index):
        class_id = int(model.names[int(idx)])  # 3-a. YOLO 번호 -> 약 ID
        rows.append(
            {
                "class_id": class_id,
                "name": names[class_id],
                "AP50": round(float(ap[i, 0]), 3),  # 3-b.
                "AP75-95": round(float(ap[i, 5:].mean()), 3),
            }
        )

    # 4. 낮은 순
    return pd.DataFrame(rows).sort_values("AP75-95")


def plot_curves():
    """epoch 별 loss, mAP 곡선을 그린다

    입력: 없음 (runs/<NAME>/results.csv 를 읽는다)

    반환:
        Path: 저장한 그림 경로  예) runs/baseline_analysis/curves.png

    동작:
        1. results.csv 를 읽는다.
        2. 세 칸에 그린다.
           box loss (train / val), cls loss (train / val), val mAP50 / mAP50-95
        3. png 로 저장한다.
    """
    # 1. 학습 로그
    df = pd.read_csv(RUNS / NAME / "results.csv")

    # 2. 세 칸 (칸 제목, 그릴 열 이름들)
    panels = [
        ("box loss", ["train/box_loss", "val/box_loss"]),
        ("cls loss", ["train/cls_loss", "val/cls_loss"]),
        ("val mAP", ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]),
    ]
    plt.figure(figsize=(12, 4))
    for i, panel in enumerate(panels):
        title, cols = panel
        ax = plt.subplot(1, 3, i + 1)
        for col in cols:
            ax.plot(df["epoch"], df[col], label=col)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()

    # 3. 저장
    plt.tight_layout()
    path = ANALYSIS / "curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ---------- 실행 ----------
def main():
    """전체 순서

    동작:
        1. 결과 폴더, 장치, 모델, 원본 정답 준비
        2. val 예측 -> 이미지마다 match()
        3. 분류별 개수 출력 + 실패 사례 그림 저장
        4. 클래스별 AP 표 저장 (낮은 10개 출력)
        5. 학습 곡선 저장
    """
    # 1. 준비
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model = YOLO(WEIGHTS)
    ann = load_annotations()

    # 2. 예측 + 짝짓기
    preds = predict_val(model, device)
    all_results = {}
    for name in preds:
        all_results[name] = match(preds[name], ann[name])

    # 3. 개수 + 실패 그림
    print(count_status(all_results))
    failed, path = save_failures(all_results)
    print(f"실패 포함 이미지 {len(failed)} / {len(all_results)}장:", path)

    # 4. 클래스별 AP
    table = class_table(model, device, ann)
    table.to_csv(ANALYSIS / "class_ap.csv", index=False)
    print(table.head(10).to_string(index=False))

    # 5. 학습 곡선
    print("학습 곡선:", plot_curves())


if __name__ == "__main__":
    main()
