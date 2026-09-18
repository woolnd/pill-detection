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
WEIGHTS = RUNS / NAME / "weights" / "best.pt"
VAL_DIR = OUT / "images" / "val"
ANALYSIS = RUNS / f"{NAME}_analysis"

CONF = 0.25
MATCH_IOU = 0.5
GOOD_IOU = 0.75


# 분류별 박스 색
COLORS = {
    "OK": "lime",
    "LOC": "orange",
    "CLS": "red",
    "DUP": "cyan",
    "FP": "magenta",
    "FN": "yellow",
}


# 글자 뒤에 까는 반투명 검은 배경
TEXT_BOX = {
    "facecolor": "black",
    "alpha": 0.6,
    "pad": 1,
    "edgecolor": "none",
}


# ---------- 1. 예측 ----------
def predict_val(model, device):
    """val 이미지 전체를 예측한다

    입력:
        model (YOLO): 학습된 모델
        device (str): get_device() 결과

    반환:
        dict: 이미지 파일명 -> 예측 박스 리스트

    동작:
        1. val 이미지를 정렬해서 하나씩 예측한다.
        2. 박스마다
           a. 좌표(xyxy)를 정답과 같은 x, y, w, h 로 바꾼다.
           b. YOLO 번호를 원본 약 ID 로 되돌린다.
           c. 딕셔너리로 만들어 리스트에 넣는다.
    """

    # 원본 annotation에서
    # "약 이름" -> "원본 class_id" 변환표를 만든다.
    ann = load_annotations()

    category_map = {}

    for boxes in ann.values():
        for box in boxes:
            category_map[box["class_name"]] = int(box["class_id"])

    preds = {}

    # 1. 이미지마다 예측
    for path in sorted(VAL_DIR.glob("*.png")):
        result = model.predict(
            path,
            imgsz=IMGSZ,
            conf=CONF,
            device=device,
            verbose=False,
        )[0]

        boxes = result.boxes

        # 2. 박스마다
        rows = []

        for (x1, y1, x2, y2), cls, score in zip(
            boxes.xyxy.tolist(),
            boxes.cls.tolist(),
            boxes.conf.tolist(),
        ):
            class_name = model.names[int(cls)]

            # YOLO 번호 -> 약 이름 -> 원본 약 ID
            class_id = category_map[class_name]

            rows.append(
                {
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "class_id": class_id,
                    "score": float(score),
                }
            )

        preds[path.name] = rows

    return preds


# ---------- 2. 예측과 정답 짝짓기 ----------
def compute_iou(a, b):
    """두 박스가 얼마나 겹치는지(IoU) 계산한다

    입력:
        a (dict): 박스 1개
        b (dict): 박스 1개

    반환:
        float: 0~1 사이의 IoU
    """

    # 1. 겹치는 영역의 네 변
    left = max(a["x"], b["x"])
    top = max(a["y"], b["y"])
    right = min(a["x"] + a["w"], b["x"] + b["w"])
    bottom = min(a["y"] + a["h"], b["y"] + b["h"])

    # 2. 겹친 넓이
    inter = max(0, right - left) * max(0, bottom - top)

    # 3. 합친 넓이
    union = (
        a["w"] * a["h"]
        + b["w"] * b["h"]
        - inter
    )

    # 4. IoU
    if union <= 0:
        return 0.0

    return inter / union


def match(preds, gts):
    """이미지 1장의 예측과 정답을 짝지어 분류한다

    반환:
        list: 결과 딕셔너리 리스트
    """

    results = []
    used = []

    # 1. 신뢰도 높은 순
    preds = sorted(
        preds,
        key=lambda p: p["score"],
        reverse=True,
    )

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
                compute_iou(p, gt)
                for gt in gts
            )

            if max_iou >= MATCH_IOU:
                status = "DUP"
            else:
                status = "FP"

            results.append(
                {
                    "status": status,
                    "pred": p,
                    "gt": None,
                    "iou": max_iou,
                }
            )

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

        results.append(
            {
                "status": status,
                "pred": p,
                "gt": gt,
                "iou": best_iou,
            }
        )

    # 3. 남은 정답 -> FN
    for j, gt in enumerate(gts):

        if j not in used:
            results.append(
                {
                    "status": "FN",
                    "pred": None,
                    "gt": gt,
                    "iou": 0.0,
                }
            )

    return results


# ---------- 3. 개수 세기 / 그림 ----------
def count_status(all_results):
    """분류별 개수를 센다"""

    counts = {
        status: 0
        for status in COLORS
    }

    for results in all_results.values():

        for r in results:
            counts[r["status"]] += 1

    return counts


def draw(ax, image_path, results):
    """이미지 1장에 정답 박스와 분류별 예측 박스를 그린다"""

    # 1. 이미지
    ax.imshow(
        plt.imread(image_path)
    )

    for r in results:

        color = COLORS[r["status"]]

        p = r["pred"]
        gt = r["gt"]

        # 2-a. 짝지어진 정답
        if gt is not None and r["status"] != "FN":

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

            ax.text(
                gt["x"],
                gt["y"] + gt["h"] + 35,
                f"GT {gt['class_id']}",
                color="white",
                fontsize=7,
                bbox=TEXT_BOX,
            )

        # 2-b. 놓친 정답(FN)
        if r["status"] == "FN":

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
                gt["x"],
                gt["y"] + gt["h"] + 70,
                f"FN GT {gt['class_id']}",
                color=color,
                fontsize=7,
                bbox=TEXT_BOX,
            )

        # 2-c. 예측
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
                text_y = p["y"] - 43
            else:
                text_y = p["y"] - 8

            text = (
                f"{r['status']} "
                f"{p['class_id']} "
                f"{p['score']:.2f} "
                f"iou{r['iou']:.2f}"
            )

            ax.text(
                p["x"],
                text_y,
                text,
                color=color,
                fontsize=7,
                bbox=TEXT_BOX,
            )

    # 3. 제목
    ax.set_title(
        image_path.name[:40],
        fontsize=7,
    )

    ax.axis("off")


def save_failures(all_results, n=6):
    """실패가 하나라도 있는 이미지를 모아서 그림으로 저장한다"""

    # 1. 실패 포함 이미지
    failed = []

    for name, results in all_results.items():

        for r in results:

            if r["status"] != "OK":
                failed.append(name)
                break

    # 2. 2행 3열
    plt.figure(figsize=(12, 10))

    for i, name in enumerate(failed[:n]):

        ax = plt.subplot(
            2,
            3,
            i + 1,
        )

        draw(
            ax,
            VAL_DIR / name,
            all_results[name],
        )

    # 3. 저장
    plt.tight_layout()

    path = ANALYSIS / "failures.png"

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    return failed, path


# ---------- 4. 클래스별 점수 / 학습 곡선 ----------
def class_table(model, device, ann):
    """클래스별 AP 표를 만든다"""

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

        for box in boxes:
            names[box["class_id"]] = box["class_name"]

    # 3. 클래스마다 한 행
    ap = metrics.box.all_ap

    rows = []

    for i, idx in enumerate(
        metrics.box.ap_class_index
    ):

        # YOLO 내부 클래스 번호
        yolo_class_index = int(idx)

        # YOLO 번호 -> 한글 약 이름
        class_name = model.names[
            yolo_class_index
        ]

        # 한글 약 이름 -> 원본 약 ID
        class_id = None

        for original_id, name in names.items():

            if name == class_name:
                class_id = int(original_id)
                break

        if class_id is None:
            continue

        rows.append(
            {
                "class_id": class_id,
                "name": class_name,
                "AP50": round(
                    float(ap[i, 0]),
                    3,
                ),
                "AP75-95": round(
                    float(ap[i, 5:].mean()),
                    3,
                ),
            }
        )

    # 4. 낮은 순
    return pd.DataFrame(
        rows
    ).sort_values(
        "AP75-95"
    )


def plot_curves():
    """epoch 별 loss, mAP 곡선을 그린다"""

    # 1. 학습 로그
    df = pd.read_csv(
        RUNS / NAME / "results.csv"
    )

    # 2. 세 칸
    panels = [
        (
            "box loss",
            [
                "train/box_loss",
                "val/box_loss",
            ],
        ),
        (
            "cls loss",
            [
                "train/cls_loss",
                "val/cls_loss",
            ],
        ),
        (
            "val mAP",
            [
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
            ],
        ),
    ]

    plt.figure(
        figsize=(12, 4)
    )

    for i, panel in enumerate(panels):

        title, cols = panel

        ax = plt.subplot(
            1,
            3,
            i + 1,
        )

        for col in cols:

            ax.plot(
                df["epoch"],
                df[col],
                label=col,
            )

        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()

    # 3. 저장
    plt.tight_layout()

    path = ANALYSIS / "curves.png"

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    return path


# ---------- 실행 ----------
def main():
    """전체 순서

    1. 결과 폴더, 장치, 모델, 원본 정답 준비
    2. val 예측 -> 이미지마다 match()
    3. 분류별 개수 출력 + 실패 사례 그림 저장
    4. 클래스별 AP 표 저장
    5. 학습 곡선 저장
    """

    # 1. 준비
    ANALYSIS.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print("분석 모델:", WEIGHTS)
    print("분석 장치:", device)
    print("Validation 이미지:", VAL_DIR)

    model = YOLO(WEIGHTS)

    ann = load_annotations()

    # 2. 예측 + 짝짓기
    preds = predict_val(
        model,
        device,
    )

    all_results = {}

    for name in preds:

        all_results[name] = match(
            preds[name],
            ann[name],
        )

    # 3. 개수 + 실패 그림
    print(
        "분류 결과:",
        count_status(all_results),
    )

    failed, path = save_failures(
        all_results
    )

    print(
        f"실패 포함 이미지 "
        f"{len(failed)} / "
        f"{len(all_results)}장:",
        path,
    )

    # 4. 클래스별 AP
    table = class_table(
        model,
        device,
        ann,
    )

    table.to_csv(
        ANALYSIS / "class_ap.csv",
        index=False,
    )

    print(
        table.head(10).to_string(
            index=False
        )
    )

    # 5. 학습 곡선
    print(
        "학습 곡선:",
        plot_curves(),
    )


if __name__ == "__main__":
    main()