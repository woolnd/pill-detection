"""앙상블(WBF) 예측 결과 시각화
   - test 이미지: 앙상블 예측 박스만 오버레이 (정답 없음, 순수 박스+클래스+신뢰도 표시)
   - val 이미지: 앙상블 예측 vs 정답으로 실패 사례 분석 (visualize.py와 같은 분류: OK/LOC/CLS/DUP/FP/FN)
   - 학습 곡선: 앙상블 자체는 학습을 새로 하지 않으므로 메인 체크포인트(모델 B)의 곡선을 참고용으로 표시

실행: uv run python -m src.analysis.visualize_ensemble
결과: runs/ensemble_analysis/
        ├── test_boxes.png   test 이미지 몇 장에 앙상블 예측 박스만 그린 것
        ├── failures.png     val 이미지 중 앙상블 기준 실패가 있는 것들
        └── curves.png       모델 B(메인 체크포인트) 학습 곡선 참고용
"""

import random

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from ultralytics import YOLO

from src.analysis.predict import WEIGHTS_A, WEIGHTS_B, predict_ensemble
from src.analysis.visualize import COLORS, TEXT_BOX, count_status, draw_boxes, match_boxes
from src.annotations import load_annotations
from src.config import KAGGLE_DIR, RUNS_DIR, YOLO_DIR, get_device
from src.train.train_yolo import NAME

# ===== 설정값 =====
TEST_DIR = KAGGLE_DIR / "test_images"
VAL_DIR = YOLO_DIR / "images" / "val"
ANALYSIS_DIR = RUNS_DIR / "ensemble_analysis"
N_TEST_SAMPLES = 6      # test 박스 오버레이 그릴 장 수
N_FAILURE_SAMPLES = 6   # 실패 사례 그릴 장 수
RANDOM_SEED = 42


# ---------- 1. test 이미지: 예측 박스만 그리기 (정답 없음) ----------
def draw_pred_only(ax, image_path, result, names):
    """정답 비교 없이 예측 박스 + 클래스 + 신뢰도만 그린다"""
    ax.imshow(plt.imread(image_path))
    xyxy = result.boxes.xyxy.tolist()
    classes = result.boxes.cls.tolist()
    scores = result.boxes.conf.tolist()

    for i in range(len(scores)):
        x1, y1, x2, y2 = xyxy[i]
        cls_id = int(names[int(classes[i])])
        score = scores[i]
        ax.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2)
        )
        ax.text(x1, y1 - 8, f"{cls_id} {score:.2f}", color="lime", fontsize=7, bbox=TEXT_BOX)

    ax.set_title(image_path.name[:30], fontsize=7)
    ax.axis("off")


def save_test_boxes(model_a, model_b, device, names):
    """test 이미지 중 N_TEST_SAMPLES장을 뽑아 앙상블 예측 박스만 그려서 저장한다"""
    paths = sorted(TEST_DIR.glob("*.png"), key=lambda p: int(p.stem))
    random.seed(RANDOM_SEED)
    sample = random.sample(paths, min(N_TEST_SAMPLES, len(paths)))

    plt.figure(figsize=(12, 10))
    for i, path in enumerate(sample):
        result = predict_ensemble(model_a, model_b, path, device)
        ax = plt.subplot(2, 3, i + 1)
        draw_pred_only(ax, path, result, names)

    plt.tight_layout()
    out = ANALYSIS_DIR / "test_boxes.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


# ---------- 2. val 이미지: 앙상블 기준 실패 사례 ----------
def predict_val_ensemble(model_a, model_b, device, names):
    """val 이미지 전체를 앙상블로 예측한다 (visualize.py의 predict_val과 같은 모양으로 반환)"""
    preds = {}
    for path in sorted(VAL_DIR.glob("*.png")):
        result = predict_ensemble(model_a, model_b, path, device)
        xyxy = result.boxes.xyxy.tolist()
        classes = result.boxes.cls.tolist()
        scores = result.boxes.conf.tolist()
        rows = []
        for i in range(len(scores)):
            x1, y1, x2, y2 = xyxy[i]
            rows.append(
                {
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "class_id": int(names[int(classes[i])]),
                    "score": scores[i],
                }
            )
        preds[path.name] = rows
    return preds


def save_ensemble_failures(all_results, n=N_FAILURE_SAMPLES):
    """visualize.py의 save_failures와 동일한 로직 (앙상블 결과 버전)"""
    failed = []
    for name, results in all_results.items():
        for r in results:
            if r["status"] != "OK":
                failed.append(name)
                break

    plt.figure(figsize=(12, 10))
    for i, name in enumerate(failed[:n]):
        ax = plt.subplot(2, 3, i + 1)
        draw_boxes(ax, VAL_DIR / name, all_results[name])

    plt.tight_layout()
    out = ANALYSIS_DIR / "failures.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return failed, out


# ---------- 3. 학습 곡선 (참고용: 메인 체크포인트=모델 B 기준) ----------
def plot_curves():
    """앙상블은 별도 학습이 없으므로, 메인 체크포인트(NAME=모델 B)의 학습 곡선을 참고용으로 그린다"""
    df = pd.read_csv(RUNS_DIR / NAME / "results.csv")
    panels = [
        ("box loss", ["train/box_loss", "val/box_loss"]),
        ("cls loss", ["train/cls_loss", "val/cls_loss"]),
        ("val mAP", ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]),
    ]
    plt.figure(figsize=(12, 4))
    for i, (title, cols) in enumerate(panels):
        ax = plt.subplot(1, 3, i + 1)
        for col in cols:
            ax.plot(df["epoch"], df[col], label=col)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()

    plt.tight_layout()
    out = ANALYSIS_DIR / "curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print("모델 A:", WEIGHTS_A)
    print("모델 B:", WEIGHTS_B)
    model_a = YOLO(WEIGHTS_A)
    model_b = YOLO(WEIGHTS_B)

    # 두 모델이 같은 data.yaml 기준으로 학습됐다는 전제 (클래스 매핑이 다르면 결과가 틀어짐)
    assert model_a.names == model_b.names, "두 모델의 클래스 매핑이 다릅니다"
    names = model_b.names

    # 1. test 이미지 박스 오버레이
    path1 = save_test_boxes(model_a, model_b, device, names)
    print("test 박스 오버레이:", path1)

    # 2. val 기준 실패 사례
    ann = load_annotations()
    preds = predict_val_ensemble(model_a, model_b, device, names)
    all_results = {name: match_boxes(preds[name], ann[name]) for name in preds}
    print(count_status(all_results))
    failed, path2 = save_ensemble_failures(all_results)
    print(f"실패 포함 이미지 {len(failed)} / {len(all_results)}장:", path2)

    # 3. 학습 곡선 (모델 B 기준 참고용)
    path3 = plot_curves()
    print("학습 곡선(모델 B 기준):", path3)


if __name__ == "__main__":
    main()