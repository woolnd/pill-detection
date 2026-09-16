from pathlib import Path
import numpy as np
from ultralytics import YOLO

from train import IMGSZ, get_device


# Experiment 05 모델
WEIGHTS = (
    Path(__file__).resolve().parent.parent
    / "runs"
    / "experiment_05_model_s"
    / "weights"
    / "best.pt"
)

# Validation 이미지
VAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "yolo"
    / "images"
    / "val"
)

# Validation 정답 라벨
VAL_LABEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "yolo"
    / "labels"
    / "val"
)

# 테스트할 confidence
CONFIDENCES = [
    0.001,
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
]


def main():
    device = get_device()

    print("===== Experiment 05 Confidence Analysis =====")
    print()
    print("모델:", WEIGHTS)
    print("Val 이미지:", VAL_DIR)
    print("Val 라벨:", VAL_LABEL_DIR)
    print("장치:", device)
    print()

    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {WEIGHTS}"
        )

    if not VAL_DIR.exists():
        raise FileNotFoundError(
            f"Val 이미지 폴더를 찾을 수 없습니다: {VAL_DIR}"
        )

    if not VAL_LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Val 라벨 폴더를 찾을 수 없습니다: {VAL_LABEL_DIR}"
        )

    model = YOLO(WEIGHTS)

    results = []

    print("===== Confidence별 평가 =====")
    print()

    for conf in CONFIDENCES:

        print(f"Confidence {conf} 평가 중...")

        metrics = model.val(
            data=str(
                Path(__file__).resolve().parent.parent
                / "data"
                / "yolo"
                / "data.yaml"
            ),
            imgsz=IMGSZ,
            batch=8,
            conf=conf,
            device=device,
            verbose=False,
        )

        map50 = float(
            metrics.box.map50
        )

        map50_95 = float(
            metrics.box.map
        )

        # IoU 0.75 ~ 0.95 평균
        ap = metrics.box.all_ap

        map75_95 = float(
            ap[:, 5:].mean()
        )

        results.append(
            {
                "confidence": conf,
                "mAP50": map50,
                "mAP50-95": map50_95,
                "mAP75-95": map75_95,
            }
        )

        print(
            f"  mAP50     : {map50:.4f}"
        )
        print(
            f"  mAP50-95  : {map50_95:.4f}"
        )
        print(
            f"  mAP75-95  : {map75_95:.4f}"
        )
        print()

    print()
    print("===== 최종 비교 =====")
    print()

    print(
        f"{'Confidence':<12}"
        f"{'mAP50':<12}"
        f"{'mAP50-95':<14}"
        f"{'mAP75-95':<12}"
    )

    print("-" * 50)

    for result in results:
        print(
            f"{result['confidence']:<12.3f}"
            f"{result['mAP50']:<12.4f}"
            f"{result['mAP50-95']:<14.4f}"
            f"{result['mAP75-95']:<12.4f}"
        )


if __name__ == "__main__":
    main()