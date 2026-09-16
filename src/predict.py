import pandas as pd
from ultralytics import YOLO

from annotations import RAW
from train import IMGSZ, RUNS, get_device

WEIGHTS = RUNS / "experiment_01" / "weights" / "best.pt"
TEST_DIR = RAW / "test_images"
OUT_CSV = RUNS / "experiment_01" / "submission.csv"
CONF = 0.001


def to_rows(result, image_id, names):
    rows = []
    boxes = result.boxes

    for (x1, y1, x2, y2), cls, score in zip(
        boxes.xyxy.tolist(),
        boxes.cls.tolist(),
        boxes.conf.tolist(),
    ):
        rows.append(
            {
                "image_id": image_id,
                "category_id": int(names[int(cls)]),
                "bbox_x": round(x1),
                "bbox_y": round(y1),
                "bbox_w": round(x2 - x1),
                "bbox_h": round(y2 - y1),
                "score": round(score, 4),
            }
        )

    return rows


def predict_all(weights, image_dir, device):
    model = YOLO(weights)

    paths = sorted(
        image_dir.glob("*.png"),
        key=lambda p: int(p.stem)
    )

    rows = []

    for path in paths:
        result = model.predict(
            path,
            imgsz=IMGSZ,
            conf=CONF,
            device=device,
            verbose=False,
        )[0]

        rows += to_rows(
            result,
            int(path.stem),
            model.names,
        )

    df = pd.DataFrame(rows)
    df.insert(0, "annotation_id", range(1, len(df) + 1))

    return df


def main():
    device = get_device()

    print("모델:", WEIGHTS)
    print("테스트 이미지:", TEST_DIR)
    print("장치:", device)

    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {WEIGHTS}"
        )

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"테스트 이미지 폴더를 찾을 수 없습니다: {TEST_DIR}"
        )

    df = predict_all(
        WEIGHTS,
        TEST_DIR,
        device,
    )

    OUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUT_CSV,
        index=False,
    )

    n_images = len(
        list(TEST_DIR.glob("*.png"))
    )

    predicted_images = (
        df["image_id"].nunique()
        if not df.empty
        else 0
    )

    print()
    print("===== 예측 완료 =====")
    print(f"박스: {len(df)}개")
    print(f"예측된 이미지: {predicted_images}개")
    print(f"전체 테스트 이미지: {n_images}개")
    print()
    print("제출 파일:", OUT_CSV)
    print()
    print(df.head())


if __name__ == "__main__":
    main()
