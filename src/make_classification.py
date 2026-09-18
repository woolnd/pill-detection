from pathlib import Path
import shutil

from PIL import Image

from make_yolo import OUT


# ===== 경로 =====
ROOT = Path(__file__).resolve().parent.parent
CLASSIFICATION = ROOT / "data" / "classification"

TRAIN_IMAGES = OUT / "images" / "train"
VAL_IMAGES = OUT / "images" / "val"


def make_dataset():
    # 기존 Classification 데이터셋 삭제
    if CLASSIFICATION.exists():
        shutil.rmtree(CLASSIFICATION)

    train_count = 0
    val_count = 0

    for split, image_dir in [
        ("train", TRAIN_IMAGES),
        ("val", VAL_IMAGES),
    ]:
        labels_dir = OUT / "labels" / split

        for label_file in sorted(labels_dir.glob("*.txt")):
            image_file = image_dir / f"{label_file.stem}.png"

            if not image_file.exists():
                continue

            image = Image.open(image_file).convert("RGB")
            width, height = image.size

            lines = label_file.read_text(encoding="utf-8").splitlines()

            for index, line in enumerate(lines):
                values = line.split()

                if len(values) != 5:
                    continue

                # YOLO label
                class_id = int(values[0])
                x_center = float(values[1]) * width
                y_center = float(values[2]) * height
                box_width = float(values[3]) * width
                box_height = float(values[4]) * height

                # YOLO 중심좌표 → 좌상단/우하단
                x1 = max(0, int(x_center - box_width / 2))
                y1 = max(0, int(y_center - box_height / 2))
                x2 = min(width, int(x_center + box_width / 2))
                y2 = min(height, int(y_center + box_height / 2))

                if x2 <= x1 or y2 <= y1:
                    continue

                # class별 폴더
                class_dir = CLASSIFICATION / split / f"class_{class_id}"
                class_dir.mkdir(parents=True, exist_ok=True)

                # 알약 Crop
                crop = image.crop((x1, y1, x2, y2))

                output_file = class_dir / f"{label_file.stem}_{index}.png"
                crop.save(output_file)

                if split == "train":
                    train_count += 1
                else:
                    val_count += 1

    print()
    print("===== Classification 데이터셋 생성 완료 =====")
    print(f"Train crop: {train_count}")
    print(f"Val crop:   {val_count}")
    print(f"저장 위치: {CLASSIFICATION}")
    print()


if __name__ == "__main__":
    make_dataset()