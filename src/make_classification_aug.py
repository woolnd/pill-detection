from pathlib import Path
import shutil

from PIL import Image
from torchvision import transforms


# ===== 경로 =====
ROOT = Path(__file__).resolve().parent.parent

SOURCE_TRAIN = ROOT / "data" / "classification" / "train"
SOURCE_VAL = ROOT / "data" / "classification" / "val"

OUTPUT = ROOT / "data" / "classification_aug"
OUTPUT_TRAIN = OUTPUT / "train"
OUTPUT_VAL = OUTPUT / "val"


# ===== 증강 설정 =====
# 적은 클래스도 최소 10장 정도가 되도록 생성
MIN_TRAIN_IMAGES = 10

transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
    ),
])


def make_train():
    count_original = 0
    count_augmented = 0

    for class_dir in sorted(SOURCE_TRAIN.iterdir()):
        if not class_dir.is_dir():
            continue

        output_class_dir = OUTPUT_TRAIN / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(class_dir.glob("*.png"))

        # 원본은 그대로 복사
        for image_path in images:
            output_path = output_class_dir / image_path.name
            shutil.copy2(image_path, output_path)
            count_original += 1

        # 부족한 클래스만 증강
        if len(images) < MIN_TRAIN_IMAGES:
            needed = MIN_TRAIN_IMAGES - len(images)

            for i in range(needed):
                source = images[i % len(images)]

                image = Image.open(source).convert("RGB")
                augmented = transform(image)

                output_path = (
                    output_class_dir
                    / f"{source.stem}_aug_{i}.png"
                )

                augmented.save(output_path)
                count_augmented += 1

    return count_original, count_augmented


def copy_val():
    count = 0

    for class_dir in sorted(SOURCE_VAL.iterdir()):
        if not class_dir.is_dir():
            continue

        output_class_dir = OUTPUT_VAL / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(class_dir.glob("*.png")):
            shutil.copy2(
                image_path,
                output_class_dir / image_path.name,
            )
            count += 1

    return count


def main():
    # 기존 증강 데이터셋 삭제
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    original, augmented = make_train()
    val = copy_val()

    print()
    print("===== Classification 증강 데이터셋 생성 완료 =====")
    print(f"원본 Train: {original}")
    print(f"증강 Train: {augmented}")
    print(f"최종 Train: {original + augmented}")
    print(f"Val:        {val}")
    print(f"저장 위치:  {OUTPUT}")
    print()


if __name__ == "__main__":
    main()