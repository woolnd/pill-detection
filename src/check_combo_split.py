from pathlib import Path
import json
import random
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/sprint_ai_project1_data"

VAL_RATIO = 0.2
SEED = 42


def load_data():
    annotations = defaultdict(set)

    for file in RAW.glob("train_annotations/**/*.json"):
        data = json.loads(
            file.read_text(encoding="utf-8")
        )

        image = data["images"][0]
        category = data["categories"][0]

        annotations[image["file_name"]].add(
            category["id"]
        )

    return annotations


def make_combos(annotations):
    combos = defaultdict(list)

    for image, classes in annotations.items():
        combo = image.split("_")[0]
        combos[combo].append((image, classes))

    return combos


def find_split(combos):
    all_classes = {
        class_id
        for images in combos.values()
        for _, classes in images
        for class_id in classes
    }

    combo_names = list(combos)
    target_images = round(
        sum(len(images) for images in combos.values())
        * VAL_RATIO
    )

    rng = random.Random(SEED)

    best = None
    best_score = float("inf")

    for _ in range(100000):
        rng.shuffle(combo_names)

        val_combos = set()
        val_images = 0

        for combo in combo_names:
            if val_images >= target_images:
                break

            val_combos.add(combo)
            val_images += len(combos[combo])

        train_combos = set(combo_names) - val_combos

        val_classes = {
            class_id
            for combo in val_combos
            for _, classes in combos[combo]
            for class_id in classes
        }

        train_classes = {
            class_id
            for combo in train_combos
            for _, classes in combos[combo]
            for class_id in classes
        }

        missing_val = all_classes - val_classes
        missing_train = all_classes - train_classes

        image_diff = abs(
            val_images - target_images
        )

        score = (
            len(missing_val) * 10000
            + len(missing_train) * 10000
            + image_diff
        )

        if score < best_score:
            best_score = score
            best = (
                val_combos.copy(),
                train_combos.copy(),
                val_images,
                missing_train,
                missing_val,
            )

        if (
            not missing_val
            and not missing_train
            and image_diff <= 3
        ):
            return (
                val_combos,
                train_combos,
                val_images,
                missing_train,
                missing_val,
            )

    return best


def main():
    annotations = load_data()
    combos = make_combos(annotations)

    result = find_split(combos)

    (
        val_combos,
        train_combos,
        val_images,
        missing_train,
        missing_val,
    ) = result

    total_images = len(annotations)

    print()
    print("===== Combo 분할 검사 =====")
    print(f"전체 이미지: {total_images}장")
    print(f"전체 Combo: {len(combos)}개")
    print(f"목표 Val 이미지: {round(total_images * VAL_RATIO)}장")
    print()
    print(f"Train Combo: {len(train_combos)}개")
    print(f"Val Combo: {len(val_combos)}개")
    print(f"Train 이미지: {total_images - val_images}장")
    print(f"Val 이미지: {val_images}장")
    print()
    print(f"Train에 없는 클래스: {sorted(missing_train)}")
    print(f"Val에 없는 클래스: {sorted(missing_val)}")
    print()
    print("모든 클래스가 양쪽에 있음:",
          not missing_train and not missing_val)
    print("Combo 중복: 0개")
    print()


if __name__ == "__main__":
    main()