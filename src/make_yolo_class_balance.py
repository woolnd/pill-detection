from pathlib import Path
import json
import random
import shutil
from collections import defaultdict, Counter

# ============================================================
# 설정
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw" / "sprint_ai_project1_data"

OUT = ROOT / "data" / "yolo_class_balance"

SEED = 42
VAL_RATIO = 0.20

IMAGE_W = 976
IMAGE_H = 1280


# ============================================================
# Annotation 유효성 검사
# ============================================================

def is_valid_bbox(x, y, w, h):
    return (
        x >= 0
        and y >= 0
        and w > 0
        and h > 0
        and x + w <= IMAGE_W
        and y + h <= IMAGE_H
    )


# ============================================================
# Annotation 불러오기
# ============================================================

def load_annotations():
    annotation_dir = RAW / "train_annotations"

    annotations = {}

    json_files = sorted(annotation_dir.rglob("*.json"))

    print("Annotation 불러오는 중...")

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        images = {
            image["id"]: image["file_name"]
            for image in data.get("images", [])
        }

        categories = {
            category["id"]: category["name"]
            for category in data.get("categories", [])
        }

        for ann in data.get("annotations", []):
            image_id = ann["image_id"]

            if image_id not in images:
                continue

            bbox = ann["bbox"]

            x, y, w, h = bbox

            if not is_valid_bbox(x, y, w, h):
                # 해당 이미지는 나중에 전체 제외
                annotations.setdefault(
                    images[image_id],
                    {"boxes": [], "invalid": True}
                )
                continue

            class_id = ann["category_id"]
            class_name = categories.get(class_id, str(class_id))

            if images[image_id] not in annotations:
                annotations[images[image_id]] = {
                    "boxes": [],
                    "invalid": False,
                }

            annotations[images[image_id]]["boxes"].append(
                (class_name, x, y, w, h)
            )

    # invalid 이미지 제거
    valid_annotations = {}

    for image_name, info in annotations.items():
        if info["invalid"]:
            continue

        if len(info["boxes"]) == 0:
            continue

        valid_annotations[image_name] = info["boxes"]

    print(f"유효 이미지: {len(valid_annotations)}장")

    return valid_annotations


# ============================================================
# 이미지 실제 위치 찾기
# ============================================================

def find_image(image_name):
    path = RAW / "train" / image_name

    if path.exists():
        return path

    matches = list(RAW.rglob(image_name))

    if matches:
        return matches[0]

    return None


# ============================================================
# 클래스별 이미지 목록
# ============================================================

def build_class_images(annotations):
    class_images = defaultdict(set)

    for image_name, boxes in annotations.items():
        for class_name, *_ in boxes:
            class_images[class_name].add(image_name)

    return class_images


# ============================================================
# Train / Val 분할
# ============================================================

def split_images(annotations):
    rng = random.Random(SEED)

    images = list(annotations.keys())
    rng.shuffle(images)

    target_val = round(len(images) * VAL_RATIO)

    class_images = build_class_images(annotations)

    all_classes = sorted(class_images.keys())

    print()
    print("=" * 80)
    print("클래스 균형 기반 Train / Val 분할")
    print("=" * 80)

    print(f"전체 이미지: {len(images)}장")
    print(f"목표 Train: {len(images) - target_val}장")
    print(f"목표 Val:   {target_val}장")
    print(f"클래스 수:  {len(all_classes)}개")

    # --------------------------------------------------------
    # 1. Val에 들어갈 이미지 선정
    #
    # 희귀 클래스부터 하나씩 확보
    # --------------------------------------------------------

    val = set()

    # 이미지가 적은 클래스부터 처리
    classes_sorted = sorted(
        all_classes,
        key=lambda c: len(class_images[c])
    )

    for class_name in classes_sorted:

        # 이미 Val에 해당 클래스가 있으면 넘어감
        if any(
            class_name in {
                box[0] for box in annotations[image_name]
            }
            for image_name in val
        ):
            continue

        candidates = [
            image_name
            for image_name in class_images[class_name]
            if image_name not in val
        ]

        if not candidates:
            continue

        # 현재 Val에서 함께 확보할 수 있는 클래스가 많은 이미지 우선
        candidates.sort(
            key=lambda image_name: len({
                box[0]
                for box in annotations[image_name]
            }),
            reverse=True,
        )

        val.add(candidates[0])

    # --------------------------------------------------------
    # 2. Val을 목표 크기까지 채움
    # --------------------------------------------------------

    remaining = [
        image_name
        for image_name in images
        if image_name not in val
    ]

    rng.shuffle(remaining)

    while len(val) < target_val and remaining:
        val.add(remaining.pop())

    # --------------------------------------------------------
    # 3. Train 구성
    # --------------------------------------------------------

    train = set(images) - val

    # --------------------------------------------------------
    # 4. Train에 없는 클래스가 있다면 수정
    # --------------------------------------------------------

    def get_classes(image_set):
        result = set()

        for image_name in image_set:
            for box in annotations[image_name]:
                result.add(box[0])

        return result

    train_classes = get_classes(train)
    val_classes = get_classes(val)

    missing_train = set(all_classes) - train_classes

    for class_name in missing_train:

        candidates = [
            image_name
            for image_name in val
            if class_name in {
                box[0]
                for box in annotations[image_name]
            }
        ]

        if not candidates:
            continue

        # Train에 없는 클래스를 가져오되,
        # Val에서 해당 이미지를 빼도 클래스가 유지되는지 확인
        for candidate in candidates:

            candidate_classes = {
                box[0]
                for box in annotations[candidate]
            }

            temp_val = val - {candidate}
            temp_val_classes = get_classes(temp_val)

            if class_name in candidate_classes:
                # Val에서도 해당 클래스가 유지되면 이동
                if class_name in temp_val_classes:
                    val.remove(candidate)
                    train.add(candidate)
                    break

    # --------------------------------------------------------
    # 5. Val에 없는 클래스가 있다면 Train에서 이동
    # --------------------------------------------------------

    train_classes = get_classes(train)
    val_classes = get_classes(val)

    missing_val = set(all_classes) - val_classes

    for class_name in missing_val:

        candidates = [
            image_name
            for image_name in train
            if class_name in {
                box[0]
                for box in annotations[image_name]
            }
        ]

        if not candidates:
            continue

        for candidate in candidates:

            candidate_classes = {
                box[0]
                for box in annotations[candidate]
            }

            temp_train = train - {candidate}
            temp_train_classes = get_classes(temp_train)

            if class_name in candidate_classes:
                # Train에서도 해당 클래스가 유지되면 이동
                if class_name in temp_train_classes:
                    train.remove(candidate)
                    val.add(candidate)
                    break

    # --------------------------------------------------------
    # 6. 최종 검증
    # --------------------------------------------------------

    train_classes = get_classes(train)
    val_classes = get_classes(val)

    missing_train = sorted(set(all_classes) - train_classes)
    missing_val = sorted(set(all_classes) - val_classes)

    print()
    print("=" * 80)
    print("최종 분할 결과")
    print("=" * 80)

    print(f"TRAIN: {len(train)}장")
    print(f"VAL:   {len(val)}장")

    print()
    print(f"Train에 없는 클래스: {missing_train}")
    print(f"Val에 없는 클래스:   {missing_val}")

    if missing_train:
        raise RuntimeError(
            f"Train에 없는 클래스가 있습니다: {missing_train}"
        )

    if missing_val:
        raise RuntimeError(
            f"Validation에 없는 클래스가 있습니다: {missing_val}"
        )

    return sorted(train), sorted(val)


# ============================================================
# YOLO 데이터셋 생성
# ============================================================

def create_dataset(annotations, train_images, val_images):

    if OUT.exists():
        shutil.rmtree(OUT)

    train_img_dir = OUT / "images" / "train"
    val_img_dir = OUT / "images" / "val"

    train_label_dir = OUT / "labels" / "train"
    val_label_dir = OUT / "labels" / "val"

    for path in [
        train_img_dir,
        val_img_dir,
        train_label_dir,
        val_label_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    # 클래스 이름
    class_names = sorted({
        box[0]
        for boxes in annotations.values()
        for box in boxes
    })

    class_to_id = {
        name: idx
        for idx, name in enumerate(class_names)
    }

    # --------------------------------------------------------
    # 이미지 + Label 복사
    # --------------------------------------------------------

    def write_split(image_names, image_dir, label_dir):

        for image_name in image_names:

            src = find_image(image_name)

            if src is None:
                print(f"이미지 없음: {image_name}")
                continue

            shutil.copy2(
                src,
                image_dir / image_name
            )

            label_path = label_dir / (
                Path(image_name).stem + ".txt"
            )

            with open(label_path, "w", encoding="utf-8") as f:

                for class_name, x, y, w, h in annotations[image_name]:

                    class_id = class_to_id[class_name]

                    # YOLO normalized bbox
                    xc = (x + w / 2) / IMAGE_W
                    yc = (y + h / 2) / IMAGE_H

                    nw = w / IMAGE_W
                    nh = h / IMAGE_H

                    f.write(
                        f"{class_id} "
                        f"{xc:.6f} "
                        f"{yc:.6f} "
                        f"{nw:.6f} "
                        f"{nh:.6f}\n"
                    )

    write_split(
        train_images,
        train_img_dir,
        train_label_dir
    )

    write_split(
        val_images,
        val_img_dir,
        val_label_dir
    )

    # --------------------------------------------------------
    # data.yaml
    # --------------------------------------------------------

    yaml_path = OUT / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:

        f.write(f"path: {OUT.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")

        for idx, name in enumerate(class_names):
            f.write(f"  {idx}: '{name}'\n")

    # --------------------------------------------------------
    # 클래스 분포 출력
    # --------------------------------------------------------

    train_counter = Counter()
    val_counter = Counter()

    for image_name in train_images:
        for box in annotations[image_name]:
            train_counter[box[0]] += 1

    for image_name in val_images:
        for box in annotations[image_name]:
            val_counter[box[0]] += 1

    print()
    print("=" * 80)
    print("클래스별 객체 수")
    print("=" * 80)

    print(
        f"{'CLASS':>8} | "
        f"{'TRAIN':>8} | "
        f"{'VAL':>8}"
    )

    print("-" * 40)

    for class_name in class_names:

        print(
            f"{class_name:>8} | "
            f"{train_counter[class_name]:>8} | "
            f"{val_counter[class_name]:>8}"
        )

    print()
    print(f"데이터셋 생성 완료: {OUT}")


# ============================================================
# Main
# ============================================================

def main():

    annotations = load_annotations()

    train_images, val_images = split_images(
        annotations
    )

    create_dataset(
        annotations,
        train_images,
        val_images
    )


if __name__ == "__main__":
    main()