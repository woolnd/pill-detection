from pathlib import Path
import json
import random
import shutil
from collections import defaultdict


ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data" / "raw" / "sprint_ai_project1_data"
OUT = ROOT / "data" / "yolo"

VAL_RATIO = 0.2
SEED = 42

IMG_W = 976
IMG_H = 1280


def load_annotations():
    by_image = defaultdict(list)

    for f in RAW.glob("train_annotations/**/*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))

        img = data["images"][0]

        for ann in data["annotations"]:
            x, y, w, h = ann["bbox"]
            cat = data["categories"][0]

            by_image[img["file_name"]].append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "class_id": cat["id"],
                    "class_name": cat["name"],
                }
            )

    return dict(by_image)


def is_valid(box):
    x = box["x"]
    y = box["y"]
    w = box["w"]
    h = box["h"]

    return (
        x >= 0
        and y >= 0
        and w > 0
        and h > 0
        and x + w <= IMG_W
        and y + h <= IMG_H
    )


def remove_bad_images(ann):
    clean = {}

    for name, boxes in ann.items():
        if all(is_valid(b) for b in boxes):
            clean[name] = boxes

    return clean


def make_class_map(ann):
    class_ids = sorted(
        {
            box["class_id"]
            for boxes in ann.values()
            for box in boxes
        }
    )

    class_to_idx = {
        class_id: idx
        for idx, class_id in enumerate(class_ids)
    }

    class_names = {}

    for boxes in ann.values():
        for box in boxes:
            class_names[class_to_idx[box["class_id"]]] = box["class_name"]

    return class_to_idx, class_names


def get_combo(name):
    return name.split("_")[0]


def split_by_combo(names):
    combos = sorted({get_combo(name) for name in names})

    rng = random.Random(SEED)
    rng.shuffle(combos)

    val_count = round(len(combos) * VAL_RATIO)

    val_combos = set(combos[:val_count])

    train = [
        name
        for name in names
        if get_combo(name) not in val_combos
    ]

    val = [
        name
        for name in names
        if get_combo(name) in val_combos
    ]

    return sorted(train), sorted(val)


def to_yolo_line(box, class_to_idx):
    x = box["x"]
    y = box["y"]
    w = box["w"]
    h = box["h"]

    cx = x + w / 2
    cy = y + h / 2

    cx /= IMG_W
    cy /= IMG_H
    w /= IMG_W
    h /= IMG_H

    class_idx = class_to_idx[box["class_id"]]

    return (
        f"{class_idx} "
        f"{cx:.6f} "
        f"{cy:.6f} "
        f"{w:.6f} "
        f"{h:.6f}"
    )


def write_split(split_name, names, ann, class_to_idx):
    image_dir = OUT / "images" / split_name
    label_dir = OUT / "labels" / split_name

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        src = RAW / "train_images" / name
        dst = image_dir / name

        shutil.copy2(src, dst)

        label_path = label_dir / f"{Path(name).stem}.txt"

        lines = [
            to_yolo_line(box, class_to_idx)
            for box in ann[name]
        ]

        label_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )


def write_yaml(class_names):
    yaml_path = OUT / "data.yaml"

    lines = [
        "path: " + str(OUT).replace("\\", "/"),
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]

    for idx in sorted(class_names):
        lines.append(
            f"  {idx}: '{class_names[idx]}'"
        )

    yaml_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    print("[1] annotation 불러오기")

    ann = load_annotations()

    print(f"전체 이미지: {len(ann)}장")

    print("[2] 잘못된 박스 제거")

    ann = remove_bad_images(ann)

    print(f"유효 이미지: {len(ann)}장")

    print("[3] 클래스 번호 생성")

    class_to_idx, class_names = make_class_map(ann)

    print(f"클래스 수: {len(class_names)}개")

    print("[4] combo 기준 train/val 분할")

    train_names, val_names = split_by_combo(
        list(ann.keys())
    )

    print(f"TRAIN: {len(train_names)}장")
    print(f"VAL:   {len(val_names)}장")

    print("[5] 기존 YOLO 데이터 삭제")

    if OUT.exists():
        shutil.rmtree(OUT)

    print("[6] 파일 생성")

    write_split(
        "train",
        train_names,
        ann,
        class_to_idx,
    )

    write_split(
        "val",
        val_names,
        ann,
        class_to_idx,
    )

    write_yaml(class_names)

    train_classes = {
        box["class_id"]
        for name in train_names
        for box in ann[name]
    }

    missing = sorted(
        set(class_to_idx) - train_classes
    )

    print()
    print(f"클래스 {len(class_names)}개")
    print(f"TRAIN {len(train_names)}장")
    print(f"VAL {len(val_names)}장")
    print(f"TRAIN에 없는 클래스: {missing}")
    print(f"저장 위치: {OUT}")


if __name__ == "__main__":
    main()