from pathlib import Path
import json
import random
import shutil
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/sprint_ai_project1_data"
OUT = ROOT / "data/yolo"

IMG_W, IMG_H = 976, 1280
VAL_RATIO = 0.2
SEED = 42


# JSON에서 이미지별 bbox 정보 수집
def load_annotations():
    annotations = defaultdict(list)

    for file in RAW.glob("train_annotations/**/*.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        image = data["images"][0]
        category = data["categories"][0]

        for ann in data["annotations"]:
            x, y, w, h = ann["bbox"]

            annotations[image["file_name"]].append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "class_id": category["id"],
                "class_name": category["name"],
            })

    return dict(annotations)


# 잘못된 bbox만 제거
def clean_annotations(annotations):
    clean = {}
    removed = 0

    for name, boxes in annotations.items():
        valid_boxes = []

        for box in boxes:
            x, y = box["x"], box["y"]
            w, h = box["w"], box["h"]

            if (
                x >= 0
                and y >= 0
                and w > 0
                and h > 0
                and x + w <= IMG_W
                and y + h <= IMG_H
            ):
                valid_boxes.append(box)
            else:
                removed += 1
                print(
                    f"[bbox 제거] {name} / "
                    f"class {box['class_id']}"
                )

        if valid_boxes:
            clean[name] = valid_boxes

    print(f"제거된 bbox: {removed}개")
    return clean


# 클래스 번호를 0부터 다시 매핑
def make_class_map(annotations):
    class_ids = sorted({
        box["class_id"]
        for boxes in annotations.values()
        for box in boxes
    })

    class_to_idx = {
        class_id: idx
        for idx, class_id in enumerate(class_ids)
    }

    class_names = {}

    for boxes in annotations.values():
        for box in boxes:
            class_names[class_to_idx[box["class_id"]]] = box["class_name"]

    return class_to_idx, class_names


# Train과 Val 모두 모든 클래스를 포함하도록 분할
def split_data(annotations):
    names = list(annotations)
    target_val = round(len(names) * VAL_RATIO)

    class_images = defaultdict(set)

    for name, boxes in annotations.items():
        for box in boxes:
            class_images[box["class_id"]].add(name)

    all_classes = set(class_images)

    # 특정 클래스가 한 장에만 있다면 양쪽에 넣을 수 없으므로 확인
    impossible = [
        class_id
        for class_id, images in class_images.items()
        if len(images) < 2
    ]

    if impossible:
        print()
        print(
            "[주의] 양쪽에 넣을 수 없는 클래스:",
            sorted(impossible)
        )

    rng = random.Random(SEED)

    best_val = None
    best_score = float("inf")

    # 여러 번 시도해서 클래스가 양쪽에 모두 들어가는 분할 탐색
    for _ in range(3000):
        val = set(rng.sample(names, target_val))

        for _ in range(100):
            train = set(names) - val

            train_classes = {
                box["class_id"]
                for name in train
                for box in annotations[name]
            }

            val_classes = {
                box["class_id"]
                for name in val
                for box in annotations[name]
            }

            missing_train = all_classes - train_classes
            missing_val = all_classes - val_classes

            score = (
                len(missing_train) * 1000
                + len(missing_val) * 1000
            )

            if score < best_score:
                best_score = score
                best_val = val.copy()

            if not missing_train and not missing_val:
                return (
                    sorted(set(names) - val),
                    sorted(val)
                )

            # 현재 부족한 클래스가 있는 이미지 우선 탐색
            important = missing_train | missing_val

            candidates = []

            for name in names:
                if name in val:
                    continue

                classes = {
                    box["class_id"]
                    for box in annotations[name]
                }

                gain = len(classes & important)

                if gain > 0:
                    candidates.append((gain, name))

            candidates.sort(reverse=True)

            if not candidates:
                break

            # Train -> Val 이동
            add_name = candidates[0][1]

            remove_candidates = []

            for name in val:
                classes = {
                    box["class_id"]
                    for box in annotations[name]
                }

                # Val에서만 존재하는 클래스가 사라지지 않도록 보호
                if len(classes & missing_val) == 0:
                    remove_candidates.append(name)

            if not remove_candidates:
                break

            remove_name = rng.choice(remove_candidates)

            val.remove(remove_name)
            val.add(add_name)

    # 완벽한 분할을 찾지 못한 경우 가장 좋은 결과 사용
    val = best_val
    train = set(names) - val

    return sorted(train), sorted(val)


# bbox를 YOLO 형식으로 변환
def to_yolo(box, class_to_idx):
    x, y = box["x"], box["y"]
    w, h = box["w"], box["h"]

    cx = (x + w / 2) / IMG_W
    cy = (y + h / 2) / IMG_H

    return (
        f"{class_to_idx[box['class_id']]} "
        f"{cx:.6f} {cy:.6f} "
        f"{w / IMG_W:.6f} {h / IMG_H:.6f}"
    )


# 이미지와 라벨 저장
def save_split(split, names, annotations, class_to_idx):
    image_dir = OUT / "images" / split
    label_dir = OUT / "labels" / split

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        shutil.copy2(
            RAW / "train_images" / name,
            image_dir / name
        )

        label = label_dir / f"{Path(name).stem}.txt"

        label.write_text(
            "\n".join(
                to_yolo(box, class_to_idx)
                for box in annotations[name]
            ),
            encoding="utf-8",
        )


# data.yaml 생성
def save_yaml(class_names):
    lines = [
        f"path: {OUT.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]

    lines += [
        f"  {idx}: '{name}'"
        for idx, name in sorted(class_names.items())
    ]

    (OUT / "data.yaml").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# Train / Val 클래스 확인
def check_classes(train, val, annotations):
    train_classes = {
        box["class_id"]
        for name in train
        for box in annotations[name]
    }

    val_classes = {
        box["class_id"]
        for name in val
        for box in annotations[name]
    }

    all_classes = train_classes | val_classes

    missing_train = sorted(all_classes - train_classes)
    missing_val = sorted(all_classes - val_classes)

    print()
    print("===== 결과 =====")
    print(f"전체 클래스: {len(all_classes)}개")
    print(f"Train 클래스: {len(train_classes)}개")
    print(f"Val 클래스: {len(val_classes)}개")
    print(f"Train에 없는 클래스: {missing_train}")
    print(f"Val에 없는 클래스: {missing_val}")

    return not missing_train and not missing_val


def main():
    print("[1] Annotation 불러오기")
    annotations = load_annotations()
    print(f"이미지: {len(annotations)}장")

    print("\n[2] 잘못된 bbox 제거")
    annotations = clean_annotations(annotations)

    print("\n[3] 클래스 매핑")
    class_to_idx, class_names = make_class_map(annotations)
    print(f"클래스: {len(class_names)}개")

    print("\n[4] Train / Val 분할")
    train, val = split_data(annotations)
    print(f"Train: {len(train)}장")
    print(f"Val:   {len(val)}장")

    print("\n[5] YOLO 데이터 생성")

    if OUT.exists():
        shutil.rmtree(OUT)

    save_split(
        "train",
        train,
        annotations,
        class_to_idx,
    )

    save_split(
        "val",
        val,
        annotations,
        class_to_idx,
    )

    save_yaml(class_names)

    success = check_classes(
        train,
        val,
        annotations,
    )

    print()
    if success:
        print("모든 클래스가 Train / Val에 포함되었습니다.")
    else:
        print("일부 클래스가 한쪽에만 존재합니다.")

    print(f"저장 위치: {OUT}")


if __name__ == "__main__":
    main()