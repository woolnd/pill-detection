from pathlib import Path
from collections import Counter, defaultdict
import json

RAW = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "sprint_ai_project1_data"
)

TARGET_CLASS = 33009


def load_annotations():
    by_image = defaultdict(list)

    for f in RAW.glob("train_annotations/**/*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))

        img = data["images"][0]
        ann = data["annotations"][0]
        cat = data["categories"][0]

        by_image[img["file_name"]].append(
            {
                "class_id": cat["id"],
                "class_name": cat["name"],
                "x": ann["bbox"][0],
                "y": ann["bbox"][1],
                "w": ann["bbox"][2],
                "h": ann["bbox"][3],
            }
        )

    return dict(by_image)


def analyze_target(annotations):
    target_images = []
    target_objects = []

    for image_name, objects in annotations.items():
        target = [
            obj for obj in objects
            if obj["class_id"] == TARGET_CLASS
        ]

        if target:
            target_images.append(image_name)

            for obj in target:
                target_objects.append(
                    {
                        "image": image_name,
                        **obj,
                    }
                )

    print("=" * 60)
    print(f"TARGET CLASS: {TARGET_CLASS}")
    print("=" * 60)

    print(f"\n33009 객체 수: {len(target_objects)}")
    print(f"33009 포함 이미지 수: {len(target_images)}")

    print("\n[1] 33009 이미지 목록")
    for name in sorted(target_images):
        print(f"  {name}")

    print("\n[2] 33009 bbox 분석")

    if not target_objects:
        print("33009 데이터가 없습니다.")
        return

    widths = [obj["w"] for obj in target_objects]
    heights = [obj["h"] for obj in target_objects]

    areas = [
        obj["w"] * obj["h"]
        for obj in target_objects
    ]

    print(f"  width  평균: {sum(widths) / len(widths):.1f}px")
    print(f"  width  최소: {min(widths):.1f}px")
    print(f"  width  최대: {max(widths):.1f}px")

    print(f"  height 평균: {sum(heights) / len(heights):.1f}px")
    print(f"  height 최소: {min(heights):.1f}px")
    print(f"  height 최대: {max(heights):.1f}px")

    print(f"  area   평균: {sum(areas) / len(areas):.1f}px²")
    print(f"  area   최소: {min(areas):.1f}px²")
    print(f"  area   최대: {max(areas):.1f}px²")

    print("\n[3] 33009 bbox 상세")

    for obj in target_objects:
        print(
            f"  {obj['image']}"
            f" | x={obj['x']:.1f}"
            f" y={obj['y']:.1f}"
            f" w={obj['w']:.1f}"
            f" h={obj['h']:.1f}"
        )

    print("\n[4] 같은 이미지에 등장하는 클래스")

    co_classes = Counter()

    for image_name in target_images:
        for obj in annotations[image_name]:
            if obj["class_id"] != TARGET_CLASS:
                co_classes[obj["class_id"]] += 1

    if co_classes:
        for class_id, count in co_classes.most_common():
            print(f"  {class_id}: {count}회")
    else:
        print("  다른 클래스 없음")

    print("\n[5] 이미지별 객체 구성")

    for image_name in sorted(target_images):
        classes = [
            obj["class_id"]
            for obj in annotations[image_name]
        ]

        print(
            f"  {image_name}"
            f" → {classes}"
        )


def analyze_train_val_split(annotations):
    yolo_root = RAW.parent.parent / "yolo"

    train_dir = yolo_root / "images" / "train"
    val_dir = yolo_root / "images" / "val"

    train_names = {
        p.name
        for p in train_dir.glob("*.png")
    }

    val_names = {
        p.name
        for p in val_dir.glob("*.png")
    }

    target_train = []
    target_val = []

    for image_name, objects in annotations.items():
        has_target = any(
            obj["class_id"] == TARGET_CLASS
            for obj in objects
        )

        if not has_target:
            continue

        if image_name in train_names:
            target_train.append(image_name)

        elif image_name in val_names:
            target_val.append(image_name)

    print("\n" + "=" * 60)
    print("[6] TRAIN / VAL 분포")
    print("=" * 60)

    print(f"33009 TRAIN 이미지: {len(target_train)}")
    print(f"33009 VAL 이미지:   {len(target_val)}")

    print("\nTRAIN:")
    for name in sorted(target_train):
        print(f"  {name}")

    print("\nVAL:")
    for name in sorted(target_val):
        print(f"  {name}")


def main():
    annotations = load_annotations()

    print(f"전체 이미지 수: {len(annotations)}")

    total_objects = sum(
        len(objects)
        for objects in annotations.values()
    )

    print(f"전체 객체 수: {total_objects}")

    analyze_target(annotations)
    analyze_train_val_split(annotations)


if __name__ == "__main__":
    main()