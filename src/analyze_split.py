from pathlib import Path
from collections import Counter

TRAIN_LABEL_DIR = Path("data/yolo/labels/train")
VAL_LABEL_DIR = Path("data/yolo/labels/val")


def count_classes(label_dir):
    counts = Counter()

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, encoding="utf-8") as f:
            for line in f:
                values = line.strip().split()

                if len(values) != 5:
                    continue

                class_id = int(values[0])
                counts[class_id] += 1

    return counts


train_counts = count_classes(TRAIN_LABEL_DIR)
val_counts = count_classes(VAL_LABEL_DIR)

all_classes = sorted(set(train_counts) | set(val_counts))

print("클래스별 Train / Validation 데이터")
print("-" * 45)

for class_id in all_classes:
    train = train_counts[class_id]
    val = val_counts[class_id]

    print(f"Class {class_id:2d} | Train: {train:3d}개 | Val: {val:3d}개")