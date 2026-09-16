from pathlib import Path
from collections import defaultdict

LABEL_ROOT = Path("data/yolo/labels")

combo_classes = defaultdict(set)
combo_split = {}

# Train / Val 각각 확인
for split in ["train", "val"]:
    split_dir = LABEL_ROOT / split

    for label_file in split_dir.glob("*.txt"):
        # 예:
        # K-001900-016548-019607-029451_0_2_0_2_70_000_200
        # → K-001900-016548-019607-029451
        combo = label_file.stem.split("_", 1)[0]

        combo_split[combo] = split

        with open(label_file, encoding="utf-8") as f:
            for line in f:
                values = line.strip().split()

                if len(values) != 5:
                    continue

                class_id = int(values[0])
                combo_classes[combo].add(class_id)


# 조합 개수 확인
train_count = sum(
    1 for split in combo_split.values()
    if split == "train"
)

val_count = sum(
    1 for split in combo_split.values()
    if split == "val"
)

print(f"Train 조합 수: {train_count}")
print(f"Val 조합 수: {val_count}")
print(f"전체 조합 수: {len(combo_split)}")
print()


# Train / Val 중복 확인
train_combos = {
    combo for combo, split in combo_split.items()
    if split == "train"
}

val_combos = {
    combo for combo, split in combo_split.items()
    if split == "val"
}

overlap = train_combos & val_combos

print(f"Train / Val 중복 조합 수: {len(overlap)}")
print()


# Val 조합 출력
print("현재 Val에 들어간 조합")
print("-" * 70)

for combo in sorted(val_combos):
    classes = sorted(combo_classes[combo])

    print(f"{combo}")
    print(f"  Class: {classes}")