from pathlib import Path
from collections import defaultdict

LABEL_DIR = Path("data/yolo/labels")

# combo → 등장한 class 목록
combo_classes = defaultdict(set)

# train / val 폴더 안까지 모두 탐색
for label_file in LABEL_DIR.rglob("*.txt"):
    combo = label_file.stem.split("_", 1)[0]

    with open(label_file, encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])
            combo_classes[combo].add(class_id)


print(f"전체 조합 수: {len(combo_classes)}")
print()

for combo in sorted(combo_classes):
    classes = sorted(combo_classes[combo])

    print(f"{combo}")
    print(f"  Class: {classes}")