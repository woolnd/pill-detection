from pathlib import Path
from collections import Counter

LABEL_DIR = Path("data/yolo/labels")

class_counts = Counter()

for label_file in LABEL_DIR.rglob("*.txt"):
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])
            class_counts[class_id] += 1


print(f"전체 클래스 수: {len(class_counts)}")
print(f"전체 Bounding Box 개수: {sum(class_counts.values())}")
print()

print("클래스별 Bounding Box 개수")
print("-" * 30)

for class_id, count in sorted(class_counts.items()):
    print(f"Class {class_id}: {count}개")