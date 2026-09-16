from pathlib import Path

LABEL_DIR = Path("data/yolo/labels")

areas = []

for label_file in LABEL_DIR.rglob("*.txt"):
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()

            if len(values) != 5:
                continue

            _, _, _, w, h = map(float, values)

            # Bounding Box가 이미지에서 차지하는 비율
            area = w * h
            areas.append(area)

if not areas:
    print("Bounding Box 데이터를 찾지 못했습니다.")
    exit()

areas.sort()

print(f"전체 Bounding Box 개수: {len(areas)}")
print()

print(f"가장 작은 박스: {areas[0]:.4f}")
print(f"25% 지점: {areas[len(areas) // 4]:.4f}")
print(f"중앙값: {areas[len(areas) // 2]:.4f}")
print(f"75% 지점: {areas[len(areas) * 3 // 4]:.4f}")
print(f"가장 큰 박스: {areas[-1]:.4f}")