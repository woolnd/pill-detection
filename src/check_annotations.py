from pathlib import Path

LABEL_DIR = Path("data/yolo/labels")

error_count = 0

for label_file in LABEL_DIR.rglob("*.txt"):
    with open(label_file, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            values = line.strip().split()

            # 한 줄에 값이 5개인지 확인
            if len(values) != 5:
                print(f"[형식 오류] {label_file}:{line_no}")
                error_count += 1
                continue

            class_id, x, y, w, h = map(float, values)

            # 좌표가 0~1 범위인지 확인
            if not (0 <= x <= 1 and 0 <= y <= 1):
                print(f"[좌표 오류] {label_file}:{line_no}")

                error_count += 1

            # 박스 크기가 정상인지 확인
            if not (0 < w <= 1 and 0 < h <= 1):
                print(f"[크기 오류] {label_file}:{line_no}")

                error_count += 1


print()
print(f"검사 완료")
print(f"발견된 오류: {error_count}개")