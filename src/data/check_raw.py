"""원본 데이터 요약 + 박스 시각화

실행: uv run python -m src.data.check_raw
결과: 콘솔 요약 + 그림 창 6장 (변환 전 원본 JSON 기준)
"""

import random

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.annotations import load_annotations
from src.config import KAGGLE_DIR, SEED


def print_summary(ann):
    """원본 데이터가 어떻게 생겼는지 한눈에 출력한다

    입력:
        ann (dict): load_annotations() 결과. 이미지 파일명 -> 박스 리스트

    반환:
        없음. 콘솔에만 출력한다.

    동작:
        1. 모든 박스를 한 리스트로 모은다.
        2. 이미지 수 / 박스 수 / 클래스 수를 출력한다.
        3. test 이미지 수를 출력한다.
        4. 이미지당 알약 수 분포를 센다.  예) {3: 151, 4: 74, 2: 7} -> 알약 수 순서로 출력
        5. 클래스별 박스 수를 세서 많은 순 5개를 출력한다.
    """
    # 1. 박스 모으기
    boxes = []
    for image_boxes in ann.values():
        for b in image_boxes:
            boxes.append(b)

    # 2. 전체 규모
    class_ids = set()
    for b in boxes:
        class_ids.add(b["class_id"])
    print(f"이미지 {len(ann)} / 박스 {len(boxes)} / 클래스 {len(class_ids)}")

    # 3. test 이미지
    print("test 이미지", len(list(KAGGLE_DIR.glob("test_images/*.png"))))

    # 4. 이미지당 알약 수 -> 장수
    pills_per_image = {}
    for image_boxes in ann.values():
        n_pills = len(image_boxes)
        pills_per_image[n_pills] = pills_per_image.get(n_pills, 0) + 1
    print("이미지당 알약:", sorted(pills_per_image.items()))

    # 5. 약 ID -> 박스 수, 많은 순 5개
    box_counts = {}
    for b in boxes:
        box_counts[b["class_id"]] = box_counts.get(b["class_id"], 0) + 1
    top_ids = sorted(box_counts, key=box_counts.get, reverse=True)[:5]  # 박스 수가 많은 약 ID 순서
    top = []
    for class_id in top_ids:
        top.append((class_id, box_counts[class_id]))
    print("클래스별 박스:", top, "...")


def show_samples(ann, n=6):
    """원본 박스를 이미지 위에 그려서 보여준다

    입력:
        ann (dict): 이미지 파일명 -> 박스 리스트
        n (int): 그릴 장수 (2행 3열이라 6 이하)

    반환:
        없음. 그림 창을 띄운다.

    동작:
        1. 시드를 고정하고 n 장을 뽑는다. (돌릴 때마다 같은 사진이 나온다)
        2. 2행 3열 칸에 이미지를 띄운다.
        3. 박스마다 파란 사각형과 약 ID 를 그린다.
    """
    # 1. 뽑기
    random.seed(SEED)
    names = random.sample(sorted(ann), n)

    for i, name in enumerate(names):
        # 2. i+1 번째 칸 (칸 번호는 1부터)
        ax = plt.subplot(2, 3, i + 1)
        ax.imshow(plt.imread(KAGGLE_DIR / "train_images" / name))

        # 3. 박스 + 약 ID
        for b in ann[name]:
            ax.add_patch(
                Rectangle((b["x"], b["y"]), b["w"], b["h"], fill=False, edgecolor="blue")
            )
            ax.text(b["x"], b["y"], b["class_id"], color="blue")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    ann = load_annotations()
    print_summary(ann)
    show_samples(ann)
