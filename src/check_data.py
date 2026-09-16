"""원본 데이터 요약 + 박스 시각화

실행: uv run python src/check_data.py
결과: 콘솔 요약 + 그림 창 6장 (변환 전 원본 JSON 기준)
"""

import random
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from annotations import RAW, load_annotations


def summary(ann):
    """원본 데이터가 어떻게 생겼는지 한눈에 출력한다

    입력:
        ann (dict): load_annotations() 결과. 이미지 파일명 -> 박스 리스트

    반환:
        없음. 콘솔에만 출력한다.

    동작:
        1. 모든 박스를 한 리스트로 모은다.
        2. 이미지 수 / 박스 수 / 클래스 수를 센다.
        3. test 이미지 수, 이미지당 알약 수, 클래스별 박스 수를 센다.
    """
    # 1. 박스 모으기
    boxes = [b for v in ann.values() for b in v]

    # 2. 전체 규모
    print(f"이미지 {len(ann)} / 박스 {len(boxes)} / 클래스 {len({b['class_id'] for b in boxes})}")

    # 3. 세부 분포
    print("test 이미지", len(list(RAW.glob("test_images/*.png"))))
    print("이미지당 알약:", sorted(Counter(len(v) for v in ann.values()).items()))
    print("클래스별 박스:", Counter(b["class_id"] for b in boxes).most_common(5), "...")


def show(ann, n=6):
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
    random.seed(42)
    names = random.sample(sorted(ann), n)

    for i, name in enumerate(names):
        # 2. i+1 번째 칸 (칸 번호는 1부터)
        ax = plt.subplot(2, 3, i + 1)
        ax.imshow(plt.imread(RAW / "train_images" / name))

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
    summary(ann)
    show(ann)
