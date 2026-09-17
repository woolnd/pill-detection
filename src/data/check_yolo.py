"""변환 결과 확인: 원본과 숫자 비교 + 그림

실행: uv run python -m src.data.check_yolo  (make_yolo.py 먼저 실행)
"""

import random

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.annotations import load_annotations
from src.config import SEED, YOLO_DIR
from src.data.make_yolo import IMG_H, IMG_W


def compare_with_original():
    """모든 라벨을 픽셀로 되돌려서 원본 JSON 좌표와 같은지 확인한다

    입력: 없음 (data/yolo/labels 와 원본 JSON 을 읽는다)

    반환:
        없음. 결과를 출력한다.
            "라벨 220개 확인 / 불일치 박스 0개"  <- 0개여야 정상

    동작:
        1. 원본 JSON 을 load_annotations() 로 읽는다.
        2. labels/train, labels/val 의 txt 를 전부 찾는다.
        3. txt 마다
           a. 되돌린 좌표를 반올림해서 정렬한다.
           b. 같은 이미지의 원본 좌표도 정렬한다.
              (txt 줄 순서와 원본 박스 순서가 같다는 보장이 없어서 정렬 후 비교)
           c. 박스끼리 x, y, w, h 가 1px 넘게 차이 나면 불일치로 센다.
              (0.5 같은 반올림 오차가 있어서 1px 까지는 허용)
    """
    # 1. 원본
    ann = load_annotations()
    bad = 0

    # 2. 모든 라벨 파일 (* 자리에 train, val 이 들어간다)
    txts = list(YOLO_DIR.glob("labels/*/*.txt"))

    for txt in txts:
        # 3-a. 되돌린 좌표 (클래스 번호 idx 는 비교 안 함)
        got = []
        for idx, x, y, w, h in load_label(txt):
            got.append((round(x), round(y), round(w), round(h)))
        got.sort()

        # 3-b. 원본 좌표 (txt.stem = 확장자 뺀 파일명)
        want = []
        for b in ann[txt.stem + ".png"]:
            want.append((b["x"], b["y"], b["w"], b["h"]))
        want.sort()

        # 3-c. 박스끼리 x, y, w, h 를 하나씩 비교
        for got_box, want_box in zip(got, want):
            too_far = False
            for got_value, want_value in zip(got_box, want_box):
                if abs(got_value - want_value) > 1:
                    too_far = True
            if too_far:
                bad += 1
                print("  불일치:", txt.name, got_box, want_box)

    print(f"라벨 {len(txts)}개 확인 / 불일치 박스 {bad}개")


def load_label(txt_path):
    """라벨 txt 파일 하나를 읽어서 박스 리스트로 만든다

    입력:
        txt_path (Path): 라벨 파일 경로

    반환:
        list: to_pixel_box() 결과 튜플의 리스트. 알약 수만큼 들어 있다.
              예) [(0, 167, 248, 184, 182), (15, ...), ...]

    동작:
        1. 파일 전체를 읽고 끝의 줄바꿈을 지운다(strip).
        2. 줄 단위로 자른다.
        3. 줄마다 to_pixel_box() 로 바꾼다.
    """
    # 1~2. 줄 단위로 자르기
    lines = txt_path.read_text().strip().split("\n")

    # 3. 줄마다 변환
    boxes = []
    for line in lines:
        boxes.append(to_pixel_box(line))
    return boxes


def to_pixel_box(line):
    """YOLO 라벨 한 줄을 COCO 픽셀 좌표로 되돌린다 (to_yolo_line 의 반대)

    입력:
        line (str): "클래스번호 중심x 중심y 너비 높이" (0~1 비율)
                    예) "0 0.265369 0.264844 0.188525 0.142187"

    반환:
        tuple: (클래스번호, 왼쪽 위 x, 왼쪽 위 y, 너비, 높이)  전부 int (소수점 아래는 버림)
               예) (0, 167, 248, 184, 182)

    동작:
        1. 공백으로 잘라 5개 값을 꺼낸다.
        2. 비율에 이미지 크기를 곱해서 픽셀로 만든다.
        3. 왼쪽 위 = 중심 - 크기의 절반
    """
    # 1. 5개 값 꺼내기 (전부 문자열이라 float/int 로 바꿔 쓴다)
    idx, cx, cy, w, h = line.split()

    # 2. 비율 -> 픽셀
    w = float(w) * IMG_W
    h = float(h) * IMG_H

    # 3. 중심 -> 왼쪽 위
    x = float(cx) * IMG_W - (w / 2)
    y = float(cy) * IMG_H - (h / 2)
    return int(idx), int(x), int(y), int(w), int(h)  # int() 는 소수점 아래를 버린다


def show_samples(split="train", n=6):
    """변환된 라벨을 이미지 위에 그려서 보여준다

    입력:
        split (str): "train" 또는 "val"
        n (int): 그릴 장수 (2행 3열이라 6 이하)

    반환:
        없음. 그림 창을 띄운다.

    동작:
        1. 라벨 파일을 정렬하고 시드를 고정해서 n 개 뽑는다. (돌릴 때마다 같은 사진이 나온다)
        2. 2행 3열 칸에 같은 이름의 png 를 띄운다.
        3. 박스마다 초록 사각형과 YOLO 번호를 그린다.
    """
    # 1. 뽑기
    txts = sorted(YOLO_DIR.glob(f"labels/{split}/*.txt"))
    random.seed(SEED)
    picked = random.sample(txts, n)

    for i, txt in enumerate(picked):
        # 2. i+1 번째 칸 (칸 번호는 1부터)
        ax = plt.subplot(2, 3, i + 1)
        ax.imshow(plt.imread(YOLO_DIR / "images" / split / (txt.stem + ".png")))

        # 3. 박스 + 번호
        for idx, x, y, w, h in load_label(txt):
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="lime"))
            ax.text(x, y, idx, color="lime")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    compare_with_original()
    show_samples()
