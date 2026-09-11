"""
변환 결과 확인: 원본과 숫자 비교 + 그림

실행:
uv run python src/check_yolo.py

(make_yolo.py 먼저 실행)
"""

import random

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from annotations import load_annotations
from make_yolo import IMG_H, IMG_W, OUT


# ---------- 1. YOLO 좌표 -> 픽셀 좌표 ----------

def yolo_to_pixel(line):
    """YOLO 라벨 한 줄을 픽셀 좌표로 되돌린다."""

    idx, cx, cy, w, h = map(float, line.split())

    cx, w = cx * IMG_W, w * IMG_W
    cy, h = cy * IMG_H, h * IMG_H

    x = cx - w / 2
    y = cy - h / 2

    return int(idx), x, y, w, h


# ---------- 2. 라벨 txt 읽기 ----------

def read_label(txt_path):
    """라벨 txt 파일 하나를 읽어서 박스 리스트로 만든다."""

    lines = txt_path.read_text().strip().split("\n")

    return [
        yolo_to_pixel(line)
        for line in lines
    ]


# ---------- 3. 원본 JSON과 비교 ----------

def compare_with_original():
    """YOLO 라벨을 원본 JSON과 비교한다."""

    ann = load_annotations()
    bad = 0

    txts = list(OUT.glob("labels/*/*.txt"))

    for txt in txts:

        # YOLO -> 픽셀 좌표
        got = sorted(
            (round(x), round(y), round(w), round(h))
            for _, x, y, w, h in read_label(txt)
        )

        # 원본 JSON 좌표
        want = sorted(
            (b["x"], b["y"], b["w"], b["h"])
            for b in ann[txt.stem + ".png"]
        )

        # 박스 비교
        for g, e in zip(got, want):

            if any(abs(a - b) > 1 for a, b in zip(g, e)):
                bad += 1
                print("불일치:", txt.name, g, e)

    print(f"라벨 {len(txts)}개 확인 / 불일치 박스 {bad}개")


# ---------- 4. 그림으로 확인 ----------

def show(split="train", n=6):
    """라벨을 이미지 위에 그려서 확인한다."""

    txts = sorted(
        OUT.glob(f"labels/{split}/*.txt")
    )

    random.seed(42)

    picked = random.sample(
        txts,
        min(n, len(txts))
    )

    for i, txt in enumerate(picked):

        ax = plt.subplot(2, 3, i + 1)

        # 이미지 표시
        ax.imshow(
            plt.imread(
                OUT / "images" / split / (txt.stem + ".png")
            )
        )

        # 박스 표시
        for idx, x, y, w, h in read_label(txt):

            ax.add_patch(
                Rectangle(
                    (x, y),
                    w,
                    h,
                    fill=False,
                    edgecolor="lime"
                )
            )

            ax.text(
                x,
                y,
                idx,
                color="lime"
            )

        ax.axis("off")

    plt.show()


# ---------- 실행 ----------

if __name__ == "__main__":

    compare_with_original()

    show()