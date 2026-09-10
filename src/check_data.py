import random
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from annotations import RAW, load_annotations


def summary(ann):
    boxes = [b for v in ann.values() for b in v]
    print(
        "이미지",
        len(ann),
        "/ 박스",
        len(boxes),
        "/ 클래스",
        len({b["class_id"] for b in boxes}),
    )
    print("test 이미지", len(list(RAW.glob("test_images/*.png"))))
    print("이미지당 알약:", sorted(Counter(len(v) for v in ann.values()).items()))
    print("클래스별 박스:", Counter(b["class_id"] for b in boxes).most_common(5), "...")


def show(ann, n=6):
    random.seed(42)
    names = random.sample(sorted(ann), n)
    
    for i, name in enumerate(names):
        ax = plt.subplot(2, 3, i + 1)
        ax.imshow(plt.imread(RAW / "train_images" / name))

        for b in ann[name]:
            ax.add_patch(
                Rectangle(
                    (b["x"], b["y"]), b["w"], b["h"], fill=False, edgecolor="blue"
                )
            )
            ax.text(b["x"], b["y"], b["class_id"], color="blue")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    ann = load_annotations()
    summary(ann)
    show(ann)
