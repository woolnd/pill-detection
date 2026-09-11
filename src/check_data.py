import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from annotations import RAW, load_annotations

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


DATA_DIR = RAW / "sprint_ai_project1_data"
TRAIN_IMAGES = DATA_DIR / "train_images"


def find_pills(image_path):

    image = cv2.imread(str(image_path))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    height, width = image.shape[:2]

    border = np.concatenate([
        lab[:30].reshape(-1, 3),
        lab[-30:].reshape(-1, 3),
        lab[:, :30].reshape(-1, 3),
        lab[:, -30:].reshape(-1, 3),
    ])

    background = np.median(border, axis=0)
    diff = np.linalg.norm(
        lab.astype(float) - background,
        axis=2
    )

    mask = (diff > 18).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        if cv2.contourArea(contour) < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < 25 or h < 25:
            continue

        # 위쪽 모서리 반사광 제거
        if (x < 150 and y < 100) or (
            x + w > width - 150 and y < 100
        ):
            continue

        boxes.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h
        })

    return rgb, boxes


def show(ann, n=6):

    random.seed(42)
    names = random.sample(sorted(ann), n)

    plt.figure(figsize=(15, 10))

    for i, name in enumerate(names):

        ax = plt.subplot(2, 3, i + 1)

        image, boxes = find_pills(
            TRAIN_IMAGES / name
        )

        ax.imshow(image)
        for box in boxes:

            ax.add_patch(
                Rectangle(
                    (box["x"], box["y"]),
                    box["w"],
                    box["h"],
                    fill=False,
                    edgecolor="blue",
                    linewidth=2
                )
            )

        for pill in ann[name]:

            ax.text(
                pill["x"],
                pill["y"] - 10,
                pill["class_name"],
                fontsize=9,
                color="red"
            )

        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    ann = load_annotations()
    show(ann)