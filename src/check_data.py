import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from annotations import RAW, load_annotations


# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"

# 마이너스 기호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False


# 실제 데이터 폴더
DATA_DIR = RAW / "sprint_ai_project1_data"

TRAIN_IMAGES = DATA_DIR / "train_images"


def find_pills(image_path):
    """
    이미지에서 알약처럼 보이는 영역을 찾아
    바운딩 박스 목록을 반환합니다.
    """

    # 이미지 읽기
    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "이미지를 찾을 수 없습니다:",
            image_path
        )

        return None, []

    # BGR -> RGB
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # LAB 색상 공간으로 변환
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    height, width = lab.shape[:2]

    # -------------------------
    # 배경색 계산
    # -------------------------

    border_size = 30

    border = np.concatenate([

        lab[
            :border_size,
            :,
            :
        ].reshape(-1, 3),

        lab[
            -border_size:,
            :,
            :
        ].reshape(-1, 3),

        lab[
            :,
            :border_size,
            :
        ].reshape(-1, 3),

        lab[
            :,
            -border_size:,
            :
        ].reshape(-1, 3),
    ])

    # 배경 대표 색상
    background_color = np.median(
        border,
        axis=0
    )

    # -------------------------
    # 배경과 색상 차이 계산
    # -------------------------

    diff = np.linalg.norm(

        lab.astype(float)
        -
        background_color,

        axis=2
    )

    # -------------------------
    # 알약 후보 마스크
    # -------------------------

    mask = np.zeros(
        diff.shape,
        dtype=np.uint8
    )

    mask[
        diff > 18
    ] = 255

    # -------------------------
    # 노이즈 제거
    # -------------------------

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

    # -------------------------
    # 윤곽선 찾기
    # -------------------------

    contours, _ = cv2.findContours(

        mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        # 너무 작은 노이즈 제거
        if area < 500:

            continue

        # 바운딩 박스
        x, y, w, h = cv2.boundingRect(
            contour
        )

        # 너무 작은 영역 제거
        if w < 25 or h < 25:

            continue

        # 너무 길쭉한 영역 제거
        aspect_ratio = w / h

        if aspect_ratio < 0.15:

            continue

        if aspect_ratio > 6:

            continue

        # 위쪽 모서리 반사광 제거

        # 왼쪽 위
        if x < 150 and y < 100:

            continue

        # 오른쪽 위
        if (
            x + w > width - 150
            and y < 100
        ):

            continue

        boxes.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )

    return rgb, boxes


def calculate_iou(box1, box2):
    """
    두 박스가 얼마나 겹치는지 계산합니다.
    """

    x1 = max(
        box1["x"],
        box2["x"]
    )

    y1 = max(
        box1["y"],
        box2["y"]
    )

    x2 = min(

        box1["x"] + box1["w"],

        box2["x"] + box2["w"]
    )

    y2 = min(

        box1["y"] + box1["h"],

        box2["y"] + box2["h"]
    )

    # 겹치는 영역이 없는 경우
    if x2 <= x1 or y2 <= y1:

        return 0

    # 겹치는 영역 넓이
    intersection = (

        (x2 - x1)

        *

        (y2 - y1)
    )

    # 각 박스 넓이
    area1 = (
        box1["w"]
        *
        box1["h"]
    )

    area2 = (
        box2["w"]
        *
        box2["h"]
    )

    # 합집합 넓이
    union = (

        area1

        +

        area2

        -

        intersection
    )

    return intersection / union


def show(ann, n=6):
    """
    랜덤 이미지 여러 장을 보여주고

    자동으로 찾은 알약:
    -> 파란색 박스

    JSON에 이름이 있는 알약:
    -> 알약 이름 표시
    """

    random.seed(42)

    # 랜덤 이미지 선택
    names = random.sample(

        sorted(ann.keys()),

        min(
            n,
            len(ann)
        )
    )

    plt.figure(
        figsize=(18, 12)
    )

    for i, name in enumerate(names):

        ax = plt.subplot(

            2,

            3,

            i + 1
        )

        # 이미지 경로
        image_path = (
            TRAIN_IMAGES
            /
            name
        )

        # -------------------------
        # 자동 알약 탐지
        # -------------------------

        image, detected_boxes = find_pills(
            image_path
        )

        # 이미지 표시
        ax.imshow(
            image
        )

        # -------------------------
        # 자동 탐지된 박스
        # -------------------------

        for box in detected_boxes:

            ax.add_patch(

                Rectangle(

                    (
                        box["x"],
                        box["y"]
                    ),

                    box["w"],

                    box["h"],

                    fill=False,

                    edgecolor="blue",

                    linewidth=2,
                )
            )

        # -------------------------
        # JSON 정답 박스 + 이름 표시
        # -------------------------

        annotation_boxes = ann.get(
            name,
            []
        )

        for annotation_box in annotation_boxes:

            # 이름
            pill_name = annotation_box[
                "class_name"
            ]

            # 정답 박스 위치
            x = annotation_box["x"]
            y = annotation_box["y"]

            # 이름 표시
            ax.text(

                x,

                y - 10,

                pill_name,

                color="red",

                fontsize=9,

                fontweight="bold",

                bbox={

                    "facecolor": "white",

                    "alpha": 0.8,

                    "edgecolor": "none",

                    "pad": 2,
                }
            )

        # -------------------------
        # 이미지 정보
        # -------------------------

        print(
            f"\n{name}"
        )

        print(
            f"자동 탐지: "
            f"{len(detected_boxes)}개"
        )

        print(
            "알약 이름:"
        )

        for annotation_box in annotation_boxes:

            print(
                "-",
                annotation_box[
                    "class_name"
                ]
            )

        # 제목
        ax.set_title(
            name,
            fontsize=10
        )

        ax.axis(
            "off"
        )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    # JSON 어노테이션 불러오기
    ann = load_annotations()

    # 이미지 확인
    show(
        ann,
        n=6
    )