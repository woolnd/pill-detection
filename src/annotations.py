"""원본 COCO JSON 라벨 읽기 + 박스 공통 계산

사용: from src.annotations import load_annotations, compute_iou

박스 딕셔너리 모양 (정답과 예측을 같은 모양으로 맞춘다):
    정답: {"x": 167, "y": 248, "w": 184, "h": 182, "class_id": 1900, "class_name": "..."}
    예측: {"x": 166.8, "y": 247.5, "w": 185.1, "h": 181.9, "class_id": 1900, "score": 0.83}
"""

import json

from src.config import KAGGLE_DIR


def load_annotations(raw=KAGGLE_DIR):
    """JSON 라벨을 이미지 파일명 기준으로 묶는다

    입력:
        raw (Path): 대회 데이터 폴더 (기본값 KAGGLE_DIR)

    반환:
        dict: 이미지 파일명 -> 박스 리스트
              예) {"K-001900-..._70_000_200.png": [{"x": 167, "y": 248, "w": 184, "h": 182,
                                                    "class_id": 1900, "class_name": "..."}, ...]}

    동작:
        1. train_annotations 아래 JSON 을 전부 찾는다.
           (K-<조합>_json/K-<약품코드>/*.json 구조라 2단계 깊이. ** 는 하위 폴더 전부)
        2. JSON 마다 이미지 · 박스 · 클래스를 꺼낸다.
           (images / annotations / categories 전부 원소가 딱 1개인 COCO 형식)
        3. 이미지 파일명으로 묶는다. 처음 보는 파일명이면 빈 리스트부터 만든다.
           (JSON 이 (이미지 x 알약)당 1개씩이라 묶어야 한 장에 3~4개인 박스가 모인다.
            각도(_70_/_75_/_90_)가 파일명에 있어서 다른 각도끼리 안 섞인다)
    """
    by_image = {}

    # 1. JSON 전부
    for f in raw.glob("train_annotations/**/*.json"):
        # 2. 이미지 · 박스 · 클래스
        data = json.loads(f.read_text(encoding="utf-8"))
        img = data["images"][0]
        x, y, w, h = data["annotations"][0]["bbox"]  # COCO 표준: 좌상단 x, y, 너비, 높이
        cat = data["categories"][0]

        # 3. 파일명으로 묶기
        name = img["file_name"]
        if name not in by_image:
            by_image[name] = []
        by_image[name].append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "class_id": cat["id"],
                "class_name": cat["name"],
            }
        )

    return by_image


def compute_iou(a, b):
    """두 박스가 얼마나 겹치는지(IoU) 0~1 로 계산한다

    입력:
        a (dict): 박스 1개 (x, y, w, h 키가 있으면 정답이든 예측이든 된다)
        b (dict): 박스 1개

    반환:
        float: 겹친 넓이 / 합친 넓이
               예) 완전히 같으면 1.0, 안 겹치면 0.0

    동작:
        1. 겹치는 영역의 네 변을 구한다.
           왼쪽 = 두 왼쪽 중 큰 값, 오른쪽 = 두 오른쪽 중 작은 값 (위, 아래도 같은 방식)
        2. 겹친 넓이 = 가로 x 세로. 안 겹치면 음수가 나오니까 0 으로 막는다.
        3. 합친 넓이 = 두 박스 넓이 합 - 겹친 넓이 (겹친 부분이 두 번 더해져서 한 번 뺀다)
        4. 겹친 넓이 / 합친 넓이 (합친 넓이가 0 이면 0.0)
    """
    # 1. 겹치는 영역의 네 변
    left = max(a["x"], b["x"])
    top = max(a["y"], b["y"])
    right = min(a["x"] + a["w"], b["x"] + b["w"])
    bottom = min(a["y"] + a["h"], b["y"] + b["h"])

    # 2. 겹친 넓이
    inter = max(0, right - left) * max(0, bottom - top)

    # 3. 합친 넓이
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter

    # 4. IoU
    if union <= 0:
        return 0.0
    return inter / union
