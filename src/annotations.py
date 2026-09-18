"""원본 COCO JSON 라벨 읽기 + 박스 공통 계산

사용: from src.annotations import load_annotations, load_aihub_annotations, compute_iou

박스 딕셔너리 모양 (정답과 예측을 같은 모양으로 맞춘다):
    정답: {"x": 167, "y": 248, "w": 184, "h": 182, "class_id": 1900, "class_name": "..."}
    예측: {"x": 166.8, "y": 247.5, "w": 185.1, "h": 181.9, "class_id": 1900, "score": 0.83}
"""

import json

from src.config import AIHUB_DIR, KAGGLE_DIR


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
        x, y, w, h = data["annotations"][0][
            "bbox"
        ]  # COCO 표준: 좌상단 x, y, 너비, 높이
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


def load_aihub_annotations(aihub_dir=AIHUB_DIR):
    """AI Hub 조합 데이터 라벨을 이미지 파일명 기준으로 묶는다

    입력:
        aihub_dir (Path): AI Hub 폴더 (labels/TL_n/, images/TS_n/ 이 들어 있다)

    반환:
        ann (dict): 이미지 파일명 -> 박스 리스트 (load_annotations() 와 같은 모양)
        image_paths (dict): 이미지 파일명 -> png 경로
                            예) {"K-000250-..._75_000_200.png": Path(".../images/TS_1/K-000250-.../...png")}

    동작:
        1. images 아래 png 를 전부 찾아 파일명 -> 경로 표를 만든다.
           (조합마다 있는 *_index.png 는 라벨이 없어서 뒤에서 자연스럽게 빠진다)
        2. labels 아래 JSON 을 하나씩 읽는다.
           a. JSON 이 깨졌거나 bbox 가 4개가 아니면 그 이미지를 "깨짐" 으로 표시한다.
           b. 클래스는 categories 가 아니라 drug_N 에서 가져온다.
              (AI Hub 는 categories 가 전부 1 "Drug" 이고, Kaggle category_id = drug_N 의 숫자다. 예) "K-033880" -> 33880)
        3. 깨진 라벨이 하나라도 있거나 png 가 없는 이미지는 뺀다.
           (박스가 빠진 알약은 학습 때 배경으로 배워서 오히려 해롭다)
    """
    # 1. png 경로 표
    png_paths = {}
    for path in aihub_dir.glob("images/*/*/*.png"):
        png_paths[path.name] = path

    # 2. JSON 읽기
    by_image = {}
    broken = set()
    for f in aihub_dir.glob("labels/**/*.json"):
        name = f.stem + ".png"  # JSON 파일명 = 이미지 파일명 (확장자만 다름)

        # 2-a. 깨진 라벨
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            broken.add(name)
            continue
        bbox = data["annotations"][0]["bbox"]
        if len(bbox) != 4:
            broken.add(name)
            continue

        # 2-b. 박스 + drug_N 클래스
        img = data["images"][0]
        x, y, w, h = bbox
        if name not in by_image:
            by_image[name] = []
        by_image[name].append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "class_id": int(img["drug_N"][2:]),  # "K-033880" -> 33880
                "class_name": img["dl_name"],
            }
        )

    # 3. 깨졌거나 png 가 없는 이미지 빼기
    ann = {}
    image_paths = {}
    n_no_png = 0
    for name in by_image:
        if name in broken:
            continue
        if name not in png_paths:
            n_no_png += 1
            continue
        ann[name] = by_image[name]
        image_paths[name] = png_paths[name]

    print(
        f"  AI Hub: 깨진 라벨 {len(broken)}장, png 없음 {n_no_png}장 제외 -> {len(ann)}장"
    )
    return ann, image_paths


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
