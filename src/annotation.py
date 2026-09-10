import json
from pathlib import Path
from collections import defaultdict

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "sprint_ai_project1_data"

def load_annotation(raw = RAW):
    """ 이미지 이름 -> [박스, 딕셔너리, ...]

    JSON이 (이미지 x 알약) 당 1개씩이라 이미지 이름으로 묶어야 한 장에 3~4개인 박스가 모인다.
    """

    
    by_image = defaultdict(list)

    # ** 는 하위 폴더 전부 훑기. K-<조합>
    for f in raw.glob("train_annotations/**/*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))

        # images  annotations / categories 3개가 다 들어있다.
        img = data["images"][0]
        x, y, w, h = data["annotations"][0]["bbox"]
        cat = data["categories"][0]

        # 파일명이 키. 각도 (_70_/_75_/_90_)가 파일명에 있어서 다른 각도끼리 안 섞인다. 
        by_image[img["file_name"]].append(
            {"x": x, 
             "y": y, 
             "w": w, 
             "h": h, 
             "class_id": cat["id"],
             "class_name": cat["name"]
             }
        )

    return dict(by_image)

