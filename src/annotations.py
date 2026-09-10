import json
from pathlib import Path
from collections import defaultdict

Raw = Path(__file__).resolve().parent.parent / "data" / "raw"

def load_annotations(raw = Raw):
    """이미지 이름 → [박스 딕셔너리, ...]

    JSON이 (이미지 × 알약)당 1개씩이라 이미지 이름으로 묶어야
    한 장에 3~4개인 박스가 모인다.
    """

    by_iamage = defaultdict(list)

    for f in raw.rglob("train_annotations/**/*.json"):
        data = json.loads(f.read_text())
        img = data["image"][0]
        x , y , w , h = data["annotations"][0]["bbox"]
        cat = data["categories"][0]
        by_iamage[img["file_name"]].append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "class_id": cat["id"],
                "class_name": cat["name"]
            }
        )