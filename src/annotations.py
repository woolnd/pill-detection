import json
from pathlib import Path
from collections import defaultdict

# 프로젝트 루트/data/raw/... 를 가리킨다.
# __file__ 기준이라 어느 디렉터리에서 실행하든 같은 경로를 찾는다.
RAW = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "sprint_ai_project1_data"
)


def load_annotations(raw=RAW):

    by_image = defaultdict(list)

    for f in raw.glob("train_annotations/**/*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))

        img = data["images"][0]
        x, y, w, h = data["annotations"][0][
            "bbox"
        ] 
        cat = data["categories"][0]

        by_image[img["file_name"]].append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "class_id": cat["id"],
                "class_name": cat["name"],
            }
        )

    return dict(by_image)
