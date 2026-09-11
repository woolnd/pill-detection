import json
from pathlib import Path
from collections import defaultdict

# 프로젝트 루트/data/raw/... 를 가리킨다.
# __file__ 기준이라 어느 디렉터리에서 실행하든 같은 경로를 찾는다.
RAW = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "sprint_ai_project1_data"
)


def load_annotations(raw=RAW):
    """이미지 이름 → [박스 딕셔너리, ...]

    JSON이 (이미지 × 알약)당 1개씩이라 이미지 이름으로 묶어야
    한 장에 3~4개인 박스가 모인다.
    """
    # 없는 키에 접근하면 빈 리스트를 자동 생성 → append 전에 확인할 필요 없음
    by_image = defaultdict(list)

    # ** 는 하위 폴더 전부 훑기. K-<조합>_json/K-<약품코드>/*.json 구조라 2단계 깊이다.
    for f in raw.glob("train_annotations/**/*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))

        # images / annotations / categories 전부 원소가 딱 1개인 COCO 형식
        img = data["images"][0]
        x, y, w, h = data["annotations"][0][
            "bbox"
        ]  # COCO 표준: 좌상단 x, y, 너비, 높이
        cat = data["categories"][0]

        # 파일명이 키. 각도(_70_/_75_/_90_)가 파일명에 있어서 다른 각도끼리 안 섞인다.
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

    # defaultdict 그대로 반환하면 오타난 키를 조회해도 빈 리스트가 나와서 버그를 놓친다
    return dict(by_image)
