import json
from pathlib import Path  
from collections import defaultdict  

# 경로설정
RAW = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "sprint_ai_project1_data"
)


def load_annotations(raw=RAW):
    """
    이미지이름 ->[박스 딕셔너리,...]
    JSON이 (이미지 x 알약)당 1개씩이라 이미지 이름으로 묶어야
    한장에 3~4개인 박스가 모인다
    """
    
    by_image = defaultdict(list)
    
    # **는 하위 폴더 전부 훓기, K-<조합>_json/K-<약품코드>/*.json 구조라 2단계 깊이다.
    for f in raw.glob("train_annotations/**/*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))  # 한글데이터 읽기위함
        
        #images / annotations / categories 전부원소가 딱 1개만 COCO형식
        img = data["images"][0]
        x, y, w, h = data["annotations"][0]["bbox"]#COCO표준 : 최상단 x,y,너비,높이
        cat = data["categories"][0]
        #파일명이 키,각도(7-,75,90)가 파일명에 있어 다른각도까지 안섞인다
        by_image[img["file_name"]].append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "class_id": cat["id"],
                "class_name": cat["name"]
            }
        )
    #defaultdict를 그대로 반환하면 오타난 키를 조회해도 빈 리스트가 나와서 버그를 놓친다.
    return dict(by_image)

