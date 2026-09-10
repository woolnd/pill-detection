import json
from pathlib import Path


# 프로젝트의 data/raw 경로
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# 실제 데이터 폴더
DATA_DIR = RAW / "sprint_ai_project1_data"

# JSON 어노테이션 폴더
ANNOTATION_DIR = DATA_DIR / "train_annotations"


def load_annotations():
    """
    모든 JSON 파일을 읽어서

    {
        "이미지파일명.png": [
            {
                "x": ...,
                "y": ...,
                "w": ...,
                "h": ...,
                "class_id": ...,
                "class_name": ...
            }
        ]
    }

    형태로 반환합니다.
    """

    result = {}

    # 모든 JSON 파일 찾기
    json_files = list(
        ANNOTATION_DIR.rglob("*.json")
    )

    print(
        f"JSON 파일 개수: {len(json_files)}"
    )

    for json_file in json_files:

        try:
            # UTF-8로 JSON 읽기
            data = json.loads(
                json_file.read_text(
                    encoding="utf-8"
                )
            )

        except Exception as e:

            print(
                "JSON 읽기 실패:",
                json_file
            )

            continue

        # images 정보가 없으면 건너뛰기
        if "images" not in data:
            continue

        # annotations 정보가 없으면 건너뛰기
        if "annotations" not in data:
            continue

        # categories 정보가 없으면 건너뛰기
        if "categories" not in data:
            continue

        # 이미지 정보
        images = data["images"]

        # 카테고리 정보
        categories = data["categories"]

        # category_id -> 알약 이름
        category_map = {}

        for category in categories:

            category_map[
                category["id"]
            ] = category["name"]

        # 이미지별 이름 저장
        image_map = {}

        for image in images:

            image_map[
                image["id"]
            ] = image["file_name"]

        # annotation 처리
        for annotation in data["annotations"]:

            # 이미지 ID
            image_id = annotation[
                "image_id"
            ]

            # 이미지 이름
            if image_id not in image_map:
                continue

            image_name = image_map[
                image_id
            ]

            # bbox
            bbox = annotation[
                "bbox"
            ]

            x, y, w, h = bbox

            # 카테고리 ID
            category_id = annotation[
                "category_id"
            ]

            # 알약 이름
            class_name = category_map.get(
                category_id,
                "알 수 없는 알약"
            )

            # 이미지가 아직 result에 없으면 생성
            if image_name not in result:

                result[
                    image_name
                ] = []

            # 박스 정보 저장
            result[
                image_name
            ].append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "class_id": category_id,
                    "class_name": class_name,
                }
            )

    return result


if __name__ == "__main__":

    annotations = load_annotations()

    print(
        f"\n어노테이션 이미지 개수: "
        f"{len(annotations)}"
    )

    # 예시 하나 출력
    if annotations:

        first_name = next(
            iter(annotations)
        )

        print(
            "\n예시 이미지:"
        )

        print(
            first_name
        )

        print(
            annotations[first_name]
        )