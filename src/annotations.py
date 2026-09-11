import json
from pathlib import Path


RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR = RAW / "sprint_ai_project1_data"
ANNOTATION_DIR = DATA_DIR / "train_annotations"


def load_annotations():

    result = {}

    for file in ANNOTATION_DIR.rglob("*.json"):

        data = json.loads(
            file.read_text(encoding="utf-8")
        )

        categories = {
            c["id"]: c["name"]
            for c in data["categories"]
        }

        images = {
            img["id"]: img["file_name"]
            for img in data["images"]
        }

        for ann in data["annotations"]:

            image_name = images[ann["image_id"]]

            x, y, w, h = ann["bbox"]

            result.setdefault(
                image_name,
                []
            ).append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "class_id": ann["category_id"],
                    "class_name": categories[ann["category_id"]],
                }
            )

    return result


if __name__ == "__main__":

    ann = load_annotations()

    print(f"이미지 개수: {len(ann)}")

    name = next(iter(ann))

    print("\n예시:")
    print(name)
    print(ann[name])