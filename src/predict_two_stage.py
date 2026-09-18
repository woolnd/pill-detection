from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO


# ===== 경로 =====
ROOT = Path(__file__).resolve().parent.parent

YOLO_WEIGHTS = (
    ROOT
    / "runs"
    / "experiment_12_class_balance_s"
    / "weights"
    / "best.pt"
)

CLASSIFIER_WEIGHTS = (
    ROOT
    / "runs"
    / "classifier"
    / "best_classifier.pt"
)

RAW = ROOT / "data" / "raw" / "sprint_ai_project1_data"
TEST_DIR = RAW / "test_images"

OUTPUT = (
    ROOT
    / "runs"
    / "submission_two_stage.csv"
)


# ===== 설정 =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMGSZ = 960
CONF = 0.001


# ===== Classifier 전처리 =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ===== Classifier 로드 =====
checkpoint = torch.load(
    CLASSIFIER_WEIGHTS,
    map_location=DEVICE,
)

classes = checkpoint["classes"]

classifier = models.resnet18(weights=None)

classifier.fc = nn.Linear(
    classifier.fc.in_features,
    len(classes),
)

classifier.load_state_dict(
    checkpoint["model_state_dict"],
)

classifier = classifier.to(DEVICE)
classifier.eval()


# ===== YOLO 로드 =====
yolo = YOLO(YOLO_WEIGHTS)


def classify_crop(crop):
    image = transform(crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = classifier(image)
        probabilities = torch.softmax(output, dim=1)

        prediction = probabilities.argmax(dim=1).item()
        classifier_conf = probabilities[0, prediction].item()

    class_name = classes[prediction]
    class_id = int(
        class_name.replace("class_", "")
    )

    return class_id, classifier_conf


def main():
    print()
    print("===== 2-Stage Test Prediction =====")
    print(f"YOLO: {YOLO_WEIGHTS}")
    print(f"Classifier: {CLASSIFIER_WEIGHTS}")
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMGSZ}")
    print(f"YOLO confidence: {CONF}")
    print()

    rows = []
    annotation_id = 0

    image_files = sorted(TEST_DIR.glob("*.png"))

    print(f"Test 이미지: {len(image_files)}")

    results = yolo.predict(
        source=str(TEST_DIR),
        imgsz=IMGSZ,
        conf=CONF,
        device=0 if DEVICE == "cuda" else DEVICE,
        verbose=False,
    )

    for result in results:
        image_path = Path(result.path)

        image = Image.open(
            image_path
        ).convert("RGB")

        if result.boxes is None:
            continue

        boxes = result.boxes

        for box, yolo_class, yolo_conf in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.cls.cpu().tolist(),
            boxes.conf.cpu().tolist(),
        ):
            x1, y1, x2, y2 = box

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(image.width, int(x2))
            y2 = min(image.height, int(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop(
                (x1, y1, x2, y2)
            )

            classifier_class, classifier_conf = (
                classify_crop(crop)
            )

            image_id = int(
                image_path.stem
            )

            rows.append({
                "annotation_id": annotation_id,
                "image_id": image_id,
                "category_id": classifier_class,
                "bbox_x": x1,
                "bbox_y": y1,
                "bbox_w": x2 - x1,
                "bbox_h": y2 - y1,
                "score": float(yolo_conf),
            })

            annotation_id += 1

    df = pd.DataFrame(
        rows,
        columns=[
            "annotation_id",
            "image_id",
            "category_id",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "score",
        ],
    )

    df.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("===== 생성 완료 =====")
    print(f"예측 BBox: {len(df)}")
    print(f"예측 이미지: {df['image_id'].nunique()}")
    print(f"CSV: {OUTPUT}")
    print()
    print("category_id 분포:")
    print(df["category_id"].value_counts().sort_index())
    print()


if __name__ == "__main__":
    main()