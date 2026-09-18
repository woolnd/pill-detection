from pathlib import Path

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

VAL_IMAGES = ROOT / "data" / "yolo" / "images" / "val"
VAL_LABELS = ROOT / "data" / "yolo" / "labels" / "val"


# ===== 설정 =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_CONF = 0.25
YOLO_IMGSZ = 960
MATCH_IOU = 0.5


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
    checkpoint["model_state_dict"]
)

classifier = classifier.to(DEVICE)
classifier.eval()


# ===== YOLO 로드 =====
yolo = YOLO(YOLO_WEIGHTS)


def read_ground_truth(label_path, width, height):
    """YOLO label을 실제 xyxy 좌표로 변환"""
    ground_truth = []

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    for line in lines:
        values = line.split()

        if len(values) != 5:
            continue

        class_id = int(values[0])

        x_center = float(values[1]) * width
        y_center = float(values[2]) * height
        box_width = float(values[3]) * width
        box_height = float(values[4]) * height

        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2
        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2

        ground_truth.append({
            "class_id": class_id,
            "box": [x1, y1, x2, y2],
        })

    return ground_truth


def calculate_iou(box1, box2):
    """xyxy 기준 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width
        * intersection_height
    )

    area1 = max(0, box1[2] - box1[0]) * max(
        0, box1[3] - box1[1]
    )

    area2 = max(0, box2[2] - box2[0]) * max(
        0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def classify_crop(crop):
    image = transform(crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = classifier(image)
        prediction = output.argmax(dim=1).item()

    return int(classes[prediction].replace("class_", ""))


def main():
    print()
    print("===== YOLO vs 2-Stage 비교 =====")
    print(f"YOLO: {YOLO_WEIGHTS}")
    print(f"Classifier: {CLASSIFIER_WEIGHTS}")
    print(f"Device: {DEVICE}")
    print(f"YOLO confidence: {YOLO_CONF}")
    print(f"Match IoU: {MATCH_IOU}")
    print()

    total_gt = 0
    matched = 0

    yolo_correct = 0
    classifier_correct = 0

    yolo_wrong_classifier_correct = 0
    yolo_correct_classifier_wrong = 0

    both_wrong = 0

    results = yolo.predict(
        source=str(VAL_IMAGES),
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        device=0 if DEVICE == "cuda" else DEVICE,
        verbose=False,
    )

    for result in results:
        image_path = Path(result.path)
        label_path = VAL_LABELS / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")

        ground_truth = read_ground_truth(
            label_path,
            image.width,
            image.height,
        )

        total_gt += len(ground_truth)

        if result.boxes is None:
            continue

        predictions = []

        for box, cls, conf in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
        ):
            predictions.append({
                "box": box,
                "class_id": int(cls),
                "confidence": float(conf),
            })

        # confidence 높은 순서대로 매칭
        predictions.sort(
            key=lambda x: x["confidence"],
            reverse=True,
        )

        used_gt = set()

        for prediction in predictions:
            best_iou = 0.0
            best_gt_index = None

            for gt_index, gt in enumerate(ground_truth):
                if gt_index in used_gt:
                    continue

                iou = calculate_iou(
                    prediction["box"],
                    gt["box"],
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

            if (
                best_gt_index is None
                or best_iou < MATCH_IOU
            ):
                continue

            used_gt.add(best_gt_index)
            matched += 1

            gt_class = ground_truth[
                best_gt_index
            ]["class_id"]

            yolo_class = prediction["class_id"]

            # YOLO BBox로 crop
            x1, y1, x2, y2 = map(
                int,
                prediction["box"],
            )

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop(
                (x1, y1, x2, y2)
            )

            classifier_class = classify_crop(crop)

            yolo_is_correct = (
                yolo_class == gt_class
            )

            classifier_is_correct = (
                classifier_class == gt_class
            )

            if yolo_is_correct:
                yolo_correct += 1

            if classifier_is_correct:
                classifier_correct += 1

            if (
                not yolo_is_correct
                and classifier_is_correct
            ):
                yolo_wrong_classifier_correct += 1

            elif (
                yolo_is_correct
                and not classifier_is_correct
            ):
                yolo_correct_classifier_wrong += 1

            elif (
                not yolo_is_correct
                and not classifier_is_correct
            ):
                both_wrong += 1

    print()
    print("===== 결과 =====")
    print(f"전체 GT: {total_gt}")
    print(f"YOLO와 매칭된 BBox: {matched}")
    print()

    if matched > 0:
        print(
            f"YOLO class 정확도: "
            f"{yolo_correct / matched:.4f}"
        )

        print(
            f"Classifier 정확도: "
            f"{classifier_correct / matched:.4f}"
        )

        print()
        print(
            "YOLO 오답 → Classifier 정답:",
            yolo_wrong_classifier_correct,
        )

        print(
            "YOLO 정답 → Classifier 오답:",
            yolo_correct_classifier_wrong,
        )

        print(
            "둘 다 오답:",
            both_wrong,
        )

    print()


if __name__ == "__main__":
    main()