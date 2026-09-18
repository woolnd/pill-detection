from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ===== 경로 =====
ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "data" / "classification_aug" / "train"
VAL_DIR = ROOT / "data" / "classification_aug" / "val"

RUN_DIR = ROOT / "runs" / "classifier"
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RUN_DIR / "best_classifier.pt"


# ===== 학습 설정 =====
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.0001


# ===== GPU =====
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print("사용 장치:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ===== 이미지 전처리 =====
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ===== 데이터셋 =====
train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform,
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform,
)

print("Train 이미지:", len(train_dataset))
print("Val 이미지:", len(val_dataset))
print("클래스 수:", len(train_dataset.classes))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ===== 모델 =====
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(train_dataset.classes),
)

model = model.to(DEVICE)


# ===== 손실 함수 / Optimizer =====
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.01,
)


# ===== 학습 =====
best_accuracy = 0.0


for epoch in range(EPOCHS):
    # -----------------
    # Train
    # -----------------
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss /= train_total
    train_accuracy = train_correct / train_total


    # -----------------
    # Validation
    # -----------------
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss /= val_total
    val_accuracy = val_correct / val_total


    print(
        f"[{epoch + 1:02d}/{EPOCHS}] "
        f"train_loss={train_loss:.4f} "
        f"train_acc={train_accuracy:.4f} "
        f"val_loss={val_loss:.4f} "
        f"val_acc={val_accuracy:.4f}"
    )


    # ===== Best 모델 저장 =====
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": train_dataset.classes,
                "accuracy": best_accuracy,
            },
            MODEL_PATH,
        )

        print(
            f"  → best 모델 저장: "
            f"{MODEL_PATH}"
        )


print()
print("===== 학습 완료 =====")
print(f"Best Val Accuracy: {best_accuracy:.4f}")
print(f"모델 저장 위치: {MODEL_PATH}")