# """Faster R-CNN / RetinaNet 학습 + val 로컬 mAP 측정 (YOLO 비교용)
#
# 실행: uv run python -m src.train.train_rcnn   (make_yolo.py 를 먼저 실행해야 한다)
# 결과: runs/<NAME>/model.pt   학습된 가중치 (val mAP 는 화면에 출력)
# """
#
# import random
#
# import numpy as np
# import torch
# import yaml
# from PIL import Image
# from torchvision.models.detection import (
#     fasterrcnn_resnet50_fpn_v2,
#     retinanet_resnet50_fpn_v2,
# )
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.retinanet import RetinaNetClassificationHead
# from torchvision.transforms.functional import to_tensor
# from ultralytics.models.yolo.detect import DetectionValidator
# from ultralytics.utils.metrics import ap_per_class, box_iou
#
# from src.config import DATA_YAML, RUNS_DIR, SEED, YOLO_DIR, get_device
#
# # ===== 학습 설정 =====
# MODEL_TYPE = "fasterrcnn"  # "fasterrcnn" 또는 "retinanet"
# EPOCHS = 50
# IMGSZ = 640  # 긴 변을 이 크기로 맞춘다 (976x1280 -> 488x640, YOLO imgsz 와 같은 방식)
# BATCH = 4  # 16GB 맥에서 16, 8 은 메모리 부족(swap)으로 느려짐
# LR = 0.0001  # AdamW 학습률
# NAME = f"jw_{MODEL_TYPE}_ep{EPOCHS}_img{IMGSZ}_b{BATCH}"  # 실험마다 바꿔야 결과가 안 덮인다 (규칙: 이니셜_모델_ep_img_b_바꾼점)
# CONF = 0.001  # 이 신뢰도 미만 박스는 버린다 (predict.py 와 같은 값)
#
#
# def load_names():
#     """data.yaml 에서 클래스 이름 목록을 읽는다
#
#     입력: 없음
#
#     반환:
#         dict: YOLO 번호 -> 약 ID 문자열
#               예) {0: '1900', 1: '2483', ...}
#
#     동작:
#         1. data.yaml 을 읽어서 names 만 꺼낸다.
#     """
#     with open(DATA_YAML, encoding="utf-8") as f:
#         return yaml.safe_load(f)["names"]
#
#
# class PillDataset(torch.utils.data.Dataset):
#     """data/yolo 의 이미지 + 라벨을 한 장씩 꺼내주는 데이터셋
#
#     입력:
#         split (str): "train" 또는 "val"
#     """
#
#     def __init__(self, split):
#         self.image_paths = sorted((YOLO_DIR / "images" / split).glob("*.png"))
#         self.label_dir = YOLO_DIR / "labels" / split
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, i):
#         """i 번째 (이미지 텐서, 정답 dict) 를 돌려준다
#
#         입력:
#             i (int): 몇 번째 이미지인지 (DataLoader 가 넘겨준다)
#
#         반환:
#             tuple: (이미지 텐서 (3, 높이, 너비), read_label() 결과 dict)
#
#         동작:
#             1. 이미지를 RGB 로 열어서 0~1 텐서로 바꾼다.
#             2. 같은 이름의 라벨 txt 를 read_label() 로 읽는다.
#         """
#         # 1. 이미지
#         path = self.image_paths[i]
#         image = Image.open(path).convert("RGB")
#
#         # 2. 정답
#         label_path = self.label_dir / (path.stem + ".txt")
#         target = read_label(label_path, image.width, image.height)
#
#         return to_tensor(image), target
#
#
# def read_label(label_path, img_w, img_h):
#     """YOLO 라벨 txt 1개를 torchvision 형식의 정답으로 바꾼다
#
#     입력:
#         label_path (Path): 라벨 txt 경로
#         img_w (int): 이미지 너비  예) 976
#         img_h (int): 이미지 높이  예) 1280
#
#     반환:
#         dict: {"boxes": 박스 텐서 (N, 4), "labels": 클래스 텐서 (N,)}
#               예) {"boxes": tensor([[144.0, 799.0, 383.0, 1038.0], ...]), "labels": tensor([16, ...])}
#
#     동작:
#         1. 한 줄씩 읽는다. 한 줄 = "클래스 중심x 중심y 너비 높이" (0~1 비율)
#         2. 비율에 이미지 크기를 곱해 픽셀로 바꾼다.
#         3. 중심 + 너비/높이를 왼쪽 위(x1, y1) + 오른쪽 아래(x2, y2) 로 바꾼다.
#         4. 클래스 번호에 1 을 더한다. (torchvision 은 0 번을 배경으로 쓴다)
#     """
#     boxes = []
#     labels = []
#
#     # 1. 한 줄씩
#     for line in label_path.read_text().splitlines():
#         cls, cx, cy, w, h = line.split()
#
#         # 2. 픽셀로
#         cx = float(cx) * img_w
#         cy = float(cy) * img_h
#         w = float(w) * img_w
#         h = float(h) * img_h
#
#         # 3. 중심 -> 모서리
#         boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
#
#         # 4. 배경 자리 비우기
#         labels.append(int(cls) + 1)
#
#     return {
#         "boxes": torch.tensor(boxes, dtype=torch.float32),
#         "labels": torch.tensor(labels, dtype=torch.int64),
#     }
#
#
# def collate(batch):
#     """배치를 (이미지 리스트, 정답 리스트) 로 묶는다
#
#     입력:
#         batch (list): [(이미지, 정답), (이미지, 정답), ...]
#
#     반환:
#         tuple: ([이미지, 이미지, ...], [정답, 정답, ...])
#
#     동작:
#         1. 이미지와 정답을 각각 리스트에 모은다.
#            (박스 개수가 이미지마다 달라서 기본 방식으로는 하나의 텐서로 못 쌓는다)
#     """
#     images = []
#     targets = []
#     for image, target in batch:
#         images.append(image)
#         targets.append(target)
#     return images, targets
#
#
# def make_model(model_type, num_classes):
#     """COCO 사전학습 모델을 불러와서 클래스 수만 바꾼다
#
#     입력:
#         model_type (str): "fasterrcnn" 또는 "retinanet"
#         num_classes (int): 배경 포함 클래스 수  예) 57
#
#     반환:
#         nn.Module: 학습할 모델
#
#     동작:
#         1. fasterrcnn:
#             a. 사전학습 모델을 불러온다. (box_score_thresh = 이 신뢰도 미만 박스는 버림, min_size = 입력 크기)
#             b. 마지막 분류기(box_predictor)를 우리 클래스 수로 새로 만든다.
#         2. retinanet:
#             a. 사전학습 모델을 불러온다.
#             b. 분류 head 를 우리 클래스 수로 새로 만든다.
#     """
#     # 1. Faster R-CNN
#     if model_type == "fasterrcnn":
#         model = fasterrcnn_resnet50_fpn_v2(
#             weights="DEFAULT", box_score_thresh=CONF, min_size=IMGSZ, max_size=IMGSZ
#         )
#         in_features = model.roi_heads.box_predictor.cls_score.in_features
#         model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#         return model
#
#     # 2. RetinaNet
#     model = retinanet_resnet50_fpn_v2(
#         weights="DEFAULT", score_thresh=CONF, min_size=IMGSZ, max_size=IMGSZ
#     )
#     num_anchors = model.head.classification_head.num_anchors
#     model.head.classification_head = RetinaNetClassificationHead(
#         256, num_anchors, num_classes  # 256 = FPN 출력 채널 수
#     )
#     return model
#
#
# def train_one_epoch(model, loader, optimizer, device):
#     """1 epoch 학습
#
#     입력:
#         model (nn.Module): make_model() 결과
#         loader (DataLoader): train 데이터
#         optimizer (Optimizer): AdamW
#         device (str): get_device() 결과
#
#     반환:
#         float: 이번 epoch 평균 loss
#
#     동작:
#         1. 배치를 장치로 옮긴다.
#         2. 학습 모드에서는 모델이 loss 들을 dict 로 돌려준다. 전부 더한다.
#         3. 역전파 후 가중치를 업데이트한다.
#     """
#     model.train()
#     total = 0.0
#
#     for images, targets in loader:
#         # 1. 장치로
#         images = [image.to(device) for image in images]
#         targets = [
#             {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)}
#             for t in targets
#         ]
#
#         # 2. loss 합
#         loss_dict = model(images, targets)
#         loss = sum(loss_dict.values())
#
#         # 3. 업데이트
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         total += loss.item()
#
#     return total / len(loader)
#
#
# @torch.no_grad()
# def evaluate(model, dataset, device):
#     """val 데이터로 로컬 mAP 계산 (YOLO 의 model.val() 과 같은 방식)
#
#     입력:
#         model (nn.Module): 학습된 모델
#         dataset (PillDataset): val 데이터셋
#         device (str): get_device() 결과
#
#     반환:
#         dict: {"mAP50": float, "mAP50-95": float, "mAP75-95": float}
#
#     동작:
#         1. 평가 모드로 바꾸고, 짝짓기용 YOLO 평가 도구를 만든다.
#         2. val 이미지마다
#             a. 예측한다.
#             b. score_image() 로 맞았는지 표시한다.
#             c. 결과를 stats 에 모은다.
#         3. compute_map() 으로 mAP 를 계산한다.
#     """
#     # 1. 준비
#     model.eval()
#     validator = DetectionValidator()
#     stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
#
#     # 2. 이미지마다
#     for i in range(len(dataset)):
#         image, target = dataset[i]
#         pred = model([image.to(device)])[0]  # 2-a.
#         pred = {
#             "boxes": pred["boxes"].cpu(),
#             "labels": pred["labels"].cpu(),
#             "scores": pred["scores"].cpu(),
#         }
#
#         stats["tp"].append(score_image(pred, target, validator))  # 2-b.
#         stats["conf"].append(pred["scores"].numpy())  # 2-c.
#         stats["pred_cls"].append(pred["labels"].numpy())
#         stats["target_cls"].append(target["labels"].numpy())
#
#     # 3. mAP
#     return compute_map(stats)
#
#
# def score_image(pred, target, validator):
#     """이미지 1장의 예측을 정답과 비교해서 IoU 기준 10개마다 맞았는지 표시한다
#
#     입력:
#         pred (dict): 모델 예측 {"boxes": (N, 4), "labels": (N,), "scores": (N,)}
#         target (dict): 정답 {"boxes": (M, 4), "labels": (M,)}
#         validator (DetectionValidator): YOLO 평가 도구 (짝짓기 함수만 빌려 쓴다)
#
#     반환:
#         numpy.ndarray: (N, 10) True/False. 예측 i 가 IoU 기준 j 에서 맞았으면 [i, j] = True
#
#     동작:
#         1. 예측이나 정답이 없으면 전부 False
#         2. 정답 x 예측 IoU 표를 만든다.
#         3. YOLO 와 똑같은 규칙으로 짝짓는다. (같은 클래스 + IoU 높은 순으로 1:1)
#     """
#     # 1. 비교할 게 없음
#     if len(pred["boxes"]) == 0 or len(target["boxes"]) == 0:
#         return np.zeros((len(pred["boxes"]), 10), dtype=bool)
#
#     # 2. IoU 표 (정답 M x 예측 N)
#     iou = box_iou(target["boxes"], pred["boxes"])
#
#     # 3. 짝짓기
#     return validator.match_predictions(pred["labels"], target["labels"], iou).numpy()
#
#
# def compute_map(stats):
#     """모은 결과로 mAP 3종을 계산한다 (train_yolo.py 의 evaluate 와 같은 키)
#
#     입력:
#         stats (dict): {"tp": [...], "conf": [...], "pred_cls": [...], "target_cls": [...]}
#                       이미지마다 score_image() 결과와 점수, 클래스를 모은 리스트
#
#     반환:
#         dict: {"mAP50": float, "mAP50-95": float, "mAP75-95": float}
#
#     동작:
#         1. 이미지별 리스트를 하나의 배열로 이어붙인다.
#         2. ap_per_class() 로 클래스 x IoU 10개 AP 표를 만든다. (YOLO 가 쓰는 함수 그대로)
#         3. IoU 0.50 칸, 전체, 0.75~0.95 칸(5번부터)을 각각 평균 낸다.
#     """
#     # 1. 이어붙이기
#     tp = np.concatenate(stats["tp"])
#     conf = np.concatenate(stats["conf"])
#     pred_cls = np.concatenate(stats["pred_cls"])
#     target_cls = np.concatenate(stats["target_cls"])
#
#     # 2. AP 표 (반환값 6번째가 AP)
#     ap = ap_per_class(tp, conf, pred_cls, target_cls)[5]
#
#     # 3. 구간별 평균
#     return {
#         "mAP50": float(ap[:, 0].mean()),
#         "mAP50-95": float(ap.mean()),
#         "mAP75-95": float(ap[:, 5:].mean()),
#     }
#
#
# def main():
#     """전체 순서
#
#     입력: 없음
#
#     반환:
#         없음. runs/<NAME>/model.pt 를 저장하고 val mAP 를 출력한다.
#
#     동작:
#         1. 시드 고정, 장치 선택
#         2. 데이터 준비 (YOLO 와 같은 train 분할)
#         3. 모델 + optimizer 준비
#         4. epoch 마다 학습하고 loss 출력
#         5. 가중치 저장
#         6. val 로컬 mAP 출력
#     """
#     # 1. 시드, 장치
#     random.seed(SEED)
#     torch.manual_seed(SEED)
#     device = get_device()
#     out_dir = RUNS_DIR / NAME
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     # 2. 데이터
#     names = load_names()
#     train_loader = torch.utils.data.DataLoader(
#         PillDataset("train"), batch_size=BATCH, shuffle=True, collate_fn=collate
#     )
#
#     # 3. 모델
#     model = make_model(MODEL_TYPE, len(names) + 1).to(device)  # +1 = 배경
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
#
#     # 4. 학습
#     for epoch in range(1, EPOCHS + 1):
#         train = train_one_epoch(model, train_loader, optimizer, device)
#         print(f"epoch {epoch}/{EPOCHS}  train loss {train:.4f}")
#
#     # 5. 저장
#     torch.save(model.state_dict(), out_dir / "model.pt")
#
#     # 6. 로컬 mAP
#     scores = evaluate(model, PillDataset("val"), device)
#     print("val mAP:", scores)
#
#
# if __name__ == "__main__":
#     main()
