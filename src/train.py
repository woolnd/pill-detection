"""YOLO 학습 + 대회 지표(mAP[0.75:0.95]) 확인

실행: uv run python src/train.py   (make_yolo.py 를 먼저 실행해야 한다)
결과: runs/<NAME>/weights/best.pt
      runs/<NAME>/args.yaml      이번 실험에 실제로 쓰인 설정값 전체
      runs/<NAME>/results.csv    epoch 별 loss, mAP, 학습률(lr/pg0)
"""

from pathlib import Path

import torch
from ultralytics import YOLO

# ===== 경로 =====
ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "yolo" / "data.yaml"  # make_yolo.py 결과
RUNS = ROOT / "runs"  # 학습 결과 저장 폴더

# ===== 학습 설정 =====
MODEL = "yolo26n.pt"  # 사전학습 모델. 크게: "yolo26s.pt", "yolo26m.pt" (처음 실행하면 자동 다운로드)
EPOCHS = 100  # 데이터 전체를 몇 번 반복할지
IMGSZ = 640  # 학습할 때 이미지 크기. 원본이 976x1280 이라 960, 1280 도 해볼 만함
BATCH = 16  # 한 번에 넣을 이미지 수 (IMGSZ 올리다 메모리 부족하면 8, 4 로 줄이기)
SEED = 42  # 랜덤 고정
NAME = "baseline"  # 실험 이름 (runs/ 아래 폴더 이름). 실험마다 바꿔야 결과가 안 덮인다

# ===== 실험용 설정 =====
# 비워두면 ultralytics 기본값으로 학습한다 (= 베이스라인).
# 바꾸고 싶은 값만 "이름": 값 으로 넣는다. 한 번에 하나씩 바꾸고 NAME 도 같이 바꾼다.
#   예) NAME = "lr0.001"
#       EXPERIMENT = {"optimizer": "AdamW", "lr0": 0.001}
#
# IMGSZ, BATCH, EPOCHS, MODEL 은 여기 넣지 말고 위 상수에서 바꾼다. (같이 넣으면 에러)
#
# 넣을 수 있는 값 (오른쪽 숫자는 기본값)
# ----- 박스 정확도 / 클래스 -----
#   "box": 7.5            박스 위치 loss 비중. 10, 15 (대회 지표가 IoU 0.75 이상이라 중요)
#   "cls": 0.5            클래스 분류 loss 비중
#   "cls_pw": 0.0         클래스 불균형 보정. 0.5, 1.0 (박스 적은 클래스에 가중치)
# ----- 학습률 -----
#   "optimizer": "auto"   "AdamW", "SGD". auto 면 lr0, momentum 을 넣어도 무시된다!
#   "lr0": 0.01           시작 학습률. AdamW 는 0.001 근처, SGD 는 0.01 근처
#   "lrf": 0.01           마지막 학습률 = lr0 x lrf
#   "cos_lr": False       True 면 학습률을 코사인 곡선으로 줄인다 (False 는 직선)
#   "warmup_epochs": 3.0  처음 N epoch 동안 학습률을 서서히 올린다
#   "momentum": 0.937     SGD momentum / Adam beta1 (auto 면 무시)
#   "weight_decay": 0.0005  가중치가 너무 커지지 않게 (과적합 방지)
# ----- 증강 -----
#   "hsv_h": 0.015        색조 변화. 알약 색이 단서라 0.0 해보기
#   "fliplr": 0.5         좌우 반전 확률 (각인 글자가 뒤집힌다). 0.0
#   "flipud": 0.0         상하 반전 확률. 0.5
#   "degrees": 0.0        회전 각도. 10~30 (회전하면 박스가 헐거워질 수 있음)
#   "scale": 0.5          크기 변화. 0.2
#   "mosaic": 1.0         4장 이어붙이기 확률. 0.5
#   "close_mosaic": 10    마지막 N epoch 은 mosaic 끄기. 20
# ----- 기타 -----
#   "patience": 100       N epoch 동안 val 점수가 안 오르면 멈춤. 30
#   "freeze": None        앞쪽 N층 고정 (데이터 적을 때 과적합 방지). 10
EXPERIMENT = {}


def get_device():
    """학습에 쓸 장치 고르기

    입력: 없음

    반환:
        str: "cuda" (NVIDIA GPU) / "mps" (맥 GPU) / "cpu"

    동작:
        1. NVIDIA GPU 가 있으면 "cuda"
        2. 맥 GPU(mps) 가 있으면 "mps"
        3. 둘 다 없으면 "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(device):
    """사전학습 모델을 우리 데이터로 학습

    입력:
        device (str): get_device() 결과

    반환:
        Path: 가장 좋은 가중치 파일 경로
              예) runs/baseline/weights/best.pt

    동작:
        1. 사전학습 모델을 불러온다.
        2. data.yaml, 기본 설정값, EXPERIMENT 값으로 학습한다.
           (**EXPERIMENT 는 딕셔너리를 "이름=값" 인자로 풀어서 넘긴다. 비어 있으면 아무것도 안 넘김)
        3. 학습 중 val 점수가 가장 좋았던 best.pt 경로를 반환한다.
    """

    model = YOLO(MODEL)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        seed=SEED,
        deterministic=True,
        project=str(RUNS),
        name=NAME,
        exist_ok=True,
        **EXPERIMENT,
    )

    return Path(model.trainer.best)


def evaluate(weights, device):
    """val 데이터로 평가하고 대회 지표 계산

    입력:
        weights (Path): best.pt 경로
        device (str): get_device() 결과

    반환:
        dict: {"mAP50": float, "mAP50-95": float, "mAP75-95": float}
              mAP75-95 가 대회 지표와 같은 구간이다. (채점 방식이 달라 참고용)

    동작:
        1. best.pt 를 불러온다.
        2. val 데이터로 평가한다.
        3. all_ap (클래스 수 x IoU 10개) 에서 IoU 0.75~0.95 칸만 평균 낸다.
           IoU 10개 = 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
                                                    └── 5번 칸부터 ──┘
    """

    model = YOLO(weights)

    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        split="val",
        project=str(RUNS),
        name=f"{NAME}_val",
        exist_ok=True,
    )

    ap = metrics.box.all_ap
    return {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "mAP75-95": float(ap[:, 5].mean()),
    }


def main():
    """전체 순서

    동작:
        1. 장치 선택, 이번 실험 이름과 설정 출력
        2. 학습
        3. best.pt 로 val 평가 후 점수 출력
    """
    # 1. 장치, 실험 정보
    device = get_device()
    print("장치:", device)
    print("실험 이름:", NAME)
    print("실험 설정:", EXPERIMENT if EXPERIMENT else "기본값")

    # 2. 학습
    weights = train(device)
    print("best.pt:", weights)

    # 3. 평가
    scores = evaluate(weights, device)
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
