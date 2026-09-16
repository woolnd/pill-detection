"""YOLO 학습 + 대회 지표(mAP[0.75:0.95]) 확인

실행: uv run python src/train.py   (make_yolo.py 를 먼저 실행해야 한다)
결과: runs/<NAME>/weights/best.pt
      runs/<NAME>/args.yaml      이번 실험에 실제로 쓰인 설정값 전체
      runs/<NAME>/results.csv    epoch 별 loss, mAP, 학습률(lr/pg0)
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO

from sheet import post_to_sheet
from ultralytics.utils.metrics import DetMetrics

# ===== 경로 =====
ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "yolo" / "data.yaml"  # make_yolo.py 결과
RUNS = ROOT / "runs"  # 학습 결과 저장 폴더

# ===== 학습 설정 =====
MODEL = "yolo26n.pt"  # 사전학습 모델. 크게: "yolo26s.pt", "yolo26m.pt" (처음 실행하면 자동 다운로드)
EPOCHS = 100  # 데이터 전체를 몇 번 반복할지
IMGSZ = 960  # 학습할 때 이미지 크기. 원본이 976x1280 이라 960, 1280 도 해볼 만함
BATCH = 8  # 한 번에 넣을 이미지 수 (IMGSZ 올리다 메모리 부족하면 8, 4 로 줄이기)
SEED = 42  # 랜덤 고정
NAME = "jw_ep100_img960_batch8_lr0.001_clspw0.5"  # 실험 이름 (runs/ 아래 폴더 이름). 실험마다 바꿔야 결과가 안 덮인다
MEMO = "epochs: 100 / imgsz: 960 / batch: 8 / lr: 0.001 / cls_pw 0.5 (희소 클래스 17종이 train 박스 2~3개)"  # 이번 실험에서 무엇을 왜 바꿨는지 한 줄 (시트·csv 에 기록)

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
EXPERIMENT = {"optimizer": "AdamW", "lr0": 0.001}


def get_device():
    """학습에 쓸 장치 고르기

    입력: 없음

    반환:
        str: "0" (NVIDIA GPU) / "mps" (맥 GPU) / "cpu"

    동작:
        1. NVIDIA GPU 가 있으면 "0"
        2. 맥 GPU(mps) 가 있으면 "mps"
        3. 둘 다 없으면 "cpu"
    """
    if torch.cuda.is_available():
        return "0"

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_train_args(device):
    """model.train() 에 넘길 기본 인자를 딕셔너리로 만든다

    입력:
        device (str): get_device() 결과

    반환:
        dict: 인자 이름 -> 값
              예) {"data": ".../data.yaml", "epochs": 50, "imgsz": 640, "batch": 16, "device": "mps",
                   "seed": 42, "deterministic": True, "project": ".../runs", "name": "baseline", "exist_ok": True}

    동작:
        1. 위 설정값(상수)으로 딕셔너리를 만든다.
           학습(train)과 기록(save_scores)이 이 딕셔너리 하나를 같이 써서, 기록된 값 = 실제 학습한 값이 된다.
        (EXPERIMENT 는 여기 넣지 않고 따로 넘긴다. 같은 이름이 겹치면 에러가 나서 실수를 막아준다)
    """
    return {
        "data": str(DATA_YAML),
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "device": device,
        "seed": SEED,
        "deterministic": True,  # 같은 설정이면 같은 결과
        "project": str(RUNS),
        "name": NAME,
        "exist_ok": True,  # 같은 이름 폴더가 있으면 덮어쓰기
    }


def train(args):
    """사전학습 모델을 우리 데이터로 학습

    입력:
        args (dict): make_train_args() 결과

    반환:
        Path: 가장 좋은 가중치 파일 경로
              예) runs/baseline/weights/best.pt

    동작:
        1. 사전학습 모델을 불러온다.
        2. 기본 인자(args)와 EXPERIMENT 값으로 학습한다.
           (**딕셔너리 는 딕셔너리를 "이름=값" 인자로 풀어서 넘긴다)
        3. 학습 중 val 점수가 가장 좋았던 best.pt 경로를 반환한다.
    """
    # 1. 사전학습 모델
    model = YOLO(MODEL)

    # 2. 학습
    model.train(**args, **EXPERIMENT)

    # 3. best.pt 경로
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
        "mAP75-95": float(ap[:, 5:].mean()),
    }


# ===== best.pt 선택 기준 =====
def fitness75(self):
    """best.pt 를 고를 때 쓸 점수를 대회 지표(mAP75-95)로 계산한다

    입력:
        self (DetMetrics): ultralytics 가 매 epoch val 을 끝내고 넘겨주는 지표 객체

    반환:
        float: IoU 0.75~0.95 구간의 평균 AP. 아직 예측이 없으면 0.0

    동작:
        1. all_ap (클래스 수 x IoU 10칸) 가 비어 있으면 0.0 을 준다. (학습 초반)
        2. 5번 칸부터(= IoU 0.75, 0.80, 0.85, 0.90, 0.95) 만 평균을 낸다.

    ultralytics 기본값은 mAP50-95 라서, 대회가 점수를 주지 않는 IoU 0.50~0.70
    구간까지 보고 best.pt 를 고른다. patience 조기종료도 같은 값을 본다.
    """
    # 1. 아직 예측이 없을 때
    if not self.box.all_ap.size:
        return 0.0

    # 2. IoU 0.75~0.95 만 평균
    return float(self.box.all_ap[:, 5:].mean())


DetMetrics.fitness = property(fitness75)


def save_scores(scores, args):
    """이번 실험의 점수와 학습 인자를 csv 와 구글 시트에 기록한다

    입력:
        scores (dict): evaluate() 결과
                       예) {"mAP50": 0.7876, "mAP50-95": 0.7550, "mAP75-95": 0.7225}
        args (dict): make_train_args() 결과 (학습에 실제로 넘긴 인자)

    반환:
        Path: csv 기록 파일 경로  예) runs/experiments.csv

    동작:
        1. 시간, 메모, 점수, 모델 이름, 실험 설정(글자)으로 한 줄을 시작한다.
        2. 학습 인자(args)와 실험용 인자(EXPERIMENT)를 뒤에 붙인다.
        3. 기존 csv 가 있으면 읽어서 아래에 합친다. (열이 달라도 concat 이 맞춰준다. 빈칸 = 기본값)
        4. csv 를 저장한다.
        5. 같은 줄을 구글 시트에도 보낸다.
           (시트는 정해진 열만 받는다. 인터넷 문제로 실패해도 csv 는 이미 저장돼서 기록이 사라지지 않는다)
    """
    path = RUNS / "experiments.csv"

    # 1. 기본 정보 + 점수
    row = {
        "time": datetime.now().strftime("%m-%d %H:%M"),
        "author": os.environ.get("AUTHOR", ""),  # 각자 .env 의 AUTHOR (없으면 빈칸)
        "memo": MEMO,
        "mAP50": round(scores["mAP50"], 4),
        "mAP50-95": round(scores["mAP50-95"], 4),
        "mAP75-95": round(scores["mAP75-95"], 4),
        "model": MODEL,
        "experiment": str(EXPERIMENT),  # 시트용: 바꾼 값을 한 칸에
    }

    # 2. 학습 인자 붙이기
    row.update(args)  # model.train() 기본 인자 (epochs, imgsz, batch, name ...)
    row.update(EXPERIMENT)  # 실험용 인자 (lr0, box ...)

    # 3. 기존 기록과 합치기
    table = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path, encoding="utf-8-sig")
        table = pd.concat([old, table], ignore_index=True)

    # 4. csv 저장
    table.to_csv(path, index=False, encoding="utf-8-sig")

    # 5. 구글 시트 전송
    try:
        print("시트 전송:", post_to_sheet({"action": "append", "row": row}))
    except Exception as e:
        print("시트 전송 실패 (csv 에는 저장됨):", e)
    return path


def main():
    """전체 순서

    동작:
        1. 장치 선택, 학습 인자 만들기, 실험 정보 출력
        2. 학습
        3. best.pt 로 val 평가 후 점수 출력 + csv·구글 시트에 기록
    """
    # 1. 장치, 학습 인자, 실험 정보
    device = get_device()
    args = make_train_args(device)
    print("실험 이름:", NAME)
    print("메모:", MEMO)
    print("학습 인자:", args)
    print("실험 설정:", EXPERIMENT if EXPERIMENT else "기본값")

    # 2. 학습
    weights = train(args)
    print("best.pt:", weights)

    # 3. 평가 + 기록
    scores = evaluate(weights, device)
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")
    print("실험 기록:", save_scores(scores, args))


if __name__ == "__main__":
    main()
