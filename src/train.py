"""YOLO 학습 + 대회 지표(mAP[0.75:0.95]) 확인

실행: uv run python src/train.py   (make_yolo.py 를 먼저 실행해야 한다)
결과: runs/<NAME>/weights/best.pt
      runs/<NAME>/args.yaml      이번 실험에 실제로 쓰인 설정값 전체
      runs/<NAME>/results.csv    epoch 별 loss, mAP, 학습률(lr/pg0)
"""
from pathlib import Path

import torch
from ultralytics import YOLO
===== 경로 =====
ROOT = Path(file).resolve().parent.parent
DATA_YAML = ROOT / "data" / "yolo" / "data.yaml"   # make_yolo.py 결과
RUNS = ROOT / "runs"                               # 학습 결과 저장 폴더

===== 학습 설정 =====
MODEL = "yolo26n.pt"   # 사전학습 모델. 크게: "yolo26s.pt", "yolo26m.pt" (처음 실행하면 자동 다운로드)
EPOCHS = 100           # 데이터 전체를 몇 번 반복할지
IMGSZ = 640            # 학습할 때 이미지 크기. 원본이 976x1280 이라 960, 1280 도 해볼 만함
BATCH = 16             # 한 번에 넣을 이미지 수 (IMGSZ 올리다 메모리 부족하면 8, 4 로 줄이기)
SEED = 42              # 랜덤 고정
NAME = "baseline"      # 실험 이름 (runs/ 아래 폴더 이름). 실험마다 바꿔야 결과가 안 덮인다

===== 실험용 설정 =====
비워두면 ultralytics 기본값으로 학습한다 (= 베
바꾸고 싶은 값만 "이름": 값 으로 넣는다. 한 번에 하나씩 바꾸고 NAME 도 같이 바꾼다.
예) NAME = "lr0.001"
EXPERIMENT = {"optimizer": "AdamW", "lr0": 0.001}
#
IMGSZ, BATCH, EPOCHS, MODEL 은 여기 넣지 말고 위 상수에서 바꾼다. (같이 넣으면 에러)
#
넣을 수 있는 값 (오른쪽 숫자는 기본값)
----- 박스 정확도 / 클래스 -----
"box": 7.5            박스 위치 loss 비중. 10, 15 (대회 지표가 IoU 0.75 이상이라 중요)
"cls": 0.5