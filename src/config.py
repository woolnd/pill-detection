"""프로젝트 공통 경로 · 설정

사용: from src.config import KAGGLE_DIR, YOLO_DIR, RUNS_DIR, SEED, get_device
      (스크립트는 프로젝트 루트에서 uv run python -m src.폴더.파일 로 실행한다)
      모든 경로는 __file__ 기준이라 어느 폴더에서 실행하든 같은 곳을 가리킨다.
      (plot_experiments.py 는 GitHub Actions 에서 torch 없이 돌아서 이 파일을 쓰지 않는다)
"""

from pathlib import Path

import torch

# ===== 경로 =====
ROOT = Path(__file__).resolve().parent.parent  # 프로젝트 루트
RAW_DIR = ROOT / "data" / "raw"  # download_data.py 가 받는 위치
KAGGLE_DIR = (
    RAW_DIR / "sprint_ai_project1_data"
)  # train_images/, train_annotations/, test_images/
AIHUB_DIR = ROOT / "data" / "aihub"  # AI Hub 조합 데이터 (labels/TL_n/, images/TS_n/)

# AI Hub 데이터를 train 에 더할지 (make_yolo, 학습, 분석이 모두 이 값을 따른다)
#   True  -> data/yolo_aihub/ (Kaggle + AI Hub)
#   False -> data/yolo/       (Kaggle 만, 이전 실험과 같은 데이터)
USE_AIHUB = False
if USE_AIHUB:
    YOLO_DIR = ROOT / "data" / "yolo_aihub"
else:
    YOLO_DIR = ROOT / "data" / "yolo"
DATA_YAML = YOLO_DIR / "data.yaml"  # YOLO 학습 설정
RUNS_DIR = ROOT / "runs"  # 학습 결과 저장 폴더

# ===== 공통 설정 =====
SEED = 42  # 랜덤 고정 -> 누가 돌려도 같은 결과


def get_device():
    """학습 · 예측에 쓸 장치 고르기

    입력: 없음

    반환:
        str: "cuda" (NVIDIA GPU) / "mps" (맥 GPU) / "cpu"
             ultralytics(device=) 와 torch(.to()) 둘 다 이 글자를 그대로 받는다.

    동작:
        1. NVIDIA GPU 가 있으면 "cuda"
        2. 맥 GPU 가 있으면 "mps"
        3. 둘 다 없으면 "cpu"
    """
    # 1. NVIDIA GPU
    if torch.cuda.is_available():
        return "cuda"

    # 2. 맥 GPU
    if torch.backends.mps.is_available():
        return "mps"

    # 3. CPU
    return "cpu"
