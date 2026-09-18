"""
YOLO 학습 + 대회 지표(mAP[0.75:0.95]) 확인

실행: uv run python src/train.py
      (make_yolo.py 를 먼저 실행해야 한다)

결과: runs/<NAME>/weights/best.pt
      runs/<NAME>/args.yaml
      runs/<NAME>/results.csv
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO

from sheet import post_to_sheet


# ===== 경로 =====
ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "yolo" / "data.yaml"
RUNS = ROOT / "runs"


# ===== 학습 설정 =====
MODEL = "yolo26s.pt"
EPOCHS = 50
IMGSZ = 960
BATCH = 8
SEED = 42

NAME = "experiment_12_class_balance_s"
MEMO = "S 모델 기반 클래스 균형 데이터 분할 실험"


# ===== 실험용 설정 =====
EXPERIMENT = {
    "mosaic": 1,
}


def get_device():
    if torch.cuda.is_available():
        return "0"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def make_train_args(device):
    return {
        "data": str(DATA_YAML),
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "device": device,
        "seed": SEED,
        "deterministic": True,
        "project": str(RUNS),
        "name": NAME,
        "exist_ok": True,
    }


def train(args):
    model = YOLO(MODEL)

    model.train(
        **args,
        **EXPERIMENT,
    )

    return Path(model.trainer.best)


def evaluate(weights, device):
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


def save_scores(scores, args):
    path = RUNS / "experiments.csv"

    row = {
        "time": datetime.now().strftime("%m-%d %H:%M"),
        "author": os.environ.get("AUTHOR", ""),
        "memo": MEMO,
        "mAP50": round(scores["mAP50"], 4),
        "mAP50-95": round(scores["mAP50-95"], 4),
        "mAP75-95": round(scores["mAP75-95"], 4),
        "model": MODEL,
        "experiment": str(EXPERIMENT),
    }

    row.update(args)
    row.update(EXPERIMENT)

    table = pd.DataFrame([row])

    if path.exists():
        old = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

        table = pd.concat(
            [old, table],
            ignore_index=True,
        )

    table.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    try:
        print(
            "시트 전송:",
            post_to_sheet(
                {
                    "action": "append",
                    "row": row,
                }
            ),
        )

    except Exception as e:
        print(
            "시트 전송 실패 (csv 에는 저장됨):",
            e,
        )

    return path


def main():
    device = get_device()

    args = make_train_args(device)

    print("실험 이름:", NAME)
    print("메모:", MEMO)
    print("학습 인자:", args)
    print(
        "실험 설정:",
        EXPERIMENT if EXPERIMENT else "기본값",
    )

    weights = train(args)

    print("best.pt:", weights)

    scores = evaluate(
        weights,
        device,
    )

    for key, value in scores.items():
        print(
            f"{key}: {value:.4f}"
        )

    print(
        "실험 기록:",
        save_scores(
            scores,
            args,
        ),
    )


if __name__ == "__main__":
    main()