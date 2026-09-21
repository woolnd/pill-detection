"""여러 예측을 WBF 로 합쳐서 제출 파일을 만든다

실행: uv run python -m src.analysis.ensemble
결과: runs/<NAME>/submission_wbf.csv

WBF (Weighted Boxes Fusion, Solovyev et al. 2019):
    NMS 는 겹치는 박스 중 하나를 '고르고' 나머지를 버린다.
    WBF 는 겹치는 박스들을 신뢰도로 가중평균해서 '새 박스를 만든다'.
    대회 지표가 mAP[0.75:0.95] 라 박스가 조금만 헐거워도 크게 깎이는데,
    좌표를 평균내면 한쪽으로 치우친 오차가 줄어서 이 구간에 직접 효과가 있다.
    구현은 논문 저자의 ensemble-boxes 패키지를 쓴다.

무엇을 합치나 (SOURCES):
    모델이 하나뿐이면 같은 가중치를 조건만 바꿔 예측해서 합친다. (해상도 960 / 1280 / TTA)
    팀원의 다른 모델을 더하면 다양성이 커져서 효과가 더 크다. -> EXTRA_CSV 에 제출 파일 경로를 넣는다.
"""

import numpy as np
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO

from src.config import KAGGLE_DIR, RUNS_DIR, get_device
from src.train.train_yolo import NAME

# ===== 설정값 =====
WEIGHTS = RUNS_DIR / NAME / "weights" / "best.pt"
TEST_DIR = KAGGLE_DIR / "test_images"
OUT_CSV = RUNS_DIR / NAME / "submission_wbf.csv"
CONF = 0.001  # predict.py 와 같은 값
IOU_LIMIT = 0.6  # 이 이상 겹치면 같은 알약을 가리킨 것으로 보고 한 덩어리로 묶는다
SKIP_LIMIT = 0.0001  # 합친 뒤 이 점수 미만은 버린다

# 한 모델을 조건만 바꿔 예측: (이미지 크기, TTA 여부)
SOURCES = [(960, False), (1280, False), (960, True)]

# 팀원이 만든 제출 파일도 같이 합치려면 여기에 경로를 넣는다 (없으면 빈 리스트)
EXTRA_CSV = []


def predict_one(model, paths, imgsz, augment, device):
    """조건 하나로 test 전체를 예측한다

    입력:
        model (YOLO): 학습된 모델
        paths (list): test 이미지 경로 리스트
        imgsz (int): 예측할 이미지 크기
        augment (bool): True 면 ultralytics TTA (좌우반전·다중 스케일)
        device (str): get_device() 결과

    반환:
        dict: 이미지 번호 -> (0~1 로 정규화한 박스, 신뢰도, 약 ID) 배열 3개
              (WBF 는 정규화된 좌표를 받는다)

    동작:
        1. 이미지마다 예측한다.
        2. 좌표를 이미지 크기로 나눠 0~1 로 만든다.
        3. YOLO 번호를 약 ID 로 되돌린다. (제출 파일과 같은 번호 체계로 맞춘다)
    """
    out = {}
    for path in paths:
        r = model.predict(
            path, imgsz=imgsz, conf=CONF, augment=augment, device=device, verbose=False
        )[0]

        # 2. 정규화
        height, width = r.orig_shape
        boxes = r.boxes.xyxy.cpu().numpy() / np.array([width, height, width, height])

        # 3. 약 ID
        ids = [int(model.names[int(c)]) for c in r.boxes.cls.cpu().numpy()]
        out[int(path.stem)] = (boxes, r.boxes.conf.cpu().numpy(), np.array(ids))
    return out


def load_submission(path, image_ids, size):
    """남이 만든 제출 csv 를 predict_one() 과 같은 모양으로 읽는다

    입력:
        path (Path): 제출 파일 경로
        image_ids (list): 맞춰야 할 이미지 번호 목록
        size (tuple): (너비, 높이)  예) (976, 1280)

    반환:
        dict: predict_one() 과 같은 모양

    동작:
        1. csv 를 읽어 이미지별로 나눈다.
        2. x, y, w, h 를 x1, y1, x2, y2 로 바꾸고 0~1 로 정규화한다.
    """
    width, height = size
    df = pd.read_csv(path)
    out = {}

    # 1~2. 이미지마다
    for image_id in image_ids:
        g = df[df.image_id == image_id]
        boxes = np.stack(
            [
                g.bbox_x / width,
                g.bbox_y / height,
                (g.bbox_x + g.bbox_w) / width,
                (g.bbox_y + g.bbox_h) / height,
            ],
            axis=1,
        ) if len(g) else np.zeros((0, 4))
        out[image_id] = (boxes, g.score.to_numpy(), g.category_id.to_numpy())
    return out


def make_rows(preds, size):
    """예측 여러 벌을 이미지마다 WBF 로 합쳐 제출 행을 만든다

    입력:
        preds (list): predict_one() / load_submission() 결과 리스트
        size (tuple): (너비, 높이) 원래 픽셀로 되돌릴 때 쓴다

    반환:
        list: 제출 행 dict 리스트

    동작:
        1. 이미지마다 예측 벌들을 리스트로 모아 weighted_boxes_fusion 에 넘긴다.
        2. 0~1 좌표를 픽셀로 되돌리고 x, y, w, h 로 바꾼다.
    """
    width, height = size
    rows = []

    for image_id in sorted(preds[0]):
        # 1. 합치기 (클래스 구분은 라이브러리가 labels 로 알아서 한다)
        fused, scores, labels = weighted_boxes_fusion(
            [p[image_id][0].tolist() for p in preds],
            [p[image_id][1].tolist() for p in preds],
            [p[image_id][2].tolist() for p in preds],
            iou_thr=IOU_LIMIT,
            skip_box_thr=SKIP_LIMIT,
        )

        # 2. 픽셀로 되돌리기
        for box, score, label in zip(fused, scores, labels):
            x1, y1, x2, y2 = box * np.array([width, height, width, height])
            rows.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox_x": round(float(x1)),
                    "bbox_y": round(float(y1)),
                    "bbox_w": round(float(x2 - x1)),
                    "bbox_h": round(float(y2 - y1)),
                    "score": round(float(score), 4),
                }
            )
    return rows


def main():
    """전체 순서

    입력: 없음

    반환:
        없음. runs/<NAME>/submission_wbf.csv 를 저장한다.

    동작:
        1. 모델·이미지 목록 준비
        2. SOURCES 조건마다 test 전체를 예측
        3. EXTRA_CSV 가 있으면 같이 읽는다
        4. WBF 로 합쳐 저장한다
    """
    # 1. 준비
    device = get_device()
    model = YOLO(WEIGHTS)
    paths = sorted(TEST_DIR.glob("*.png"), key=lambda p: int(p.stem))
    print("모델:", WEIGHTS, "/ 이미지", len(paths), "장")

    # 2. 조건마다 예측
    preds = []
    for imgsz, augment in SOURCES:
        print(f"  예측 중: imgsz {imgsz} / TTA {augment}")
        preds.append(predict_one(model, paths, imgsz, augment, device))

    # 3. 남의 제출 파일
    image_ids = sorted(preds[0])
    size = (976, 1280)  # train·test 전체가 같은 크기
    for path in EXTRA_CSV:
        print("  제출 파일 추가:", path)
        preds.append(load_submission(path, image_ids, size))

    # 4. 합치기 + 저장
    df = pd.DataFrame(make_rows(preds, size))
    df.insert(0, "annotation_id", range(1, len(df) + 1))
    df.to_csv(OUT_CSV, index=False)
    print(f"예측 {len(preds)}벌 -> 박스 {len(df)}개 / 이미지 {df['image_id'].nunique()}장")
    print("저장 위치:", OUT_CSV)


if __name__ == "__main__":
    main()
