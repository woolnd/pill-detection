"""test 이미지 예측 -> Kaggle 제출 파일(csv) 만들기

실행: uv run python -m src.analysis.predict
결과: USE_ENSEMBLE=True 면 runs/ensemble_f2_960_ratio_submission.csv
      False 면 runs/<NAME>/submission.csv  (train_yolo.py 로 학습을 먼저 끝내야 한다)

세 가지 예측 모드 (우선순위: 앙상블 > TTA > 단일):
    1. USE_ENSEMBLE=True   서로 다른 두 체크포인트(모델 A+B)를 WBF로 합쳐 예측 (최종 제출 방식)
    2. USE_TTA=True        같은 체크포인트를 여러 스케일로 반복 예측 후 WBF로 합침
    3. 둘 다 False          NAME 체크포인트로 단일 스케일 예측

제출 형식 (Kaggle Evaluation 탭):
    annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
    - 한 행 = 박스 1개
    - image_id    = test 파일명 숫자        예) "123.png" -> 123
    - category_id = 원래 약 ID              예) 1900 (YOLO 번호 0 이 아님)
    - bbox        = 왼쪽 위 x, y, 너비, 높이 (원본 이미지 픽셀)
    - annotation_id = 1 부터 행마다 고유한 번호
"""

from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO

from src.config import KAGGLE_DIR, RUNS_DIR, get_device
from src.train.train_yolo import IMGSZ, NAME

# ===== 공통 설정값 =====
WEIGHTS = RUNS_DIR / NAME / "weights" / "best.pt"  # 단일 모델 모드에서 쓸 체크포인트
TEST_DIR = KAGGLE_DIR / "test_images"  # test 이미지 842장
OUT_CSV = RUNS_DIR / NAME / "submission.csv"  # USE_ENSEMBLE=True면 아래에서 덮어씀
CONF = 0.001  # 이 신뢰도 미만 박스는 버린다. mAP 는 낮은 점수 박스까지 보고 계산해서 낮게 둔다

# ===== 멀티스케일 TTA 설정 (같은 모델, 스케일만 바꿔 WBF) =====
USE_TTA = False  # USE_ENSEMBLE=True면 무시됨 (앙상블이 우선)
TTA_SCALES = [IMGSZ - 128, IMGSZ, IMGSZ + 128]  # 예) IMGSZ=960 이면 832, 960, 1088
WBF_IOU_THR = 0.55  # 이 이상 겹치면 같은 박스로 보고 합친다
WBF_SKIP_BOX_THR = 0.001  # 이 신뢰도 미만은 WBF에서 아예 제외

# ===== 크로스모델 WBF 앙상블 설정 (최종 제출 방식) =====
# runs/finals/ 에 최종 제출 두 모델이 들어 있다 (git 에 포함). 직접 학습한 모델을 쓰려면 경로를 runs/<NAME>/ 으로 바꾼다
USE_ENSEMBLE = True  # 끄고 싶으면 False

WEIGHTS_A = RUNS_DIR / "finals" / "rahui_ep50_img1280_batch4_aihubdata_oversample" / "weights" / "best.pt"  # 개별 0.61802
IMGSZ_A = 1280

WEIGHTS_B = RUNS_DIR / "finals" / "rahui_ep30_img960_batch8_aihubdata_oversample_f2" / "weights" / "best.pt"  # 개별 0.62041 (최고)
IMGSZ_B = 960

ENSEMBLE_WEIGHTS_RATIO = [1.0, 1.3]  # 두 모델 비중 (B가 더 강해서 더 크게 반영)

if USE_ENSEMBLE:
    OUT_CSV = RUNS_DIR / "ensemble_f2_960_ratio_submission.csv"  # 기존 submission.csv 안 덮어씀


def predict_test(weights, image_dir, device):
    """test 이미지 전체를 예측해서 제출 표를 만든다

    입력:
        weights (Path): best.pt 경로 (USE_ENSEMBLE=True면 무시되고 WEIGHTS_A/B를 대신 씀)
        image_dir (Path): test 이미지 폴더
        device (str): get_device() 결과

    반환:
        DataFrame: 제출 형식 그대로의 표 (열 8개)

    동작:
        1. 예측 방식을 정한다. (모델, imgsz) 목록 + WBF 비중
           - 앙상블: 모델 A·B 를 각자 학습 imgsz 로 1번씩
           - TTA: 같은 모델을 TTA_SCALES 마다 1번씩
           - 단일: 목록 없이 model.predict() 한 번
        2. 이미지 경로를 파일명 숫자 순서로 정렬한다. (문자열 정렬이면 1, 10, 100 순서가 된다)
        3. 이미지마다 예측하고 to_rows() 로 행을 모은다.
        4. 표로 만들고 맨 앞에 annotation_id (1, 2, 3 ...) 열을 넣는다.
    """
    # 1. 예측 방식
    if USE_ENSEMBLE:
        model_a = YOLO(WEIGHTS_A)
        model_b = YOLO(WEIGHTS_B)
        assert model_a.names == model_b.names, "두 모델 클래스 매핑이 다름 - make_yolo 재실행 여부 확인 필요"
        model = model_a  # names 를 꺼낼 때 쓴다
        runs = [(model_a, IMGSZ_A), (model_b, IMGSZ_B)]
        ratio = ENSEMBLE_WEIGHTS_RATIO
    else:
        model = YOLO(weights)
        runs = [(model, scale) for scale in TTA_SCALES] if USE_TTA else None
        ratio = None

    # 2. 파일명 숫자 순서
    paths = sorted(image_dir.glob("*.png"), key=lambda p: int(p.stem))

    # 3. 이미지마다 예측
    rows = []
    for path in paths:
        if runs:
            result = predict_wbf(runs, path, device, ratio)
        else:
            result = model.predict(
                path, imgsz=IMGSZ, conf=CONF, device=device, verbose=False
            )[0]
        rows += to_rows(result, int(path.stem), model.names)

    # 4. 표 + annotation_id
    df = pd.DataFrame(rows)
    df.insert(0, "annotation_id", range(1, len(df) + 1))
    return df


def predict_wbf(runs, path, device, ratio=None):
    """(모델, imgsz) 마다 1번씩 예측한 다음 WBF(Weighted Boxes Fusion)로 하나로 합친다

    입력:
        runs (list): [(YOLO, imgsz), ...]
                     앙상블 예) [(모델A, 1280), (모델B, 960)] / TTA 예) [(모델, 832), (모델, 960), (모델, 1088)]
        path (Path): 예측할 이미지 경로
        device (str): get_device() 결과
        ratio (list): runs 마다 WBF 비중. None 이면 모두 같게  예) [1.0, 1.3]

    반환:
        SimpleNamespace: to_rows() 가 model.predict() 결과처럼 .boxes.xyxy/.cls/.conf 로 쓸 수 있는 객체

    동작:
        1. 원본 이미지 크기를 구한다. (WBF 는 0~1 정규화 좌표를 요구해서 필요)
        2. runs 마다 예측 -> 박스를 0~1 정규화 좌표로 바꿔 리스트에 모은다.
        3. weighted_boxes_fusion 으로 하나로 합친다.
        4. 합쳐진 정규화 좌표를 다시 원본 픽셀 좌표로 되돌린다.
    """
    # 1. 원본 크기
    h, w = cv2.imread(str(path)).shape[:2]

    # 2. 예측 + 정규화
    boxes_list, scores_list, labels_list = [], [], []
    for model, imgsz in runs:
        r = model.predict(path, imgsz=imgsz, conf=CONF, device=device, verbose=False)[0]
        xyxy = r.boxes.xyxy.cpu().numpy() / [w, h, w, h]
        boxes_list.append(xyxy.tolist())
        scores_list.append(r.boxes.conf.cpu().numpy().tolist())
        labels_list.append(r.boxes.cls.cpu().numpy().tolist())

    # 3. WBF
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=ratio,
        iou_thr=WBF_IOU_THR,
        skip_box_thr=WBF_SKIP_BOX_THR,
    )

    # 4. 픽셀 좌표로
    boxes = np.asarray(boxes).reshape(-1, 4) * [w, h, w, h]
    return SimpleNamespace(boxes=SimpleNamespace(xyxy=boxes, cls=labels, conf=scores))


def to_rows(result, image_id, names):
    """이미지 1장의 예측 결과를 제출 행 리스트로 바꾼다

    입력:
        result (Results): model.predict() 결과 1장
        image_id (int): 이미지 번호  예) 123
        names (dict): YOLO 번호 -> 약 ID 문자열  예) {0: '1900', 1: '2483', ...}

    반환:
        list: 박스마다 dict 1개
              예) [{"image_id": 123, "category_id": 1900,
                    "bbox_x": 156, "bbox_y": 247, "bbox_w": 211, "bbox_h": 456, "score": 0.91}, ...]

    동작:
        1. 박스마다 좌표(xyxy), 클래스 번호, 신뢰도를 꺼낸다.
           (xyxy = 왼쪽 위 x1, y1 + 오른쪽 아래 x2, y2. 원본 이미지 픽셀 기준)
        2. YOLO 번호를 names 로 약 ID 로 되돌린다.
        3. x2, y2 를 너비, 높이로 바꾼다. (너비 = x2 - x1)
        4. 제출 예시처럼 좌표는 정수로 반올림한다.
    """
    # 1. 좌표, 클래스 번호, 신뢰도를 리스트로
    xyxy = result.boxes.xyxy.tolist()  # 예) [[156.2, 247.1, 367.5, 703.4], ...]
    classes = result.boxes.cls.tolist()  # 예) [0.0, 15.0, ...]
    scores = result.boxes.conf.tolist()  # 예) [0.91, 0.85, ...]

    rows = []
    for i in range(len(scores)):
        x1, y1, x2, y2 = xyxy[i]
        cls = classes[i]
        score = scores[i]
        rows.append(
            {
                "image_id": image_id,
                "category_id": int(names[int(cls)]),  # 2. YOLO 번호 -> 약 ID
                "bbox_x": round(x1),  # 3~4. 왼쪽 위 x
                "bbox_y": round(y1),  #      왼쪽 위 y
                "bbox_w": round(x2 - x1),  #      너비
                "bbox_h": round(y2 - y1),  #      높이
                "score": round(score, 4),
            }
        )
    return rows


def main():
    """전체 순서

    입력: 없음

    반환:
        없음. USE_ENSEMBLE 여부에 따라 OUT_CSV 위치에 결과를 저장한다.

    동작:
        1. 장치 선택
        2. test 전체 예측
        3. csv 저장 후 요약 출력 (박스가 하나도 안 나온 이미지 수도 확인)
    """
    # 1. 장치
    device = get_device()
    if not USE_ENSEMBLE:
        print("모델:", WEIGHTS)
    print("TTA:", USE_TTA, TTA_SCALES if USE_TTA else "")
    print("앙상블:", USE_ENSEMBLE, [str(WEIGHTS_A), str(WEIGHTS_B)] if USE_ENSEMBLE else "")

    # 2. 예측
    df = predict_test(WEIGHTS, TEST_DIR, device)

    # 3. 저장 + 요약
    df.to_csv(OUT_CSV, index=False)
    n_images = len(list(TEST_DIR.glob("*.png")))
    print(df.head())
    print(
        f"박스 {len(df)}개 / 예측된 이미지 {df['image_id'].nunique()}장 / 전체 {n_images}장"
    )
    print("저장 위치:", OUT_CSV)


if __name__ == "__main__":
    main()
