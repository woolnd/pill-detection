"""test 이미지 예측 -> Kaggle 제출 파일(csv) 만들기 (헷갈리는 클래스 threshold 실험용)

기존 predict.py와 다른 점:
    1. WEIGHTS 를 train.py의 NAME 대신 yolo26n_clean_v2_imgsz1280(0.41738)로 고정
       -> NAME이 지금 오버샘플링 실험명으로 바뀌어 있어서, 그대로 두면 엉뚱한 가중치를 씀
    2. conf/agnostic_nms/max_det는 전부 원본(0.41738)과 동일하게 유지
       -> 이번 실험에서 바꾸는 변수는 "클래스별 conf threshold" 하나뿐
    3. 헷갈리는 클래스 7개만 confidence 기준을 높여서(0.5) 오탐만 추가로 거른다

실행: uv run python src/predict.py
결과: runs/yolo26n_clean_v2_imgsz1280/submission_allrisky_thresh.csv

제출 형식 (Kaggle Evaluation 탭):
    annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
    - 한 행 = 박스 1개
    - image_id    = test 파일명 숫자        예) "123.png" -> 123
    - category_id = 원래 약 ID              예) 1900 (YOLO 번호 0 이 아님)
    - bbox        = 왼쪽 위 x, y, 너비, 높이 (원본 이미지 픽셀)
    - annotation_id = 1 부터 행마다 고유한 번호
"""

import pandas as pd
from ultralytics import YOLO

from annotations import RAW
from train import IMGSZ, RUNS, get_device  # NAME은 안 씀 (지금 오버샘플링 실험명이라 혼동 방지)

# ===== 설정값 =====
WEIGHTS = (
    RUNS / "yolo26n_clean_v2_imgsz1280" / "weights" / "best.pt"
)  # 0.41738 받은 원본 가중치 고정 (NAME 안 씀)
TEST_DIR = RAW / "test_images"  # test 이미지 842장
OUT_CSV = RUNS / "yolo26n_clean_v2_imgsz1280" / "submission_allrisky_thresh.csv"  # 원본 submission.csv는 안 건드림

BASE_CONF = 0.15  # 일단 낮게 뽑아서, 나중에 클래스별로 다시 거를 여지를 만든다
DEFAULT_CONF = 0.25  # 헷갈리는 클래스가 아닌 나머지는 원본과 동일한 기준 유지

# 헷갈리는 클래스 7개: confusion matrix·visualize.py에서 반복 확인된 클래스만
# 기본(0.25)보다 높은 기준(0.5)을 적용해서, 확신 낮은 오탐만 추가로 걸러낸다
RISKY_THRESHOLDS = {
    20238: 0.5,  # 20238 <-> 38162
    38162: 0.5,
    19232: 0.5,  # 19232 <-> 32310
    32310: 0.5,
    29667: 0.5,  # 29667 <-> 16548
    16548: 0.5,  # + 라벨 버그 이력
    35206: 0.5,  # 배경을 이 약으로 착각하는 오탐 잦음
}


def apply_class_thresholds(df, thresholds, default=DEFAULT_CONF):
    """클래스별로 다른 confidence 기준을 적용해서 낮은 점수 박스를 거른다

    입력:
        df (DataFrame): image_id, category_id, score 등을 포함한 예측 결과 표
                         (annotation_id 넣기 전, BASE_CONF로 낮게 뽑은 상태)
        thresholds (dict): 약 ID -> 이 값 미만이면 버릴 기준       예) {35206: 0.5}
        default (float): thresholds에 없는 나머지 클래스에 적용할 기준

    반환:
        DataFrame: 클래스별 기준을 통과한 박스만 남은 표

    동작:
        1. 각 행의 category_id를 thresholds에서 찾아 기준값을 매긴다.
        2. thresholds에 없는 클래스는 default(기존 CONF와 동일한 값)를 쓴다.
        3. score가 기준보다 낮은 행은 버린다.

    헷갈리는 클래스만 기준을 높여서 그 클래스들의 오탐(확신 낮은 예측)만 추가로 줄이고,
    나머지 클래스는 원래 재현율을 그대로 유지하기 위한 후처리.
    """
    cutoff = df["category_id"].map(thresholds).fillna(default)
    return df[df["score"] >= cutoff]


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

    rows = []
    boxes = result.boxes

    for (x1, y1, x2, y2), cls, score in zip(
        boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
    ):
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


def remove_duplicate_class(df):
    """같은 이미지 안에서 같은 클래스가 중복 예측되면 점수 높은 것만 남긴다

    입력:
        df (DataFrame): image_id, category_id, score 등을 포함한 예측 결과 표
                         (annotation_id 넣기 전)

    반환:
        DataFrame: 이미지당 클래스당 최고 score 박스만 남은 표

    이 데이터셋(조합경구약제)은 한 이미지 안에 같은 약이 두 번 나올 수 없다.
    그런데도 모델이 같은 class를 한 이미지에서 두 번 예측하면, 점수 낮은 쪽은
    100% 오탐(다른 약을 착각한 것)이라 지워도 손해가 없다.
    """
    before = len(df)
    df = (
        df.sort_values("score", ascending=False)
        .drop_duplicates(subset=["image_id", "category_id"], keep="first")
        .sort_index()  # 원래 이미지 순서로 되돌리기
    )
    removed = before - len(df)
    if removed:
        print(f"같은 이미지 내 클래스 중복 제거: {removed}개")
    return df


def predict_all(weights, image_dir, device):
    """test 이미지 전체를 예측해서 제출 표를 만든다

    입력:
        weights (Path): best.pt 경로
        image_dir (Path): test 이미지 폴더
        device (str): get_device() 결과

    반환:
        DataFrame: 제출 형식 그대로의 표 (열 8개)

    동작:
        1. 모델을 불러온다.
        2. 이미지 경로를 파일명 숫자 순서로 정렬한다.
        3. 이미지마다 BASE_CONF(낮은 기준)로 예측하고 to_rows() 로 행을 모은다.
        4. apply_class_thresholds() 로 헷갈리는 클래스만 기준을 높여서 다시 거른다.
        5. remove_duplicate_class() 로 같은 이미지 내 클래스 중복을 제거한다.
        6. 표로 만들고 맨 앞에 annotation_id (1, 2, 3 ...) 열을 넣는다.

    conf/agnostic_nms은 원본(0.41738) 실험과 동일하게 유지 — 이번 실험에서
    바뀌는 변수는 "예측 후 클래스별 threshold" 하나뿐이다.
    """

    model = YOLO(weights)
    paths = sorted(image_dir.glob("*.png"), key=lambda p: int(p.stem))

    rows = []
    for path in paths:
        result = model.predict(
            path,
            imgsz=IMGSZ,
            conf=BASE_CONF,      # 0.25 대신 0.15로 낮게 뽑아서 threshold 필터링 여지를 만든다
            iou=0.6,
            agnostic_nms=True,   # 원본과 동일하게 유지 (이번 실험 변수 아님)
            device=device,
            verbose=False,
        )[0]
        rows += to_rows(result, int(path.stem), model.names)

    df = pd.DataFrame(rows)
    df = apply_class_thresholds(df, RISKY_THRESHOLDS)  # 헷갈리는 클래스만 기준 상향
    df = remove_duplicate_class(df)  # 같은 이미지 내 클래스 중복 제거
    df.insert(0, "annotation_id", range(1, len(df) + 1))
    return df


def main():
    """전체 순서

    동작:
        1. 장치 선택
        2. test 전체 예측
        3. csv 저장 후 요약 출력
    """
    # 1. 장치
    device = get_device()
    print("모델:", WEIGHTS)

    # 2. 예측
    df = predict_all(WEIGHTS, TEST_DIR, device)

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