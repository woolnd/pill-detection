"""test 이미지 예측 -> Kaggle 제출 파일(csv) 만들기

실행: uv run python src/predict.py   (train.py 로 학습을 먼저 끝내야 한다)
결과: runs/<NAME>/submission.csv

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
from train import IMGSZ, NAME, RUNS, get_device

# ===== 설정값 =====
WEIGHTS = (
    RUNS / NAME / "weights" / "best.pt"
)  # 제출할 모델 (train.py 의 NAME 실험 결과)
TEST_DIR = RAW / "test_images"  # test 이미지 842장
OUT_CSV = RUNS / NAME / "submission.csv"  # 제출 파일 (실험 폴더 안에 같이 저장)
CONF = 0.001  # 이 신뢰도 미만 박스는 버린다. mAP 는 낮은 점수 박스까지 보고 계산해서 낮게 둔다 (0.25 와 비교 제출해보기)


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
        2. 이미지 경로를 파일명 숫자 순서로 정렬한다. (문자열 정렬이면 1, 10, 100 순서가 된다)
        3. 이미지마다 예측하고 to_rows() 로 행을 모은다.
        4. 표로 만들고 맨 앞에 annotation_id (1, 2, 3 ...) 열을 넣는다.
    """

    model = YOLO(weights)
    paths = sorted(image_dir.glob("*.png"), key=lambda p: int(p.stem))

    rows = []
    for path in paths:
        result = model.predict(
            path, imgsz=IMGSZ, conf=CONF, device=device, verbose=False
        )[0]
        rows += to_rows(result, int(path.stem), model.names)

    df = pd.DataFrame(rows)
    df.insert(0, "annotation_id", range(1, len(df) + 1))
    return df


def main():
    """전체 순서

    동작:
        1. 장치 선택
        2. test 전체 예측
        3. csv 저장 후 요약 출력 (박스가 하나도 안 나온 이미지 수도 확인)
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
