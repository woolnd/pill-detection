"""test 예측을 의사 라벨로 삼아 train 을 늘린 데이터셋을 만든다

실행: uv run python -m src.data.make_pseudo   (predict.py 로 제출 파일을 먼저 만들어야 한다)
결과: data/yolo_pseudo/   기존 train + test 의사 라벨, val 은 그대로
      학습하려면 src/config.py 의 YOLO_DIR 을 이 폴더로 바꾼다.

왜 하나:
    test 842장은 라벨만 없을 뿐 우리가 맞혀야 할 바로 그 분포다.
    현재 모델의 확신 있는 예측을 라벨로 삼아 다시 학습시키면 그 분포에 모델을 맞출 수 있다.
    (Global Wheat Detection 대회 우승자 전원이 쓴 방법. 그들은 test 를 보지도 못한 채 적용했다)

주의:
    틀린 예측을 라벨로 굳히면 그 오류가 강화된다. CONF 를 높게 잡아 확신 있는 것만 쓴다.
    val 은 손대지 않는다. 의사 라벨을 val 에 넣으면 점수가 자기 자신을 평가하게 된다.
"""

import os
import shutil

import pandas as pd
import yaml

from src.config import DATA_YAML, KAGGLE_DIR, ROOT, YOLO_DIR
from src.train.train_yolo import NAME

# ===== 설정값 =====
SUBMISSION = ROOT / "runs" / NAME / "submission.csv"  # 의사 라벨로 쓸 예측. 앙상블 결과면 runs/ensemble_f2_960_ratio_submission.csv
OUT_DIR = ROOT / "data" / "yolo_pseudo"  # 만들 데이터셋
TEST_DIR = KAGGLE_DIR / "test_images"
CONF = 0.5  # 이 신뢰도 이상 박스만 라벨로 쓴다 (낮추면 라벨은 늘지만 오류도 늘어난다)
IMG_W, IMG_H = 976, 1280  # 모든 이미지 크기 (make_yolo 와 같은 값)


def load_class_map():
    """약 ID -> YOLO 번호 표를 만든다

    입력: 없음 (현재 YOLO_DIR 의 data.yaml 을 읽는다)

    반환:
        dict: 약 ID -> YOLO 번호  예) {1900: 0, 2483: 1, ...}

    동작:
        1. data.yaml 의 names (YOLO 번호 -> 약 ID) 를 읽는다.
        2. 뒤집는다.
    """
    # 1. names
    with open(DATA_YAML, encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]

    # 2. 뒤집기
    return {int(v): int(k) for k, v in names.items()}


def copy_existing():
    """기존 데이터셋(train·val)을 새 폴더로 옮겨 담는다

    입력: 없음

    반환:
        없음. OUT_DIR 아래에 images/labels 를 만든다.

    동작:
        1. 이전 결과가 있으면 지운다. (같은 입력 -> 같은 출력)
        2. train/val 의 이미지는 하드링크로 연결하고(용량 절약) 라벨은 복사한다.
    """
    # 1. 새로 만들기
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    # 2. 이미지는 하드링크, 라벨은 복사
    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True)
        for src in (YOLO_DIR / "images" / split).glob("*.png"):
            os.link(src, OUT_DIR / "images" / split / src.name)
        for src in (YOLO_DIR / "labels" / split).glob("*.txt"):
            shutil.copy(src, OUT_DIR / "labels" / split / src.name)


def add_pseudo(class_map):
    """test 이미지와 의사 라벨을 train 에 더한다

    입력:
        class_map (dict): load_class_map() 결과

    반환:
        (int, int): 더한 이미지 수, 박스 수

    동작:
        1. 제출 파일에서 CONF 이상 박스만 고른다.
        2. 이미지마다
           a. 박스를 YOLO 줄(번호 + 중심점 비율)로 바꾼다. 박스가 하나도 없으면 건너뛴다.
              (빈 라벨을 넣으면 그 사진 전체가 '배경'이 되어 학습을 망친다)
           b. 이미지는 하드링크, 라벨은 txt 로 저장한다. 이름이 겹치지 않게 앞에 test_ 를 붙인다.
    """
    # 1. 확신 있는 박스만
    df = pd.read_csv(SUBMISSION)
    df = df[df.score >= CONF]

    n_images = 0
    n_boxes = 0

    # 2. 이미지마다
    for image_id, group in df.groupby("image_id"):
        lines = []
        for r in group.itertuples():
            if r.category_id not in class_map:  # 학습에 없는 클래스면 버린다
                continue
            # 2-a. 좌상단 픽셀 -> 중심점 비율 (make_yolo.to_yolo_line 과 같은 계산)
            cx = (r.bbox_x + r.bbox_w / 2) / IMG_W
            cy = (r.bbox_y + r.bbox_h / 2) / IMG_H
            w = r.bbox_w / IMG_W
            h = r.bbox_h / IMG_H
            lines.append(f"{class_map[r.category_id]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not lines:
            continue

        # 2-b. 저장
        name = f"test_{image_id}"
        os.link(TEST_DIR / f"{image_id}.png", OUT_DIR / "images" / "train" / f"{name}.png")
        (OUT_DIR / "labels" / "train" / f"{name}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        n_images += 1
        n_boxes += len(lines)

    return n_images, n_boxes


def save_yaml():
    """data.yaml 을 새 폴더에 맞춰 저장한다

    입력: 없음

    반환:
        없음. OUT_DIR/data.yaml 을 만든다.

    동작:
        1. 기존 data.yaml 을 읽는다. (클래스 목록은 그대로 쓴다)
        2. path 만 새 폴더로 바꿔 저장한다.
    """
    # 1. 기존 설정
    with open(DATA_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 2. 경로만 교체
    lines = [f"path: {OUT_DIR}", "train: images/train", "val: images/val", "names:"]
    for idx in sorted(data["names"], key=int):
        lines.append(f"  {idx}: '{data['names'][idx]}'")
    (OUT_DIR / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """전체 순서

    입력: 없음

    반환:
        없음. data/yolo_pseudo/ 를 만든다.

    동작:
        1. 약 ID -> YOLO 번호 표
        2. 기존 train/val 복사
        3. test 의사 라벨 추가
        4. data.yaml 저장 + 요약 출력
    """
    # 1~2. 준비
    print("의사 라벨로 쓸 예측:", SUBMISSION)
    class_map = load_class_map()
    copy_existing()
    before = len(list((OUT_DIR / "images" / "train").glob("*.png")))

    # 3. 추가
    n_images, n_boxes = add_pseudo(class_map)

    # 4. 저장
    save_yaml()
    print(f"train {before}장 + test 의사 라벨 {n_images}장 = {before + n_images}장")
    print(f"의사 라벨 박스 {n_boxes}개 (신뢰도 {CONF} 이상)")
    print("만든 곳:", OUT_DIR)
    print(f"학습하려면 src/config.py 의 YOLO_DIR 을 {OUT_DIR.name} 으로 바꾼다.")


if __name__ == "__main__":
    main()
