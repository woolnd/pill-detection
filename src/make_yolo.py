"""COCO JSON -> YOLO 형식 변환 + train/val 분할

실행: uv run python src/make_yolo.py
결과: data/yolo/
        ├── images/train/, images/val/   이미지 복사본
        ├── labels/train/, labels/val/   YOLO 라벨 txt
        └── data.yaml                    학습 설정

박스 딕셔너리 b 의 모양 (annotations.py 참고):
    {"x": 167, "y": 248, "w": 184, "h": 182, "class_id": 1900, "class_name": "..."}
"""

import random
import shutil
from pathlib import Path

from annotations import RAW, load_annotations

# ===== 설정값 =====
OUT = Path(__file__).resolve().parent.parent / "data" / "yolo"  # 결과 저장 위치
VAL_RATIO = 0.2  # val 비율 (조합 기준 20%)
SEED = 42  # 랜덤 고정 -> 누가 돌려도 같은 분할
IMG_W, IMG_H = 976, 1280  # 모든 이미지 크기 (train/test 1074장 전부 확인함)


# ---------- 1. 이상한 박스 걸러내기 ----------
def is_valid(b):
    """박스가 이미지 안에 제대로 들어있는지 검사

    입력:
        b (dict): 박스 1개
    반환:
        bool: 네 조건을 모두 만족하면 True, 하나라도 어기면 False
            - 왼쪽 끝(x)이 0 이상
            - 위쪽 끝(y)이 0 이상
            - 오른쪽 끝(x + w)이 이미지 너비 이하
            - 아래쪽 끝(y + h)이 이미지 높이 이하
    """
        
    return (
        b["x"] >= 0
        and b["y"] >= 0
        and b["x"] + b["w"] <= IMG_W
        and b["y"] + b["h"] <= IMG_H
    )


def remove_bad_images(ann):
    """이상한 박스가 하나라도 있는 이미지를 통째로 뺀다

    입력:
        ann (dict): load_annotations() 결과. 이미지 파일명 -> 박스 리스트

    반환:
        dict: 같은 모양의 dict. 이상한 이미지만 빠져 있다.

    동작:
        1. 이미지를 하나씩 보면서
        2. 그 이미지의 박스가 전부 is_valid() 이면 결과에 넣고
        3. 하나라도 아니면 넣지 않고 파일명을 출력한다.

    박스 하나만 빼지 않고 이미지째 빼는 이유:
        박스만 빼면 그 알약이 '라벨 없는 배경'으로 학습돼서 모델이 헷갈린다.
        지금은 x=6567 짜리 1장이 해당되고, 같은 조합의 다른 각도 사진이 있어서 손해가 없다.
    """
    clean = {}

    for name, boxes in ann.items():
        if all(is_valid(b) for b in boxes):
            clean[name] = boxes

    return clean


# ---------- 2. 클래스 번호 만들기 ----------
def make_class_map(ann):
    """약 ID(1900, 2483, ...)를 YOLO 번호(0, 1, 2, ...)로 바꾸는 표를 만든다

    입력:
        ann (dict): 이미지 파일명 -> 박스 리스트

    반환:
        class_ids (list): 약 ID를 작은 순서로 정렬한 리스트
                          예) [1900, 2483, 3351, ..., 41768]
                          -> 리스트 위치(인덱스)가 곧 YOLO 번호다. data.yaml 만들 때 쓴다.
        class_to_idx (dict): 약 ID -> YOLO 번호
                          예) {1900: 0, 2483: 1, 3351: 2, ...}
                          -> 라벨 txt 쓸 때 약 ID를 번호로 바꾸는 데 쓴다.

    동작:
        1. 모든 박스의 class_id 를 set 으로 모아 중복을 없앤다.
        2. 정렬한다. (정렬해야 누가 돌려도 같은 번호가 나온다)
        3. 순서대로 0, 1, 2 ... 번호를 붙인다.

    YOLO는 0부터 빈틈 없이 이어지는 번호만 받는다.
    """
    # 1~2. 모든 박스의 약 ID -> 중복 제거 -> 정렬
    class_ids = sorted(set(b["class_id"] for boxes in ann.values() for b in boxes))

    # 3. enumerate 로 (번호, 약 ID) 를 꺼내서 {약 ID: 번호} 로 뒤집는다
    class_to_idx = {id: i for i, id in enumerate(class_ids)}
    return class_ids, class_to_idx


# ---------- 3. 좌표 변환 ----------
def to_yolo_line(b, idx):
    """박스 1개를 YOLO 라벨 한 줄(문자열)로 바꾼다

    입력:
        b (dict): 박스 1개 (COCO 좌표: 왼쪽 위 x, y, 너비, 높이 / 픽셀)
        idx (int): 이 박스의 YOLO 클래스 번호 (class_to_idx 로 바꾼 값)

    반환:
        str: "클래스번호 중심x 중심y 너비 높이" (좌표는 0~1 비율, 소수점 6자리)
             예) "0 0.265369 0.264844 0.188525 0.142187"

    동작:
        1. 중심 = 왼쪽 위 + 크기의 절반
        2. 픽셀 값을 이미지 크기로 나눠서 0~1 비율로 만든다.
        3. 공백으로 이어 붙인다.
    """
    cx = (
        b["x"] + (b["w"] / 2)
    ) / IMG_W  # 중심 x: 왼쪽 끝 + 너비 절반, 이미지 너비로 나눔
    cy = (
        b["y"] + (b["h"] / 2)
    ) / IMG_H  # 중심 y: 위쪽 끝 + 높이 절반, 이미지 높이로 나눔
    w = b["w"] / IMG_W  # 너비 비율
    h = b["h"] / IMG_H  # 높이 비율
    return f"{idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# ---------- 4. train / val 나누기 ----------
def get_combo(name):
    """파일명에서 조합 ID 부분만 꺼낸다

    입력:
        name (str): 이미지 파일명
                    예) "K-001900-016548-019607-029451_0_2_0_2_70_000_200.png"

    반환:
        str: 첫 번째 "_" 앞부분
             예) "K-001900-016548-019607-029451"

    같은 조합을 각도만 바꿔 찍은 사진 3장은 이 값이 같다.
    """
    return name.split("_", 1)[0]


def split_by_combo(names):
    """이미지 목록을 조합 단위로 train / val 로 나눈다

    입력:
        names (list): 이미지 파일명 리스트 (정렬된 상태로 넣는다)

    반환:
        train (list): train 에 들어갈 이미지 파일명 리스트
        val (list): val 에 들어갈 이미지 파일명 리스트

    동작:
        1. 파일명마다 조합 ID를 뽑고 중복을 없애서 정렬한다. (114개)
        2. 시드를 고정하고 섞는다.
        3. 앞에서 20% 조합을 val 로 정한다.
        4. 이미지마다 조합이 val 쪽이면 val, 아니면 train 에 넣는다.

    이미지가 아니라 조합으로 나누는 이유:
        같은 조합의 70°/75°/90° 사진이 train과 val에 나뉘어 들어가면
        거의 같은 사진을 보고 시험 치는 셈이라 val 점수가 실제보다 좋게 나온다.
    """
    # 1. 조합 ID 목록 (정렬해야 섞기 전 순서가 매번 같다)
    combos = sorted(set(get_combo(name) for name in names))

    # 2. 시드 고정 후 섞기
    random.seed(SEED)
    random.shuffle(combos)

    # 3. 앞에서 20%를 val 조합으로
    n_val = int(len(combos) * VAL_RATIO)
    val_combos = set(combos[:n_val])

    # 4. 이미지를 조합에 따라 나누기
    train = [name for name in names if get_combo(name) not in val_combos]
    val = [name for name in names if get_combo(name) in val_combos]
    return train, val


# ---------- 5. 파일 쓰기 ----------
def write_split(ann, names, split, class_to_idx):
    """한 분할(train 또는 val)의 이미지와 라벨 파일을 만든다

    입력:
        ann (dict): 이미지 파일명 -> 박스 리스트
        names (list): 이 분할에 들어갈 이미지 파일명 리스트
        split (str): "train" 또는 "val" (폴더 이름으로 쓴다)
        class_to_idx (dict): 약 ID -> YOLO 번호

    반환:
        없음. 파일만 만든다.
            data/yolo/images/<split>/<이미지명>.png   원본 복사
            data/yolo/labels/<split>/<이미지명>.txt   박스 1개 = 한 줄

    동작:
        1. images/<split>, labels/<split> 폴더를 만든다.
        2. 이미지마다
           a. 원본 png 를 images 폴더로 복사한다.
           b. 박스마다 to_yolo_line() 으로 한 줄씩 만든다.
           c. 줄들을 합쳐서 같은 이름의 .txt 로 저장한다.
    """
    # 1. 폴더 만들기 (이미 있어도 에러 안 나게 exist_ok=True)
    img_dir = OUT / "images" / split
    lbl_dir = OUT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        # 2-a. 이미지 복사
        shutil.copy(RAW / "train_images" / name, img_dir / name)

        # 2-b. 박스마다: 약 ID -> YOLO 번호로 바꾸고 한 줄로 변환
        lines = [to_yolo_line(b, class_to_idx[b["class_id"]]) for b in ann[name]]

        # 2-c. 줄바꿈으로 이어서 저장. 파일명은
        (lbl_dir / name.replace(".png", ".txt")).write_text("\n".join(lines) + "\n")


def write_yaml(class_ids):
    """YOLO 학습 설정 파일(data.yaml)을 만든다

    입력:
        class_ids (list): 정렬된 약 ID 리스트 (make_class_map 결과)

    반환:
        없음. data/yolo/data.yaml 파일을 만든다. 내용 예:
            path: /.../data/yolo
            train: images/train
            val: images/val
            names:
              0: '1900'
              1: '2483'
              ...

    동작:
        1. 데이터 위치(path, train, val) 세 줄을 쓴다.
        2. names 아래에 "번호: '약 ID'" 를 클래스 수만큼 쓴다.
           -> 학습 때는 클래스 이름표, 제출 때는 번호를 약 ID로 되돌리는 표로 쓴다.
    """
    # 1. 데이터 위치
    lines = [
        f"path: {OUT}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    # 2. 번호 -> 약 ID
    for i, cid in enumerate(class_ids):
        lines.append(f"  {i}: '{cid}'")
    (OUT / "data.yaml").write_text("\n".join(lines) + "\n")


# ---------- 실행 ----------
def main():
    """전체 변환 순서

    입력: 없음
    반환: 없음. data/yolo/ 를 새로 만들고 요약을 출력한다.

    동작:
        0. 이전 결과(data/yolo/)가 있으면 지운다.
        1. JSON 읽기 -> 이상한 이미지 빼기
        2. 클래스 번호표 만들기
        3. 조합 단위로 train/val 나누기
        4. train, val 파일 쓰기 + data.yaml 쓰기
        5. 요약 출력 (train 에 한 번도 안 나온 클래스도 확인)
    """
    # 0. 이전 결과가 남아 있으면 예전 분할과 섞이니까 지우고 새로 만든다
    if OUT.exists():
        shutil.rmtree(OUT)

    print("[1] 이상한 박스 걸러내기")
    ann = remove_bad_images(load_annotations())

    print("[2] 클래스 번호 만들기")
    class_ids, class_to_idx = make_class_map(ann)

    print("[3] train/val 나누기")
    train, val = split_by_combo(sorted(ann))  # sorted(ann) = 이미지 파일명 정렬 리스트

    print("[4] 파일 쓰기")
    write_split(ann, train, "train", class_to_idx)
    write_split(ann, val, "val", class_to_idx)
    write_yaml(class_ids)

    # 5. 확인: val 에만 있고 train 에 없는 클래스는 학습이 안 된다
    train_classes = set(b["class_id"] for n in train for b in ann[n])
    missing = sorted(set(class_ids) - train_classes)
    print(f"\n클래스 {len(class_ids)}개 / train {len(train)}장 / val {len(val)}장")
    print("train에 없는 클래스 (학습 안 됨):", missing)
    print("저장 위치:", OUT)


if __name__ == "__main__":
    main()
