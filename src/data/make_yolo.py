"""COCO JSON -> YOLO 형식 변환 + train/val 분할

실행: uv run python -m src.data.make_yolo
결과: data/yolo/ (USE_AIHUB = True 이면 data/yolo_aihub/)
        ├── images/train/, images/val/   이미지 (원본 하드링크)
        ├── labels/train/, labels/val/   YOLO 라벨 txt
        └── data.yaml                    학습 설정

박스 딕셔너리 b 의 모양 (annotations.py 참고):
    {"x": 167, "y": 248, "w": 184, "h": 182, "class_id": 1900, "class_name": "..."}
"""

import os
import random
import shutil

from src.annotations import compute_iou, load_aihub_annotations, load_annotations
from src.config import KAGGLE_DIR, SEED, USE_AIHUB, YOLO_DIR

# ===== 설정값 =====
VAL_RATIO = 0.2  # val 비율 (조합 기준 20%)
AIHUB_ANGLES = [
    "70",
    "75",
    "90",
]  # AI Hub 에서 쓸 촬영 각도. 학습 시간을 줄이려면 75° 와 거의 같은 "70" 을 뺀다
IMG_W, IMG_H = 976, 1280  # 모든 이미지 크기 (train/test 1074장 전부 확인함)


# ---------- 1. 문제 있는 이미지 빼기 ----------
def remove_bad_images(ann):
    """문제가 있는 이미지를 통째로 뺀다

    입력:
        ann (dict): load_annotations() 결과. 이미지 파일명 -> 박스 리스트

    반환:
        dict: 같은 모양의 dict. 문제 있는 이미지만 빠져 있다.

    동작:
        1. 이미지를 파일명 순서로 하나씩 보면서 세 가지를 검사한다. (파일명 순서라 출력 순서가 매번 같다)
           a. 이미지 밖으로 나간 박스가 있음 (is_valid)
           b. 사진 속 약 중 라벨이 빠진 약이 있음 (has_missing_label)
           c. 한 알약에 라벨이 2개 겹침 (has_duplicate_box)
        2. 하나라도 걸리면 결과에 넣지 않고, 이유와 파일명을 출력한다.
        3. 모두 통과한 이미지만 결과에 넣는다.

    박스 하나만 빼지 않고 이미지째 빼는 이유:
        박스만 빼면 그 알약이 '라벨 없는 배경'으로 학습돼서 모델이 헷갈린다.
        지금은 12장이 해당된다. (이미지 밖 1 + 라벨 빠짐 8 + 라벨 겹침 3)
        빼도 train 에서 사라지는 클래스는 없다.
    """
    clean = {}

    for name in sorted(ann):
        boxes = ann[name]

        # 1-a. 이미지 밖 박스가 하나라도 있는지
        has_outside_box = False
        for b in boxes:
            if not is_valid(b):
                has_outside_box = True

        # 1-b, 1-c. 나머지 검사
        if has_outside_box:
            reason = "박스가 이미지 밖"
        elif has_missing_label(name, boxes):
            reason = "라벨 빠짐"
        elif has_duplicate_box(boxes):
            reason = "라벨 겹침"
        else:
            # 3. 통과
            clean[name] = boxes
            continue

        # 2. 제외 이유 출력
        print(f"  제외 ({reason}): {name}")

    return clean


def is_valid(b):
    """박스가 이미지 안에 제대로 들어있는지 검사

    입력:
        b (dict): 박스 1개

    반환:
        bool: 다섯 조건을 모두 만족하면 True, 하나라도 어기면 False

    동작:
        1. 왼쪽 끝(x)과 위쪽 끝(y)이 0 이상인지
        2. 너비와 높이가 0 보다 큰지
        3. 오른쪽 끝(x + w)이 이미지 너비 이하, 아래쪽 끝(y + h)이 이미지 높이 이하인지
    """
    return (
        b["x"] >= 0
        and b["y"] >= 0
        and b["w"] > 0
        and b["h"] > 0
        and b["x"] + b["w"] <= IMG_W
        and b["y"] + b["h"] <= IMG_H
    )


def has_missing_label(name, boxes):
    """사진에 있는 약 중에 라벨이 없는 약이 있는지 검사

    입력:
        name (str): 이미지 파일명
        boxes (list): 그 이미지의 박스 리스트

    반환:
        bool: 파일명에는 있는데 박스에는 없는 약이 하나라도 있으면 True

    동작:
        1. 박스들의 약 ID 를 모은다.
        2. 파일명의 약 ID 를 하나씩 보면서 박스 목록에 없으면 True
        3. 끝까지 다 있으면 False

    라벨이 빠진 알약은 학습 때 '배경'으로 배워서 모델이 그 알약을 무시하게 된다.
    """
    # 1. 라벨이 붙은 약 ID
    labeled = []
    for b in boxes:
        labeled.append(b["class_id"])

    # 2. 파일명의 약이 라벨에 있는지
    for code in get_codes(name):
        if code not in labeled:
            return True

    # 3. 전부 있음
    return False


def get_codes(name):
    """파일명에서 사진에 들어 있는 약 ID 목록을 꺼낸다

    입력:
        name (str): 이미지 파일명 또는 조합 ID
                    예) "K-001900-016548-019607-033009_0_2_0_2_70_000_200.png"

    반환:
        list: 약 ID 숫자 리스트
              예) [1900, 16548, 19607, 33009]

    동작:
        1. get_combo() 로 조합 ID 만 꺼낸다.  예) "K-001900-016548-019607-033009"
        2. "-" 로 나누고 맨 앞 "K" 는 뺀다.
        3. 나머지를 숫자로 바꾼다. ("001900" -> 1900)
    """
    # 1. 조합 ID
    combo = get_combo(name)

    # 2~3. "K" 빼고 숫자로
    codes = []
    for code in combo.split("-")[1:]:
        codes.append(int(code))
    return codes


def get_combo(name):
    """파일명에서 조합 ID 부분만 꺼낸다

    입력:
        name (str): 이미지 파일명
                    예) "K-001900-016548-019607-029451_0_2_0_2_70_000_200.png"

    반환:
        str: 첫 번째 "_" 앞부분
             예) "K-001900-016548-019607-029451"

    동작:
        1. 첫 번째 "_" 에서 한 번만 자르고 앞부분을 돌려준다.
           (같은 조합을 각도만 바꿔 찍은 사진 3장은 이 값이 같다)
    """
    return name.split("_", 1)[0]


def has_duplicate_box(boxes, threshold=0.9):
    """한 알약 자리에 라벨이 두 개 붙었는지 검사

    입력:
        boxes (list): 이미지 1장의 박스 리스트
        threshold (float): 이만큼 겹치면 중복으로 본다 (기본 0.9)

    반환:
        bool: 겹치는 박스 짝이 하나라도 있으면 True

    동작:
        1. 박스를 두 개씩 짝지어 본다.
        2. IoU 가 threshold 이상이면 True
        3. 끝까지 없으면 False

    알약은 서로 떨어져 놓여 있어서 정상이면 IoU 가 0 에 가깝다.
    0.9 넘게 겹쳤다면 한 알약에 라벨 두 개가 붙은 것이고, 그러면 다른 알약은 라벨이 없다.
    예) 16548 과 33009 가 완전히 같은 좌표를 가진 원본 JSON 오류.
    """
    # 1~2. 두 개씩 짝지어 비교
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if compute_iou(boxes[i], boxes[j]) >= threshold:
                return True

    # 3. 겹침 없음
    return False


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
        1. 모든 박스의 class_id 를 set 에 모아 중복을 없앤다.
        2. 정렬한다. (정렬해야 누가 돌려도 같은 번호가 나온다)
        3. 순서대로 0, 1, 2 ... 번호를 붙인다.

    YOLO는 0부터 빈틈 없이 이어지는 번호만 받는다.
    """
    # 1. 중복 없이 모으기
    unique_ids = set()
    for boxes in ann.values():
        for b in boxes:
            unique_ids.add(b["class_id"])

    # 2. 정렬
    class_ids = sorted(unique_ids)

    # 3. 번호 붙이기
    class_to_idx = {}
    for i, class_id in enumerate(class_ids):
        class_to_idx[class_id] = i
    return class_ids, class_to_idx


# ---------- 3. train / val 나누기 ----------
def split_by_combo(names):
    """이미지 목록을 조합 단위로 train / val 로 나눈다

    입력:
        names (list): 이미지 파일명 리스트 (정렬된 상태로 넣는다)

    반환:
        train (list): train 에 들어갈 이미지 파일명 리스트
        val (list): val 에 들어갈 이미지 파일명 리스트

    동작:
        1. 파일명마다 조합 ID를 뽑고 중복을 없애서 정렬한다.
        2. 시드를 고정하고 섞는다.
        3. 앞에서 20% 조합을 val 후보로 정한다.
        4. train 에서 아예 사라지는 약이 생기면, 그 약이 든 val 조합을 train 으로 되돌린다.
           없어질 때까지 반복한다.
        5. 이미지마다 조합이 val 쪽이면 val, 아니면 train 에 넣는다.

    이미지가 아니라 조합으로 나누는 이유:
        같은 조합의 70°/75°/90° 사진이 train과 val에 나뉘어 들어가면
        거의 같은 사진을 보고 시험 치는 셈이라 val 점수가 실제보다 좋게 나온다.
    """
    # 1. 조합 ID 목록 (정렬해야 섞기 전 순서가 매번 같다)
    unique_combos = set()
    for name in names:
        unique_combos.add(get_combo(name))
    combos = sorted(unique_combos)

    # 2. 시드 고정 후 섞기 (전역 시드를 건드리지 않게 따로 만든다)
    random.Random(SEED).shuffle(combos)

    # 3. 앞에서 20%를 val 후보로
    n_val = int(len(combos) * VAL_RATIO)
    val_combos = set(combos[:n_val])

    # 4. train 에서 사라지는 약이 있으면 그 약이 든 val 조합을 train 으로 되돌린다.
    #    train 박스가 0개인 약은 학습 자체가 불가능한데 val 에서는 점수만 깎인다.
    #    (56종 중 17종이 조합 1개뿐이라 시드에 따라 매번 다른 약이 이렇게 죽는다)
    all_codes = set()
    for combo in combos:
        for code in get_codes(combo):
            all_codes.add(code)

    while True:
        # 4-a. 지금 train 에 있는 약
        train_codes = set()
        for combo in combos:
            if combo not in val_combos:
                for code in get_codes(combo):
                    train_codes.add(code)

        # 4-b. 사라진 약이 없으면 끝
        missing = all_codes - train_codes
        if not missing:
            break

        # 4-c. 사라진 약이 들어 있는 val 조합 하나를 train 으로 되돌린다 (정렬해서 매번 같은 조합을 고른다)
        for combo in sorted(val_combos):
            if has_any(get_codes(combo), missing):
                val_combos.remove(combo)
                break

    # 5. 이미지를 조합에 따라 나누기
    train = []
    val = []
    for name in names:
        if get_combo(name) in val_combos:
            val.append(name)
        else:
            train.append(name)
    return train, val


def has_any(codes, targets):
    """codes 중에 targets 에 들어 있는 값이 하나라도 있는지 검사

    입력:
        codes (list): 약 ID 리스트  예) [1900, 16548, 19607, 33009]
        targets (set): 찾을 약 ID  예) {33009}

    반환:
        bool: 하나라도 있으면 True

    동작:
        1. codes 를 하나씩 보면서 targets 에 있으면 True
        2. 끝까지 없으면 False
    """
    # 1. 하나씩 확인
    for code in codes:
        if code in targets:
            return True

    # 2. 없음
    return False


# ---------- 4. 파일 쓰기 ----------
def save_split(ann, names, split, class_to_idx, image_paths):
    """한 분할(train 또는 val)의 이미지와 라벨 파일을 만든다

    입력:
        ann (dict): 이미지 파일명 -> 박스 리스트
        names (list): 이 분할에 들어갈 이미지 파일명 리스트
        split (str): "train" 또는 "val" (폴더 이름으로 쓴다)
        class_to_idx (dict): 약 ID -> YOLO 번호
        image_paths (dict): 이미지 파일명 -> 원본 png 경로 (Kaggle, AI Hub 위치가 달라서 따로 받는다)

    반환:
        없음. 파일만 만든다.
            data/yolo/images/<split>/<이미지명>.png   원본 연결 (하드링크)
            data/yolo/labels/<split>/<이미지명>.txt   박스 1개 = 한 줄

    동작:
        1. images/<split>, labels/<split> 폴더를 만든다.
        2. 이미지마다
           a. 원본 png 를 images 폴더에 하드링크로 만든다. (AI Hub 는 약 20GB 라 복사하면 디스크를 두 배로 쓴다)
              하드링크가 안 되는 경우(다른 디스크 등)에만 복사한다.
           b. 박스마다 to_yolo_line() 으로 한 줄씩 만든다.
           c. 줄들을 합쳐서 같은 이름의 .txt 로 저장한다.
    """
    # 1. 폴더 만들기 (이미 있어도 에러 안 나게 exist_ok=True)
    img_dir = YOLO_DIR / "images" / split
    lbl_dir = YOLO_DIR / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        # 2-a. 이미지 연결 (하드링크 = 같은 파일을 가리키는 또 하나의 이름. 디스크를 더 쓰지 않는다)
        try:
            os.link(image_paths[name], img_dir / name)
        except OSError:
            shutil.copy(image_paths[name], img_dir / name)

        # 2-b. 박스마다 약 ID -> YOLO 번호로 바꾸고 한 줄로 변환
        lines = []
        for b in ann[name]:
            lines.append(to_yolo_line(b, class_to_idx[b["class_id"]]))

        # 2-c. 줄바꿈으로 이어서 저장 (파일명은 이미지와 같고 확장자만 .txt)
        text = "\n".join(lines) + "\n"
        (lbl_dir / name.replace(".png", ".txt")).write_text(text, encoding="utf-8")


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
    # 1~2. 중심과 크기를 비율로
    cx = (b["x"] + b["w"] / 2) / IMG_W
    cy = (b["y"] + b["h"] / 2) / IMG_H
    w = b["w"] / IMG_W
    h = b["h"] / IMG_H

    # 3. 한 줄로
    return f"{idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def save_yaml(class_ids):
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
        3. 파일로 저장한다.
    """
    # 1. 데이터 위치
    lines = [
        f"path: {YOLO_DIR}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]

    # 2. 번호 -> 약 ID
    for i, class_id in enumerate(class_ids):
        lines.append(f"  {i}: '{class_id}'")

    # 3. 저장
    (YOLO_DIR / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- 5. AI Hub 데이터 더하기 ----------
def add_aihub(ann, train, image_paths):
    """AI Hub 이미지를 train 에만 더한다 (val 은 Kaggle 이미지 그대로)

    입력:
        ann (dict): Kaggle 이미지 파일명 -> 박스 리스트 (이 함수가 AI Hub 이미지를 추가한다)
        train (list): Kaggle train 이미지 파일명 리스트 (이 함수가 AI Hub 이미지를 추가한다)
        image_paths (dict): 이미지 파일명 -> 원본 png 경로 (이 함수가 AI Hub 이미지를 추가한다)

    반환:
        int: 추가한 AI Hub 이미지 수

    동작:
        1. AI Hub 라벨을 읽는다. (깨진 라벨 · png 없는 이미지는 여기서 빠진다)
        2. Kaggle 과 같은 기준(remove_bad_images)으로 문제 이미지를 뺀다.
        3. AIHUB_ANGLES 에 있는 각도의 이미지만 ann, image_paths, train 에 추가한다.
    """
    # 1. 라벨
    aihub_ann, aihub_paths = load_aihub_annotations()

    # 2. 문제 이미지 빼기
    aihub_ann = remove_bad_images(aihub_ann)

    # 3. 쓸 각도만 추가
    n_added = 0
    for name in sorted(aihub_ann):
        if get_angle(name) not in AIHUB_ANGLES:
            continue
        ann[name] = aihub_ann[name]
        image_paths[name] = aihub_paths[name]
        train.append(name)
        n_added += 1
    print(f"  AI Hub train 에 {n_added}장 추가 (각도 {AIHUB_ANGLES})")
    return n_added


def get_angle(name):
    """파일명에서 카메라 각도를 꺼낸다

    입력:
        name (str): 이미지 파일명
                    예) "K-001900-016548-019607-029451_0_2_0_2_75_000_200.png"

    반환:
        str: 각도  예) "75"

    동작:
        1. "_" 로 나눴을 때 6번째 값(인덱스 5)이 각도다.
           ("조합ID_0_2_0_2_각도_000_200.png" 형식)
    """
    return name.split("_")[5]


# ---------- 실행 ----------
def main():
    """전체 변환 순서

    입력: 없음

    반환:
        없음. data/yolo/ (USE_AIHUB 이면 data/yolo_aihub/) 를 새로 만들고 요약을 출력한다.

    동작:
        0. 이전 결과 폴더가 있으면 지운다. (남아 있으면 예전 분할과 섞인다)
        1. Kaggle JSON 읽기 -> 문제 있는 이미지 빼기
        2. Kaggle 이미지만 조합 단위로 train/val 나누기 (val 은 USE_AIHUB 와 상관없이 항상 같다)
        3. USE_AIHUB 이면 AI Hub 이미지를 train 에만 더하기
        4. 클래스 번호표 만들기 (train/val 에 있는 약 전체)
        5. train, val 파일 쓰기 + data.yaml 쓰기
        6. 요약 출력 (train 에 한 번도 안 나온 클래스도 확인)
    """
    # 0. 이전 결과 지우기
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)

    print("[1] 이상한 박스 걸러내기")
    ann = remove_bad_images(load_annotations())
    image_paths = {}
    for name in ann:
        image_paths[name] = KAGGLE_DIR / "train_images" / name

    print("[2] train/val 나누기 (Kaggle 이미지만)")
    train, val = split_by_combo(sorted(ann))  # sorted(ann) = 이미지 파일명 정렬 리스트
    n_kaggle_train = len(train)

    # 3. AI Hub
    n_aihub = 0
    if USE_AIHUB:
        print("[3] AI Hub 이미지를 train 에 더하기")
        n_aihub = add_aihub(ann, train, image_paths)

    print("[4] 클래스 번호 만들기")
    class_ids, class_to_idx = make_class_map(ann)

    print("[5] 파일 쓰기")
    save_split(ann, train, "train", class_to_idx, image_paths)
    save_split(ann, val, "val", class_to_idx, image_paths)
    save_yaml(class_ids)

    # 6. 확인: val 에만 있고 train 에 없는 클래스는 학습이 안 된다
    train_classes = set()
    for name in train:
        for b in ann[name]:
            train_classes.add(b["class_id"])
    missing = sorted(set(class_ids) - train_classes)

    print(
        f"\n클래스 {len(class_ids)}개 / train {len(train)}장 (Kaggle {n_kaggle_train} + AI Hub {n_aihub}) / val {len(val)}장"
    )
    print("train에 없는 클래스 (학습 안 됨):", missing)
    print("저장 위치:", YOLO_DIR)


if __name__ == "__main__":
    main()
