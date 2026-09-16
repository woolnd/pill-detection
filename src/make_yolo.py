"""COCO JSON -> YOLO 형식 변환 + train/val 분할

실행:
    uv run python src/make_yolo.py

결과:
    data/yolo/
        ├── images/train/
        ├── images/val/
        ├── labels/train/
        ├── labels/val/
        └── data.yaml

전처리:
    1. 라벨 없는 이미지도 유지
    2. 완전히 동일한 중복 라벨은 하나만 유지
    3. 이미지 밖으로 나간 bbox는 이미지 안으로 보정
    4. 모든 클래스가 train에 최소 1번 이상 포함되도록 분할
    5. 같은 combo의 이미지는 train/val에 섞이지 않도록 유지
"""

import random
import shutil
from collections import Counter
from pathlib import Path

from annotations import RAW, load_annotations


# ===== 설정값 =====
OUT = Path(__file__).resolve().parent.parent / "data" / "yolo"

VAL_RATIO = 0.2
SEED = 42

# 모든 이미지 크기
IMG_W, IMG_H = 976, 1280


# ---------- 1. bbox 보정 ----------
def fix_bbox(b):
    """이미지 밖으로 나간 bbox를 이미지 안쪽으로 보정한다.

    입력:
        b (dict):
            {
                "x": ...,
                "y": ...,
                "w": ...,
                "h": ...,
                "class_id": ...,
                "class_name": ...
            }

    반환:
        dict:
            보정된 bbox

        None:
            이미지와 완전히 겹치지 않아 사용할 수 없는 bbox
    """

    # bbox의 왼쪽 위 좌표
    x1 = max(0, b["x"])
    y1 = max(0, b["y"])

    # bbox의 오른쪽 아래 좌표
    x2 = min(IMG_W, b["x"] + b["w"])
    y2 = min(IMG_H, b["y"] + b["h"])

    # 보정된 너비와 높이
    w = x2 - x1
    h = y2 - y1

    # 이미지와 완전히 겹치지 않는 bbox
    if w <= 0 or h <= 0:
        return None

    # 기존 bbox를 복사해서 수정
    fixed = b.copy()

    fixed["x"] = x1
    fixed["y"] = y1
    fixed["w"] = w
    fixed["h"] = h

    return fixed


# ---------- 2. 데이터 전처리 ----------
def preprocess_annotations(ann):
    """데이터를 최대한 유지하면서 학습 가능한 형태로 전처리한다.

    전처리 기준:
        - 라벨 없는 이미지: 유지
        - 완전히 동일한 중복 라벨: 중복 제거
        - 이미지 밖 bbox: 이미지 안으로 보정
        - 보정 후 사용할 수 없는 bbox: 해당 bbox만 제외
        - bbox가 모두 사라져도 이미지 자체는 유지
    """

    clean = {}

    # 통계 출력용
    no_label_count = 0
    duplicate_count = 0
    clipped_count = 0
    removed_box_count = 0

    for name, boxes in ann.items():

        # --------------------------------
        # 1. 라벨 없는 이미지도 유지
        # --------------------------------
        if not boxes:
            clean[name] = []
            no_label_count += 1
            continue

        new_boxes = []
        seen = set()

        for b in boxes:

            # --------------------------------
            # 2. 완전히 동일한 중복 라벨 제거
            # --------------------------------
            key = (
                b["x"],
                b["y"],
                b["w"],
                b["h"],
                b["class_id"],
            )

            if key in seen:
                duplicate_count += 1
                continue

            seen.add(key)

            # --------------------------------
            # 3. bbox가 이미지 밖으로 나간 경우 보정
            # --------------------------------
            original = (
                b["x"],
                b["y"],
                b["w"],
                b["h"],
            )

            fixed = fix_bbox(b)

            # 이미지와 완전히 겹치지 않는 bbox
            if fixed is None:
                removed_box_count += 1
                continue

            fixed_box = (
                fixed["x"],
                fixed["y"],
                fixed["w"],
                fixed["h"],
            )

            if original != fixed_box:
                clipped_count += 1

            new_boxes.append(fixed)

        # bbox가 전부 없어졌더라도 이미지 자체는 유지
        clean[name] = new_boxes

    # 전처리 결과 출력
    print("\n[전처리 결과]")
    print(f"라벨 없는 이미지: {no_label_count}장")
    print(f"중복 라벨 제거: {duplicate_count}개")
    print(f"이미지 밖 bbox 보정: {clipped_count}개")
    print(f"사용할 수 없어 제거된 bbox: {removed_box_count}개")

    return clean


# ---------- 3. 클래스 번호 만들기 ----------
def make_class_map(ann):
    """약 ID를 YOLO 클래스 번호로 변환하는 표를 만든다.

    예:
        약 ID
        1900
        2483
        3351

        ↓

        YOLO 클래스 번호
        0
        1
        2
    """

    # 모든 bbox에서 class_id를 가져와 중복 제거 후 정렬
    class_ids = sorted(
        {
            b["class_id"]
            for boxes in ann.values()
            for b in boxes
        }
    )

    # 약 ID -> YOLO 클래스 번호
    class_to_idx = {
        class_id: i
        for i, class_id in enumerate(class_ids)
    }

    return class_ids, class_to_idx

def print_class_counts(ann):
    """클래스별 bbox 개수를 출력한다."""

    counts = Counter(
        b["class_id"]
        for boxes in ann.values()
        for b in boxes
    )

    print("\n[클래스별 데이터 개수]")
    print(f"전체 클래스 수: {len(counts)}개")

    for class_id, count in sorted(counts.items()):
        print(f"class_id {class_id}: {count}개")


# ---------- 4. COCO -> YOLO 좌표 변환 ----------
def to_yolo_line(b, idx):
    """bbox 하나를 YOLO 형식 한 줄로 변환한다.

    YOLO 형식:
        class_id center_x center_y width height

    좌표는 모두 0~1 사이의 비율로 변환한다.
    """

    # 중심 좌표
    cx = (
        b["x"] + (b["w"] / 2)
    ) / IMG_W

    cy = (
        b["y"] + (b["h"] / 2)
    ) / IMG_H

    # 너비 / 높이
    w = b["w"] / IMG_W
    h = b["h"] / IMG_H

    return f"{idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# ---------- 5. combo ID 가져오기 ----------
def get_combo(name):
    """파일명에서 combo ID를 가져온다.

    예:
        K-001900-016548-019607-029451_0_2_0_2_70_000_200.png

    결과:
        K-001900-016548-019607-029451
    """

    return name.split("_", 1)[0]


# ---------- 6. train / val 분할 ----------
def split_by_combo(names, ann):
    """combo 단위로 train / val을 나눈다.

    조건:
        1. 같은 combo는 train/val에 섞이지 않는다.
        2. 기본적으로 val은 20%로 구성한다.
        3. 모든 클래스가 train에 최소 1번 이상 등장하도록 조정한다.
    """

    # --------------------------------
    # 1. combo 목록 만들기
    # --------------------------------
    combos = sorted(
        set(get_combo(name) for name in names)
    )

    # --------------------------------
    # 2. 랜덤하게 섞기
    # --------------------------------
    random.seed(SEED)
    random.shuffle(combos)

    # --------------------------------
    # 3. 기본 val combo 결정
    # --------------------------------
    n_val = int(len(combos) * VAL_RATIO)

    val_combos = set(combos[:n_val])

    # --------------------------------
    # 4. 전체 클래스 확인
    # --------------------------------
    all_classes = {
        b["class_id"]
        for boxes in ann.values()
        for b in boxes
    }

    # --------------------------------
    # 5. 현재 train에 존재하는 클래스 확인
    # --------------------------------
    train_classes = {
        b["class_id"]
        for name in names
        if get_combo(name) not in val_combos
        for b in ann[name]
    }

    # train에 없는 클래스
    missing = all_classes - train_classes

    print("\n[클래스 확인]")
    print(f"전체 클래스: {len(all_classes)}개")
    print(f"현재 train 클래스: {len(train_classes)}개")

    if missing:
        print(
            f"train에 없는 클래스: {len(missing)}개"
        )

        # --------------------------------
        # 6. missing class를 가지고 있는
        #    val combo를 train으로 이동
        # --------------------------------
        while missing:

            candidates = []

            for combo in val_combos:

                combo_classes = {
                    b["class_id"]
                    for name in names
                    if get_combo(name) == combo
                    for b in ann[name]
                }

                covered = combo_classes & missing

                if covered:
                    candidates.append(
                        (
                            len(covered),
                            combo,
                            covered,
                        )
                    )

            # 후보 combo가 없는 경우
            if not candidates:
                print(
                    "경고: train에 추가할 수 있는 combo를 찾지 못했습니다."
                )
                break

            # 가장 많은 missing class를 해결하는 combo 선택
            _, selected_combo, covered = max(
                candidates,
                key=lambda x: x[0],
            )

            # val -> train으로 이동
            val_combos.remove(selected_combo)

            # 해결된 클래스 제거
            missing -= covered

            print(
                f"combo 이동: {selected_combo} "
                f"→ train "
                f"(클래스 {len(covered)}개 확보)"
            )

    # --------------------------------
    # 7. 최종 train / val 이미지 생성
    # --------------------------------
    train = [
        name
        for name in names
        if get_combo(name) not in val_combos
    ]

    val = [
        name
        for name in names
        if get_combo(name) in val_combos
    ]

    # --------------------------------
    # 8. 최종 클래스 확인
    # --------------------------------
    train_classes = {
        b["class_id"]
        for name in train
        for b in ann[name]
    }

    missing = all_classes - train_classes

    print("\n[최종 분할 결과]")
    print(f"train: {len(train)}장")
    print(f"val: {len(val)}장")
    print(f"train 클래스: {len(train_classes)}개")
    print(f"train에 없는 클래스: {sorted(missing)}")

    return train, val


# ---------- 7. 이미지 + YOLO 라벨 저장 ----------
def write_split(ann, names, split, class_to_idx):
    """train 또는 val 데이터를 저장한다."""

    img_dir = OUT / "images" / split
    lbl_dir = OUT / "labels" / split

    img_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    lbl_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name in names:

        # --------------------------------
        # 1. 이미지 복사
        # --------------------------------
        shutil.copy(
            RAW / "train_images" / name,
            img_dir / name,
        )

        # --------------------------------
        # 2. YOLO 라벨 생성
        # --------------------------------
        lines = [
            to_yolo_line(
                b,
                class_to_idx[b["class_id"]],
            )
            for b in ann[name]
        ]

        # --------------------------------
        # 3. txt 저장
        #
        # 라벨이 없는 이미지라면
        # 빈 txt 파일이 만들어진다.
        # --------------------------------
        label_path = (
            lbl_dir
            / name.replace(".png", ".txt")
        )

        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )


# ---------- 8. data.yaml 생성 ----------
def write_yaml(class_ids):
    """YOLO 학습 설정 파일을 만든다."""

    lines = [
        f"path: {OUT}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]

    for i, class_id in enumerate(class_ids):
        lines.append(
            f"  {i}: '{class_id}'"
        )

    (OUT / "data.yaml").write_text(
        "\n".join(lines) + "\n"
    )


# ---------- 9. 전체 실행 ----------
def main():
    """COCO -> YOLO 전체 변환 과정."""

    # --------------------------------
    # 0. 이전 결과 삭제
    # --------------------------------
    if OUT.exists():
        shutil.rmtree(OUT)

    # --------------------------------
    # 1. 원본 annotation 불러오기
    # --------------------------------
    print("[1] 원본 annotation 불러오기")

    raw_ann = load_annotations()

    print(f"원본 이미지: {len(raw_ann)}장")

    # --------------------------------
    # 2. 전처리
    # --------------------------------
    print("\n[2] 데이터 전처리")

    ann = preprocess_annotations(raw_ann)

    print(f"전처리 후 이미지: {len(ann)}장")

    # --------------------------------
    # 3. 클래스 번호 만들기
    # --------------------------------
    print("\n[3] 클래스 번호 만들기")

    class_ids, class_to_idx = make_class_map(ann)

    print(f"전체 클래스: {len(class_ids)}개")

    print_class_counts(ann)

    # --------------------------------
    # 4. train / val 분할
    # --------------------------------
    print("\n[4] train / val 나누기")

    train, val = split_by_combo(
        sorted(ann),
        ann,
    )

    # --------------------------------
    # 5. 파일 저장
    # --------------------------------
    print("\n[5] YOLO 파일 생성")

    write_split(
        ann,
        train,
        "train",
        class_to_idx,
    )

    write_split(
        ann,
        val,
        "val",
        class_to_idx,
    )

    # --------------------------------
    # 6. data.yaml 생성
    # --------------------------------
    write_yaml(class_ids)

    # --------------------------------
    # 7. 최종 확인
    # --------------------------------
    train_classes = {
        b["class_id"]
        for name in train
        for b in ann[name]
    }

    missing = sorted(
        set(class_ids) - train_classes
    )

    print("\n==============================")
    print("최종 결과")
    print("==============================")
    print(f"클래스: {len(class_ids)}개")
    print(f"전체 이미지: {len(ann)}장")
    print(f"train: {len(train)}장")
    print(f"val: {len(val)}장")
    print(
        f"train에 없는 클래스: {missing}"
    )
    print(f"저장 위치: {OUT}")
    print("==============================")


if __name__ == "__main__":
    main()

