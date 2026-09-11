import random
import shutil
from pathlib import Path

from annotations import RAW, load_annotations


# ===== 설정값 =====
OUT = Path(__file__).resolve().parent.parent / "data" / "yolo"
VAL_RATIO = 0.2
IMG_W, IMG_H = 976, 1280


# ---------- 1. 이상한 박스 걸러내기 ----------

def is_valid(b):
    """박스가 이미지 안에 제대로 들어있는지 검사"""

    x, y, w, h = b["x"], b["y"], b["w"], b["h"]

    return (
        x >= 0
        and y >= 0
        and x + w <= IMG_W
        and y + h <= IMG_H
    )


def remove_bad_images(ann):
    """이상한 박스가 하나라도 있는 이미지를 제거"""

    clean = {}

    for name, boxes in ann.items():
        if all(is_valid(b) for b in boxes):
            clean[name] = boxes

    return clean


# ---------- 2. 클래스 번호 만들기 ----------

def make_class_map(ann):
    """약 ID를 YOLO 번호(0, 1, 2...)로 변환"""

    class_ids = sorted({
        b["class_id"]
        for boxes in ann.values()
        for b in boxes
    })

    class_to_idx = {
        class_id: idx
        for idx, class_id in enumerate(class_ids)
    }

    return class_ids, class_to_idx


# ---------- 3. COCO → YOLO 변환 ----------

def convert_bbox(b):
    """COCO 좌표를 YOLO 좌표로 변환"""

    x, y, w, h = b["x"], b["y"], b["w"], b["h"]

    x_center = (x + w / 2) / IMG_W
    y_center = (y + h / 2) / IMG_H
    width = w / IMG_W
    height = h / IMG_H

    return x_center, y_center, width, height


def to_yolo_line(b, class_to_idx):
    """박스 하나를 YOLO 라벨 한 줄로 변환"""

    class_id = class_to_idx[b["class_id"]]

    x_center, y_center, width, height = convert_bbox(b)

    return (
        f"{class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{width:.6f} "
        f"{height:.6f}"
    )


# ---------- 4. train / val 나누기 ----------

def get_combo(name):
    """파일명에서 조합 ID 부분만 꺼낸다"""

    return name.split("_")[0]


def split_by_combo(names):
    """이미지 목록을 조합 단위로 train / val로 나눈다

    같은 조합의 사진이 train과 val에 동시에 들어가는 것을 방지한다.
    """

    # 1. 조합 ID 목록 -> 중복 제거 -> 정렬
    combos = sorted({get_combo(name) for name in names})

    # 2. 시드 고정 후 섞기
    random.seed(42)
    random.shuffle(combos)

    # 3. 앞에서 20%를 val 조합으로
    n_val = int(len(combos) * VAL_RATIO)
    val_combos = set(combos[:n_val])

    # 4. 이미지를 조합에 따라 나누기
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

    return train, val


# ---------- 5. 파일 쓰기 ----------

def write_split(ann, names, split, class_to_idx):
    """한 분할(train 또는 val)의 이미지와 라벨 파일을 만든다

    이미지:
        data/yolo/images/<split>/

    라벨:
        data/yolo/labels/<split>/
    """

    # 1. 폴더 만들기
    img_dir = OUT / "images" / split
    lbl_dir = OUT / "labels" / split

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for name in names:

        # 2-a. 이미지 복사
        shutil.copy(
            RAW / "sprint_ai_project1_data" / "train_images" / name,
            img_dir / name
        )

        # 2-b. 박스를 YOLO 형식으로 변환
        lines = [
            to_yolo_line(b, class_to_idx)
            for b in ann[name]
        ]

        # 2-c. 라벨 txt 저장
        label_file = lbl_dir / f"{Path(name).stem}.txt"

        label_file.write_text(
            "\n".join(lines) + "\n"
        )


def write_yaml(class_ids):
    """YOLO 학습 설정 파일(data.yaml)을 만든다"""

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

    (OUT / "data.yaml").write_text(
        "\n".join(lines) + "\n"
    )


# ---------- 6. 실행 ----------

def main():
    """전체 변환 순서"""

    # 0. 이전 결과 삭제
    if OUT.exists():
        shutil.rmtree(OUT)

    # 1. 이상한 박스 제거
    print("[1] 이상한 박스 걸러내기")

    ann = remove_bad_images(
        load_annotations()
    )

    # 2. 클래스 번호 만들기
    print("[2] 클래스 번호 만들기")

    class_ids, class_to_idx = make_class_map(ann)

    # 3. Train / Validation 나누기
    print("[3] train/val 나누기")

    train, val = split_by_combo(
        sorted(ann)
    )

    # 4. 파일 쓰기
    print("[4] 파일 쓰기")

    write_split(
        ann,
        train,
        "train",
        class_to_idx
    )

    write_split(
        ann,
        val,
        "val",
        class_to_idx
    )

    write_yaml(class_ids)

    # 5. Train에 없는 클래스 확인
    train_classes = {
        b["class_id"]
        for name in train
        for b in ann[name]
    }

    missing = sorted(
        set(class_ids) - train_classes
    )

    # 결과 출력
    print("\n변환 완료!")
    print("클래스 수:", len(class_ids))
    print("Train 이미지:", len(train))
    print("Validation 이미지:", len(val))

    if missing:
        print("Train에 없는 클래스:", missing)
    else:
        print("모든 클래스가 Train에 포함되어 있습니다.")


if __name__ == "__main__":
    main()