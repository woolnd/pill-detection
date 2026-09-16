import sys
from pathlib import Path
from collections import Counter, defaultdict
import random


# ============================================================
# src 폴더를 Python 모듈 검색 경로에 추가
# ============================================================

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))


from annotations import load_annotations


# ============================================================
# 설정
# ============================================================

VAL_RATIO = 0.2
SEED = 42

IMG_W = 976
IMG_H = 1280


# ============================================================
# 잘못된 박스 제거
# ============================================================

def is_valid(b):
    return (
        b["x"] >= 0
        and b["y"] >= 0
        and b["x"] + b["w"] <= IMG_W
        and b["y"] + b["h"] <= IMG_H
    )


def remove_bad_images(ann):
    clean = {}

    for name, boxes in ann.items():
        if all(is_valid(b) for b in boxes):
            clean[name] = boxes

    return clean


# ============================================================
# combo 관련
# ============================================================

def get_combo(name):
    return name.split("_", 1)[0]


def make_combo_groups(names):
    groups = defaultdict(list)

    for name in names:
        groups[get_combo(name)].append(name)

    return {
        combo: sorted(images)
        for combo, images in groups.items()
    }


# ============================================================
# 클래스별 객체 수
# ============================================================

def get_class_counts(ann, names):
    counts = Counter()

    for name in names:
        for box in ann[name]:
            counts[box["class_id"]] += 1

    return counts


# ============================================================
# Exp05 분할 재현
# ============================================================

def split_exp05(ann):

    names = sorted(ann)

    combo_images = make_combo_groups(names)

    combos = sorted(combo_images)

    random.seed(SEED)
    random.shuffle(combos)

    val_combo_count = round(
        len(combos) * VAL_RATIO
    )

    val_combos = set(
        combos[:val_combo_count]
    )

    val = []

    for combo in val_combos:
        val.extend(
            combo_images[combo]
        )

    val = sorted(val)

    val_set = set(val)

    train = sorted(
        name
        for name in names
        if name not in val_set
    )

    return train, val


# ============================================================
# 현재 make_yolo.py 방식 재현
# ============================================================

def split_current(ann):

    random.seed(SEED)

    names = sorted(ann)

    combo_images = make_combo_groups(names)

    combo_class_counts = {}

    for combo, images in combo_images.items():

        counts = Counter()

        for name in images:
            for box in ann[name]:
                counts[box["class_id"]] += 1

        combo_class_counts[combo] = counts

    total_class_counts = get_class_counts(
        ann,
        names,
    )

    target_val_images = round(
        len(names) * VAL_RATIO
    )

    target_val_counts = {
        class_id: count * VAL_RATIO
        for class_id, count in total_class_counts.items()
    }

    combos = list(combo_images.keys())

    random.shuffle(combos)

    val_images = []
    val_combos = set()

    current_val_counts = Counter()

    while len(val_images) < target_val_images:

        candidates = []

        for combo in combos:

            if combo in val_combos:
                continue

            combo_counts = combo_class_counts[combo]

            # 이 combo를 VAL로 보내면
            # TRAIN에서 해당 클래스가 사라지는지 확인
            valid = True

            for class_id, count in combo_counts.items():

                remaining = (
                    total_class_counts[class_id]
                    - current_val_counts[class_id]
                    - count
                )

                if remaining <= 0:
                    valid = False
                    break

            if not valid:
                continue

            # VAL에서 부족한 클래스에 점수 부여
            gain = 0.0

            for class_id, count in combo_counts.items():

                target = target_val_counts[class_id]
                current = current_val_counts[class_id]

                if current < target:

                    shortage = target - current

                    rarity_weight = (
                        1.0
                        / total_class_counts[class_id]
                    )

                    gain += (
                        shortage
                        * count
                        * rarity_weight
                        * 100
                    )

            new_image_count = (
                len(val_images)
                + len(combo_images[combo])
            )

            overflow = max(
                0,
                new_image_count - target_val_images,
            )

            score = gain - overflow * 10

            candidates.append(
                (
                    score,
                    random.random(),
                    combo,
                )
            )

        if not candidates:
            break

        candidates.sort(
            key=lambda x: (x[0], x[1]),
            reverse=True,
        )

        _, _, selected_combo = candidates[0]

        val_combos.add(selected_combo)

        for name in combo_images[selected_combo]:
            val_images.append(name)

        for class_id, count in combo_class_counts[
            selected_combo
        ].items():

            current_val_counts[class_id] += count

    val = sorted(val_images)

    val_set = set(val)

    train = sorted(
        name
        for name in names
        if name not in val_set
    )

    return train, val


# ============================================================
# 클래스 분포 비교
# ============================================================

def print_class_comparison(
    ann,
    exp05_train,
    exp05_val,
    current_train,
    current_val,
):

    exp05_train_counts = get_class_counts(
        ann,
        exp05_train,
    )

    exp05_val_counts = get_class_counts(
        ann,
        exp05_val,
    )

    current_train_counts = get_class_counts(
        ann,
        current_train,
    )

    current_val_counts = get_class_counts(
        ann,
        current_val,
    )

    all_classes = sorted(
        set(exp05_train_counts)
        | set(exp05_val_counts)
        | set(current_train_counts)
        | set(current_val_counts)
    )

    print()
    print("=" * 80)
    print("클래스별 객체 수 비교")
    print("=" * 80)

    print(
        f"{'CLASS':>8} | "
        f"{'Exp05 TRAIN':>11} | "
        f"{'Exp05 VAL':>9} | "
        f"{'현재 TRAIN':>11} | "
        f"{'현재 VAL':>9}"
    )

    print("-" * 80)

    for class_id in all_classes:

        print(
            f"{class_id:>8} | "
            f"{exp05_train_counts[class_id]:>11} | "
            f"{exp05_val_counts[class_id]:>9} | "
            f"{current_train_counts[class_id]:>11} | "
            f"{current_val_counts[class_id]:>9}"
        )


# ============================================================
# 메인
# ============================================================

def main():

    print("=" * 80)
    print("Exp05 분할 vs 현재 클래스 분포 기반 분할 비교")
    print("=" * 80)

    # Raw annotation 불러오기
    ann = load_annotations()

    # 잘못된 박스 제거
    ann = remove_bad_images(ann)

    print()
    print(f"전체 이미지: {len(ann)}장")

    # --------------------------------------------------------
    # Exp05 방식
    # --------------------------------------------------------

    exp05_train, exp05_val = split_exp05(ann)

    # --------------------------------------------------------
    # 현재 방식
    # --------------------------------------------------------

    current_train, current_val = split_current(ann)

    # --------------------------------------------------------
    # 분할 크기
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("분할 크기")
    print("=" * 80)

    print(
        f"Exp05  TRAIN: {len(exp05_train)}장"
    )

    print(
        f"Exp05  VAL:   {len(exp05_val)}장"
    )

    print(
        f"현재   TRAIN: {len(current_train)}장"
    )

    print(
        f"현재   VAL:   {len(current_val)}장"
    )

    # --------------------------------------------------------
    # VAL 이미지 비교
    # --------------------------------------------------------

    exp05_val_set = set(exp05_val)
    current_val_set = set(current_val)

    removed_from_val = sorted(
        exp05_val_set - current_val_set
    )

    added_to_val = sorted(
        current_val_set - exp05_val_set
    )

    same_val = sorted(
        exp05_val_set & current_val_set
    )

    print()
    print("=" * 80)
    print("VAL 이미지 비교")
    print("=" * 80)

    print(
        f"공통 VAL 이미지: {len(same_val)}장"
    )

    print(
        f"Exp05 → 현재에서 빠진 이미지: "
        f"{len(removed_from_val)}장"
    )

    print(
        f"현재 방식에서 새로 들어온 이미지: "
        f"{len(added_to_val)}장"
    )

    # --------------------------------------------------------
    # 빠진 이미지
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("① Exp05에는 있었지만 현재 VAL에서는 빠진 이미지")
    print("-" * 80)

    if removed_from_val:

        for name in removed_from_val:
            print(name)

    else:
        print("(없음)")

    # --------------------------------------------------------
    # 추가된 이미지
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("② 현재 VAL에 새로 들어온 이미지")
    print("-" * 80)

    if added_to_val:

        for name in added_to_val:
            print(name)

    else:
        print("(없음)")

    # --------------------------------------------------------
    # 클래스 비교
    # --------------------------------------------------------

    print_class_comparison(
        ann,
        exp05_train,
        exp05_val,
        current_train,
        current_val,
    )

    # --------------------------------------------------------
    # 클래스 존재 여부
    # --------------------------------------------------------

    exp05_train_classes = set(
        get_class_counts(
            ann,
            exp05_train,
        )
    )

    exp05_val_classes = set(
        get_class_counts(
            ann,
            exp05_val,
        )
    )

    current_train_classes = set(
        get_class_counts(
            ann,
            current_train,
        )
    )

    current_val_classes = set(
        get_class_counts(
            ann,
            current_val,
        )
    )

    print()
    print("=" * 80)
    print("클래스 존재 여부 비교")
    print("=" * 80)

    print(
        "Exp05 TRAIN에 없는 클래스:",
        sorted(
            exp05_val_classes
            - exp05_train_classes
        ),
    )

    print(
        "현재 TRAIN에 없는 클래스:",
        sorted(
            current_val_classes
            - current_train_classes
        ),
    )

    # --------------------------------------------------------
    # VAL 클래스 개수
    # --------------------------------------------------------

    print()
    print(
        "Exp05 VAL 클래스 종류:",
        len(exp05_val_classes),
    )

    print(
        "현재 VAL 클래스 종류:",
        len(current_val_classes),
    )

    print()
    print("=" * 80)
    print("비교 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()

