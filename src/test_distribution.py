from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter


# ==========================================
# 경로
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data" / "raw" / "sprint_ai_project1_data"

TRAIN_DIR = ROOT / "data" / "yolo" / "images" / "train"
VAL_DIR = ROOT / "data" / "yolo" / "images" / "val"
TEST_DIR = RAW / "test_images"


# ==========================================
# 이미지 분석
# ==========================================

def analyze_images(image_dir, name):
    files = list(image_dir.rglob("*.png"))

    if not files:
        print(f"\n[{name}] 이미지가 없습니다.")
        print(f"확인 경로 : {image_dir}")
        return None

    widths = []
    heights = []
    brightness = []
    contrast = []

    for file in files:
        try:
            img = Image.open(file).convert("RGB")
            arr = np.array(img).astype(np.float32)

            h, w = arr.shape[:2]

            widths.append(w)
            heights.append(h)

            gray = arr.mean(axis=2)

            brightness.append(gray.mean())
            contrast.append(gray.std())

        except Exception as e:
            print(f"읽기 실패 : {file.name} / {e}")

    result = {
        "files": files,
        "count": len(files),
        "widths": widths,
        "heights": heights,
        "brightness": brightness,
        "contrast": contrast,
    }

    print("\n" + "=" * 50)
    print(f"{name} 이미지 분석")
    print("=" * 50)

    print(f"이미지 수 : {len(files)}")

    print("\n[해상도]")
    print(f"가로 : {min(widths)} ~ {max(widths)}")
    print(f"세로 : {min(heights)} ~ {max(heights)}")
    print(
        f"평균 : "
        f"{np.mean(widths):.1f} x "
        f"{np.mean(heights):.1f}"
    )

    print("\n[밝기]")
    print(f"최소 : {min(brightness):.2f}")
    print(f"최대 : {max(brightness):.2f}")
    print(f"평균 : {np.mean(brightness):.2f}")

    print("\n[대비]")
    print(f"최소 : {min(contrast):.2f}")
    print(f"최대 : {max(contrast):.2f}")
    print(f"평균 : {np.mean(contrast):.2f}")

    return result


# ==========================================
# 파일명 그룹 분석
# ==========================================

def get_combo(name):
    """
    이미지 파일명에서 첫 번째 '_' 앞부분을 그룹으로 사용합니다.

    예:
    123_001.png → 123
    """

    return Path(name).stem.split("_")[0]


def analyze_groups(data, name):
    if data is None:
        return

    combos = [
        get_combo(file.name)
        for file in data["files"]
    ]

    counter = Counter(combos)

    print("\n" + "=" * 50)
    print(f"{name} 그룹 분석")
    print("=" * 50)

    print(f"그룹 수 : {len(counter)}")

    print("\n[그룹별 이미지 수]")

    for combo, count in sorted(
        counter.items(),
        key=lambda x: (-x[1], x[0])
    ):
        print(f"{combo:15} : {count}")


# ==========================================
# Train / Val / Test 비교
# ==========================================

def compare_datasets(train, val, test):

    datasets = {
        "Train": train,
        "Validation": val,
        "Test": test,
    }

    print("\n" + "=" * 50)
    print("Train / Validation / Test 비교")
    print("=" * 50)

    print("\n[이미지 수]")

    for name, data in datasets.items():
        if data:
            print(f"{name:12} : {data['count']}")

    print("\n[평균 해상도]")

    for name, data in datasets.items():
        if data:
            print(
                f"{name:12} : "
                f"{np.mean(data['widths']):.1f} x "
                f"{np.mean(data['heights']):.1f}"
            )

    print("\n[평균 밝기]")

    for name, data in datasets.items():
        if data:
            print(
                f"{name:12} : "
                f"{np.mean(data['brightness']):.2f}"
            )

    print("\n[평균 대비]")

    for name, data in datasets.items():
        if data:
            print(
                f"{name:12} : "
                f"{np.mean(data['contrast']):.2f}"
            )

    if train and val and test:

        train_brightness = np.mean(
            train["brightness"]
        )
        val_brightness = np.mean(
            val["brightness"]
        )
        test_brightness = np.mean(
            test["brightness"]
        )

        train_contrast = np.mean(
            train["contrast"]
        )
        val_contrast = np.mean(
            val["contrast"]
        )
        test_contrast = np.mean(
            test["contrast"]
        )

        print("\n[Train 기준 차이]")

        print(
            f"밝기 차이  Train ↔ Val  : "
            f"{abs(train_brightness - val_brightness):.2f}"
        )

        print(
            f"밝기 차이  Train ↔ Test : "
            f"{abs(train_brightness - test_brightness):.2f}"
        )

        print(
            f"대비 차이  Train ↔ Val  : "
            f"{abs(train_contrast - val_contrast):.2f}"
        )

        print(
            f"대비 차이  Train ↔ Test : "
            f"{abs(train_contrast - test_contrast):.2f}"
        )


# ==========================================
# 실행
# ==========================================

def main():

    print("=" * 50)
    print("Experiment 10 - Test Distribution Analysis")
    print("=" * 50)

    train = analyze_images(
        TRAIN_DIR,
        "Train"
    )

    val = analyze_images(
        VAL_DIR,
        "Validation"
    )

    test = analyze_images(
        TEST_DIR,
        "Test"
    )

    compare_datasets(
        train,
        val,
        test
    )

    analyze_groups(
        train,
        "Train"
    )

    analyze_groups(
        val,
        "Validation"
    )

    analyze_groups(
        test,
        "Test"
    )


if __name__ == "__main__":
    main()

