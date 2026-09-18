"""헷갈리는 클래스가 포함된 학습 이미지를 오버샘플링

주의: make_yolo.py를 다시 돌리면 data/yolo/가 새로 만들어지면서
      여기서 만든 복제본이 사라짐 -> 매번 make_yolo.py 다음, train.py 전에 실행

실행: uv run python src/oversample.py
"""
from pathlib import Path
import shutil
import yaml

YOLO_DIR = Path(__file__).resolve().parent.parent / "data" / "yolo"
IMG_DIR = YOLO_DIR / "images" / "train"
LBL_DIR = YOLO_DIR / "labels" / "train"
DATA_YAML = YOLO_DIR / "data.yaml"

RISKY_IDS = {20238, 38162, 19232, 32310, 29667, 16548, 35206}  # 실제 약 ID
N_COPIES = 2  # 몇 배로 늘릴지 (2 = 원본 포함 2번 노출)

names = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))["names"]
if isinstance(names, list):  # names가 리스트 형식인 경우 대비
    names = dict(enumerate(names))
id_to_idx = {str(v): k for k, v in names.items()}  # 값을 문자열로 통일해서 매칭
risky_idx = {id_to_idx[str(i)] for i in RISKY_IDS if str(i) in id_to_idx}

missing = [i for i in RISKY_IDS if str(i) not in id_to_idx]
if missing:
    print(f"경고: data.yaml에서 못 찾은 약 ID: {missing} (오타/클래스 미포함 확인)")

count = 0
for lbl_path in LBL_DIR.glob("*.txt"):
    classes_in_label = {
        int(line.split()[0]) for line in lbl_path.read_text().splitlines() if line.strip()
    }
    if classes_in_label & risky_idx:
        img_path = IMG_DIR / f"{lbl_path.stem}.png"
        for i in range(1, N_COPIES):
            shutil.copy(img_path, IMG_DIR / f"{lbl_path.stem}_dup{i}.png")
            shutil.copy(lbl_path, LBL_DIR / f"{lbl_path.stem}_dup{i}.txt")
        count += 1

print(f"오버샘플링 대상 이미지 {count}장, 각 {N_COPIES - 1}장씩 복제")