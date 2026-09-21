"""제출 파일의 클래스를 대조학습 임베딩으로 다시 정한다 (2단계 분류)

실행: uv run python -m src.analysis.predict_contrastive
      (predict.py 로 submission.csv 를, train_contrastive.py 로 head.pt 를 먼저 만들어야 한다)
결과: runs/<NAME>/submission_contrastive.csv   클래스만 바꾼 제출 파일
      runs/<NAME>/contrastive_changes.csv      바뀐 박스 목록 (원래 클래스 -> 바뀐 클래스)

무엇을 하나:
    1. YOLO 가 찾은 박스는 그대로 두고(위치 유지) 클래스만 다시 고른다.
       박스를 잘라 DINOv2 + 학습된 헤드로 임베딩하고, train 알약의 클래스 중심 중 가장 가까운 것을 고른다.
    2. 가장 가까운 클래스와도 별로 안 닮았으면(유사도 < UNKNOWN_LIMIT) '모르는 약'으로 보고 점수를 낮춘다.
       test 에는 train 56종에 없는 약이 섞여 있는데, 그걸 아는 약으로 찍으면 그 클래스의 오탐이 되어
       원래 맞히던 알약의 점수까지 깎는다. 순위를 내려서 그 오염만 막는다.
       ('미분류' 라는 칸이 제출 형식에 없어서, 점수를 바닥으로 낮추는 방식으로 같은 효과를 낸다)

왜 이렇게 하나:
    val 오류를 보면 위치(LOC)는 0 이고 틀리는 것은 분류였다. 그래서 분류만 교체한다.
    신뢰도가 낮은 박스는 어차피 순위가 낮아 mAP 에 거의 영향이 없어서 MIN_SCORE 이상만 바꾼다.

주의:
    val 은 114 크롭뿐이고 사전학습 임베딩만으로도 이미 0.99 라 로컬에서는 효과를 잴 수 없다.
    좋아졌는지는 Kaggle 제출 점수로만 확인할 수 있다.
"""

import numpy as np
import pandas as pd
import torch
import yaml

from src.analysis.filter_unknown import compute_embedding, crop_pill, load_dino
from src.config import DATA_YAML, KAGGLE_DIR, RUNS_DIR, YOLO_DIR, get_device
from src.train.train_contrastive import CACHE_DIR, CROP, DIM, MARGIN
from src.train.train_contrastive import NAME as HEAD_NAME
from src.train.train_yolo import NAME

# ===== 설정값 =====
SUBMISSION = RUNS_DIR / NAME / "submission.csv"  # predict.py 결과
OUT_CSV = RUNS_DIR / NAME / "submission_contrastive.csv"
CHANGES_CSV = RUNS_DIR / NAME / "contrastive_changes.csv"
HEAD_PT = RUNS_DIR / f"{HEAD_NAME}_{YOLO_DIR.name}" / "head.pt"  # train_contrastive.py 결과 (데이터셋별로 나뉜다)
TEST_DIR = KAGGLE_DIR / "test_images"
MIN_SCORE = 0.1  # 이 점수 이상 박스만 다시 분류한다 (낮은 박스는 순위가 낮아 영향이 작다)
BATCH = 256  # 한 번에 크롭해서 임베딩할 박스 수 (메모리 때문에 나눠서 처리)

# 모르는 약 처리: 가장 가까운 클래스와의 유사도가 이보다 낮으면 '아는 약이 아니다'로 보고 점수를 낮춘다.
# 0 으로 두면 이 처리를 끈다. (필터 켠 버전 / 끈 버전을 둘 다 제출해서 비교한다)
# 0.82 는 val 크롭 114개(전부 아는 약)의 유사도 하위 1% 지점이다. 아는 약을 잘못 거르는 비율 약 1%.
UNKNOWN_LIMIT = 0.82
DOWN = 0.01  # 모르는 약 박스 점수에 곱할 값 (지우지 않고 순위만 내린다)


def load_head(device):
    """학습된 투영 헤드를 불러온다

    입력:
        device (str): get_device() 결과

    반환:
        Module: 768 -> DIM 헤드 (평가 모드)

    동작:
        1. train_contrastive 와 같은 모양으로 만든다.
        2. 저장된 가중치를 넣는다.
    """
    # 1. 같은 모양
    head = torch.nn.Linear(768, DIM)

    # 2. 가중치
    head.load_state_dict(torch.load(HEAD_PT, map_location="cpu"))
    return head.to(device).eval()


def make_centers(head, device):
    """train 임베딩으로 약 ID 별 중심 벡터를 만든다

    입력:
        head (Module): load_head() 결과
        device (str): get_device() 결과

    반환:
        (np.ndarray, list):
            centers (클래스 수, DIM) 길이 1 로 맞춘 중심 벡터
            drug_ids 같은 순서의 약 ID 리스트  예) [1900, 2483, ...]

    동작:
        1. train_contrastive 가 저장한 임베딩 캐시를 읽는다.
        2. 헤드에 통과시킨다.
        3. 클래스마다 평균을 내고 길이를 1 로 맞춘다.
        4. YOLO 번호를 data.yaml 로 약 ID 로 바꾼다. (제출 파일은 약 ID 를 쓴다)
    """
    # 1. 캐시 (train_contrastive.py 를 먼저 돌려야 생긴다)
    cache = CACHE_DIR / f"{YOLO_DIR.name}_train.npz"
    data = np.load(cache)
    X, y = data["X"], data["y"]

    # 2. 헤드 통과
    with torch.no_grad():
        z = head(torch.from_numpy(X).float().to(device))
        z = torch.nn.functional.normalize(z, dim=1).cpu().numpy()

    # 3~4. 클래스별 중심 + 약 ID
    with open(DATA_YAML, encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]

    centers = []
    drug_ids = []
    for class_idx in sorted(set(int(c) for c in y)):
        center = z[y == class_idx].mean(axis=0)
        centers.append(center / np.linalg.norm(center))  # 3.
        drug_ids.append(int(names[class_idx]))  # 4.

    return np.stack(centers), drug_ids


def reclassify(rows, head, dino, centers, drug_ids, device):
    """박스마다 가장 가까운 중심의 약 ID 를 고른다

    입력:
        rows (list): 제출 파일 행 dict 리스트 (image_id, bbox_x, bbox_y, bbox_w, bbox_h)
        head (Module): load_head() 결과
        dino: load_dino() 결과
        centers (np.ndarray): make_centers() 결과
        drug_ids (list): make_centers() 결과
        device (str): get_device() 결과

    반환:
        (list, list):
            picked 행마다 새 약 ID  예) [1900, 33009, ...]
            sims   그때의 유사도 (-1~1). 낮을수록 아는 약 중 닮은 게 없다는 뜻

    동작:
        1. BATCH 개씩 묶어서
           a. 박스를 잘라낸다.
           b. DINOv2 + 헤드로 임베딩한다.
           c. 중심들과의 코사인 유사도가 가장 큰 클래스를 고르고, 그 유사도도 같이 남긴다.
    """
    picked = []
    sims = []

    # 1. 묶음마다
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]

        # 1-a. 크롭
        crops = []
        for r in chunk:
            path = TEST_DIR / f"{r['image_id']}.png"
            box = {"x": r["bbox_x"], "y": r["bbox_y"], "w": r["bbox_w"], "h": r["bbox_h"]}
            crops.append(crop_pill(path, box, CROP, MARGIN))

        # 1-b. 임베딩
        with torch.no_grad():
            z = head(torch.from_numpy(compute_embedding(dino, crops)).float().to(device))
            z = torch.nn.functional.normalize(z, dim=1).cpu().numpy()

        # 1-c. 가장 가까운 중심 + 그때의 유사도
        similarity = z @ centers.T
        for row in similarity:
            i = row.argmax()
            picked.append(drug_ids[i])
            sims.append(float(row[i]))

        print(f"  {min(start + BATCH, len(rows))}/{len(rows)}")

    return picked, sims


def main():
    """전체 순서

    입력: 없음

    반환:
        없음. submission_contrastive.csv 와 contrastive_changes.csv 를 저장한다.

    동작:
        1. 장치·모델·중심 준비
        2. 제출 파일에서 MIN_SCORE 이상 박스만 고른다.
        3. 다시 분류한다. (클래스 + 유사도)
        4. 판정 표를 만든다. 유사도가 UNKNOWN_LIMIT 미만이면 '모르는 약'으로 표시한다.
        5. category_id 를 바꾸고, 모르는 약은 점수에 DOWN 을 곱해서 저장한다.
    """
    # 1. 준비
    device = get_device()
    head = load_head(device)
    dino = load_dino()
    centers, drug_ids = make_centers(head, device)
    print(f"클래스 {len(drug_ids)}종 / 제출 파일: {SUBMISSION}")

    # 2. 대상 박스
    submission = pd.read_csv(SUBMISSION)
    targets = submission[submission["score"] >= MIN_SCORE]
    rows = targets.to_dict("records")
    print(f"다시 분류할 박스: {len(rows)} / 전체 {len(submission)}")

    # 3. 재분류
    picked, sims = reclassify(rows, head, dino, centers, drug_ids, device)

    # 4. 판정 표
    flags = pd.DataFrame(
        {
            "annotation_id": [r["annotation_id"] for r in rows],
            "image_id": [r["image_id"] for r in rows],
            "score": [r["score"] for r in rows],
            "before": [r["category_id"] for r in rows],
            "after": picked,
            "similarity": [round(s, 3) for s in sims],
        }
    )
    flags["unknown"] = flags["similarity"] < UNKNOWN_LIMIT  # 4. 아는 약 중 닮은 게 없다

    # 5-a. 클래스 바꾸기
    out = submission.set_index("annotation_id")
    out.loc[flags["annotation_id"], "category_id"] = flags["after"].values

    # 5-b. 모르는 약은 점수 낮추기 (지우지 않는다. 잘못 걸러도 순위만 내려가서 손해가 작다)
    unknown_ids = flags[flags["unknown"]]["annotation_id"]
    out.loc[unknown_ids, "score"] = (out.loc[unknown_ids, "score"] * DOWN).round(6)

    out.reset_index().to_csv(OUT_CSV, index=False)
    flags.to_csv(CHANGES_CSV, index=False)

    changed = flags[flags["before"] != flags["after"]]
    print(f"클래스가 바뀐 박스: {len(changed)} / {len(rows)}")
    print(f"모르는 약으로 점수 낮춘 박스: {flags['unknown'].sum()} / {len(rows)}  (기준 {UNKNOWN_LIMIT})")
    print("저장 위치:", OUT_CSV)
    print("좋아졌는지는 Kaggle 제출 점수로만 확인할 수 있다.")


if __name__ == "__main__":
    main()
