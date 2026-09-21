"""알약 크롭 임베딩을 대조학습(SupCon)으로 이 데이터에 맞춘다

실행: uv run python -m src.train.train_contrastive   (make_yolo.py 를 먼저 실행해야 한다)
결과: runs/<NAME>/head.pt                 학습된 투영 헤드
      runs/contrastive/<split>.npz        DINOv2 임베딩 캐시 (다시 돌리면 재사용해서 빠르다)

무엇을 하나:
    DINOv2 백본은 그대로 두고(freeze) 그 위에 작은 투영 헤드만 학습한다.
    같은 약끼리는 가깝게, 다른 약끼리는 멀게 만드는 학습(SupCon)이다.

왜 하나:
    filter_unknown.py 는 사전학습 DINOv2 임베딩을 그대로 썼고 Kaggle +0.002 에 그쳤다.
    알약은 색·모양이 거의 같고 각인만 다른 도메인이라 범용 임베딩이 잘 갈리지 않는다는 뜻이다.
    임베딩을 이 데이터에 맞추면 갈리는지 먼저 숫자로 확인한다.

무엇을 보고 판단하나:
    val 크롭을 train 클래스 중심과 비교한 최근접 이웃 정확도를 학습 전 / 후로 출력한다.
    올라가지 않으면 YOLO 에 붙이지 않는다. (이 스크립트는 제출 파이프라인과 연결되어 있지 않다)
"""

import random

import numpy as np
import torch

from src.analysis.filter_unknown import compute_embedding, crop_pill, load_dino
from src.config import RUNS_DIR, SEED, YOLO_DIR, get_device

# ===== 설정값 =====
NAME = "jw_dinov2_s500_b128_d128"  # runs/<NAME>/ (실험마다 바꿔야 결과가 안 덮인다)
CROP = 224  # 크롭 한 변 픽셀 (DINOv2 입력 크기, filter_unknown 과 같은 값)
MARGIN = 1.2  # 박스 긴 변의 몇 배로 자를지 (filter_unknown 과 같은 값)
MAX_PER_CLASS = 40  # 클래스당 최대 크롭 수. AI Hub 를 쓰면 박스가 10,431 개라 전부 뽑으면 오래 걸린다
DIM = 128  # 투영 헤드 출력 차원 (DINOv2 는 768)
CLASSES_PER_BATCH = 32  # 한 배치에 넣을 클래스 수
CROPS_PER_CLASS = 4  # 그 클래스마다 넣을 크롭 수 (배치 크기 = 32 x 4 = 128)
STEPS = 500  # 학습 반복 횟수
LR = 0.001  # AdamW 학습률
TEMPERATURE = 0.1  # SupCon 온도. 낮을수록 어려운 짝에 집중한다

CACHE_DIR = RUNS_DIR / "contrastive"  # 임베딩 캐시 폴더


# ---------- 1. 크롭 모으기 ----------
def load_boxes(split):
    """YOLO 라벨을 읽어서 (이미지 경로, 박스, 클래스) 목록을 만든다

    입력:
        split (str): "train" 또는 "val"

    반환:
        list: (path, box, class_idx) 튜플 리스트
              예) [(PosixPath('.../K-001900....png'),
                    {"x": 167, "y": 248, "w": 184, "h": 182}, 48), ...]

    동작:
        1. 이미지를 하나씩 열어 크기를 읽는다. (라벨이 0~1 비율이라 픽셀로 되돌리려면 필요하다)
        2. 같은 이름의 라벨 파일에서 줄마다
           a. 클래스 번호와 중심점 비율 좌표를 꺼낸다.
           b. 픽셀 좌상단 x, y, 너비, 높이로 되돌린다. (make_yolo 의 반대 방향)
    """
    from PIL import Image  # 크기만 읽으면 되어서 여기서만 쓴다

    items = []

    # 1. 이미지마다
    for path in sorted((YOLO_DIR / "images" / split).glob("*.png")):
        label_path = YOLO_DIR / "labels" / split / f"{path.stem}.txt"
        if not label_path.exists():
            continue
        width, height = Image.open(path).size

        # 2. 라벨 줄마다
        for line in label_path.read_text().split("\n"):
            if not line.strip():
                continue

            # 2-a. 클래스 번호 + 비율 좌표
            parts = line.split()
            class_idx = int(parts[0])
            cx, cy, w, h = [float(v) for v in parts[1:5]]

            # 2-b. 픽셀 좌상단 기준으로 되돌리기
            box = {
                "x": (cx - w / 2) * width,
                "y": (cy - h / 2) * height,
                "w": w * width,
                "h": h * height,
            }
            items.append((path, box, class_idx))

    return items


def sample_by_class(items, max_per_class):
    """클래스마다 최대 max_per_class 개만 남긴다

    입력:
        items (list): load_boxes() 결과
        max_per_class (int): 클래스당 최대 개수

    반환:
        list: 줄어든 items

    동작:
        1. 클래스별로 모은다.
        2. 클래스마다 섞어서 앞에서 max_per_class 개만 가져온다. (시드 고정이라 항상 같은 표본)

    클래스마다 장수가 크게 다르면(AI Hub 수백 장 vs Kaggle 2~3 장) 많은 쪽으로 학습이 쏠린다.
    """
    # 1. 클래스별로 모으기
    by_class = {}
    for item in items:
        class_idx = item[2]
        if class_idx not in by_class:
            by_class[class_idx] = []
        by_class[class_idx].append(item)

    # 2. 클래스마다 잘라내기
    rng = random.Random(SEED)
    picked = []
    for class_idx in sorted(by_class):
        group = by_class[class_idx][:]
        rng.shuffle(group)
        picked += group[:max_per_class]

    return picked


# ---------- 2. 임베딩 ----------
def load_embeddings(split, dino):
    """크롭을 DINOv2 임베딩으로 바꾼다 (캐시가 있으면 읽어 쓴다)

    입력:
        split (str): "train" 또는 "val"
        dino: load_dino() 결과

    반환:
        (np.ndarray, np.ndarray):
            X (크롭 수, 768) 길이 1 로 맞춘 임베딩
            y (크롭 수,) 클래스 번호

    동작:
        1. 캐시 파일이 있으면 그대로 읽어서 반환한다. (백본이 고정이라 매번 다시 뽑을 필요가 없다)
        2. 없으면 박스를 모아 클래스별로 표본을 줄이고, 박스마다 크롭한다.
        3. DINOv2 로 임베딩을 뽑아 캐시에 저장한다.
    """
    # 1. 캐시 (Kaggle 만 / AI Hub 포함이 섞이지 않게 데이터셋 폴더 이름을 붙인다)
    cache = CACHE_DIR / f"{YOLO_DIR.name}_{split}.npz"
    if cache.exists():
        data = np.load(cache)
        print(f"{split}: 캐시 사용 {data['X'].shape[0]}개")
        return data["X"], data["y"]

    # 2. 크롭
    items = sample_by_class(load_boxes(split), MAX_PER_CLASS)
    print(f"{split}: 크롭 {len(items)}개 준비 중 (처음 한 번만 오래 걸린다)")
    crops = []
    labels = []
    for path, box, class_idx in items:
        crops.append(crop_pill(path, box, CROP, MARGIN))
        labels.append(class_idx)

    # 3. 임베딩 + 저장
    X = compute_embedding(dino, crops)
    y = np.array(labels)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=y)
    print(f"{split}: 임베딩 {X.shape} 저장", cache)
    return X, y


# ---------- 3. 대조학습 ----------
def compute_supcon_loss(z, labels, temperature):
    """SupCon 손실 (같은 약끼리 가깝게, 다른 약끼리 멀게)

    입력:
        z (Tensor): (배치, DIM) 길이 1 로 맞춘 벡터
        labels (Tensor): (배치,) 클래스 번호
        temperature (float): 낮을수록 어려운 짝에 집중

    반환:
        Tensor: 스칼라 손실

    동작:
        1. 모든 쌍의 코사인 유사도를 구하고 온도로 나눈다.
        2. 자기 자신과의 쌍은 빼고(-inf) 행마다 log-softmax 를 계산한다.
        3. 같은 클래스인 쌍(양성)의 로그 확률만 평균낸다. 부호를 뒤집은 것이 손실이다.
           (양성이 하나도 없는 행은 배치에 혼자 들어온 클래스라 건너뛴다)

    2 에서 넣은 -inf 는 곱하지 말고 0 으로 덮어써야 한다. -inf x False 는 nan 이 된다.
    """
    # 1. 쌍별 유사도
    sim = z @ z.T / temperature

    # 2. 자기 자신 제외 + log-softmax
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    # 3. 양성 쌍만 평균
    positive = (labels[:, None] == labels[None, :]) & ~self_mask
    count = positive.sum(dim=1)
    has_positive = count > 0
    positive_log_prob = log_prob.masked_fill(
        ~positive, 0.0
    )  # 양성이 아닌 칸(-inf 포함)은 0 으로
    loss = -positive_log_prob.sum(dim=1)[has_positive] / count[has_positive]
    return loss.mean()


def make_batch(by_class, X, device):
    """클래스 균형을 맞춘 배치 하나를 만든다

    입력:
        by_class (dict): 클래스 번호 -> X 에서의 행 번호 리스트
        X (Tensor): (크롭 수, 768) 전체 임베딩
        device (str): get_device() 결과

    반환:
        (Tensor, Tensor): (배치, 768) 임베딩, (배치,) 클래스 번호

    동작:
        1. 크롭이 CROPS_PER_CLASS 개 이상인 클래스 중 CLASSES_PER_BATCH 개를 고른다.
        2. 고른 클래스마다 CROPS_PER_CLASS 개를 뽑는다.
           (한 배치에 같은 클래스가 여러 장 있어야 '가깝게 당길 짝'이 생긴다)
    """
    # 1. 클래스 고르기
    usable = [c for c in by_class if len(by_class[c]) >= CROPS_PER_CLASS]
    chosen = random.sample(usable, min(CLASSES_PER_BATCH, len(usable)))

    # 2. 클래스마다 몇 장씩
    rows = []
    labels = []
    for class_idx in chosen:
        rows += random.sample(by_class[class_idx], CROPS_PER_CLASS)
        labels += [class_idx] * CROPS_PER_CLASS

    return X[rows].to(device), torch.tensor(labels, device=device)


def train_head(X, y, device):
    """투영 헤드를 SupCon 으로 학습한다

    입력:
        X (np.ndarray): (크롭 수, 768) train 임베딩
        y (np.ndarray): (크롭 수,) 클래스 번호
        device (str): get_device() 결과

    반환:
        Module: 학습된 헤드 (768 -> DIM)

    동작:
        1. 클래스별 행 번호 표를 만든다.
        2. 헤드와 옵티마이저를 만든다. (백본은 이미 임베딩으로 고정되어 있어 학습 대상이 아니다)
        3. STEPS 번 반복하며
           a. 클래스 균형 배치를 만든다.
           b. 헤드를 통과시키고 길이를 1 로 맞춘다.
           c. SupCon 손실로 한 걸음 학습한다.
    """
    # 1. 클래스별 행 번호
    by_class = {}
    for row, class_idx in enumerate(y):
        class_idx = int(class_idx)
        if class_idx not in by_class:
            by_class[class_idx] = []
        by_class[class_idx].append(row)

    # 2. 헤드
    X = torch.from_numpy(X).float()
    head = torch.nn.Linear(X.shape[1], DIM).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR)

    # 3. 학습
    for step in range(1, STEPS + 1):
        # 3-a. 배치
        batch, labels = make_batch(by_class, X, device)

        # 3-b. 통과 + 길이 1
        z = torch.nn.functional.normalize(head(batch), dim=1)

        # 3-c. 한 걸음
        loss = compute_supcon_loss(z, labels, TEMPERATURE)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"  step {step}/{STEPS}  loss {loss.item():.4f}")

    return head


# ---------- 4. 평가 ----------
def apply_head(head, X, device):
    """임베딩을 헤드에 통과시킨다 (head 가 None 이면 원본 그대로)

    입력:
        head (Module 또는 None): train_head() 결과
        X (np.ndarray): (크롭 수, 768)
        device (str): get_device() 결과

    반환:
        np.ndarray: (크롭 수, DIM) 또는 (크롭 수, 768) 길이 1 로 맞춘 벡터

    동작:
        1. head 가 없으면 학습 전 비교용이므로 그대로 돌려준다.
        2. 있으면 통과시키고 길이를 1 로 맞춘다.
    """
    # 1. 학습 전
    if head is None:
        return X

    # 2. 학습 후
    with torch.no_grad():
        z = head(torch.from_numpy(X).float().to(device))
        return torch.nn.functional.normalize(z, dim=1).cpu().numpy()


def evaluate(X_train, y_train, X_val, y_val):
    """val 크롭을 train 클래스 중심과 비교한 최근접 이웃 정확도

    입력:
        X_train, y_train: train 임베딩과 클래스 번호
        X_val, y_val: val 임베딩과 클래스 번호

    반환:
        (float, int): 정확도(0~1), train 에 없어서 맞출 수 없는 val 크롭 수

    동작:
        1. 클래스마다 train 벡터의 평균(중심)을 구하고 길이를 1 로 맞춘다.
        2. val 벡터와 모든 중심의 코사인 유사도를 구해 가장 가까운 클래스를 고른다.
        3. 정답과 비교해 맞은 비율을 낸다.
           train 에 아예 없는 클래스는 무엇을 골라도 틀리므로 따로 센다.
    """
    # 1. 클래스 중심
    classes = sorted(set(int(c) for c in y_train))
    centers = []
    for class_idx in classes:
        vectors = X_train[y_train == class_idx]
        center = vectors.mean(axis=0)
        centers.append(center / np.linalg.norm(center))
    centers = np.stack(centers)

    # 2. 가장 가까운 중심
    predicted = [classes[i] for i in (X_val @ centers.T).argmax(axis=1)]

    # 3. 정확도 + 못 맞추는 크롭
    correct = sum(1 for p, t in zip(predicted, y_val) if p == int(t))
    missing = sum(1 for t in y_val if int(t) not in classes)
    return correct / len(y_val), missing


# ---------- 5. 자체 점검 ----------
def check_supcon_loss():
    """손실 함수가 제대로 동작하는지 확인한다

    입력: 없음

    반환:
        없음. 틀리면 AssertionError 로 멈춘다.

    동작:
        1. 같은 클래스끼리 완전히 겹치고 클래스끼리는 직각인, 가장 좋은 배치를 만든다.
        2. 모든 벡터가 같은, 가장 나쁜 배치를 만든다.
        3. 좋은 배치의 손실이 나쁜 배치보다 작아야 한다.
    """
    labels = torch.tensor([0, 0, 1, 1])

    # 1. 가장 좋은 경우
    best = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    # 2. 가장 나쁜 경우 (전부 같은 방향이라 같은 약과 다른 약이 구분되지 않는다)
    worst = torch.tensor([[1.0, 0.0]] * 4)

    # 3. 비교
    best_loss = compute_supcon_loss(best, labels, TEMPERATURE).item()
    worst_loss = compute_supcon_loss(worst, labels, TEMPERATURE).item()
    assert best_loss < worst_loss, f"손실이 뒤집혔다: {best_loss} >= {worst_loss}"


# ---------- 실행 ----------
def main():
    """전체 순서

    입력: 없음

    반환:
        없음. runs/<NAME>/head.pt 를 저장하고 학습 전 / 후 정확도를 출력한다.

    동작:
        1. 손실 함수 자체 점검, 시드 고정, 장치 선택
        2. train / val 임베딩 준비 (캐시)
        3. 학습 전 정확도 (사전학습 DINOv2 그대로)
        4. 헤드 학습
        5. 학습 후 정확도 + 가중치 저장
    """
    # 1. 준비
    check_supcon_loss()
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = get_device()
    print("장치:", device, "/ 데이터:", YOLO_DIR)

    # 2. 임베딩
    dino = load_dino()
    X_train, y_train = load_embeddings("train", dino)
    X_val, y_val = load_embeddings("val", dino)

    # 3. 학습 전
    before, missing = evaluate(X_train, y_train, X_val, y_val)
    print(f"\n학습 전 (사전학습 DINOv2): {before:.3f}")
    print(
        f"  val 크롭 {len(y_val)}개 중 train 에 없는 클래스 {missing}개는 맞출 수 없다"
    )

    # 4. 학습
    print("\n대조학습 시작")
    head = train_head(X_train, y_train, device)

    # 5. 학습 후
    after, _ = evaluate(
        apply_head(head, X_train, device),
        y_train,
        apply_head(head, X_val, device),
        y_val,
    )
    out_dir = (
        RUNS_DIR / f"{NAME}_{YOLO_DIR.name}"
    )  # Kaggle 만(yolo) / AI Hub 포함(yolo_aihub) 을 따로 저장
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), out_dir / "head.pt")

    print(
        f"\n학습 전 {before:.3f} -> 학습 후 {after:.3f}  (차이 {after - before:+.3f})"
    )
    print("저장 위치:", out_dir / "head.pt")
    print("\n올라가지 않으면 YOLO 에 붙이지 않는다. (판단은 사람이 한다)")


if __name__ == "__main__":
    main()
