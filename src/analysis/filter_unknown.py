"""제출 파일에서 '모르는 알약'으로 보이는 박스의 점수를 낮춘다

실행: uv run python -m src.analysis.filter_unknown   (predict.py 로 submission.csv 를 먼저 만들어야 한다)
결과: runs/<NAME>/submission_filtered.csv   점수 낮춘 제출 파일
      runs/<NAME>/unknown_flags.csv         박스별 읽은 글자, 점수, 표시 여부
      runs/vlm_reads.json                   VLM 이 읽은 글자 저장 (다시 실행하면 이어서 쓴다)

왜 필요한가:
    test 에는 train 56종에 없는 약도 섞여 있다. 모델은 그런 약도 비슷한 약으로 높은 신뢰도로 예측해서 오답이 된다.

판단 방법 (둘 중 하나라도 걸리면 '모르는 약'):
    1. 글자: VLM(Qwen3-VL-2B)이 각인을 읽고, 예측한 약의 train 글자 사전과 비교한다.   글자 점수 < 0.537
    2. 모양: DINOv2 로 예측한 약의 train 사진과 얼마나 닮았는지 본다.                  모양 점수 < 0.907
    기준값은 train 시뮬레이션에서 아는 약을 잘못 거르는 비율이 약 2% 가 되도록 고른 값이다.

박스를 지우지 않고 점수만 낮추는 이유:
    아는 약을 잘못 걸러도 순위만 내려가서 mAP 손해가 작다.
"""

import difflib
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.annotations import load_annotations
from src.config import KAGGLE_DIR, RUNS_DIR, get_device
from src.data.make_yolo import remove_bad_images
from src.train.train_yolo import NAME

# ===== 설정값 =====
SUBMISSION = RUNS_DIR / NAME / "submission.csv"  # 필터를 걸 제출 파일
OUT_CSV = RUNS_DIR / NAME / "submission_filtered.csv"
FLAGS_CSV = RUNS_DIR / NAME / "unknown_flags.csv"
READS_JSON = RUNS_DIR / "vlm_reads.json"  # 글자 읽기가 느려서(알약당 0.6초) 저장해두고 다시 쓴다
MIN_SCORE = 0.1  # 이 점수 이상 박스만 검사 (낮은 박스는 어차피 순위가 낮다)
TEXT_LIMIT = 0.537  # 글자 점수가 이보다 낮으면 모르는 약
SHAPE_LIMIT = 0.907  # 모양 점수가 이보다 낮으면 모르는 약
DOWN = 0.01  # 모르는 약 박스 점수에 곱할 값
VLM_NAME = "Qwen/Qwen3-VL-2B-Instruct"
QUESTION = (
    "Read the characters engraved or printed on this pill (they may be rotated or embossed). "
    "Reply with only the characters, or NONE."
)
DEVICE = get_device()

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/"
LOOK_ALIKE = str.maketrans("0156824", "OISGBZA")  # VLM 이 자주 헷갈리는 숫자 -> 글자 (0 과 O 를 같게 본다)


# ---------- 1. 모델 · 저장된 글자 불러오기 ----------
def load_dino():
    """DINOv2 모델을 불러온다 (처음 실행하면 약 330MB 다운로드)

    입력: 없음

    반환:
        DINOv2 ViT-B/14 모델 (평가 모드)

    동작:
        1. torch.hub 에서 모델을 받는다.
        2. 장치로 옮기고 평가 모드로 바꾼다.
    """
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    return model.to(DEVICE).eval()


def load_vlm():
    """Qwen3-VL-2B 를 불러온다 (처음 실행하면 약 4.5GB 다운로드)

    입력: 없음

    반환:
        model: 이미지 + 글자를 받아 답을 만드는 모델 (평가 모드)
        processor: 이미지와 질문을 모델 입력으로 바꾸는 도구

    동작:
        1. 숫자 정밀도를 고른다. (GPU 는 float16 으로 메모리 절약, CPU 는 float16 이 느려서 float32)
        2. processor 와 모델을 받는다.
        3. 모델을 장치로 옮기고 평가 모드로 바꾼다.
    """
    # 1. 정밀도
    if DEVICE == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.float16

    # 2~3. 불러오기
    processor = AutoProcessor.from_pretrained(VLM_NAME)
    model = AutoModelForImageTextToText.from_pretrained(VLM_NAME, dtype=dtype)
    model = model.to(DEVICE).eval()
    return model, processor


def load_reads():
    """저장해둔 VLM 글자를 불러온다

    입력: 없음 (READS_JSON 파일을 읽는다)

    반환:
        dict: "파일명|x|y|w|h" -> 읽은 글자  예) {"541.png|148|220|250|222": "SK"}

    동작:
        1. 파일이 있으면 읽어서 돌려주고, 없으면 빈 dict 를 돌려준다.
    """
    if READS_JSON.exists():
        return json.loads(READS_JSON.read_text(encoding="utf-8"))
    return {}


# ---------- 2. train 으로 약별 모양 벡터 · 글자 사전 만들기 ----------
def make_train_data(dino, model, processor, reads):
    """train 알약으로 약별 모양 벡터와 글자 사전을 만든다

    입력:
        dino: load_dino() 결과
        model, processor: load_vlm() 결과
        reads (dict): load_reads() 결과

    반환:
        (dict, dict):
            vectors_by_class: 약 ID -> (사진 수, 768) 벡터  예) {3483: array(...)}
            dict_by_class: 약 ID -> 글자 사전  예) {3483: ["GAO", "SK", "GAV", ...]}

    동작:
        1. 전처리와 같은 기준으로 train 박스를 모은다. (remove_bad_images)
        2. 박스마다
           a. 224 크기로 잘라 모양 벡터용으로 모은다.
           b. 글자를 읽어 split_words() 결과를 그 약 사전에 추가한다. (앞면·뒷면 글자가 모두 들어간다)
        3. 약별로 벡터를 계산한다.
    """
    # 1. train 박스
    ann = remove_bad_images(load_annotations())

    crops_by_class = {}
    dict_by_class = {}
    for name in sorted(ann):
        path = KAGGLE_DIR / "train_images" / name
        for box in ann[name]:
            class_id = box["class_id"]
            if class_id not in crops_by_class:
                crops_by_class[class_id] = []
                dict_by_class[class_id] = []

            # 2-a. 모양용
            crops_by_class[class_id].append(crop_pill(path, box, 224, 1.2))

            # 2-b. 글자 사전
            read = cached_read(reads, model, processor, path, box)
            for word in split_words(read):
                if word not in dict_by_class[class_id]:
                    dict_by_class[class_id].append(word)

    # 3. 약별 벡터
    vectors_by_class = {}
    for class_id in crops_by_class:
        vectors_by_class[class_id] = compute_embedding(dino, crops_by_class[class_id])

    READS_JSON.write_text(json.dumps(reads, ensure_ascii=False), encoding="utf-8")
    return vectors_by_class, dict_by_class


def crop_pill(image_path, box, size, margin):
    """알약 박스를 정사각형으로 잘라낸다

    입력:
        image_path (Path): 원본 이미지 경로
        box (dict): {"x": 왼쪽 위 x, "y": 왼쪽 위 y, "w": 너비, "h": 높이}
        size (int): 결과 이미지 한 변 픽셀  예) 224
        margin (float): 박스 긴 변의 몇 배로 자를지  예) 1.2

    반환:
        Image: size x size 크기 RGB 이미지

    동작:
        1. 박스 중심에서 긴 변 x margin 크기의 정사각형 영역을 정한다.
        2. 배경색(이미지 위쪽 20줄의 중앙값)으로 채운 빈 캔버스를 만든다.
           (박스가 이미지 끝에 붙어 있으면 잘린 부분이 비어서 배경색으로 채운다)
        3. 원본에서 잘라 캔버스에 붙이고 size 로 줄인다.
    """
    image = Image.open(image_path).convert("RGB")

    # 1. 정사각형 영역 (왼쪽 위 = 박스 중심 - 한 변의 절반)
    side = int(margin * max(box["w"], box["h"]))
    left = int(box["x"] + box["w"] / 2 - side / 2)
    top = int(box["y"] + box["h"] / 2 - side / 2)

    # 2. 배경색 캔버스 (위쪽 20줄의 픽셀들을 R, G, B 별 중앙값으로)
    top_rows = np.asarray(image)[:20].reshape(-1, 3)  # (20 x 너비, 3)
    median = np.median(top_rows, axis=0)  # 예) [212.0, 208.0, 199.0]
    background = (int(median[0]), int(median[1]), int(median[2]))
    canvas = Image.new("RGB", (side, side), background)

    # 3-a. 원본에서 이미지 안에 들어오는 부분만 자른다
    crop_left = max(left, 0)
    crop_top = max(top, 0)
    crop_right = min(left + side, image.width)
    crop_bottom = min(top + side, image.height)
    piece = image.crop((crop_left, crop_top, crop_right, crop_bottom))

    # 3-b. 캔버스에 붙일 위치 (영역이 이미지 왼쪽·위로 넘어갔으면 넘어간 만큼 띄운다)
    paste_x = crop_left - left
    paste_y = crop_top - top
    canvas.paste(piece, (paste_x, paste_y))

    # 3-c. 크기 맞추기
    return canvas.resize((size, size), Image.LANCZOS)


def cached_read(reads, model, processor, image_path, box):
    """저장된 글자가 있으면 쓰고, 없으면 읽어서 저장한다

    입력:
        reads (dict): load_reads() 결과 (이 함수가 새 값을 추가한다)
        model, processor: load_vlm() 결과
        image_path (Path): 이미지 경로
        box (dict): {"x", "y", "w", "h"}

    반환:
        str: 읽은 글자

    동작:
        1. 파일명과 박스 좌표로 키를 만든다.
        2. 저장된 적이 없으면 448 크기로 잘라서 읽고 reads 에 넣는다.
        3. reads 에서 꺼내 돌려준다.
    """
    # 1. 키
    key = f"{image_path.name}|{box['x']}|{box['y']}|{box['w']}|{box['h']}"

    # 2. 처음 보는 박스면 읽기
    if key not in reads:
        crop = crop_pill(image_path, box, 448, 1.1)
        reads[key] = read_text(model, processor, crop)

    # 3. 꺼내기
    return reads[key]


def read_text(model, processor, crop):
    """알약 각인 글자를 읽는다

    입력:
        model, processor: load_vlm() 결과
        crop (Image): crop_pill(size=448, margin=1.1)

    반환:
        str: 읽은 글자 그대로  예) "G40", "TYLENOL\\nER", "NONE"

    동작:
        1. 이미지 + 질문을 대화 형식으로 만든다.
        2. 답을 최대 16토큰 생성한다. (do_sample=False -> 매번 같은 답)
        3. 질문 부분을 빼고 답만 글자로 되돌린다.
    """
    # 1. 대화 형식 (사용자가 이미지 1장 + 질문 1개를 보낸 대화)
    content = [
        {"type": "image", "image": crop},
        {"type": "text", "text": QUESTION},
    ]
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)

    # 2. 생성 (**inputs = dict 를 "이름=값" 인자로 풀어서 넘긴다)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)

    # 3. 답만 (output 앞부분은 질문 토큰이 그대로 들어 있어서 질문 길이만큼 잘라낸다)
    question_length = inputs["input_ids"].shape[1]
    answer_tokens = output[:, question_length:]
    return processor.batch_decode(answer_tokens, skip_special_tokens=True)[0]


def split_words(read):
    """VLM 답을 비교할 단어 리스트로 바꾼다

    입력:
        read (str): read_text() 결과  예) "TYLENOL\\nER"

    반환:
        list: 정리한 단어들 + 전체를 붙인 글자 (2글자 이상만)  예) ["TYLENOL", "ER", "TYLENOLER"]
              글자가 없으면 []

    동작:
        1. "NONE" 이면 빈 리스트.
        2. 공백·줄바꿈으로 나눈 단어마다 clean_text() 해서 2글자 이상이면 넣는다.
        3. 전체를 붙인 글자도 넣는다. (각인이 줄바꿈돼 있어도 비교되게)
    """
    # 1. 글자 없음
    if read.strip().upper() == "NONE":
        return []

    # 2. 단어별
    words = []
    for part in read.split():
        word = clean_text(part)
        if len(word) >= 2 and word not in words:
            words.append(word)

    # 3. 전체
    whole = clean_text(read)
    if len(whole) >= 2 and whole not in words:
        words.append(whole)
    return words


def clean_text(text):
    """비교하기 좋게 글자를 정리한다

    입력:
        text (str): 예) "g-40"

    반환:
        str: 대문자, 영문·숫자·/ 만 남기고, 헷갈리는 숫자는 글자로  예) "GAO"

    동작:
        1. 대문자로 바꾸고 LETTERS 에 있는 글자만 남긴다.
        2. LOOK_ALIKE 표로 헷갈리는 숫자를 글자로 바꾼다. (0 -> O, 1 -> I ...)
    """
    # 1. 남길 글자만
    kept = ""
    for ch in text.upper():
        if ch in LETTERS:
            kept = kept + ch

    # 2. 숫자 -> 글자
    return kept.translate(LOOK_ALIKE)


def compute_embedding(dino, crops):
    """이미지들을 특징 벡터로 바꾼다

    입력:
        dino: load_dino() 결과
        crops (list): crop_pill(size=224) 이미지 리스트

    반환:
        np.ndarray: (이미지 수, 768) 벡터. 길이를 1로 맞춰서 두 벡터를 곱하면 바로 닮은 정도(-1~1)가 된다.

    동작:
        1. 32장씩 묶어서
           a. 0~1 로 바꾸고 ImageNet 평균/표준편차로 정규화한다.
           b. 모델에 넣고 CLS 토큰(이미지 전체 요약 벡터)을 꺼낸다.
           c. 길이를 1로 맞춘다.
        2. 전부 이어 붙인다.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    vectors = []

    for start in range(0, len(crops), 32):
        # 1-a. 정규화
        batch = []
        for crop in crops[start:start + 32]:
            pixels = np.asarray(crop, dtype=np.float32) / 255
            batch.append(torch.from_numpy(pixels).permute(2, 0, 1))  # (높이, 너비, 3) -> (3, 높이, 너비)
        x = (torch.stack(batch) - mean) / std

        # 1-b. CLS 토큰
        with torch.no_grad():
            cls_token = dino.forward_features(x.to(DEVICE))["x_norm_clstoken"]

        # 1-c. 길이 1
        vectors.append(torch.nn.functional.normalize(cls_token, dim=1).cpu())

    return torch.cat(vectors).numpy()


# ---------- 3. 박스별 점수 ----------
def compute_text_score(words, dictionary):
    """읽은 글자가 예측한 약의 글자와 맞는지 (0~1, 낮을수록 모르는 약)

    입력:
        words (list): split_words() 결과  예) ["TYLENOL", "ER", "TYLENOLER"]
        dictionary (list): 그 약의 train 글자들  예) ["SUSPEN", "ER", "SUSPENER"]

    반환:
        float: 예) 0.47

    동작:
        1. 글자를 못 읽었으면 판단 근거가 없으니 1.0 (통과)
        2. train 에서 글자가 한 번도 안 읽힌 약인데 글자를 읽었으면 0.0
        3. 맞는 정도 = 둘 중 큰 값
           a. 가장 긴 글자(전체)가 사전과 맞는 정도
           b. 단어 중 가장 안 맞는 단어의 맞는 정도 (모든 단어가 사전에 있어야 높다)
        4. 짧은 글자는 VLM 이 잘못 읽기 쉬워서 6글자 미만이면 감점을 줄인다.
           예) 2글자가 완전히 달라도 1 - 1 x 2/6 = 0.67
    """
    # 1. 못 읽음
    if len(words) == 0:
        return 1.0
    # 2. 사전 비어 있음
    if len(dictionary) == 0:
        return 0.0

    # 3-a. 가장 긴 글자
    longest = words[0]
    for word in words:
        if len(word) > len(longest):
            longest = word
    whole_match = find_best_match(longest, dictionary)

    # 3-b. 가장 안 맞는 단어
    worst_word = 1.0
    for word in words:
        word_match = find_best_match(word, dictionary)
        if word_match < worst_word:
            worst_word = word_match

    match = max(whole_match, worst_word)

    # 4. 짧은 글자 감점 줄이기
    weight = min(len(longest), 6) / 6
    return 1 - (1 - match) * weight


def find_best_match(word, dictionary):
    """단어 하나와 사전에서 가장 비슷한 값

    입력:
        word (str): 단어
        dictionary (list): 그 약의 train 글자들

    반환:
        float: 가장 큰 compute_similarity() 값

    동작:
        1. 사전 글자마다 비슷한 정도를 계산해서 가장 큰 값을 남긴다.
    """
    best = 0.0
    for known in dictionary:
        similarity = compute_similarity(word, known)
        if similarity > best:
            best = similarity
    return best


def compute_similarity(a, b):
    """두 글자가 얼마나 같은지 (0~1)

    입력:
        a, b (str): clean_text() 결과

    반환:
        float: 1.0 = 같음  예) compute_similarity("GAO", "GAO") -> 1.0

    동작:
        1. 3글자 이상이고 한쪽이 다른 쪽에 들어 있으면 1.0 (각인 일부만 읽은 경우)
           (2글자는 "ER" 이 "SUSPENER" 안에 들어가듯 우연히 겹치기 쉬워서 뺀다)
        2. 아니면 difflib 로 글자 순서가 얼마나 같은지 비율을 낸다.
    """
    # 1. 일부 포함
    if min(len(a), len(b)) >= 3 and (a in b or b in a):
        return 1.0
    # 2. 비율
    return difflib.SequenceMatcher(None, a, b).ratio()


def compute_shape_score(dino, crop, class_vectors):
    """test 알약이 예측한 약의 train 사진과 얼마나 닮았는지

    입력:
        dino: load_dino() 결과
        crop (Image): test 알약 crop_pill(size=224)
        class_vectors (np.ndarray): 그 약의 train 사진 벡터들 (사진 수, 768)

    반환:
        float: 가장 닮은 정도 (1 에 가까울수록 같은 약)  예) 0.97

    동작:
        1. 알약이 어느 방향으로 놓였을지 모르니 30도씩 12번 돌린 이미지를 만든다.
        2. 12개 벡터와 train 벡터를 모두 곱해서 (= 닮은 정도) 가장 큰 값을 고른다.
    """
    # 1. 12방향
    rotated = []
    for angle in range(0, 360, 30):
        rotated.append(crop.rotate(angle))
    vectors = compute_embedding(dino, rotated)

    # 2. 가장 닮은 값 (12 x 사진 수 표에서 최댓값)
    return float((vectors @ class_vectors.T).max())


# ---------- 실행 ----------
def main():
    """전체 순서

    입력: 없음 (runs/<NAME>/submission.csv 를 읽는다)

    반환:
        없음. submission_filtered.csv, unknown_flags.csv, vlm_reads.json 을 저장한다.

    동작:
        1. 모델 2개와 저장된 글자를 불러온다.
        2. train 으로 약별 모양 벡터·글자 사전을 만든다.
        3. 제출 파일에서 MIN_SCORE 이상 박스마다 글자 점수·모양 점수를 계산한다. (100개마다 글자 저장)
        4. 둘 중 하나라도 기준 미만이면 점수에 DOWN 을 곱한다.
        5. 저장하고 요약을 출력한다.
    """
    # 1. 준비
    dino = load_dino()
    model, processor = load_vlm()
    reads = load_reads()

    # 2. train
    vectors_by_class, dict_by_class = make_train_data(dino, model, processor, reads)
    print("train 준비 완료:", len(dict_by_class), "종")

    # 3. 박스별 점수
    submission = pd.read_csv(SUBMISSION)
    targets = submission[submission["score"] >= MIN_SCORE]
    print("검사할 박스:", len(targets))

    rows = []
    records = targets.to_dict("records")  # 표를 행마다 dict 1개인 리스트로
    for i, r in enumerate(records):
        path = KAGGLE_DIR / "test_images" / f"{r['image_id']}.png"
        box = {"x": r["bbox_x"], "y": r["bbox_y"], "w": r["bbox_w"], "h": r["bbox_h"]}
        class_id = r["category_id"]

        # 3-a. 글자 점수
        read = cached_read(reads, model, processor, path, box)
        t_score = compute_text_score(split_words(read), dict_by_class[class_id])

        # 3-b. 모양 점수
        crop = crop_pill(path, box, 224, 1.2)
        s_score = compute_shape_score(dino, crop, vectors_by_class[class_id])

        # 4. 판단 (둘 중 하나라도 기준 미만이면 모르는 약)
        unknown = t_score < TEXT_LIMIT or s_score < SHAPE_LIMIT
        rows.append(
            {
                "annotation_id": r["annotation_id"],
                "image_id": r["image_id"],
                "category_id": class_id,
                "score": r["score"],
                "read": read,
                "text_score": round(t_score, 3),
                "shape_score": round(s_score, 3),
                "unknown": unknown,
            }
        )

        if i % 100 == 0:
            READS_JSON.write_text(json.dumps(reads, ensure_ascii=False), encoding="utf-8")
            print(f"{i}/{len(targets)}")

    # 4. 점수 낮추기
    flags = pd.DataFrame(rows)
    unknown_ids = flags[flags["unknown"]]["annotation_id"]  # unknown 이 True 인 행의 annotation_id 만
    filtered = submission.copy()
    is_unknown = filtered["annotation_id"].isin(unknown_ids)  # 행마다 True / False
    filtered.loc[is_unknown, "score"] = filtered.loc[is_unknown, "score"] * DOWN  # True 인 행의 score 만 바꾼다

    # 5. 저장 + 요약
    READS_JSON.write_text(json.dumps(reads, ensure_ascii=False), encoding="utf-8")
    flags.to_csv(FLAGS_CSV, index=False)
    filtered.to_csv(OUT_CSV, index=False)
    confident = flags[flags["score"] >= 0.5]
    n_unknown = flags["unknown"].sum()
    confident_rate = confident["unknown"].mean() * 100
    print(f"모르는 약 표시: {n_unknown} / {len(flags)} 박스 (점수 0.5 이상 중 {confident_rate:.1f}%)")
    print("저장 위치:", OUT_CSV)


if __name__ == "__main__":
    main()
