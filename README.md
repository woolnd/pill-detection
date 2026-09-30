# pill-detection

YOLO로 이미지 속 알약을 탐지하고 종류를 분류하는 프로젝트입니다. (코드잇 AI 엔지니어 과정 초급 프로젝트)

- **입력**: 여러 알약이 함께 놓인 이미지 (976 × 1280, 한 장에 3~4알)
- **출력**: 알약별 바운딩 박스 + 클래스 + 신뢰도
- **데이터**: Kaggle 대회 `ai14-level-project` 제공 데이터 + AI Hub 「경구약제 이미지 데이터」 조합 데이터
- **평가**: Kaggle 리더보드 **mAP[0.75:0.95]**, 최종 순위는 Private Score 기준
- **팀**: 팀 파이리 4명 / 2026.09.10 ~ 09.29
- **최종 결과**: Kaggle **0.62550** (AI Hub 데이터 + 오버샘플링 + YOLO26n 두 모델 WBF 앙상블)

실험 과정 · 분석 · 결론은 [보고서 · 발표 자료](#-보고서--발표-자료)에 정리되어 있습니다.

## 📚 목차

| 🚀 실행 | 🗂️ 코드 | 📈 실험 | 👥 팀 |
|:---|:---|:---|:---|
| [환경 설정](#환경-설정)<br>[⚡ 바로 체험하기](#-학습-없이-바로-체험하기)<br>[실행 방법](#실행-방법) | [파일 구조](#파일-구조)<br>[협업 방식 · 컨벤션](#협업-방식--컨벤션) | [실험 기록](#실험-기록) | [팀 소개](#-팀)<br>[보고서 · 발표](#-보고서--발표-자료)<br>[협업 일지](#-협업-일지) |

## 실행 방법

### 환경 설정

```bash
git clone <repo-url>
cd pill-detection
uv sync
```

`uv sync`만 하면 팀원 전원이 동일한 환경이 됩니다. 패키지는 `uv add <pkg>`로 추가하고 `uv.lock`을 꼭 커밋합니다. (`pip` 사용 금지)

프로젝트 루트에 `.env`를 만듭니다. `.env`는 gitignore 되어 있으며 **절대 커밋하지 않습니다.**

```
KAGGLE_USERNAME=<캐글 아이디>
KAGGLE_KEY=<API 키>
AI_HUB_KEY=<AI Hub API 키>                  # AI Hub 데이터 다운로드용
SHEET_URL=<실험 기록 구글 시트 웹앱 URL>   # 팀 채팅으로 공유
AUTHOR=<본인 이름>                          # 실험 기록의 author 칸
```

### ⚡ 학습 없이 바로 체험하기

최종 제출(Kaggle 0.62550)에 쓴 두 모델이 `runs/finals/` 에 들어 있어서, 학습 없이 앙상블 예측까지 바로 돌려볼 수 있습니다. (맥 M2 Pro 기준 약 1분 40초)

```bash
uv sync
uv run --env-file .env python -m src.data.download_data   # Kaggle test 이미지 842장 받기
uv run python -m src.analysis.predict                    # 두 모델 WBF 앙상블 → 제출 파일
```

```
runs/finals/
├── rahui_ep50_img1280_batch4_aihubdata_oversample/   모델 A (1280) · 단독 0.61802
│   ├── weights/best.pt
│   ├── args.yaml       학습에 쓴 설정 전체
│   └── results.csv     epoch 별 loss · mAP
└── rahui_ep30_img960_batch8_aihubdata_oversample_f2/ 모델 B (960) · 단독 0.62041
```

결과는 `runs/ensemble_f2_960_ratio_submission.csv` (박스 5,019개 · test 842장 전부)에 저장됩니다.

### 데이터 준비 → 학습 → 제출

```bash
uv run --env-file .env python -m src.data.download_data   # 1. Kaggle 데이터 다운로드
uv run python -m src.data.download_aihub                 #    AI Hub 조합 데이터 다운로드 (약 21GB, .env 의 AI_HUB_KEY 필요)
uv run python -m src.data.make_yolo                      # 2. 라벨 정리 → YOLO 변환 → train/val 분할 → 오버샘플링
uv run python -m src.data.check_yolo                     #    변환 결과 검증
uv run python -m src.train.train_yolo                    # 3. 학습 + val 평가 + 실험 기록(csv · 구글 시트)
uv run python -m src.analysis.visualize                  # 4. val 예측 시각화 · 실패 사례 · 클래스별 점수
uv run python -m src.analysis.predict                    # 5. test 예측 → Kaggle 제출 파일
```

- 모든 스크립트는 **프로젝트 루트에서 `python -m src.폴더.파일`** 형태로 실행합니다.
- `src/config.py` 의 `USE_AIHUB` 로 데이터셋을 고릅니다. `True` = Kaggle + AI Hub 118종 (최종), `False` = Kaggle 56종.
- ⛔ AI Hub `TL_2_조합` · `TS_2_조합` 은 대회 train/test 원본이라 받지 않습니다 (다운로드 스크립트에서 제외).

### 최종 제출(0.62550) 재현

두 모델을 학습한 뒤 `predict.py` 가 WBF(비중 1 : 1.3)로 합칩니다.

| | EPOCHS | IMGSZ | BATCH | close_mosaic | NAME |
|---|---|---|---|---|---|
| 모델 B (`train_yolo.py` 기본값) | 30 | 960 | 8 | 3 | `rahui_ep30_img960_batch8_aihubdata_oversample_f2` |
| 모델 A | 50 | 1280 | 4 | 5 | `rahui_ep50_img1280_batch4_aihubdata_oversample` |

**1) 모델 B 학습** · `src/train/train_yolo.py` 기본값 그대로 실행합니다.

```python
# src/train/train_yolo.py
EPOCHS = 30
IMGSZ = 960
BATCH = 8
NAME = "rahui_ep30_img960_batch8_aihubdata_oversample_f2"
EXPERIMENT = {
    "optimizer": "AdamW",
    "lr0": 0.001,
    "cos_lr": True,
    "warmup_epochs": 1.5,
    "close_mosaic": 3,
    "box": 12.0,
    "cls": 1.0,
}
```

```bash
uv run python -m src.train.train_yolo
```

**2) 모델 A 학습** · 같은 파일에서 아래 5곳만 바꿔 한 번 더 실행합니다. 나머지는 그대로 둡니다.

```diff
-EPOCHS = 30
+EPOCHS = 50
-IMGSZ = 960
+IMGSZ = 1280
-BATCH = 8
+BATCH = 4
-NAME = "rahui_ep30_img960_batch8_aihubdata_oversample_f2"
+NAME = "rahui_ep50_img1280_batch4_aihubdata_oversample"
 EXPERIMENT = {
     ...
-    "close_mosaic": 3,
+    "close_mosaic": 5,
     ...
 }
```

```bash
uv run python -m src.train.train_yolo
```

**3) 앙상블 예측** · `src/analysis/predict.py` 는 기본으로 `runs/finals/` 의 두 모델을 합칩니다. 직접 학습한 모델(`runs/<NAME>/`)로 합치려면 두 경로에서 `"finals" /` 만 지웁니다.

```python
# src/analysis/predict.py
USE_ENSEMBLE = True   # 두 모델 WBF 앙상블 (False 면 train_yolo.py 의 NAME 모델 하나로 예측)

WEIGHTS_A = RUNS_DIR / "finals" / "rahui_ep50_img1280_batch4_aihubdata_oversample" / "weights" / "best.pt"
IMGSZ_A = 1280
WEIGHTS_B = RUNS_DIR / "finals" / "rahui_ep30_img960_batch8_aihubdata_oversample_f2" / "weights" / "best.pt"
IMGSZ_B = 960

ENSEMBLE_WEIGHTS_RATIO = [1.0, 1.3]   # 더 강한 모델 B 에 비중
WBF_IOU_THR = 0.55                    # 이 이상 겹치면 같은 알약으로 보고 합침
```

```bash
uv run python -m src.analysis.predict   # → runs/ensemble_f2_960_ratio_submission.csv
```

**4) 제출** · Kaggle에 **팀 계정으로** 업로드합니다 (하루 10회).

직접 학습한 결과는 `runs/<NAME>/` 에 저장되어 `runs/finals/` 원본을 덮어쓰지 않습니다. 맥 GPU(MPS)는 같은 시드로도 결과가 조금씩 달라 점수가 정확히 같지 않을 수 있습니다.

## 파일 구조

```
data/raw/        Kaggle 원본 데이터 (git 제외)
data/aihub/      AI Hub 조합 데이터 (git 제외, 라벨 labels/TL_n · 이미지 images/TS_n)
data/yolo/       Kaggle 만으로 만든 YOLO 데이터셋 (git 제외)
data/yolo_aihub/ Kaggle + AI Hub YOLO 데이터셋 (git 제외)
runs/            학습 결과 · 가중치 · 분석 그림 · 제출 파일 (git 제외, runs/finals/ 최종 두 모델만 포함)
src/             공통 모듈 (config · annotations · sheet)
src/data/        데이터 다운로드 · 확인 · YOLO 변환
src/train/       학습 (YOLO, Faster R-CNN · RetinaNet 비교용, 대조학습)
src/analysis/    예측 · 앙상블 · 시각화 · README 그래프
docs/images/     README 실험 그래프 (GitHub Actions가 자동 갱신)
.github/         이슈·PR 템플릿, README 그래프 갱신 워크플로
```

| 파일 | 역할 |
|---|---|
| `src/config.py` | 공통 경로(`KAGGLE_DIR`, `YOLO_DIR`, `RUNS_DIR` 등) · `USE_AIHUB` · `SEED` · `get_device` |
| `src/annotations.py` | (이미지 × 알약)당 JSON 1개인 라벨을 이미지 파일명 기준으로 묶기 · `compute_iou` |
| `src/sheet.py` | 구글 시트(Apps Script 웹앱)로 실험 기록 전송 |
| `src/data/download_data.py` | Kaggle 대회 데이터를 `data/raw/`에 다운로드 |
| `src/data/download_aihub.py` | AI Hub 조합 데이터를 `data/aihub/`에 다운로드 (받기 → 조각 병합 → 압축 해제) |
| `src/data/check_raw.py` | 원본 데이터 요약 + 박스 시각화 |
| `src/data/make_yolo.py` | 문제 있는 이미지 제외 → YOLO 라벨 변환 → 조합 단위 train/val 분할 → 취약 클래스 오버샘플링 |
| `src/data/check_yolo.py` | YOLO 라벨을 픽셀로 되돌려 원본과 비교 |
| `src/data/make_pseudo.py` | test 예측을 의사 라벨로 train 에 더한 데이터셋 만들기 (실험용) |
| `src/train/train_yolo.py` | YOLO 학습, val 평가(mAP75-95), 실험 기록 |
| `src/train/train_rcnn.py` | Faster R-CNN · RetinaNet 학습 (모델 비교용) |
| `src/train/train_contrastive.py` | 알약 크롭 임베딩 대조학습 (실험용) |
| `src/analysis/visualize.py` | val 예측/정답 비교 그림, 오류 유형 집계, 클래스별 AP, 학습 곡선 |
| `src/analysis/predict.py` | test 842장 예측 → 제출 형식 csv (단일 / 멀티스케일 TTA / 두 모델 WBF 앙상블) |
| `src/analysis/visualize_ensemble.py` | 앙상블 예측 시각화 (test 박스, val 실패 사례) |
| `src/analysis/filter_unknown.py` | 모르는 알약으로 보이는 박스 점수 낮추기 (실험용, 미사용) |
| `src/analysis/predict_contrastive.py` | 대조학습 임베딩으로 예측 클래스 재분류 (실험용) |
| `src/analysis/plot_experiments.py` | 시트 기록으로 README 실험 그래프(SVG) 생성 |

## 실험 기록

<!-- 실험그래프 시작 -->
### 🏆 Kaggle 점수 TOP 5

![Kaggle 점수 순위](docs/images/experiments_best.svg)

| 순위 | kaggle_score | name | author | val mAP75-95 | 시간 | memo |
|---|---|---|---|---|---|---|
| 1 | 0.6255 | rahui_ep30_img960_batch8_aihubdata_oversample_f2 | 김라희 | 0.9892 | 09-20 23:04 | 오버샘플링유지 + imgsz 1280→960, batch 4→8, epoch 50→30, close_mosaic 5→3(10% 비율 유지) |
| 2 | 0.62067 | jw_ep30_img960_batch8_lr0.001_cos_lr_aihubdata | 신재민 | 0.983 | 09-18 17:16 | epochs: 30 / imgsz: 960 / batch: 8 / lr: 0.001 / cos_lr: True / warmup 1 · close_mosaic 3 (100 epoch 와 같은 비율) / AI Hub 조합 데이터 추가 (train 10,394장, 118종) |
| 3 | 0.61712 | rahui_ep50_img1280_batch4_aihubdata_oversample | 김라희 | 0.9903 | 09-20 02:43 | v2(box=12.0/cls=1.0) 설정 그대로 유지, 35206·3832(각인의존형 취약 클래스) 학습 이미지 2배 오버샘플링 추가,conf=0.25 |
| 4 | 0.61084 | rahui_ep50_img1280_batch4_aihubdata_boxcls_v2 | 김라희 | 0.9924 | 09-19 05:07 | 0.60543 조합(img1280·batch4·warmup1.5·close_mosaic5) 그대로 + box12.0·cls1.0 재추가 (단일 변수 격리 테스트) |
| 5 | 0.61074 | jw_ep30_img960_batch8_lr0.001_cos_lr_aihubdata | 엄재웅 | 0.9929 | 09-18 11:52 | epochs: 30 / imgsz: 960 / batch: 8 / lr: 0.001 / cos_lr: True / warmup 1 · close_mosaic 3 (100 epoch 와 같은 비율) / AI Hub 조합 데이터 추가 (train 10,394장, 118종) |
<!-- 실험그래프 끝 -->


## 협업 방식 · 컨벤션

- 단계별로 브랜치 1개를 만들고 4명이 같은 브랜치에서 작업합니다.
- 작업 시작 전 `git pull`, 완료 즉시 `git push`.
- 같은 파일을 수정하기 전에 팀 채팅에 먼저 공유합니다.
- 재현 가능하도록 랜덤 시드를 고정합니다.

**컨벤션**

타입은 `feat` / `fix` / `chore` / `design` / `docs` / `refactor` 6개입니다.

| 대상 | 형식 | 예시 |
|---|---|---|
| Branch | `타입/#이슈번호-작업내용` | `feat/#11-json-to-yolo` |
| Commit | `타입: 작업 내용` | `feat: JSON→YOLO 라벨 변환 구현` |
| Issue / PR | `[타입] 작업 내용` | `[Feat] JSON→YOLO 라벨 변환` |

```
# Branch
feat/#11-json-to-yolo
fix/#12-label-bbox-bug

# Commit
feat: JSON→YOLO 라벨 변환 구현
fix: 바운딩 박스 좌표 오류 수정
chore: 개발 환경 세팅
docs: README 업데이트
refactor: 전처리 스크립트 리팩토링

# Issue / PR
[Feat] JSON→YOLO 라벨 변환
[Fix] 바운딩 박스 좌표 오류 수정
```

## 👥 팀

<div align="center">

### 팀 파이리 (파트2_1팀)
<img width="500" height="275" alt="i15089145892" src="https://github.com/user-attachments/assets/3c0e71b5-b6ab-41ea-831f-7c286efc85e4" />


| <img src="https://github.com/user-attachments/assets/ba8bb80e-1b2d-4ac7-96e7-bf58ce9bedb9" width="150" height="150"> | <img src="https://avatars.githubusercontent.com/u/312283986?v=4" width="150" height="150"> | <img src="https://avatars.githubusercontent.com/u/207137531?v=4" width="150" height="150"> | <img src="https://avatars.githubusercontent.com/u/310539671?v=4" width="150" height="150"> | 
|:---:|:---:|:---:|:---:|
| **[엄재웅](https://github.com/woolnd)** | **[윤성원](https://github.com/UchimataDev)** | **[김라희](https://github.com/KIMRAHUI)** | **[신재민](https://github.com/soul8327)** |
| 팀장 | 모델 개발 | 모델 개발 | 모델 개발 |

</div>

### 📑 보고서 · 발표 자료

| 자료 | 링크 |
|:---:|:---:|
| 초급 프로젝트 보고서 (1팀) | [📄 PDF](docs/%E1%84%8E%E1%85%A9%E1%84%80%E1%85%B3%E1%86%B8%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3_%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%281%E1%84%90%E1%85%B5%E1%86%B7%29.pdf) |
| 초급 프로젝트 최종 발표 (1팀) | [📊 PDF](docs/%E1%84%8E%E1%85%A9%E1%84%80%E1%85%B3%E1%86%B8%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3_%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%E1%84%87%E1%85%A1%E1%86%AF%E1%84%91%E1%85%AD%281%E1%84%90%E1%85%B5%E1%86%B7%29.pdf) |

### 📝 협업 일지

| 날짜 | 엄재웅 | 윤성원 | 김라희 | 신재민 |
|:---:|:---:|:---:|:---:|:---:|
| 09-10 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-10-3d777fa64f7f8011a5fff6741aa963e0?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-10-3d777fa64f7f807fbe4ccff5b42ed249?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/09-10Daily-3d777fa64f7f8038a483e9c31208eccb?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-10-3d777fa64f7f80bcab5cc5a66ead2e4a?source=copy_link) |
| 09-11 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-11-3d877fa64f7f806980ddd7e45f4a2ca3?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-11-3d877fa64f7f804e9cb5dfc1a8e21cec?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/09-11Daily-3d877fa64f7f80898b9ee0aa0779e13f?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-11-3d877fa64f7f80ff9004eaf0e55b2328?source=copy_link) |
| 09-14 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-14-3db77fa64f7f801592fde3c9620f62ea?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-14-3db77fa64f7f80309ad5ee063b2dbd33?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/09-11Daily-3db77fa64f7f80d69c09d018888a360a?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-14-3db77fa64f7f800483c3e382ec222b57?source=copy_link) |
| 09-15 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-15-3dc77fa64f7f802db9c7c4ece66ef915?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-15-3dc77fa64f7f8060a4ddf31fe213a441?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/09-15Daily-3dc77fa64f7f8024a953ea2277fea760?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-15-3dc77fa64f7f80618184ec4d933baba7?source=copy_link) |
| 09-16 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-16-3dd77fa64f7f809d8e75cd6c818f31b7?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-16-3dd77fa64f7f80b38f52d031552e57f6?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/9-16-3dd77fa64f7f80149b63c1c5b4c55878?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-16-3dd77fa64f7f804ba599c6265f48a574?source=copy_link) |
| 09-17 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-17-3de77fa64f7f80cab749f6116eca64d8?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-17-3de77fa64f7f809581cbfda45f9d7e6c?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-18-3df77fa64f7f80ddbc3be88e8238dc76?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-17-3de77fa64f7f801d963bccfecc295c7e?source=copy_link) |
| 09-18 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-18-3df77fa64f7f80c89704cd200328dcb2?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-18-3df77fa64f7f80dd8d92f11bb7810e57?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/9-18-3df77fa64f7f8044b67ecb0f45ab1f2f?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-18-3df77fa64f7f80ddbc3be88e8238dc76?source=copy_link) |
| 09-21 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-21-3e277fa64f7f809cbc3bd7aa520ae056?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-21-3e277fa64f7f8007992bd41fadf14e0f?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/9-21-3e277fa64f7f80f48786d8b9bc6ee7e4?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-21-3e277fa64f7f80e38b87f8dc3194d3a1?source=copy_link) |
| 09-22 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-22-3e377fa64f7f801a9288dbb16ff0788f?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-22-3e377fa64f7f8076bc33f18574aed28c?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/9-22-3e377fa64f7f80bb928ee958bb95e440?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-22-3e377fa64f7f80cb8e7bd7ac50b26d57?source=copy_link) |
| 09-23 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-23-3e477fa64f7f8078a2ddea40adbe8050?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-23-3e477fa64f7f8021a1a6ff3892523a28?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/9-23-3e477fa64f7f802cae15f1a28ec4b7ba?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-23-3e477fa64f7f80719ee4e7dfbf729906?source=copy_link) |
| 09-28 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-28-3e977fa64f7f800db39ffc90a29a7b4f?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-28-3e977fa64f7f80aeb421c77f5cf76688?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/9-28-3e977fa64f7f809eb4abdd19b2c93bc5?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-28-3e977fa64f7f80dc983ffbbd82deca2d?source=copy_link) |
| 09-29 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-29-3ea77fa64f7f8087ab0de56f1b97b676?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-29-3ea77fa64f7f808daeafd1cdad5b079f?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/9-29-3ea77fa64f7f8055821ac4f450403f0a?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-29-3ea77fa64f7f80b49c37e8cff61a4c29?source=copy_link) |
| 09-30 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-30-3eb77fa64f7f8002b3c0dd1efceb8669?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-30-3eb77fa64f7f80eca088ca982cc4d8c7?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/9-30-3eb77fa64f7f80a6a1fbcf132378118e?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-30-3eb77fa64f7f80629664d18c52ee0e1e?source=copy_link) |
| **전체** | [📂 전체 보기](https://smoggy-gymnast-0ed.notion.site/3ce77fa64f7f808b98c9c8ee73af6b2a?source=copy_link) | [📂 전체 보기](https://smoggy-gymnast-0ed.notion.site/3ce77fa64f7f806bbfe0d8104be5278f?source=copy_link) | [📂 전체 보기](https://smoggy-gymnast-0ed.notion.site/3ce77fa64f7f8009b24ae54025455ab6?source=copy_link) | [📂 전체 보기](https://smoggy-gymnast-0ed.notion.site/3ce77fa64f7f80c2a82af7e97a665164?source=copy_link) |



