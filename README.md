# pill-detection

YOLO로 이미지 속 알약을 탐지하고 종류를 분류하는 프로젝트입니다.

- **입력**: 여러 알약이 함께 놓인 이미지
- **출력**: 알약별 바운딩 박스 + 클래스 + 신뢰도
- **데이터**: Kaggle 대회 `ai14-level-project` 제공 데이터 (AI Hub 「경구약제 이미지 데이터」 조합경구약제 기반)
- **평가**: Kaggle 리더보드 **mAP[0.75:0.95]**, 최종 순위는 Private Score 기준
- **팀**: 4명 / 기간 약 4일

Kaggle 점수 TOP 5 순위는 [실험 기록](#실험-기록)에서 볼 수 있습니다.

## 시작하기

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
SHEET_URL=<실험 기록 구글 시트 웹앱 URL>   # 팀 채팅으로 공유
AUTHOR=<본인 이름>                          # 실험 기록의 author 칸
```

## 전체 실행 순서

```bash
uv run --env-file .env python -m src.data.download_data   # 1. Kaggle 데이터 다운로드
uv run python -m src.data.make_yolo                      # 2. 라벨 정리 + COCO JSON → YOLO 변환 + train/val 분할
uv run python -m src.data.check_yolo                     #    변환 결과 검증
uv run python -m src.train.train_yolo                    # 3. 학습 + val 평가 + 실험 기록(csv·구글 시트)
uv run python -m src.analysis.visualize                  # 4. val 예측 시각화 · 실패 사례 · 클래스별 점수
uv run python -m src.analysis.predict                    # 5. test 예측 → Kaggle 제출 파일
```

모든 스크립트는 **프로젝트 루트에서 `python -m src.폴더.파일`** 형태로 실행합니다. (`python src/data/make_yolo.py` 처럼 경로로 실행하면 `from src.config import ...` 를 찾지 못합니다)

## 디렉터리

```
data/raw/      Kaggle 원본 데이터 (git 제외)
data/yolo/     YOLO 형식으로 변환된 데이터셋 (git 제외)
runs/          학습 결과 · 가중치 · 분석 그림 · 제출 파일 (git 제외)
src/           공통 모듈 (config · annotations · sheet)
src/data/      데이터 다운로드 · 확인 · YOLO 변환
src/train/     학습 (YOLO, Faster R-CNN · RetinaNet 비교용)
src/analysis/  예측 · 시각화 · 모르는 약 필터 · README 그래프
docs/images/   README 실험 그래프 (GitHub Actions가 자동 갱신)
.github/       이슈·PR 템플릿, README 그래프 갱신 워크플로
```

| 파일 | 역할 |
|---|---|
| `src/config.py` | 공통 경로(`KAGGLE_DIR`, `YOLO_DIR`, `RUNS_DIR` 등) · `SEED` · `get_device` |
| `src/annotations.py` | (이미지 × 알약)당 JSON 1개인 라벨을 이미지 파일명 기준으로 묶기 · `compute_iou` |
| `src/sheet.py` | 구글 시트(Apps Script 웹앱)로 실험 기록 전송 |
| `src/data/download_data.py` | Kaggle 대회 데이터를 `data/raw/`에 다운로드 |
| `src/data/check_raw.py` | 원본 데이터 요약 + 박스 시각화 |
| `src/data/make_yolo.py` | 문제 있는 이미지 제외 → YOLO 라벨 변환 → 조합 단위 train/val 분할 |
| `src/data/check_yolo.py` | YOLO 라벨을 픽셀로 되돌려 원본과 비교 |
| `src/train/train_yolo.py` | YOLO 학습, val 평가(mAP75-95), 실험 기록 |
| `src/analysis/visualize.py` | val 예측/정답 비교 그림, 오류 유형 집계, 클래스별 AP, 학습 곡선 |
| `src/analysis/predict.py` | test 842장 예측 → 제출 형식 csv |
| `src/analysis/plot_experiments.py` | 시트 기록으로 README 실험 그래프(SVG) 생성 |

## 데이터

```
data/raw/sprint_ai_project1_data/
├── train_images/        학습 이미지 (.png)
├── train_annotations/   K-<조합>_json/K-<약품코드>/*.json
└── test_images/         제출용 이미지 (.png, 라벨 없음, 파일명 = 숫자)
```

어노테이션은 **(이미지 × 알약)당 JSON 1개**인 COCO 형식이고, bbox는 `(x, y, w, h)` 픽셀 좌표(좌상단 기준)입니다.

| 항목 | 값 |
|---|---|
| train 이미지 | 232장 |
| 박스 (JSON) | 763개 |
| 클래스 | 56개 |
| test 이미지 | 842장 |
| 이미지당 알약 수 | 2알 7장 / 3알 151장 / 4알 74장 |
| 이미지 크기 | 976 × 1280 (train·test 전체) |

외부 데이터는 사용 가능하지만, AI Hub의 아래 2개는 **사용 금지**입니다. (경진대회 train/test 데이터의 원본)

- `Training > 라벨링데이터 > 경구약제조합 5000종 > TL_2_조합.zip`
- `Training > 원천데이터 > 경구약제조합 5000종 > TS_2_조합.zip`

## 전처리

`src/data/make_yolo.py`가 아래 순서로 `data/yolo/`를 만듭니다.

**1. 문제 있는 이미지 제외 (232장 → 220장)**

| 제외 이유 | 장수 | 판단 방법 |
|---|---|---|
| 라벨 빠짐 | 8 | 파일명의 약 ID(`K-003351-032310-038162` → 3351·32310·38162) 중 박스 라벨이 없는 약이 있음 |
| 라벨 겹침 | 3 | 한 알약 박스에 서로 다른 약 라벨 2개가 같은 좌표로 붙어 있음 |
| 박스가 이미지 밖 | 1 | 좌표가 이미지 크기를 넘음 (x=6567) |

박스만 빼지 않고 이미지째 뺍니다. 라벨이 빠진 알약은 학습 때 '배경'으로 배워서 모델이 그 알약을 무시하게 되기 때문입니다. 제외해도 train에서 사라지는 클래스는 없습니다.

**2. 클래스 번호** · 약 ID(1900, 2483, …)를 정렬해 YOLO 번호 0~55로 매핑. `data.yaml`의 `names`가 번호 → 약 ID 표입니다.

**3. train/val 분할** · **조합 단위**로 8:2 (seed 42). 같은 조합을 70°/75°/90°로 찍은 사진이 train과 val에 섞이면 val 점수가 부풀려지기 때문입니다.

| 결과 | 값 |
|---|---|
| train / val | 183장 / 37장 |
| train에 없는 클래스 | 없음 (val에만 들어간 약 `33009`의 조합을 train으로 되돌림) |
| 변환 검증 (`check_yolo.py`) | 라벨 220개, 원본과 불일치 0개 |

## 학습

`src/train/train_yolo.py` 위쪽 설정만 바꿔서 실험합니다.

```python
MODEL  = "yolo26n.pt"     # 모델 크기
EPOCHS = 100              # 학습 바퀴 수
IMGSZ  = 960              # 입력 이미지 크기
BATCH  = 8                # 한 번에 넣는 사진 수 (64장마다 1번 업데이트는 BATCH와 무관하게 유지)
NAME   = "jw_yolo26n_ep100_img960_b8_base"  # 실험 이름 = runs/<NAME>/ (실험마다 새 이름)
MEMO   = "무엇을 왜 바꿨는지 한 줄"
EXPERIMENT = {"optimizer": "AdamW", "lr0": 0.001}  # 나머지 하이퍼파라미터
```

- **한 실험에서는 설정 하나만** 바꾸고, NAME은 `이니셜_모델_ep_img_b_바꾼점`으로 짓습니다(예: `jw_yolo26n_ep50_img640_b4_compare`). 같은 NAME이면 결과가 덮어써집니다.
- `optimizer`가 기본값 `auto`면 `lr0`·`momentum`을 넣어도 무시됩니다. 학습률을 바꿀 땐 `optimizer`도 같이 지정합니다.
- EPOCHS가 다르면 워밍업 비율·mosaic 끄는 시점도 달라집니다. 같은 EPOCHS끼리 비교합니다.
- 16GB 맥에서 IMGSZ 960·BATCH 16은 메모리가 부족해 스왑이 발생했습니다. 960에서는 BATCH 8을 씁니다.

학습이 끝나면 best.pt로 val을 평가해 mAP50 / mAP50-95 / **mAP75-95**(대회 지표 구간)를 출력하고, `runs/experiments.csv`와 구글 시트에 한 줄 기록합니다.

| 결과 파일 | 내용 |
|---|---|
| `runs/<NAME>/weights/best.pt` | val mAP50-95가 가장 높았던 가중치 |
| `runs/<NAME>/args.yaml` | 실제로 쓰인 설정값 전체 (기본값 포함) |
| `runs/<NAME>/results.csv` | epoch별 loss · mAP · 학습률 |
| `runs/<NAME>_val/` | confusion matrix, PR 곡선 |
| `runs/experiments.csv` | 실험마다 점수 + 학습 인자 (로컬 백업) |

## 평가 · 시각화

```bash
uv run python -m src.analysis.visualize   # train_yolo.py 의 NAME 실험을 분석
```

val 예측을 정답과 IoU로 짝지어 분류하고 `runs/<NAME>_analysis/`에 저장합니다.

| 분류 | 뜻 |
|---|---|
| OK | 클래스·위치 모두 맞음 (IoU 0.75 이상) |
| LOC | 클래스는 맞고 위치 부정확 (IoU 0.75 미만) |
| CLS | 위치는 맞고 클래스 틀림 |
| DUP | 같은 알약에 예측을 2개 냄 |
| FP | 알약이 없는 곳을 예측 |
| FN | 정답 알약을 놓침 |

- `failures.png` · 실패가 있는 이미지 6장. 박스 **위 글자 = 모델 예측**, **아래 글자 = 정답 라벨**
- `class_ap.csv` · 클래스별 AP50 · AP75-95 (낮은 순)
- `curves.png` · epoch별 box/cls loss, val mAP

## 결과

| 실험 | 데이터 | 설정 | val mAP50 | val mAP50-95 | val mAP75-95 | Kaggle Public |
|---|---|---|---|---|---|---|
| baseline | 정리 전 (val 42장) | yolo26n · 640 · 50 epoch · BATCH 16 · 기본값 | 0.7876 | 0.7550 | 0.7225 | 0.2356 ※ |
| jw_ep100_img960_batch8_lr0.00 | 정리 전 (val 42장) | yolo26n · 960 · 100 epoch · BATCH 8 · AdamW lr0 0.001 | 0.9456 | 0.9388 | 0.9321 | 0.39139 |

※ 같은 baseline 설정을 다른 컴퓨터에서 학습한 실행(val mAP75-95 0.7179)의 제출 점수입니다.
라벨 정리·재분할 후에는 val이 37장으로 바뀌어 위 val 점수와 직접 비교하지 않습니다. 전체 기록은 아래 「실험 기록」과 팀 구글 시트를 기준으로 합니다.

**val 오류 유형 (`visualize.py`, 신뢰도 0.25 이상, 정답 박스 131개)**

| 실험 | OK | LOC | CLS | DUP | FP | FN |
|---|---|---|---|---|---|---|
| baseline | 93 | 0 | 24 | 17 | 3 | 14 |
| 960 · 100 epoch · AdamW | 123 | 0 | 3 | 2 | 3 | 5 |

- 두 실험 모두 **LOC 0개** · 박스 위치는 처음부터 정확했고, 점수를 깎은 것은 분류와 중복 예측이었습니다.
- 960 · 100 epoch 조합에서 CLS 24 → 3, DUP 17 → 2로 줄었습니다. (세 설정을 함께 바꿔 각각의 효과는 분리되지 않음)
- 이 조합에 남은 오류 13개 중 FP 3개 전부, CLS 2개, FN 1개는 **라벨 오류 이미지**에서, FN 3개는 train에 없는 **33009**에서 나왔습니다. → 전처리에서 라벨 오류 이미지를 제외한 이유
- val mAP75-95 0.93과 Kaggle 0.39의 차이가 커서, 이후 실험은 Kaggle 점수를 최종 판단 기준으로 봅니다.

## 실험 기록

<!-- 실험그래프 시작 -->
### 🏆 Kaggle 점수 TOP 5

![Kaggle 점수 순위](docs/images/experiments_best.svg)

| 순위 | kaggle_score | name | author | val mAP75-95 | 시간 | memo |
|---|---|---|---|---|---|---|
| 1 | 0.61074 | jw_ep30_img960_batch8_lr0.001_cos_lr_aihubdata | 엄재웅 | 0.9929 | 09-18 11:52 | epochs: 30 / imgsz: 960 / batch: 8 / lr: 0.001 / cos_lr: True / warmup 1 · close_mosaic 3 (100 epoch 와 같은 비율) / AI Hub 조합 데이터 추가 (train 10,394장, 118종) |
| 2 | 0.41738 | yolo26n_clean_v2_imgsz1280 | 김라희 | 0.995 | 09-17 18:09 | yolo26n_clean_v2와 EXPERIMENT·epochs(150)·patience(60) 전부 동일, imgsz만 960→1280으로 변경해 크기 효과만 확인 (기존 1280_v3는 degrees=15·cls 1.5도 같이 바뀌어 크기 효과와 분리 안 됐음) |
| 3 | 0.4155 | jw_ep100_img1280_batch4_lr0.001_cos_lr | 엄재웅 | 0.9784 | 09-17 19:58 | epochs: 100 / imgsz: 1280 / batch: 4 / lr: 0.001 / cos_lr: True |
| 4 | 0.41504 | jw_ep100_img960_batch8_lr0.001_resplit | 엄재웅 | 0.984 | 09-16 09:59 | epochs: 100 / imgsz: 960 / batch: 8 / lr: 0.001 / val 재분할(train에 모든 클래스가 학습될 수 있도록) |
| 5 | 0.41264 | imgsz1024, ep150, batch=4 | 윤성원 | 0.9918 | 09-16 16:11 | imgsz=1024, batch=4 가 현재까지 점수가 제일 좋아 EPOCHS=150으로 값 조정 |
<!-- 실험그래프 끝 -->

## 실험 기록 자동화

```
train_yolo.py 학습 종료
  └─ 구글 시트 experiments 탭에 한 줄 추가 (Apps Script 웹앱)
       └─ GitHub에 신호 → Actions가 시트를 읽어 README 그래프 갱신
            └─ docs/readme-graph 브랜치로 PR 생성 → 팀장이 확인 후 머지
```

| 시트 열 | 입력 |
|---|---|
| time, author, name, memo, mAP50, mAP50-95, mAP75-95, model, epochs, imgsz, batch, seed, experiment | 자동 (`train_yolo.py`) |
| kaggle_score, 결론 / 다음 실험 | **직접 입력** |

- `SHEET_URL`은 `.env`와 저장소 Secret에만 두고 공개하지 않습니다.
- README 아래 「실험 기록」의 표시 영역은 자동으로 바뀌므로 직접 수정하지 않습니다.
- 다음 실험 방향은 시트를 보고 팀이 판단합니다.

## 예측 · 제출

```bash
uv run python -m src.analysis.predict   # train_yolo.py 의 NAME 실험 best.pt 로 예측 → runs/<NAME>/submission.csv
```

```
annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
```

- `image_id` = test 파일명 숫자, `category_id` = 원래 약 ID (YOLO 번호 아님), bbox = 원본 이미지 픽셀 좌표
- `CONF = 0.001` (mAP는 낮은 점수 박스까지 반영해 계산)

**제출은 사람이 Kaggle 웹에서 직접 합니다.**

1. **팀 계정으로** 업로드 (개인 제출 금지, 팀 전체 하루 10회)
2. 설명칸에 `NAME / conf0.001` 형식으로 작성
3. 점수가 나오면 구글 시트의 해당 행 `kaggle_score`에 입력

## 알려진 이슈

- 클래스 `33009`는 조합 단위 분할 결과 val에만 있어 학습되지 않습니다.
- val(37장)이 작아 0.01~0.02 차이는 우연일 수 있고, val과 Kaggle 점수 차이가 큽니다.
- 맥 GPU(mps)에서는 일부 연산이 비결정적이라, 장치가 다르면 같은 설정·시드로도 점수가 조금 다릅니다. (같은 맥에서는 50 epoch baseline 두 번 모두 0.7225)

## 진행 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| 1. 데이터 준비 | Kaggle 데이터 다운로드, 어노테이션 로드·시각화 | ✅ 완료 (클래스 56개 전체 사용) |
| 2. 전처리 | 라벨 오류 이미지 제외 → COCO JSON→YOLO 변환 → 조합 단위 분할 → 변환 검증 | ✅ 완료 |
| 3. 학습 | YOLO 베이스라인 → 하이퍼파라미터 조정 → 비교 모델 | 🔬 실험 중 (구현 완료 · 기준 조합 확보, 하이퍼파라미터·비교 모델 실험 진행) |
| 4. 평가 | mAP[0.75:0.95], 클래스별 AP, 시각화, confusion matrix, 실패 사례 | ✅ 완료 (분석 자동화, 실험마다 실행) |
| 5. 제출·문서화 | Kaggle 제출(1일 10회 제한), README 정리 및 재현성 검증 | 🟡 진행 중 (제출·기록 자동화 완료 · 최종 모델 선택, 재현성 검증, 보고서 남음) |

## 협업 방식

- 단계별로 브랜치 1개를 만들고 4명이 같은 브랜치에서 작업합니다.
- 작업 시작 전 `git pull`, 완료 즉시 `git push`.
- 같은 파일을 수정하기 전에 팀 채팅에 먼저 공유합니다.
- 재현 가능하도록 랜덤 시드를 고정합니다.

### 컨벤션

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

### 📝 협업 일지

| 날짜 | 엄재웅 | 윤성원 | 김라희 | 신재민 |
|:---:|:---:|:---:|:---:|:---:|
| 09-10 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-10-3d777fa64f7f8011a5fff6741aa963e0?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-10-3d777fa64f7f807fbe4ccff5b42ed249?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/09-10Daily-3d777fa64f7f8038a483e9c31208eccb?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-10-3d777fa64f7f80bcab5cc5a66ead2e4a?source=copy_link) |
| 09-11 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-11-3d877fa64f7f806980ddd7e45f4a2ca3?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-11-3d877fa64f7f804e9cb5dfc1a8e21cec?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/09-11Daily-3d877fa64f7f80898b9ee0aa0779e13f?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-11-3d877fa64f7f80ff9004eaf0e55b2328?source=copy_link) |
| 09-14 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-14-3db77fa64f7f801592fde3c9620f62ea?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-14-3db77fa64f7f80309ad5ee063b2dbd33?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/09-11Daily-3db77fa64f7f80d69c09d018888a360a?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-14-3db77fa64f7f800483c3e382ec222b57?source=copy_link) |
| 09-15 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-15-3dc77fa64f7f802db9c7c4ece66ef915?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-15-3dc77fa64f7f8060a4ddf31fe213a441?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/09-15Daily-3dc77fa64f7f8024a953ea2277fea760?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-15-3dc77fa64f7f80618184ec4d933baba7?source=copy_link) |
| 09-16 | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-16-3dd77fa64f7f809d8e75cd6c818f31b7?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-16-3dd77fa64f7f80b38f52d031552e57f6?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/9-16-3dd77fa64f7f80149b63c1c5b4c55878?source=copy_link) | [📝](https://smoggy-gymnast-0ed.notion.site/26-09-16-3dd77fa64f7f804ba599c6265f48a574?source=copy_link) |
