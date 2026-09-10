# pill-detection

YOLO로 이미지 속 알약을 탐지하고 종류를 분류하는 프로젝트입니다.

- **입력**: 여러 알약이 함께 놓인 이미지
- **출력**: 알약별 바운딩 박스 + 클래스 + 신뢰도
- **데이터**: Kaggle 대회 `ai14-level-project` 제공 데이터 (AI Hub 「경구약제 이미지 데이터」 조합경구약제 기반)
- **평가**: Kaggle 리더보드 **mAP[0.75:0.95]**, 최종 순위는 Private Score 기준
- **팀**: 4명 / 기간 약 4일

## 시작하기

```bash
git clone <repo-url>
cd pill-detection
uv sync
```

`uv sync`만 하면 팀원 전원이 동일한 환경이 됩니다. 패키지는 `uv add <pkg>`로 추가하고 `uv.lock`을 꼭 커밋합니다. (`pip` 사용 금지)

## 디렉터리

```
data/raw/    Kaggle 원본 데이터 (git 제외)
data/yolo/   YOLO 형식으로 변환된 데이터셋 (git 제외)
src/         전처리 · 학습 · 평가 스크립트
docs/        클래스 선정 기준, 분석 문서
```

데이터는 용량이 커서 커밋하지 않습니다. 아래 「데이터 확보」를 따라 `data/raw/`에 받습니다.

## 데이터 확보

### 1. Kaggle API 키 설정

Kaggle 대회 페이지에서 참가(규칙 동의) 후, 프로젝트 루트에 `.env`를 만듭니다. `.env`는 gitignore 되어 있으며 **절대 커밋하지 않습니다.**

```
KAGGLE_USERNAME=<캐글 아이디>
KAGGLE_KEY=<API 키>
```

### 2. 다운로드

```bash
uv run --env-file .env python src/download_data.py
```

아래 구조로 받아집니다.

```
data/raw/sprint_ai_project1_data/
├── train_images/        학습 이미지 (.png)
├── train_annotations/   K-<조합>_json/K-<약품코드>/*.json
└── test_images/         제출용 이미지 (.png, 라벨 없음)
```

어노테이션은 **(이미지 × 알약)당 JSON 1개**인 COCO 형식이고, bbox는 `(x, y, w, h)` 픽셀 좌표(좌상단 기준)입니다. `src/annotations.py`의 `load_annotations()`가 이미지 파일명 기준으로 박스를 묶어 줍니다.

### 데이터 현황 (`src/check_data.py` 실행 결과)

| 항목 | 값 |
|---|---|
| train 이미지 | 232장 |
| 박스 (JSON) | 763개 |
| 클래스 | 56개 |
| test 이미지 | 842장 |
| 이미지당 알약 수 | 2알 7장 / 3알 151장 / 4알 74장 |

### 외부 데이터 사용 규칙

외부 데이터는 사용 가능하지만, AI Hub의 아래 2개는 **사용 금지**입니다.

- `Training > 라벨링데이터 > 경구약제조합 5000종 > TL_2_조합.zip`
- `Training > 원천데이터 > 경구약제조합 5000종 > TS_2_조합.zip`

외부 데이터를 추가로 쓸 경우 출처와 파일명을 `docs/`에 기록합니다.

## 데이터 확인

```bash
cd src
uv run python check_data.py
```

데이터 요약(이미지·박스·클래스 수)을 출력하고, train 이미지 6장을 랜덤(seed 42 고정)으로 골라 바운딩 박스와 클래스 ID를 그려 보여줍니다.

## 진행 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| 1. 데이터 준비 | Kaggle 데이터 다운로드, 어노테이션 로드·시각화, 클래스 선정 | 🟡 진행 중 (다운로드·시각화 완료) |
| 2. 전처리 | 클래스 필터링 → COCO JSON→YOLO 변환 → train/val 분할 | ⬜ |
| 3. 학습 | YOLO 베이스라인 → 하이퍼파라미터 조정 → (여유 시) 비교 모델 | ⬜ |
| 4. 평가 | mAP[0.75:0.95], 클래스별 Precision/Recall, 시각화, confusion matrix, 실패 사례 | ⬜ |
| 5. 제출·문서화 | Kaggle 제출(1일 10회 제한), README 정리 및 재현성 검증 | ⬜ |

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
