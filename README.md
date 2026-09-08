# pill-detection

YOLO로 이미지 속 알약을 탐지하고 종류를 분류하는 프로젝트입니다.

- **입력**: 여러 알약이 함께 놓인 이미지
- **출력**: 알약별 바운딩 박스 + 클래스 + 신뢰도
- **데이터**: AI Hub 「경구약제 이미지 데이터」 중 조합경구약제 (클래스 10~15개)
- **팀**: 4명 / 기간 약 4일

## 시작하기

```bash
git clone <repo-url>
cd pill-detection
uv sync
```

`uv sync`만 하면 팀원 전원이 동일한 환경이 됩니다. 패키지는 `uv add <pkg>`로 추가하고 `uv.lock`을 꼭 커밋합니다.

## 디렉터리

```
data/raw/    AI Hub 원본 데이터 (git 제외)
data/yolo/   YOLO 형식으로 변환된 데이터셋 (git 제외)
src/         전처리 · 학습 · 평가 스크립트
docs/        클래스 선정 기준, 분석 문서
```

데이터는 용량이 커서 커밋하지 않습니다. AI Hub에서 직접 내려받아 `data/raw/`에 둡니다.

## 진행 단계

| 단계 | 내용 |
|---|---|
| 1. 데이터 준비 | aihubshell 다운로드, 클래스 10~15개 선정 |
| 2. 전처리 | 이미지 정제 → JSON 정제 → 클래스 필터링 → COCO→YOLO 변환 → train/val 분할 |
| 3. 학습 | YOLO 베이스라인 → 하이퍼파라미터 조정 → (여유 시) 비교 모델 |
| 4. 평가 | mAP, 클래스별 Precision/Recall, 시각화, confusion matrix, 실패 사례 |
| 5. 문서화 | README 정리 및 재현성 검증 |

## 협업 방식

- 단계별로 브랜치 1개를 만들고 4명이 같은 브랜치에서 작업합니다.
- 작업 시작 전 `git pull`, 완료 즉시 `git push`.
- 같은 파일을 수정하기 전에 팀 채팅에 먼저 공유합니다.

### 컨벤션

타입은 `feat` / `fix` / `chore` / `design` / `docs` / `refactor` 6개입니다.

| 대상 | 형식 | 예시 |
|---|---|---|
| Branch | `타입/#이슈번호-작업내용` | `feat/#11-coco-to-yolo` |
| Commit | `타입: 작업 내용` | `feat: COCO→YOLO 라벨 변환 구현` |
| Issue / PR | `[타입] 작업 내용` | `[Feat] COCO→YOLO 라벨 변환` |

```
# Branch
feat/#11-coco-to-yolo
fix/#12-label-bbox-bug

# Commit
feat: COCO→YOLO 라벨 변환 구현
fix: 바운딩 박스 좌표 오류 수정
chore: 개발 환경 세팅
docs: README 업데이트
refactor: 전처리 스크립트 리팩토링

# Issue / PR
[Feat] COCO→YOLO 라벨 변환
[Fix] 바운딩 박스 좌표 오류 수정
```
