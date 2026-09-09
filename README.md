# pill-detection

YOLO로 이미지 속 알약을 탐지하고 종류를 분류하는 프로젝트입니다.

- **입력**: 여러 알약이 함께 놓인 이미지
- **출력**: 알약별 바운딩 박스 + 클래스 + 신뢰도
- **데이터**: AI Hub 「경구약제 이미지 데이터」(dataSetSn=576) 중 조합경구약제 (클래스 10~15개)
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
data/raw/    원본 이미지 + 어노테이션 CSV (git 제외)
data/yolo/   YOLO 형식으로 변환된 데이터셋 (git 제외)
src/         전처리 · 학습 · 평가 스크립트
docs/        클래스 선정 기준, 분석 문서
```

데이터는 용량이 커서 커밋하지 않습니다. 아래 「데이터 확보」를 참고해 `data/raw/`에 둡니다.

## 데이터 확보

전년도 프로젝트에서 **조합경구약제만 필터링·정제한 어노테이션**을 확보했습니다. 이미지 통합, JSON 정제, 손상 이미지 제거가 이미 끝난 상태라 이걸 1순위로 사용합니다.

- 이미지 2,050장 (4알 1,906장 / 3알 144장), 박스 8,056개, 클래스 82개
- `image_annotations.csv` — bbox는 COCO와 같은 `(x, y, w, h)` 픽셀 좌표 (`w * h == area`로 검증)
- 이미지 실물은 별도 공유 드라이브에 있음

원본이 필요하거나 위 경로를 못 쓰게 되면 AI Hub에서 직접 내려받습니다.

```bash
curl -o aihubshell https://api.aihub.or.kr/api/aihubshell.do && chmod +x aihubshell
./aihubshell -mode l -datasetkey 576                       # 파일 목록에서 조합경구약제 filekey 확인
./aihubshell -mode d -datasetkey 576 -filekey <키> -aihubapikey '<발급키>'
```

전체를 받으면 수백 GB이므로 반드시 `-mode l`로 조합경구약제 filekey를 골라서 받습니다. API 키는 커밋하지 않고 환경변수로 둡니다. 어떤 filekey를 받았는지는 `docs/`에 기록해 팀원 간 데이터가 어긋나지 않게 합니다.

## 진행 단계

| 단계 | 내용 |
|---|---|
| 1. 데이터 준비 | 어노테이션 CSV·이미지 확보, 클래스 10~15개 선정 |
| 2. 전처리 | 클래스 필터링 → CSV→YOLO 변환 → train/val 분할 |
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
| Branch | `타입/#이슈번호-작업내용` | `feat/#11-csv-to-yolo` |
| Commit | `타입: 작업 내용` | `feat: CSV→YOLO 라벨 변환 구현` |
| Issue / PR | `[타입] 작업 내용` | `[Feat] CSV→YOLO 라벨 변환` |

```
# Branch
feat/#11-csv-to-yolo
fix/#12-label-bbox-bug

# Commit
feat: CSV→YOLO 라벨 변환 구현
fix: 바운딩 박스 좌표 오류 수정
chore: 개발 환경 세팅
docs: README 업데이트
refactor: 전처리 스크립트 리팩토링

# Issue / PR
[Feat] CSV→YOLO 라벨 변환
[Fix] 바운딩 박스 좌표 오류 수정
```
