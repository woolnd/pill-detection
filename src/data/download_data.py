"""Kaggle 대회 데이터 → data/raw/

실행: uv run --env-file .env python -m src.data.download_data
"""

from pathlib import Path

import kagglehub
from dotenv import load_dotenv

from src.config import RAW_DIR

COMPETITION = "ai14-level-project"


def sort_by_count(counts):
    """개수 dict 를 많은 순서의 (값, 개수) 리스트로 바꾼다

    입력:
        counts (dict): 값 -> 개수  예) {".png": 1074, ".json": 763}

    반환:
        list: 많은 순서  예) [(".png", 1074), (".json", 763)]

    동작:
        1. 개수 기준으로 키를 많은 순서로 정렬한다. (개수가 같으면 먼저 나온 순서 유지)
        2. (값, 개수) 짝으로 리스트에 넣는다.
    """
    # 1. 많은 순 정렬
    keys = sorted(counts, key=counts.get, reverse=True)

    # 2. 짝 만들기
    pairs = []
    for key in keys:
        pairs.append((key, counts[key]))
    return pairs


def main():
    """대회 데이터를 받아서 파일 구성을 출력한다

    입력: 없음 (.env 의 KAGGLE_USERNAME, KAGGLE_KEY 를 쓴다)

    반환:
        없음. data/raw/ 에 데이터를 받고 요약을 출력한다.

    동작:
        1. .env 의 Kaggle 인증값을 환경변수로 읽는다.
        2. data/raw/ 폴더를 만들고 대회 데이터를 받는다.
        3. 전체 파일 수와 확장자별 개수를 출력한다.
        4. 파일이 많은 폴더 5개를 출력한다.
    """
    # 1. 인증값
    load_dotenv()

    # 2. 다운로드
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(kagglehub.competition_download(COMPETITION, output_dir=str(RAW_DIR)))
    print("경로:", path)

    # 3. 파일 수 (폴더는 빼고 파일만), 확장자별 개수
    files = []
    for p in path.rglob("*"):
        if p.is_file():
            files.append(p)

    suffix_counts = {}
    for p in files:
        suffix_counts[p.suffix] = suffix_counts.get(p.suffix, 0) + 1
    print(f"총 {len(files)}개 / {sort_by_count(suffix_counts)}")

    # 4. 폴더별 파일 수, 많은 순 5개
    folder_counts = {}
    for p in files:
        folder_counts[p.parent.name] = folder_counts.get(p.parent.name, 0) + 1
    for folder, n in sort_by_count(folder_counts)[:5]:
        print(f"  {folder}  {n}")


if __name__ == "__main__":
    main()
