"""AI Hub 조합 데이터 다운로드 -> data/aihub/labels/TL_n, data/aihub/images/TS_n

실행: uv run --env-file .env python -m src.data.download_aihub
      (.env 에 AI_HUB_KEY 필요. aihub.or.kr 에서 데이터 이용 신청 · API 키 발급 후)
결과: data/aihub/labels/TL_1, TL_3~8/   라벨 JSON (약 55MB)
      data/aihub/images/TS_1, TS_3~8/   이미지 png (약 21GB)

⛔ TL_2_조합(66066), TS_2_조합(66155) 은 대회 train/test 원본이라 받지 않는다. (대회 규칙)

공식 aihubshell 을 쓰지 않는 이유:
    aihubshell v0.6 은 macOS 기본 bash(3.2) 에서 한글 파일명 조각을 합치지 못해
    빈 zip 을 만들고 조각 파일을 지운다. 그래서 같은 API 를 직접 호출하고 병합도 직접 한다.

파일 하나씩 받고 -> 조각 합치고 -> 압축 풀고 -> zip 을 지우므로 디스크 여유는 약 25GB 면 된다.
중간에 멈춰도 이미 풀린 폴더는 건너뛰고 이어서 받는다.
"""

import os
import re
import shutil
import sys
import tarfile
import urllib.request
import zipfile

from src.config import AIHUB_DIR

# ===== 설정값 =====
DOWNLOAD_URL = "https://api.aihub.or.kr/down/0.6/576.do?fileSn={filekey}"
PARTS = [1, 3, 4, 5, 6, 7, 8]  # 받을 조합 파일 번호 (2 는 대회 규칙상 금지)
LABEL_KEYS = {1: 66065, 3: 66067, 4: 66068, 5: 66069, 6: 66070, 7: 66071, 8: 66072}  # TL_n_조합.zip
IMAGE_KEYS = {1: 66154, 3: 66156, 4: 66157, 5: 66158, 6: 66159, 7: 66160, 8: 66161}  # TS_n_조합.zip
TAR_PATH = AIHUB_DIR / "download.tar"  # 받는 중인 파일 (풀면 지운다)


def download(filekey, api_key):
    """filekey 파일 하나를 download.tar 로 받는다

    입력:
        filekey (int): AI Hub 파일 번호  예) 66065
        api_key (str): AI Hub API 키

    반환:
        없음. data/aihub/download.tar 를 만든다.

    동작:
        1. apikey 를 헤더에 넣어 요청한다. (아이디·비밀번호가 아니라 키로 인증한다)
        2. 응답을 조금씩 읽어 파일로 저장한다. (3GB 를 메모리에 올리지 않게)
    """
    # 1. 요청
    request = urllib.request.Request(DOWNLOAD_URL.format(filekey=filekey), headers={"apikey": api_key})

    # 2. 조금씩 저장
    with urllib.request.urlopen(request) as response, open(TAR_PATH, "wb") as out:
        shutil.copyfileobj(response, out, 16 * 1024 * 1024)  # 16MB 씩


def merge_parts():
    """download.tar 를 풀고 조각 파일(*.partN)을 번호 순서대로 합친다

    입력: 없음 (data/aihub/download.tar 를 읽는다)

    반환:
        list: 만들어진 zip 경로 리스트

    동작:
        1. tar 를 푼다. (안에 zip 이 통째로 또는 *.part0, *.part1 ... 조각으로 들어 있다)
        2. 조각을 이름별로 모아 번호 순서대로 이어 붙인다.
        3. zip 이 깨지지 않았는지 검사하고, 통과하면 조각 파일을 지운다.
        4. download.tar 를 지우고 zip 경로를 돌려준다.
    """
    # 1. tar 풀기
    with tarfile.open(TAR_PATH) as tar:
        tar.extractall(AIHUB_DIR, filter="data")

    # 2. 조각 모으기 (이름 -> [(번호, 경로), ...])
    groups = {}
    for part in AIHUB_DIR.rglob("*.part*"):
        found = re.match(r"(.*)\.part(\d+)$", part.name)
        if found:
            target = part.parent / found.group(1)
            if target not in groups:
                groups[target] = []
            groups[target].append((int(found.group(2)), part))

    zips = []
    for target in sorted(groups):
        parts = sorted(groups[target])

        # 2. 번호 순서대로 이어 붙이기
        with open(target, "wb") as out:
            for number, part in parts:
                with open(part, "rb") as src:
                    shutil.copyfileobj(src, out, 16 * 1024 * 1024)

        # 3. 손상 검사 후 조각 삭제
        broken = zipfile.ZipFile(target).testzip()
        if broken is not None:
            sys.exit(f"zip 손상: {target} ({broken}). 조각 파일은 지우지 않았으니 다시 실행하세요")
        for number, part in parts:
            part.unlink()
        zips.append(target)

    # 4. 정리
    for path in AIHUB_DIR.rglob("*.zip"):
        if path not in zips:
            zips.append(path)  # 조각이 아니라 통째로 들어 있던 zip
    TAR_PATH.unlink()
    return zips


def unzip(zip_path, out_dir):
    """zip 을 out_dir 에 풀고 zip 을 지운다

    입력:
        zip_path (Path): zip 경로
        out_dir (Path): 풀 위치  예) data/aihub/labels/TL_1

    반환:
        int: 풀린 파일 수

    동작:
        1. zip 을 out_dir 에 푼다.
        2. zip 파일을 지운다. (디스크를 두 배로 쓰지 않게)
    """
    # 1. 풀기
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
        n_files = len(z.namelist())

    # 2. zip 삭제
    zip_path.unlink()
    return n_files


def get_one(number, filekey, out_dir):
    """파일 하나를 받아서 풀기까지 한다

    입력:
        number (int): 조합 파일 번호  예) 3
        filekey (int): AI Hub 파일 번호  예) 66156
        out_dir (Path): 풀 위치  예) data/aihub/images/TS_3

    반환:
        없음. out_dir 을 만든다.

    동작:
        0. out_dir 이 이미 있으면 건너뛴다. (중간에 멈췄을 때 이어서 받기)
        1. 받는다 -> 2. 조각 합치기 -> 3. 압축 풀기
    """
    # 0. 이미 있음
    if out_dir.exists():
        print(f"  {out_dir.name} 이미 있음, 건너뜀")
        return

    print(f"  {out_dir.name} 받는 중 (filekey {filekey})")
    download(filekey, os.environ["AI_HUB_KEY"])

    zips = merge_parts()
    for zip_path in zips:
        n_files = unzip(zip_path, out_dir)
        print(f"  {out_dir.name} 완료: 파일 {n_files}개")


def main():
    """전체 순서

    입력: 없음 (.env 의 AI_HUB_KEY 를 쓴다)

    반환:
        없음. data/aihub/labels/, data/aihub/images/ 를 만든다.

    동작:
        1. API 키가 있는지 확인한다.
        2. 라벨(TL) 7개를 받는다. (약 55MB)
        3. 이미지(TS) 7개를 받는다. (각 약 3GB)
        4. 다음에 할 일을 출력한다.
    """
    # 1. 키 확인
    if not os.environ.get("AI_HUB_KEY"):
        raise SystemExit(
            "AI_HUB_KEY 가 없습니다. .env 에 AI_HUB_KEY=... 를 넣고\n"
            "  uv run --env-file .env python -m src.data.download_aihub\n"
            "로 실행하세요. (키는 aihub.or.kr 에서 데이터 이용 신청 후 발급)"
        )
    AIHUB_DIR.mkdir(parents=True, exist_ok=True)

    # 2. 라벨
    print("[1] 라벨 (TL) 받기")
    for number in PARTS:
        get_one(number, LABEL_KEYS[number], AIHUB_DIR / "labels" / f"TL_{number}")

    # 3. 이미지
    print("[2] 이미지 (TS) 받기 - 파일당 약 3GB")
    for number in PARTS:
        get_one(number, IMAGE_KEYS[number], AIHUB_DIR / "images" / f"TS_{number}")

    # 4. 다음 순서
    print(f"\n완료: {AIHUB_DIR}")
    print("다음: src/config.py 의 USE_AIHUB = True 확인 후")
    print("  uv run python -m src.data.make_yolo")
    print("  uv run python -m src.data.check_yolo")


if __name__ == "__main__":
    main()
