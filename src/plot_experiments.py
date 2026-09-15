"""구글 시트 실험 기록 -> README 실험 그래프(Mermaid) 갱신

실행: uv run --env-file .env python src/plot_experiments.py
      (GitHub Actions 도 같은 파일을 실행한다. 이때 SHEET_URL 은 저장소 Secret)
결과: README.md 의 <!-- 실험그래프 시작 --> ~ <!-- 실험그래프 끝 --> 사이를 새 그래프 + 표로 바꾼다

표준 라이브러리만 쓴다. (Actions 에서 uv sync 로 torch 까지 설치하지 않아도 되게)
"""

import json
import os
import urllib.request
from pathlib import Path

# ===== 설정값 =====
README = Path(__file__).resolve().parent.parent / "README.md"
START = "<!-- 실험그래프 시작 -->"  # README 에서 자동으로 바꿀 영역의 시작 표시
END = "<!-- 실험그래프 끝 -->"  # 끝 표시
RECENT = 10  # 최근 몇 개 실험을 그릴지


def fetch_rows():
    """구글 시트의 실험 기록을 전부 읽는다

    입력: 없음 (환경변수 SHEET_URL 을 쓴다)

    반환:
        list: 행마다 dict 1개 (시트 제목 -> 값), 시트에 쌓인 순서 = 시간순
              예) [{"time": "...", "author": "재웅", "name": "jw_base", "mAP75-95": 0.7225, ...}, ...]

    동작:
        1. SHEET_URL 이 없으면 멈춘다.
        2. GET 으로 요청하면 Code.gs 의 doGet 이 시트 내용을 JSON 으로 준다.
           (302 로 한 번 넘겨주는데 urllib 이 알아서 따라간다)
        3. rows 만 꺼내서 반환한다.
    """
    # 1. 주소
    url = os.environ.get("SHEET_URL")
    if not url:
        raise SystemExit("SHEET_URL 환경변수가 없음 (.env 또는 저장소 Secret 확인)")

    # 2. 읽기
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    # 3. 행 목록
    return data["rows"]


def recent_rows(rows):
    """그래프에 쓸 최근 실험만 고른다

    입력:
        rows (list): fetch_rows() 결과

    반환:
        list: 최근 RECENT 개 행 (오래된 것 -> 최근 순서)

    동작:
        1. name 이나 mAP75-95 가 빈 행(사람이 만든 빈 줄 등)은 뺀다.
        2. 뒤에서 RECENT 개만 자른다. (시트는 기록된 순서대로 쌓여서 뒤쪽이 최근)
    """

    # 1. 값이 있는 행만
    valid = []

    for row in rows:
        if row.get("name") and row.get("mAP75-95"):
            valid.append(row)

    # 2. 최근 N개
    return valid[-RECENT:]



def make_chart(rows):
    """Mermaid 꺾은선 그래프 코드를 만든다 (GitHub 가 README 에서 그래프로 그려줌)

    입력:
        rows (list): recent_rows() 결과

    반환:
        str: ```mermaid 로 시작하는 여러 줄 글자

    동작:
        1. 실험 이름은 x축, mAP75-95 는 선의 값으로 모은다.
        2. xychart-beta 형식으로 줄을 만들어 이어 붙인다.
    """

    # 1. x축 이름 ("jw_base" 처럼 따옴표로 감쌈), 값
    names = []
    scores = []
    for row in rows:
        names.append('"' + str(row["name"]) + '"')
        scores.append(str(round(float(row["mAP75-95"]), 4)))

    # 2. Mermaid 코드
    lines = [
        "```mermaid",
        "xychart-beta",
        '    title "최근 실험 ' + str(len(rows)) + '개 val mAP75-95 (시간순)"',
        "    x-axis [" + ", ".join(names) + "]",
        '    y-axis "mAP75-95" 0 --> 1',
        "    line [" + ", ".join(scores) + "]",
        "```",
    ]
    return "\n".join(lines)


def make_table(rows):
    """그래프 아래에 붙일 표를 만든다 (정확한 숫자와 메모 확인용)

    입력:
        rows (list): recent_rows() 결과

    반환:
        str: 마크다운 표

    동작:
        1. 제목 줄 2개를 만든다.
        2. 행마다 한 줄씩 추가한다.
           메모에 | 가 있으면 표 칸이 깨져서 \\| 로, 줄바꿈은 공백으로 바꾼다.
    """

    # 1. 제목
    lines = [
        "| 순서 | name | author | memo | val mAP75-95 | kaggle_score |",
        "|---|---|---|---|---|---|",
    ]

    # 2. 행마다
    for i, row in enumerate(rows):
        memo = str(row["memo"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {i + 1} | {row['name']} | {row['author']} | {memo} | {row['mAP75-95']} | {row['kaggle_score']} |"
        )
    return "\n".join(lines)


def update_readme(block):
    """README 의 표시 영역 사이만 새 내용으로 바꾼다

    입력:
        block (str): 그래프 + 표

    반환:
        bool: 바뀌었으면 True, 이전과 같으면 False

    동작:
        1. README 를 읽고 시작/끝 표시가 있는지 확인한다.
        2. 시작 표시 앞부분 + 새 내용 + 끝 표시 뒷부분으로 새 글을 만든다.
        3. 이전과 같으면 저장하지 않는다. (Actions 가 불필요한 PR 을 안 만들게)
    """
    # 1. 읽기 + 표시 확인
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"README 에 {START} / {END} 표시가 없음")

    # 2. 앞 + 새 내용 + 뒤
    before = text.split(START)[0]
    after = text.split(END)[1]
    new_text = before + START + "\n" + block + "\n" + END + after

    # 3. 같으면 저장 안 함
    if new_text == text:
        return False
    README.write_text(new_text, encoding="utf-8")
    return True

def main():
    """전체 순서

    동작:
        1. 시트 읽기 -> 최근 실험 고르기
        2. 그래프 + 표 만들기
        3. README 갱신 후 결과 출력
    """
    # 1. 최근 실험
    rows = recent_rows(fetch_rows())
    if not rows:
        print("시트에 기록된 실험이 없음")
        return

    # 2. 그래프 + 표
    block = make_chart(rows) + "\n\n" + make_table(rows)

    # 3. README
    if update_readme(block):
        print(f"README 갱신: 최근 실험 {len(rows)}개")
    else:
        print("README 변경 없음")


if __name__ == "__main__":
    main()