"""
구글 시트 실험 기록 -> README 실험 그래프(Mermaid) 갱신

실행: uv run --env-file .env python src/plot_experiments.py
      (GitHub Actions 도 같은 파일을 실행한다. 이때 SHEET_URL 은 저장소 Secret)

결과: README.md 의 <!-- 실험그래프 시작 --> ~ <!-- 실험그래프 끝 -->
      사이를 새 그래프 + 표로 바꾼다

표준 라이브러리만 쓴다.
(Actions 에서 uv sync 로 torch 까지 설치하지 않아도 되게)
"""

import json
import os
import urllib.request
from pathlib import Path


# ===== 설정값 =====
README = Path(__file__).resolve().parent.parent / "README.md"
START = "<!-- 실험그래프 시작 -->"
END = "<!-- 실험그래프 끝 -->"
RECENT = 10


def fetch_rows():
    """구글 시트의 실험 기록을 전부 읽는다."""

    # 1. 주소
    url = os.environ.get("SHEET_URL")
    if not url:
        raise SystemExit(
            "SHEET_URL 환경변수가 없음 (.env 또는 저장소 Secret 확인)"
        )

    # 2. 읽기
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    # 3. 행 목록
    return data["rows"]


def recent_rows(rows):
    """그래프에 쓸 최근 실험만 고른다."""

    # 1. 값이 있는 행만
    valid = []

    for row in rows:
        if not row.get("name") or not row.get("mAP75-95"):
            continue

        valid.append(row)

    # 2. 최근 N개
    return valid[-RECENT:]


def make_chart(rows):
    """Mermaid 꺾은선 그래프 코드를 만든다."""

    # 1. x축 이름과 값
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
    """그래프 아래에 붙일 표를 만든다."""

    # 1. 제목
    lines = [
        "| 순서 | name | author | memo | val mAP75-95 | kaggle_score |",
        "|---|---|---|---|---|---|",
    ]

    # 2. 행마다
    for i, row in enumerate(rows):
        memo = str(row["memo"]).replace("|", "\\|").replace("\n", " ")

        lines.append(
            f"| {i + 1} | {row['name']} | {row['author']} | "
            f"{memo} | {row['mAP75-95']} | {row['kaggle_score']} |"
        )

    return "\n".join(lines)


def update_readme(block):
    """README 의 표시 영역 사이만 새 내용으로 바꾼다."""

    # 1. 읽기 + 표시 확인
    text = README.read_text(encoding="utf-8")

    if START not in text or END not in text:
        raise SystemExit(
            f"README 에 {START} / {END} 표시가 없음"
        )

    # 2. 앞 + 새 내용 + 뒤
    before = text.split(START)[0]
    after = text.split(END)[1]

    new_text = (
        before
        + START
        + "\n"
        + block
        + "\n"
        + END
        + after
    )

    # 3. 같으면 저장 안 함
    if new_text == text:
        return False

    README.write_text(new_text, encoding="utf-8")
    return True


def main():
    """전체 순서."""

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