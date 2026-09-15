"""구글 시트 실험 기록 -> README 실험 그래프(SVG) + 표 갱신

실행: uv run --env-file .env python src/plot_experiments.py
      (GitHub Actions 도 같은 파일을 실행한다. 이때 SHEET_URL 은 저장소 Secret)
결과: docs/images/experiments_today.svg   가장 최근 실험한 날의 val mAP75-95 (시간순)
      docs/images/experiments_best.svg    Kaggle 점수 높은 순 상위 실험
      README.md 의 <!-- 실험그래프 시작 --> ~ <!-- 실험그래프 끝 --> 사이를 그래프 + 표로 바꾼다

표준 라이브러리만 쓴다. (Actions 에서 uv sync 로 torch 까지 설치하지 않아도 되게)
그래프는 SVG 글자로 직접 그린다. (matplotlib 없이도 점 + 선 + 값 표시 그래프를 만들 수 있다)
"""

import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.sax.saxutils import escape

# ===== 설정값 =====
ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
IMAGES = ROOT / "docs" / "images"  # 그래프 파일 저장 폴더
START = "<!-- 실험그래프 시작 -->"  # README 에서 자동으로 바꿀 영역의 시작 표시
END = "<!-- 실험그래프 끝 -->"  # 끝 표시
TOP = 10  # Kaggle 순위 그래프에 몇 개까지 그릴지
KST = timezone(timedelta(hours=9))  # 한국 시간 (시트 time 은 UTC 로 온다)

# ===== 그래프 모양 =====
WIDTH, HEIGHT = 760, 380  # 그림 전체 크기
LEFT, RIGHT, TOP_MARGIN, BOTTOM = 70, 50, 80, 80  # 그래프 영역 바깥 여백
INNER = 40  # 첫 점, 마지막 점을 그래프 영역 안쪽으로 들이는 거리
LINE_COLOR = "#3b8fd9"  # 선 색
AREA_COLOR = "#e9f1fa"  # 선 아래 음영 색
FONT = "Pretendard, 'Apple SD Gothic Neo', 'Malgun Gothic', 'Noto Sans KR', sans-serif"


# ---------- 1. 시트 읽기 ----------
def fetch_rows():
    """구글 시트의 실험 기록을 전부 읽는다

    입력: 없음 (환경변수 SHEET_URL 을 쓴다)

    반환:
        list: 행마다 dict 1개 (시트 제목 -> 값), 시트에 쌓인 순서 = 시간순
              예) [{"time": "2026-09-15T01:17:00.000Z", "author": "재웅", "name": "jw_base",
                    "mAP75-95": 0.7225, "kaggle_score": "", ...}, ...]

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


def to_kst(value):
    """시트의 time 값을 한국 시간 datetime 으로 바꾼다

    입력:
        value (str): 시트 time 값
                     예) "2026-09-15T01:17:00.000Z" (시트가 날짜로 바꾼 경우, UTC)
                         "09-15 10:17"              (글자 그대로 남은 경우, 한국 시간)

    반환:
        datetime: 한국 시간  예) 2026-09-15 10:17

    동작:
        1. "T" 가 있으면 UTC 날짜 글자라서 읽은 뒤 한국 시간(+9시간)으로 바꾼다.
        2. 아니면 "월-일 시:분" 글자라서 올해 연도를 붙여 읽는다.
    """
    text = str(value)

    # 1. UTC 날짜 글자 (끝의 Z 는 UTC 라는 뜻, +00:00 으로 바꿔야 읽힌다)
    if "T" in text:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(KST)

    # 2. "09-15 10:17"
    year = datetime.now(KST).year
    return datetime.strptime(f"{year}-{text}", "%Y-%m-%d %H:%M").replace(tzinfo=KST)


# ---------- 2. 그래프에 쓸 행 고르기 ----------
def valid_rows(rows):
    """이름과 val 점수가 있는 행만 남기고, 한국 시간을 붙인다

    입력:
        rows (list): fetch_rows() 결과

    반환:
        list: 같은 dict 에 "kst" (한국 시간 datetime) 키를 추가한 리스트

    동작:
        1. name, mAP75-95, time 중 하나라도 빈 행은 뺀다. (사람이 만든 빈 줄 등)
        2. 남은 행에 한국 시간을 붙인다.
    """
    result = []
    for row in rows:
        # 1. 빈 행 빼기
        if row["name"] == "" or row["mAP75-95"] == "" or row["time"] == "":
            continue

        # 2. 한국 시간
        row["kst"] = to_kst(row["time"])
        result.append(row)
    return result


def latest_day_rows(rows):
    """가장 최근 실험한 날의 실험만 시간순으로 고른다

    입력:
        rows (list): valid_rows() 결과

    반환:
        day (date): 가장 최근 실험 날짜  예) 2026-09-15
        picked (list): 그날 실험 행 (시간순)

    동작:
        1. 모든 행의 날짜 중 가장 늦은 날을 찾는다.
           (오늘 실험이 있으면 오늘, 없으면 마지막으로 실험한 날. Kaggle 점수만 입력한 날에도 그래프가 비지 않게)
        2. 그날 행만 모아 시간순으로 정렬한다.
    """
    # 1. 가장 최근 날짜
    day = max(row["kst"].date() for row in rows)

    # 2. 그날 행만, 시간순
    picked = []
    for row in rows:
        if row["kst"].date() == day:
            picked.append(row)
    picked = sorted(picked, key=lambda row: row["kst"])
    return day, picked


def best_rows(rows):
    """Kaggle 점수가 있는 실험을 점수 높은 순으로 고른다

    입력:
        rows (list): valid_rows() 결과

    반환:
        list: kaggle_score 높은 순 상위 TOP 개 (없으면 빈 리스트)

    동작:
        1. kaggle_score 가 빈 행은 뺀다.
        2. 점수 높은 순으로 정렬해서 앞에서 TOP 개만 자른다.
    """
    # 1. 점수 있는 행만
    scored = []
    for row in rows:
        if row["kaggle_score"] != "":
            scored.append(row)

    # 2. 높은 순 TOP 개
    scored = sorted(scored, key=lambda row: float(row["kaggle_score"]), reverse=True)
    return scored[:TOP]


# ---------- 3. SVG 그래프 ----------
def y_range(values):
    """y축 아래·위 끝을 정한다 (점수 차이가 작아도 선이 잘 보이게)

    입력:
        values (list): 그릴 점수들  예) [0.0, 0.7939, 0.1101]

    반환:
        low (float), high (float): y축 범위 (0~1 안)

    동작:
        1. 가장 작은 값과 큰 값에서 여유(차이의 15%, 최소 0.02)를 더한다.
        2. 점수는 0~1 이라서 그 밖으로 나가지 않게 자른다.
    """
    # 1. 여유 더하기
    low = min(values)
    high = max(values)
    pad = max((high - low) * 0.15, 0.02)

    # 2. 0~1 안으로
    return max(0.0, low - pad), min(1.0, high + pad)


def make_svg(title, subtitle, labels, sublabels, values):
    """점을 선으로 이은 꺾은선 그래프를 SVG 글자로 만든다

    입력:
        title (str): 제목  예) "9월 15일 실험"
        subtitle (str): 제목 아래 설명  예) "val mAP75-95 · 시간순"
        labels (list): x축 첫째 줄 (실험 이름)  예) ["jw_base", "rh_ep100"]
        sublabels (list): x축 둘째 줄 (작성자 등)  예) ["엄재웅", "김라희"]
        values (list): 점수  예) [0.7225, 0.7939]

    반환:
        str: <svg> ... </svg> 글자

    동작:
        1. 점마다 그림 위 좌표(x, y)를 계산한다.
           x = 왼쪽 여백부터 같은 간격 (점이 1개면 가운데)
           y = 점수가 높을수록 위 (SVG 는 y 가 아래로 커져서 뒤집는다)
        2. 배경, 제목, 가로 눈금선을 그린다.
        3. 선 아래 음영 -> 선 -> 네모 점 -> 점 위 점수 -> x축 이름 순서로 그린다.
           (나중에 그린 것이 위에 보여서 점과 글자가 선에 가리지 않는다)
    """
    plot_w = WIDTH - LEFT - RIGHT  # 그래프 영역 가로
    plot_h = HEIGHT - TOP_MARGIN - BOTTOM  # 그래프 영역 세로
    base_y = TOP_MARGIN + plot_h  # 그래프 영역 바닥 y
    low, high = y_range(values)

    # 1. 점 좌표 (양 끝 점과 글자가 테두리에 붙지 않게 안쪽 여백 INNER 를 둔다)
    points = []
    for i, value in enumerate(values):
        if len(values) == 1:
            x = LEFT + plot_w / 2
        else:
            x = LEFT + INNER + (plot_w - 2 * INNER) * i / (len(values) - 1)
        y = base_y - plot_h * (value - low) / (high - low)
        points.append((round(x, 1), round(y, 1)))

    # 2. 배경 + 제목
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" font-family="{FONT}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="12" fill="#ffffff" stroke="#e5e7eb"/>',
        f'<text x="{LEFT - 40}" y="38" font-size="20" font-weight="700" fill="#1f2937">{escape(title)}</text>',
        f'<text x="{LEFT - 40}" y="60" font-size="12" fill="#6b7280">{escape(subtitle)}</text>',
    ]

    # 2. 가로 눈금선 4칸 + 왼쪽 눈금 숫자
    for k in range(5):
        tick = low + (high - low) * k / 4
        ty = round(base_y - plot_h * k / 4, 1)
        parts.append(f'<line x1="{LEFT - 10}" y1="{ty}" x2="{WIDTH - RIGHT + 10}" y2="{ty}" stroke="#f1f3f5"/>')
        parts.append(f'<text x="{LEFT - 16}" y="{ty + 4}" font-size="11" fill="#9ca3af" text-anchor="end">{tick:.2f}</text>')

    # 3-a. 선 아래 음영 (점이 2개 이상일 때)
    line_points = " ".join(f"{x},{y}" for x, y in points)
    if len(points) > 1:
        area = f"{points[0][0]},{base_y} {line_points} {points[-1][0]},{base_y}"
        parts.append(f'<polygon points="{area}" fill="{AREA_COLOR}"/>')
        # 3-b. 선
        parts.append(
            f'<polyline points="{line_points}" fill="none" stroke="{LINE_COLOR}" stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    # 바닥선
    parts.append(f'<line x1="{LEFT - 10}" y1="{base_y}" x2="{WIDTH - RIGHT + 10}" y2="{base_y}" stroke="#9ca3af"/>')

    for (x, y), value, label, sublabel in zip(points, values, labels, sublabels):
        # 3-c. 네모 점 (흰 칸 + 진한 테두리)
        parts.append(f'<rect x="{x - 6}" y="{y - 6}" width="12" height="12" fill="#ffffff" stroke="#374151" stroke-width="2"/>')
        # 3-d. 점 위 점수
        parts.append(f'<text x="{x}" y="{y - 14}" font-size="13" font-weight="700" fill="#111827" text-anchor="middle">{value:.4f}</text>')
        # 3-e. x축 이름 두 줄 (점 10개일 때 옆 이름과 안 겹치게 11자까지, 전체 이름은 아래 표에 있음)
        short = label if len(label) <= 11 else label[:10] + "…"
        parts.append(f'<text x="{x}" y="{base_y + 24}" font-size="12" fill="#374151" text-anchor="middle">{escape(short)}</text>')
        parts.append(f'<text x="{x}" y="{base_y + 42}" font-size="11" fill="#9ca3af" text-anchor="middle">{escape(sublabel)}</text>')

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def save_svg(filename, svg):
    """SVG 글자를 docs/images/ 에 저장한다

    입력:
        filename (str): 파일 이름  예) "experiments_today.svg"
        svg (str): make_svg() 결과

    반환:
        str: README 에 쓸 상대 경로  예) "docs/images/experiments_today.svg"

    동작:
        1. 폴더가 없으면 만든다.
        2. 파일로 저장한다.
    """
    # 1. 폴더
    IMAGES.mkdir(parents=True, exist_ok=True)

    # 2. 저장
    (IMAGES / filename).write_text(svg, encoding="utf-8")
    return f"docs/images/{filename}"


# ---------- 4. 표 ----------
def make_table(rows, first_column):
    """그래프 아래에 붙일 표를 만든다 (정확한 숫자와 메모 확인용)

    입력:
        rows (list): 그래프에 쓴 행 (그래프와 같은 순서)
        first_column (str): 첫 열 이름  예) "순서", "순위"

    반환:
        str: 마크다운 표

    동작:
        1. 제목 줄 2개를 만든다.
        2. 행마다 한 줄씩 추가한다.
           메모에 | 가 있으면 표 칸이 깨져서 \\| 로, 줄바꿈은 공백으로 바꾼다.
    """
    # 1. 제목
    lines = [
        f"| {first_column} | 시간 | name | author | memo | val mAP75-95 | kaggle_score |",
        "|---|---|---|---|---|---|---|",
    ]

    # 2. 행마다
    for i, row in enumerate(rows):
        memo = str(row["memo"]).replace("|", "\\|").replace("\n", " ")
        when = row["kst"].strftime("%m-%d %H:%M")
        lines.append(
            f"| {i + 1} | {when} | {row['name']} | {row['author']} | {memo} | {row['mAP75-95']} | {row['kaggle_score']} |"
        )
    return "\n".join(lines)


# ---------- 5. README ----------
def make_block(rows):
    """README 에 넣을 전체 내용(그래프 2개 + 표 2개)을 만든다

    입력:
        rows (list): valid_rows() 결과

    반환:
        str: 마크다운 글자

    동작:
        1. 가장 최근 실험한 날 그래프 + 표
        2. Kaggle 점수 순위 그래프 + 표 (점수가 하나도 없으면 안내 문구만)
    """
    # 1. 최근 실험한 날
    day, today = latest_day_rows(rows)
    title = f"{day.month}월 {day.day}일 실험"
    path = save_svg(
        "experiments_today.svg",
        make_svg(
            title,
            f"val mAP75-95 · 시간순 · {len(today)}개",
            [row["name"] for row in today],
            [row["author"] for row in today],
            [float(row["mAP75-95"]) for row in today],
        ),
    )
    lines = [f"### 📅 {title}", "", f"![{title}]({path})", "", make_table(today, "순서"), ""]

    # 2. Kaggle 점수 순위
    best = best_rows(rows)
    lines.append("### 🏆 Kaggle 점수 순위")
    lines.append("")
    if not best:
        lines.append("아직 Kaggle 점수가 입력된 실험이 없습니다.")
    else:
        path = save_svg(
            "experiments_best.svg",
            make_svg(
                "Kaggle 점수 순위",
                f"Kaggle Public Score · 높은 순 · 상위 {len(best)}개",
                [row["name"] for row in best],
                [row["author"] for row in best],
                [float(row["kaggle_score"]) for row in best],
            ),
        )
        lines += [f"![Kaggle 점수 순위]({path})", "", make_table(best, "순위")]
    return "\n".join(lines)


def update_readme(block):
    """README 의 표시 영역 사이만 새 내용으로 바꾼다

    입력:
        block (str): make_block() 결과

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
        1. 시트 읽기 -> 빈 행 빼고 한국 시간 붙이기
        2. 그래프 2개 저장 + README 에 넣을 내용 만들기
        3. README 갱신 후 결과 출력
    """
    # 1. 시트
    rows = valid_rows(fetch_rows())
    if not rows:
        print("시트에 기록된 실험이 없음")
        return

    # 2. 그래프 + 표
    block = make_block(rows)

    # 3. README
    if update_readme(block):
        print("README 갱신")
    else:
        print("README 변경 없음 (그래프 파일은 git 이 변경 여부를 판단)")


if __name__ == "__main__":
    main()
