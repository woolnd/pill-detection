"""구글 시트 실험 기록 -> README Kaggle 점수 순위 그래프(SVG) + 표 갱신

실행: uv run --env-file .env python -m src.analysis.plot_experiments
      (GitHub Actions 도 같은 파일을 실행한다. 이때 SHEET_URL 은 저장소 Secret)
결과: docs/images/experiments_best.svg    Kaggle 점수 상위 5개 (오른쪽으로 갈수록 높음, 1위 옆에 이름)
      README.md 의 <!-- 실험그래프 시작 --> ~ <!-- 실험그래프 끝 --> 사이를 그래프 + 순위표로 바꾼다

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
ROOT = Path(__file__).resolve().parent.parent.parent  # src/analysis/ 에서 세 단계 위 = 프로젝트 루트
README = ROOT / "README.md"
IMAGES_DIR = ROOT / "docs" / "images"  # 그래프 파일 저장 폴더
START = "<!-- 실험그래프 시작 -->"  # README 에서 자동으로 바꿀 영역의 시작 표시
END = "<!-- 실험그래프 끝 -->"  # 끝 표시
TOP = 5  # Kaggle 순위 몇 등까지 보여줄지
KST = timezone(timedelta(hours=9))  # 한국 시간 (시트 time 은 UTC 로 온다)

# ===== 그래프 모양 =====
WIDTH, HEIGHT = 760, 400  # 그림 전체 크기
LEFT, RIGHT, TOP_MARGIN, BOTTOM = 70, 50, 110, 80  # 그래프 영역 바깥 여백 (위는 1위 이름 자리까지)
INNER = 40  # 첫 점, 마지막 점을 그래프 영역 안쪽으로 들이는 거리
LINE_COLOR = "#3b8fd9"  # 선 색
AREA_COLOR = "#e9f1fa"  # 선 아래 음영 색
GOLD = "#b7791f"  # 1위 글자·테두리 색
GOLD_FILL = "#f6c343"  # 1위 점 색
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


def filter_valid_rows(rows):
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


# ---------- 2. README 에 넣을 내용 (그래프 + 표) ----------
def make_block(rows):
    """README 에 넣을 전체 내용(순위 그래프 + 순위표)을 만든다

    입력:
        rows (list): filter_valid_rows() 결과

    반환:
        str: 마크다운 글자

    동작:
        1. Kaggle 점수 상위 TOP 개를 고른다. 하나도 없으면 안내 문구만 넣는다.
        2. 그래프는 낮은 점수부터 그려서 오른쪽 끝이 1위가 되게 뒤집는다.
        3. 표는 1위부터 적는다.
    """
    lines = [f"### 🏆 Kaggle 점수 TOP {TOP}", ""]

    # 1. 상위 실험
    best = filter_best_rows(rows)
    if not best:
        lines.append("아직 Kaggle 점수가 입력된 실험이 없습니다.")
        return "\n".join(lines)

    # 2. 그래프 (낮은 점수 -> 1위)
    ranks = []
    names = []
    authors = []
    scores = []
    for i in range(len(best)):
        row = best[len(best) - 1 - i]  # 뒤에서부터 = 낮은 점수부터
        ranks.append(f"{len(best) - i}위")
        names.append(row["name"])
        authors.append(row["author"])
        scores.append(float(row["kaggle_score"]))

    subtitle = f"Kaggle Public Score · 상위 {len(best)}개 · 오른쪽이 1위"
    svg = make_svg("Kaggle 점수 순위", subtitle, ranks, names, authors, scores)
    path = save_svg("experiments_best.svg", svg)

    # 3. 표 (1위부터)
    lines += [f"![Kaggle 점수 순위]({path})", "", make_table(best)]
    return "\n".join(lines)


def filter_best_rows(rows):
    """Kaggle 점수가 있는 실험을 점수 높은 순으로 고른다

    입력:
        rows (list): filter_valid_rows() 결과

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


def make_svg(title, subtitle, ranks, labels, sublabels, values):
    """점을 선으로 이은 순위 그래프를 SVG 글자로 만든다 (마지막 점 = 1위)

    입력:
        title (str): 제목  예) "Kaggle 점수 순위"
        subtitle (str): 제목 아래 설명  예) "Kaggle Public Score · 상위 5개"
        ranks (list): x축 첫째 줄 (등수), 낮은 등수부터  예) ["2위", "1위"]
        labels (list): x축 둘째 줄 (실험 이름)  예) ["rh_base", "jw_ep100"]
        sublabels (list): x축 셋째 줄 (작성자)  예) ["김라희", "엄재웅"]
        values (list): 점수, 낮은 순  예) [0.2356, 0.3914]

    반환:
        str: <svg> ... </svg> 글자

    동작:
        1. 점마다 그림 위 좌표(x, y)를 계산한다.
           x = 왼쪽 여백부터 같은 간격 (점이 1개면 가운데)
           y = 점수가 높을수록 위 (SVG 는 y 가 아래로 커져서 뒤집는다)
        2. 배경, 제목, 가로 눈금선을 그린다.
        3. 선 아래 음영 -> 선 -> 네모 점 -> 점 위 점수 -> x축 등수·이름 순서로 그린다.
           (나중에 그린 것이 위에 보여서 점과 글자가 선에 가리지 않는다)
        4. 1위(마지막 점) 위에 순위표처럼 "1위 이름 · 작성자" 를 적는다.
    """
    plot_w = WIDTH - LEFT - RIGHT  # 그래프 영역 가로
    plot_h = HEIGHT - TOP_MARGIN - BOTTOM  # 그래프 영역 세로
    base_y = TOP_MARGIN + plot_h  # 그래프 영역 바닥 y
    low, high = compute_y_range(values)

    # 1. 점 좌표 (양 끝 점과 글자가 테두리에 붙지 않게 안쪽 여백 INNER 를 둔다)
    points = []
    for i in range(len(values)):
        if len(values) == 1:
            x = LEFT + plot_w / 2
        else:
            x = LEFT + INNER + (plot_w - 2 * INNER) * i / (len(values) - 1)
        y = base_y - plot_h * (values[i] - low) / (high - low)
        points.append({"x": round(x, 1), "y": round(y, 1)})

    # 2-a. 배경 + 제목
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="{FONT}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="12" fill="#ffffff" stroke="#e5e7eb"/>',
        f'<text x="{LEFT - 40}" y="38" font-size="20" font-weight="700" fill="#1f2937">{escape(title)}</text>',
        f'<text x="{LEFT - 40}" y="60" font-size="12" fill="#6b7280">{escape(subtitle)}</text>',
    ]

    # 2-b. 가로 눈금선 4칸 + 왼쪽 눈금 숫자
    for k in range(5):
        tick = low + (high - low) * k / 4
        ty = round(base_y - plot_h * k / 4, 1)
        parts.append(f'<line x1="{LEFT - 10}" y1="{ty}" x2="{WIDTH - RIGHT + 10}" y2="{ty}" stroke="#f1f3f5"/>')
        parts.append(
            f'<text x="{LEFT - 16}" y="{ty + 4}" font-size="11" fill="#9ca3af" text-anchor="end">{tick:.2f}</text>'
        )

    # 3-a. 선 아래 음영 (점이 2개 이상일 때)
    point_texts = []
    for point in points:
        point_texts.append(f"{point['x']},{point['y']}")
    line_points = " ".join(point_texts)  # 예) "110.0,250.3 330.0,180.1 ..."

    if len(points) > 1:
        first_x = points[0]["x"]
        last_x = points[-1]["x"]
        area = f"{first_x},{base_y} {line_points} {last_x},{base_y}"
        parts.append(f'<polygon points="{area}" fill="{AREA_COLOR}"/>')

        # 3-b. 선
        parts.append(
            f'<polyline points="{line_points}" fill="none" stroke="{LINE_COLOR}" '
            f'stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    # 바닥선
    parts.append(f'<line x1="{LEFT - 10}" y1="{base_y}" x2="{WIDTH - RIGHT + 10}" y2="{base_y}" stroke="#9ca3af"/>')

    for i in range(len(points)):
        x = points[i]["x"]
        y = points[i]["y"]

        # 3-c. 색 (마지막 점이 1위라서 금색, 나머지는 흰 칸 + 진한 테두리)
        if i == len(points) - 1:
            fill = GOLD_FILL
            stroke = GOLD
            rank_color = GOLD
        else:
            fill = "#ffffff"
            stroke = "#374151"
            rank_color = "#374151"
        parts.append(
            f'<rect x="{x - 6}" y="{y - 6}" width="12" height="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )

        # 3-d. 점 위 점수
        parts.append(
            f'<text x="{x}" y="{y - 14}" font-size="13" font-weight="700" fill="#111827" '
            f'text-anchor="middle">{values[i]:.4f}</text>'
        )

        # 3-e. x축 세 줄: 등수 / 이름 (18자까지, 전체 이름은 아래 표에 있음) / 작성자
        if len(labels[i]) <= 18:
            short = labels[i]
        else:
            short = labels[i][:17] + "…"
        parts.append(
            f'<text x="{x}" y="{base_y + 22}" font-size="13" font-weight="700" fill="{rank_color}" '
            f'text-anchor="middle">{escape(ranks[i])}</text>'
        )
        parts.append(
            f'<text x="{x}" y="{base_y + 40}" font-size="12" fill="#374151" text-anchor="middle">{escape(short)}</text>'
        )
        parts.append(
            f'<text x="{x}" y="{base_y + 56}" font-size="11" fill="#9ca3af" '
            f'text-anchor="middle">{escape(sublabels[i])}</text>'
        )

    # 4. 1위 이름 (점수 글자 위, 오른쪽 끝에 맞춰 그림 밖으로 안 나가게)
    x = points[-1]["x"]
    y = points[-1]["y"]
    parts.append(
        f'<text x="{x + INNER}" y="{y - 36}" font-size="14" fill="#111827" text-anchor="end">'
        f'<tspan font-weight="700" fill="{GOLD}">1위</tspan> {escape(labels[-1])} · {escape(sublabels[-1])}</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def compute_y_range(values):
    """y축 아래·위 끝을 정한다 (점수 차이가 작아도 선이 잘 보이게)

    입력:
        values (list): 그릴 점수들  예) [0.1, 0.2356, 0.3914]

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


def save_svg(filename, svg):
    """SVG 글자를 docs/images/ 에 저장한다

    입력:
        filename (str): 파일 이름  예) "experiments_best.svg"
        svg (str): make_svg() 결과

    반환:
        str: README 에 쓸 상대 경로  예) "docs/images/experiments_best.svg"

    동작:
        1. 폴더가 없으면 만든다.
        2. 파일로 저장한다.
    """
    # 1. 폴더
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # 2. 저장
    (IMAGES_DIR / filename).write_text(svg, encoding="utf-8")
    return f"docs/images/{filename}"


def make_table(rows):
    """그래프 아래에 붙일 순위표를 만든다 (정확한 숫자와 메모 확인용)

    입력:
        rows (list): filter_best_rows() 결과 (1위부터)

    반환:
        str: 마크다운 표

    동작:
        1. 제목 줄 2개를 만든다. (점수를 앞에, 긴 메모는 맨 뒤에)
        2. 행마다 한 줄씩 추가한다.
           메모에 | 가 있으면 표 칸이 깨져서 \\| 로, 줄바꿈은 공백으로 바꾼다.
    """
    # 1. 제목
    lines = [
        "| 순위 | kaggle_score | name | author | val mAP75-95 | 시간 | memo |",
        "|---|---|---|---|---|---|---|",
    ]

    # 2. 행마다
    for i, row in enumerate(rows):
        memo = str(row["memo"]).replace("|", "\\|").replace("\n", " ")
        when = row["kst"].strftime("%m-%d %H:%M")
        lines.append(
            f"| {i + 1} | {row['kaggle_score']} | {row['name']} | {row['author']} "
            f"| {row['mAP75-95']} | {when} | {memo} |"
        )
    return "\n".join(lines)


# ---------- 3. README 갱신 ----------
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


# ---------- 실행 ----------
def main():
    """전체 순서

    입력: 없음 (환경변수 SHEET_URL 을 쓴다)

    반환:
        없음. docs/images/ 그래프와 README.md 를 갱신한다.

    동작:
        1. 시트 읽기 -> 빈 행 빼고 한국 시간 붙이기
        2. 순위 그래프 저장 + README 에 넣을 내용 만들기
        3. README 갱신 후 결과 출력
    """
    # 1. 시트
    rows = filter_valid_rows(fetch_rows())
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
