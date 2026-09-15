"""구글 시트(Apps Script 웹앱)로 데이터 보내기

준비: .env 에 SHEET_URL=https://script.google.com/macros/s/.../exec, AUTHOR=이름 추가 (.env 는 커밋 금지)
사용: train.py (학습 끝나면 실험 기록 한 줄 추가), plot_experiments.py 는 같은 URL 로 읽기만 함
"""

import json
import os
import urllib.request

from dotenv import load_dotenv

load_dotenv()  # .env 의 SHEET_URL, Kaggle 인증값을 환경변수로 읽는다


def post_to_sheet(payload):
    """시트 웹앱에 JSON 을 보내고 응답을 받는다

    입력:
        payload (dict): 보낼 내용
                        예) {"action": "append", "row": {"name": "baseline", "mAP75-95": 0.7225, ...}}

    반환:
        dict: 시트 응답  예) {"ok": True, "row": 5}
        None: .env 에 SHEET_URL 이 없어서 보내지 않았을 때

    동작:
        1. 환경변수에서 SHEET_URL 을 읽는다. 없으면 건너뛴다.
        2. payload 를 JSON 글자로 바꾼다. (한글 그대로, Path 같은 값은 글자로)
        3. POST 로 보낸다.
           (Apps Script 는 결과를 다른 주소로 한 번 넘겨주는데(302), urllib 이 알아서 따라간다)
        4. 응답 JSON 을 dict 로 바꿔 반환한다.
    """

    # 1. 주소
    url = os.environ.get("SHEET_URL")
    if not url:
        print("SHEET_URL 이 .env 에 없어서 시트 전송을 건너뜀")
        return None

    # 2. JSON 글자 -> 바이트
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")

    # 3. POST
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        # 4. 응답
        return json.loads(response.read().decode("utf-8"))
