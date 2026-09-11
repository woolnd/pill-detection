# """변환 결과 확인: 원본과 숫자 비교 + 그림

# 실행: uv run python src/check_yolo.py  (make_yolo.py 먼저 실행)
# """
# import random

# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

# from annotations import load_annotations
# from make_yolo import IMG_H, IMG_W, OUT

# def yolo_to_pixel(line):
#     """YOLO 라벨 한 줄을 COCO 픽셀 좌표로 되돌린다 (to_yolo_line 의 반대)

#     입력:
#         line (str): "클래스번호 중심x 중심y 너비 높이" (0~1 비율)
#                     예) "0 0.265369 0.264844 0.188525 0.142187"

#     반환:
#         tuple: (클래스번호 int, 왼쪽 위 x, 왼쪽 위 y, 너비, 높이)  좌표는 float 픽셀
#                예) (0, 167.0, 248.0, 184.0, 182.0)

#     동작:
#         1. 공백으로 잘라 5개 값을 꺼낸다.
#         2. 비율에 이미지 크기를 곱해서 픽셀로 만든다.
#         3. 왼쪽 위 = 중심 - 크기의 절반
#     """
#     # 1. 5개 값 꺼내기 (전부 문자열이라 float/int 로 바꿔 쓴다)
#     idx, cx, cy, w, h = 

#     # 2. 비율 -> 픽셀
#     w = 
#     h = 

#     # 3. 중심 -> 왼쪽 위
#     x = 
#     y = 
#     return int(idx), x, y, w, h


# def read_label(txt_path):
#     """라벨 txt 파일 하나를 읽어서 박스 리스트로 만든다

#     입력:
#         txt_path (Path): 라벨 파일 경로

#     반환:
#         list: yolo_to_pixel() 결과 튜플의 리스트. 알약 수만큼 들어 있다.
#               예) [(0, 167.0, 248.0, 184.0, 182.0), (15, ...), ...]

#     동작:
#         1. 파일 전체를 읽고 끝의 줄바꿈을 지운다(strip).
#         2. 줄 단위로 자른다.
#         3. 줄마다 yolo_to_pixel() 로 바꾼다.
#     """
#     lines = 
#     return 


# def compare_with_original():
#     """모든 라벨을 픽셀로 되돌려서 원본 JSON 좌표와 같은지 확인한다

#     입력: 없음 (data/yolo/labels 와 원본 JSON 을 읽는다)

#     반환:
#         없음. 결과를 출력한다.
#             "라벨 231개 확인 / 불일치 박스 0개"  <- 0개여야 정상

#     동작:
#         1. 원본 JSON 을 load_annotations() 로 읽는다.
#         2. labels/train, labels/val 의 txt 를 전부 찾는다.
#         3. txt 마다
#            a. 되돌린 좌표를 반올림해서 정렬한다.
#            b. 같은 이미지의 원본 좌표도 정렬한다.
#               (txt 줄 순서와 원본 박스 순서가 같다는 보장이 없어서 정렬 후 비교)
#            c. 박스끼리 x, y, w, h 가 1px 넘게 차이 나면 불일치로 센다.
#               (0.5 같은 반올림 오차가 있어서 1px 까지는 허용)
#     """
#     # 1. 원본
#     ann = load_annotations()
#     bad = 0

#     # 2. 모든 라벨 파일 (* 자리에 train, val 이 들어간다)
#     txts = list(OUT.glob("labels/*/*.txt"))

#     for txt in txts:
#         # 3-a. 되돌린 좌표 (클래스 번호 _ 는 비교 안 함)
#         got = sorted((round(x), round(y), round(w), round(h)) for _, x, y, w, h in read_label(txt))

#         # 3-b. 원본 좌표. txt.stem 은 확장자 뺀  키로 쓴다
#         want = sorted((b["x"], b["y"], b["w"], b["h"]) for b in ann[txt.stem + ".png"])

#         # 3-c. 박스끼리 비교
#         for g, e in zip(got, want):
#             if any(abs(a - b) > 1 for a, b in zip(g, e)):
#                 bad += 1
#                 print("  불일치:", txt.name, g, e)

#     print(f"라벨 {len(txts)}개 확인 / 불일치 박스 {bad}개")


# def show(split="train", n=6):
#     """YOLO 라벨을 이미지 위에 그려서 눈으로 확인한다

#     입력:
#         split (str): "train" 또는 "val". 기본값 "train"
#         n (int): 그릴 이미지 수. 2x3 칸이라 6 이하. 기본값 6

#     반환:
#         없음. 그림 창을 띄운다. 박스가 알약에 딱 맞으면 정상.

#     동작:
#         1. 해당 분할의 라벨 파일을 정렬해서 모은다.
#         2. 시드 고정 후 n개를 무작위로 고른다.
#         3. 파일마다
#            a. 이름이 같은 이미지를 띄운다.
#            b. read_label() 로 박스를 읽어 사각형과 클래스 번호를 그린다.
#     """
#     # 1~2. 라벨 파일 n개 무작위 선택 (정렬해야 시드가 같을 때 같은 파일이 뽑힌다)
#     txts = sorted(OUT.glob(f"labels/{split}/*.txt"))
#     random.seed(42)
#     picked = random.sample(txts, n)

#     for i, txt in enumerate(picked):
#         ax = plt.subplot(2, 3, i + 1)   # subplot 번호는 1부터

#         # 3-a. 같은 이름의 이미지
#         ax.imshow(plt.imread(OUT / "images" / spl

#         # 3-b. 박스 그리기 (Rectangle 은 왼쪽 위
#         for idx, x, y, w, h in read_label(txt):
#             ax.add_patch(Rectangle((x, y), w, h, e"))
#             ax.text(x, y, idx, color="lime")
#         ax.axis("off")
#     plt.show()


# if __name__ == "__main__":
#     compare_with_original()
#     show()