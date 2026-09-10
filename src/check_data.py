import random #모든 데이터를 확인할수없으니 sample링통해 확인하기 위함
from collections import Counter

import matplotlib.pyplot as plt

# 이미지 위에 바운딩 박스를 사각형으로 표시할 때 사용합니다.
from matplotlib.patches import Rectangle

#함수불러옴
from annotations import RAW, load_annotations

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False  # 한글 폰트 쓰면 마이너스 기호 깨지는 것 방지

def summary(ann):
    boxes = [b for v in ann.values() for b in v]
    print("이미지", len(ann), "/ 박스", len(boxes), "/ 클래스", len({b["class_id"] for b in boxes}))
    print("test 이미지", len(list(RAW.glob("test_images/*.png"))))
    print("이미지당 알약:", sorted(Counter(len(v) for v in ann.values()).items()))
    print("클래스별 박스:", Counter(b["class_id"] for b in boxes).most_common(5), "...")
    

def show(ann, n=6):
    random.seed(42)
    names = random.sample(sorted(ann), n)
    for i, name in enumerate(names):
        ax = plt.subplot(2, 3, i + 1)
        ax.imshow(plt.imread(RAW / "train_images" / name))
        for b in ann[name]:
            ax.add_patch(Rectangle((b["x"], b["y"]), b["w"], b["h"], fill=False, edgecolor="blue"))
            ax.text(b["x"], b["y"], b["class_name"], color="blue")
        ax.axis("off")
    plt.savefig(RAW.parent / "sample_check.png", dpi=120)
    plt.show()

    
if __name__ == "__main__":
    ann = load_annotations()
    summary(ann)
    show(ann)

   