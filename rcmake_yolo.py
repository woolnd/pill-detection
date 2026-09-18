[1mdiff --git "a/src\\make_yolo.py" "b/src\\make_yolo_class_balance.py"[m
[1mindex c20a139..1acec9c 100644[m
[1m--- "a/src\\make_yolo.py"[m
[1m+++ "b/src\\make_yolo_class_balance.py"[m
[36m@@ -2,265 +2,530 @@[m [mfrom pathlib import Path[m
 import json[m
 import random[m
 import shutil[m
[31m-from collections import defaultdict[m
[32m+[m[32mfrom collections import defaultdict, Counter[m
 [m
[32m+[m[32m# ============================================================[m
[32m+[m[32m# 설정[m
[32m+[m[32m# ============================================================[m
 [m
[31m-ROOT = Path(__file__).resolve().parent.parent[m
[32m+[m[32mROOT = Path(__file__).resolve().parents[1][m
 [m
 RAW = ROOT / "data" / "raw" / "sprint_ai_project1_data"[m
[31m-OUT = ROOT / "data" / "yolo"[m
 [m
[31m-VAL_RATIO = 0.2[m
[32m+[m[32mOUT = ROOT / "data" / "yolo_class_balance"[m
[32m+[m
 SEED = 42[m
[32m+[m[32mVAL_RATIO = 0.20[m
[32m+[m
[32m+[m[32mIMAGE_W = 976[m
[32m+[m[32mIMAGE_H = 1280[m
[32m+[m
[32m+[m
[32m+[m[32m# ============================================================[m
[32m+[m[32m# Annotation 유효성 검사[m
[32m+[m[32m# ============================================================[m
[32m+[m
[32m+[m[32mdef is_valid_bbox(x, y, w, h):[m
[32m+[m[32m    return ([m
[32m+[m[32m        x >= 0[m
[32m+[m[32m        and y >= 0[m
[32m+[m[32m        and w > 0[m
[32m+[m[32m        and h > 0[m
[32m+[m[32m        and x + w <= IMAGE_W[m
[32m+[m[32m        and y + h <= IMAGE_H[m
[32m+[m[32m    )[m
 [m
[31m-IMG_W = 976[m
[31m-IMG_H = 1280[m
 [m
[32m+[m[32m# ============================================================[m
[32m+[m[32m# Annotation 불러오기[m
[32m+[m[32m# ============================================================[m
 [m
 def load_annotations():[m
[31m-    by_image = defaultdict(list)[m
[32m+[m[32m    annotation_dir = RAW / "train_annotations"[m
 [m
[31m-    for f in RAW.glob("train_annotations/**/*.json"):[m
[31m-        data = json.loads(f.read_text(encoding="utf-8"))[m
[32m+[m[32m    annotations = {}[m
 [m
[31m-        img = data["images"][0][m
[32m+[m[32m    json_files = sorted(annotation_dir.rglob("*.json"))[m
 [m
[31m-        for ann in data["annotations"]:[m
[31m-            x, y, w, h = ann["bbox"][m
[31m-            cat = data["categories"][0][m
[32m+[m[32m    print("Annotation 불러오는 중...")[m
[32m+[m
[32m+[m[32m    for json_file in json_files:[m
[32m+[m[32m        with open(json_file, "r", encoding="utf-8") as f:[m
[32m+[m[32m            data = json.load(f)[m
[32m+[m
[32m+[m[32m        images = {[m
[32m+[m[32m            image["id"]: image["file_name"][m
[32m+[m[32m            for image in data.get("images", [])[m
[32m+[m[32m        }[m
 [m
[31m-            by_image[img["file_name"]].append([m
[31m-                {[m
[31m-                    "x": x,[m
[31m-                    "y": y,[m
[31m-                    "w": w,[m
[31m-                    "h": h,[m
[31m-                    "class_id": cat["id"],[m
[31m-                    "class_name": cat["name"],[m
[32m+[m[32m        categories = {[m
[32m+[m[32m            category["id"]: category["name"][m
[32m+[m[32m            for category in data.get("categories", [])[m
[32m+[m[32m        }[m
[32m+[m
[32m+[m[32m        for ann in data.get("annotations", []):[m
[32m+[m[32m            image_id = ann["image_id"][m
[32m+[m
[32m+[m[32m            if image_id not in images:[m
[32m+[m[32m                continue[m
[32m+[m
[32m+[m[32m            bbox = ann["bbox"][m
[32m+[m
[32m+[m[32m            x, y, w, h = bbox[m
[32m+[m
[32m+[m[32m            if not is_valid_bbox(x, y, w, h):[m
[32m+[m[32m                # 해당 이미지는 나중에 전체 제외[m
[32m+[m[32m                annotations.setdefault([m
[32m+[m[32m                    images[image_id],[m
[32m+[m[32m                    {"boxes": [], "invalid": True}[m
[32m+[m[32m                )[m
[32m+[m[32m                continue[m
[32m+[m
[32m+[m[32m            class_id = ann["category_id"][m
[32m+[m[32m            class_name = categories.get(class_id, str(class_id))[m
[32m+[m
[32m+[m[32m            if images[image_id] not in annotations:[m
[32m+[m[32m                annotations[images[image_id]] = {[m
[32m+[m[32m                    "boxes": [],[m
[32m+[m[32m                    "invalid": False,[m
                 }[m
[32m+[m
[32m+[m[32m            annotations[images[image_id]]["boxes"].append([m
[32m+[m[32m                (class_name, x, y, w, h)[m
             )[m
 [m
[31m-    return dict(by_image)[m
[32m+[m[32m    # invalid 이미지 제거[m
[32m+[m[32m    valid_annotations = {}[m
 [m
[32m+[m[32m    for image_name, info in annotations.items():[m
[32m+[m[32m        if info["invalid"]:[m
[32m+[m[32m            continue[m
 [m
[31m-def is_valid(box):[m
[31m-    x = box["x"][m
[31m-    y = box["y"][m
[31m-    w = box["w"][m
[31m-    h = box["h"][m
[32m+[m[32m        if len(info["boxes"]) == 0:[m
[32m+[m[32m            continue[m
 [m
[31m-    return ([m
[31m-        x >= 0[m
[31m-        and y >= 0[m
[31m-        and w > 0[m
[31m-        and h > 0[m
[31m-        and x + w <= IMG_W[m
[31m-        and y + h <= IMG_H[m
[31m-    )[m
[32m+[m[32m        valid_annotations[image_name] = info["boxes"][m
 [m
[32m+[m[32m    print(f"유효 이미지: {len(valid_annotations)}장")[m
 [m
[31m-def remove_bad_images(ann):[m
[31m-    clean = {}[m
[32m+[m[32m    return valid_annotations[m
 [m
[31m-    for name, boxes in ann.items():[m
[31m-        if all(is_valid(b) for b in boxes):[m
[31m-            clean[name] = boxes[m
 [m
[31m-    return clean[m
[32m+[m[32m# ============================================================[m
[32m+[m[32m# 이미지 실제 위치 찾기[m
[32m+[m[32m# ============================================================[m
 [m
[32m+[m[32mdef find_image(image_name):[m
[32m+[m[32m    path = RAW / "train" / image_name[m
 [m
[31m-def make_class_map(ann):[m
[31m-    class_ids = sorted([m
[31m-        {[m
[31m-            box["class_id"][m
[31m-            for boxes in ann.values()[m
[31m-            for box in boxes[m
[31m-        }[m
[31m-    )[m
[32m+[m[32m    if path.exists():[m
[32m+[m[32m        return path[m
 [m
[31m-    class_to_idx = {[m
[31m-        class_id: idx[m
[31m-        for idx, class_id in enumerate(class_ids)[m
[31m-    }[m
[32m+[m[32m    matches = list(RAW.rglob(image_name))[m
[32m+[m
[32m+[m[32m    if matches:[m
[32m+[m[32m        return matches[0][m
[32m+[m
[32m+[m[32m    return None[m
 [m
[31m-    class_names = {}[m
 [m
[31m-    for boxes in ann.values():[m
[31m-        for box in boxes:[m
[31m-            class_names[class_to_idx[box["class_id"]]] = box["class_name"][m
[32m+[m[32m# ============================================================[m
[32m+[m[32m# 클래스별 이미지 목록[m
[32m+[m[32m# ============================================================[m
 [m
[31m-    return class_to_idx, class_names[m
[32m+[m[32mdef build_class_images(annotations):[m
[32m+[m[32m    class_images = defaultdict(set)[m
 [m
[32m+[m[32m    for image_name, boxes in annotations.items():[m
[32m+[m[32m        for class_name, *_ in boxes:[m
[32m+[m[32m            class_images[class_name].add(image_name)[m
 [m
[31m-def get_combo(name):[m
[31m-    return name.split("_")[0][m
[32m+[m[32m    return class_images[m
 [m
 [m
[31m-def split_by_combo(names):[m
[31m-    combos = sorted({get_combo(name) for name in names})[m
[32m+[m[32m# ============================================================[m
[32m+[m[32m# Train / Val 분할[m
[32m+[m[32m# ============================================================[m
 [m
[32m+[m[32mdef split_images(annotations):[m
     rng = random.Random(