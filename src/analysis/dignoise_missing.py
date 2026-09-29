# """
# 미등장 클래스가 왜 안 잡히는지 진단: train 이미지에서 CONF 거의 0으로 재검사
# """
# from pathlib import Path
# from ultralytics import YOLO
# import yaml

# WEIGHTS = "runs/rahui_ep50_img1280_batch4_aihubdata_oversample/weights/best.pt"  # 0.61802 모델
# DATA_YAML = "data/yolo_aihub/data.yaml"

# # 먼저 오버샘플링으로 새로 잃은 5개부터 확인 (원래 잡던 애들이라 더 흥미로운 케이스)
# TARGET_CLASS_IDS = [5000, 13004, 13161, 38954, 43233]

# with open(DATA_YAML, encoding="utf-8") as f:
#     names = yaml.safe_load(f)["names"]  # idx(str) -> class_id(str)
# id_to_idx = {int(v): int(k) for k, v in names.items()}
# idx_to_id = {v: k for k, v in id_to_idx.items()}

# LABELS_DIR = Path("data/yolo_aihub/labels/train")
# IMAGES_DIR = Path("data/yolo_aihub/images/train")

# model = YOLO(WEIGHTS)

# for cid in TARGET_CLASS_IDS:
#     idx = id_to_idx[cid]
#     sample = []
#     for txt in LABELS_DIR.glob("*.txt"):
#         for line in txt.read_text(encoding="utf-8").splitlines():
#             parts = line.split()
#             if int(parts[0]) == idx:
#                 sample.append((txt, parts))
#                 break
#         if len(sample) >= 3:
#             break

#     print(f"\n=== class_id {cid} (idx {idx}) ===")
#     for txt, gt in sample:
#         img_path = IMAGES_DIR / (txt.stem + ".png")
#         if not img_path.exists():
#             continue
#         result = model.predict(str(img_path), imgsz=1280, conf=0.0001, verbose=False)[0]
#         gt_cx, gt_cy = float(gt[1]), float(gt[2])

#         found = False
#         for box, cls, conf in zip(result.boxes.xywhn, result.boxes.cls, result.boxes.conf):
#             pcx, pcy = box[0].item(), box[1].item()
#             if abs(pcx - gt_cx) < 0.05 and abs(pcy - gt_cy) < 0.05:
#                 pred_id = idx_to_id[int(cls.item())]
#                 tag = "정답" if pred_id == cid else f"오답(진짜는 {cid})"
#                 print(f"  {img_path.name}: 위치 일치, class={pred_id} [{tag}], conf={conf.item():.4f}")
#                 found = True
#         if not found:
#             print(f"  {img_path.name}: 그 위치에 예측 자체가 없음 (conf=0.0001에도 안 뜸)")
# # 23개 전체가 몇 개의 서로 다른 조합(combo)에 등장하는지 확인
# ALL_MISSING = [250, 573, 3614, 5000, 5391, 13004, 13161, 15280, 16206, 20753,
#                23319, 24941, 26455, 26993, 28424, 29711, 33026, 37777, 38723,
#                38954, 38972, 43233, 44834]

# def get_combo(name):
#     return name.split("_", 1)[0]

# combo_map = {cid: set() for cid in ALL_MISSING}
# for txt in LABELS_DIR.glob("*.txt"):
#     classes_here = {int(l.split()[0]) for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()}
#     combo = get_combo(txt.stem)
#     for cid in ALL_MISSING:
#         idx = id_to_idx.get(cid)
#         if idx in classes_here:
#             combo_map[cid].add(combo)

# print(f"\n{'class_id':>10} {'조합 수':>8}")
# for cid in ALL_MISSING:
#     print(f"{cid:>10} {len(combo_map[cid]):>8}")