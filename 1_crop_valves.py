# import os
# import csv
# import cv2
# from ultralytics import YOLO

# # ================== CONFIG ==================

# IMAGE_PATH = "pdf_images_visual/page_2_visual.png"
# MODEL_PATH = "best.pt"

# OUT_DIR = "cropped_valves_image_2"
# CONF = 0.01
# IMG_SIZE = 1024

# os.makedirs(OUT_DIR, exist_ok=True)
# CSV_PATH = os.path.join(OUT_DIR, "valves_meta.csv")

# # ================== DETECTION ==================

# model = YOLO(MODEL_PATH)
# print("model.names =", model.names)  # debug: see all classes

# results = model.predict(
#     source=IMAGE_PATH,
#     imgsz=IMG_SIZE,
#     conf=CONF,
#     save=False,
# )

# img = cv2.imread(IMAGE_PATH)
# h, w = img.shape[:2]

# rows = []
# valve_count = 0

# for box in results[0].boxes:
#     x1, y1, x2, y2 = map(int, box.xyxy[0])

#     # clamp to image
#     x1, y1 = max(0, x1), max(0, y1)
#     x2, y2 = min(w, x2), min(h, y2)

#     crop = img[y1:y2, x1:x2]
#     if crop.size == 0:
#         continue

#     cls_id = int(box.cls[0])
#     cls_name = model.names[cls_id]  # whatever the model calls this symbol

#     # For now: keep ALL detections as valve candidates
#     valve_count += 1
#     fname = f"valve_{valve_count}_{x1}_{y1}_{x2}_{y2}.png"
#     out_path = os.path.join(OUT_DIR, fname)
#     cv2.imwrite(out_path, crop)

#     cx = (x1 + x2) / 2.0
#     cy = (y1 + y2) / 2.0

#     rows.append({
#         "valve_id": valve_count,
#         "cls_name": cls_name,
#         "file": fname,
#         "x1": x1,
#         "y1": y1,
#         "x2": x2,
#         "y2": y2,
#         "center_x": cx,
#         "center_y": cy,
#     })

# # write metadata CSV
# with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
#     writer = csv.DictWriter(
#         f,
#         fieldnames=[
#             "valve_id",
#             "cls_name",
#             "file",
#             "x1", "y1", "x2", "y2",
#             "center_x", "center_y",
#         ],
#     )
#     writer.writeheader()
#     writer.writerows(rows)

# print(f"Extracted {valve_count} valve candidates into {OUT_DIR}")
# print(f"Metadata saved to {CSV_PATH}")




import os
import csv
import cv2
from ultralytics import YOLO

# ================== CONFIG ==================
IMAGE_PATH = "pdf_images_visual/page_1_visual.png"
MODEL_PATH = "best.pt"

OUT_DIR = "cropped_valves_image_1_v1"
CSV_PATH = os.path.join(OUT_DIR, "valves_meta.csv")

CONF = 0.01           # keep low for P&ID
IMG_SIZE = 1024

MIN_W = 30            # 🔥 critical
MIN_H = 30            # 🔥 critical

os.makedirs(OUT_DIR, exist_ok=True)

# ================== LOAD MODEL ==================
model = YOLO(MODEL_PATH)
print("🔍 model.names =", model.names)

# ================== RUN DETECTION ==================
results = model.predict(
    source=IMAGE_PATH,
    imgsz=IMG_SIZE,
    conf=CONF,
    save=False,
)

img = cv2.imread(IMAGE_PATH)
H, W = img.shape[:2]

rows = []
valve_count = 0

# ================== PROCESS DETECTIONS ==================
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # clamp
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    w = x2 - x1
    h = y2 - y1

    # 🔥 reject tiny junk
    if w < MIN_W or h < MIN_H:
        continue

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    cls_id = int(box.cls[0])
    cls_name = model.names.get(cls_id, "unknown")
    conf = float(box.conf[0])

    valve_count += 1
    fname = f"valve_{valve_count}_{x1}_{y1}_{x2}_{y2}.png"
    out_path = os.path.join(OUT_DIR, fname)
    cv2.imwrite(out_path, crop)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    rows.append({
        "valve_id": valve_count,
        "cls_name": cls_name,
        "confidence": round(conf, 3),
        "file": fname,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "center_x": round(cx, 1),
        "center_y": round(cy, 1),
    })

# ================== WRITE CSV ==================
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "valve_id",
            "cls_name",
            "confidence",
            "file",
            "x1", "y1", "x2", "y2",
            "center_x", "center_y",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ Extracted {valve_count} valve candidates")
print(f"📄 Metadata saved → {CSV_PATH}")
