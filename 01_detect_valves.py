# from ultralytics import YOLO
# import cv2
# import os
# import pandas as pd

# MODEL_PATH = r"D:\Workspace\pid_valve_detection\pid_valve_detection_v1\inline_branch_clean\weights\best.pt"
# IMAGE_PATH = r"D:\Workspace\Pnid_new\valves_detection_\pid-ml\pdf_images_visual\page_2_visual.png"

# OUT_DIR  = "output_image_2_valve"
# CROP_DIR = os.path.join(OUT_DIR, "valve_crops")

# YOLO_CONF     = 0.01
# CONF_INLINE   = 0.05
# CONF_BRANCH   = 0.05

# MIN_AREA      = 30
# MIN_THICKNESS = 3
# MAX_ASPECT    = 20.0

# os.makedirs(CROP_DIR, exist_ok=True)
# os.makedirs(OUT_DIR, exist_ok=True)


# model = YOLO(MODEL_PATH)
# print("Model classes:", model.names)


# img = cv2.imread(IMAGE_PATH)
# H, W = img.shape[:2]

# results = model(
#     IMAGE_PATH,
#     conf=YOLO_CONF,
#     imgsz=640,
#     verbose=False
# )

# rows = []
# idx = 0

# raw = rej_conf = rej_area = rej_thick = rej_aspect = 0

# for r in results:
#     if r.boxes is None:
#         continue

#     for b in r.boxes:
#         raw += 1

#         cls_id = int(b.cls)
#         label  = model.names[cls_id]
#         conf   = float(b.conf)

#         if label not in ("valve_inline", "valve_branch"):
#             continue

#         if label == "valve_inline" and conf < CONF_INLINE:
#             rej_conf += 1
#             continue
#         if label == "valve_branch" and conf < CONF_BRANCH:
#             rej_conf += 1
#             continue

#         x1, y1, x2, y2 = map(int, b.xyxy[0])
#         w, h = x2 - x1, y2 - y1
#         area = w * h

#         if area < MIN_AREA:
#             rej_area += 1
#             continue

#         if min(w, h) < MIN_THICKNESS:
#             rej_thick += 1
#             continue

#         aspect = max(w / max(h, 1), h / max(w, 1))
#         if aspect > MAX_ASPECT:
#             rej_aspect += 1
#             continue

#         crop = img[y1:y2, x1:x2]
#         name = f"valve_{idx}_{x1}_{y1}_{x2}_{y2}.png"
#         cv2.imwrite(os.path.join(CROP_DIR, name), crop)

#         rows.append({
#             "valve_crop": name,
#             "valve_type": label,
#             "confidence": round(conf, 3),
#             "x1": x1, "y1": y1,
#             "x2": x2, "y2": y2,
#             "cx": round((x1 + x2) / 2, 1),
#             "cy": round((y1 + y2) / 2, 1)
#         })

#         idx += 1

# df = pd.DataFrame(rows)
# out_csv = os.path.join(OUT_DIR, "detected_valves_semantic.csv")
# df.to_csv(out_csv, index=False)

# print("\n===== SUMMARY =====")
# print("Raw YOLO boxes :", raw)
# print("Rejected conf  :", rej_conf)
# print("Rejected area  :", rej_area)
# print("Rejected thick :", rej_thick)
# print("Rejected aspect:", rej_aspect)
# print("Final valves   :", len(df))
# print("→", out_csv)











# from ultralytics import YOLO
# import cv2
# import os
# import pandas as pd

# # =========================================================
# # CONFIG
# # =========================================================
# # MODEL_PATH = r"D:\Workspace\pid_valve_detection\pid_valve_detection_v3\inline_branch_cleaned\weights\best.pt"
# MODEL_PATH=R"D:\Workspace\pid_valve_detection\pid_valve_detection_v4_new\inline_branch_cleaned\weights\best.pt"
# IMAGE_PATH = r"D:\Workspace\Pnid_new\valves_detection_\pid-ml\pdf_images_visual\page_2_visual.png"

# OUT_DIR  = "output_image_2_valves_new"
# CROP_DIR = os.path.join(OUT_DIR, "valve_crops")

# # YOLO confidence (keep LOW initially)
# YOLO_CONF = 0.15

# # geometry filters (very important)
# MIN_AREA      = 400      # removes commas, dots
# MIN_THICKNESS = 6        # removes slashes
# MAX_ASPECT    = 6.0      # removes long lines

# os.makedirs(CROP_DIR, exist_ok=True)
# os.makedirs(OUT_DIR, exist_ok=True)

# # =========================================================
# # LOAD MODEL & IMAGE
# # =========================================================
# model = YOLO(MODEL_PATH)
# print("Model classes:", model.names)

# img = cv2.imread(IMAGE_PATH)
# H, W = img.shape[:2]

# # =========================================================
# # RUN YOLO
# # =========================================================
# results = model(
#     IMAGE_PATH,
#     conf=YOLO_CONF,
#     imgsz=1024,
#     verbose=False
# )

# rows = []
# idx = 0

# raw = rej_area = rej_thick = rej_aspect = 0

# for r in results:
#     if r.boxes is None:
#         continue

#     for b in r.boxes:
#         cls_id = int(b.cls)
#         label  = model.names[cls_id]
#         conf   = float(b.conf)

#         if label not in ("inline_valve", "branch_valve"):
#             continue

#         x1, y1, x2, y2 = map(int, b.xyxy[0])
#         w, h = x2 - x1, y2 - y1
#         area = w * h

#         # ---------------- GEOMETRY FILTERS ----------------
#         if area < MIN_AREA:
#             rej_area += 1
#             continue

#         if min(w, h) < MIN_THICKNESS:
#             rej_thick += 1
#             continue

#         aspect = max(w / max(h, 1), h / max(w, 1))
#         if aspect > MAX_ASPECT:
#             rej_aspect += 1
#             continue
#         # --------------------------------------------------

#         crop = img[y1:y2, x1:x2]
#         name = f"valve_{idx}_{x1}_{y1}_{x2}_{y2}.png"
#         cv2.imwrite(os.path.join(CROP_DIR, name), crop)

#         rows.append({
#             "valve_crop": name,
#             "valve_type": label,
#             "confidence": round(conf, 3),
#             "x1": x1, "y1": y1,
#             "x2": x2, "y2": y2,
#             "cx": round((x1 + x2) / 2, 1),
#             "cy": round((y1 + y2) / 2, 1)
#         })

#         idx += 1

# # =========================================================
# # SAVE OUTPUT
# # =========================================================
# df = pd.DataFrame(rows)
# out_csv = os.path.join(OUT_DIR, "detected_valves.csv")
# df.to_csv(out_csv, index=False)

# print("\n===== SUMMARY =====")
# print("Detected valves :", len(df))
# print("Rejected area   :", rej_area)
# print("Rejected thin   :", rej_thick)
# print("Rejected aspect :", rej_aspect)
# print("→", out_csv)





from ultralytics import YOLO
import cv2
import os
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = r"best.pt"
IMAGE_PATH = r"pdf_images_visual\page_1_visual.png"

OUT_DIR  = "output_image_1_valves_MAX_RECALL"
CROP_DIR = os.path.join(OUT_DIR, "valve_crops")

# 🔥 VERY LOW confidence — intentional
YOLO_CONF = 0.05

# 🔥 MINIMAL geometry filters
MIN_AREA      = 150      # keep small branch valves
MIN_THICKNESS = 3        # allow thin inline valves
MAX_ASPECT    = 12.0     # inline valves are long

os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# LOAD MODEL & IMAGE
# =========================================================
model = YOLO(MODEL_PATH)
print("Model classes:", model.names)

img = cv2.imread(IMAGE_PATH)
H, W = img.shape[:2]

# =========================================================
# RUN YOLO
# =========================================================
results = model(
    IMAGE_PATH,
    conf=YOLO_CONF,
    imgsz=1024,
    verbose=False
)

rows = []
idx = 0

raw = rej_area = rej_thick = rej_aspect = 0

for r in results:
    if r.boxes is None:
        continue

    for b in r.boxes:
        cls_id = int(b.cls)
        label  = model.names[cls_id]
        conf   = float(b.conf)

        if label not in ("inline_valve", "branch_valve"):
            continue

        x1, y1, x2, y2 = map(int, b.xyxy[0])
        w, h = x2 - x1, y2 - y1
        area = w * h

        # ---------------- VERY LIGHT FILTERS ----------------
        if area < MIN_AREA:
            rej_area += 1
            continue

        if min(w, h) < MIN_THICKNESS:
            rej_thick += 1
            continue

        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > MAX_ASPECT:
            rej_aspect += 1
            continue
        # ---------------------------------------------------

        crop = img[y1:y2, x1:x2]
        name = f"valve_{idx}_{x1}_{y1}_{x2}_{y2}.png"
        cv2.imwrite(os.path.join(CROP_DIR, name), crop)

        rows.append({
            "valve_crop": name,
            "valve_type": label,
            "confidence": round(conf, 3),
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "cx": round((x1 + x2) / 2, 1),
            "cy": round((y1 + y2) / 2, 1)
        })

        idx += 1

# =========================================================
# SAVE OUTPUT
# =========================================================
df = pd.DataFrame(rows)
out_csv = os.path.join(OUT_DIR, "detected_valves_MAX_RECALL.csv")
df.to_csv(out_csv, index=False)

print("\n===== MAX-RECALL SUMMARY =====")
print("Total detections :", len(df))
print("Rejected area    :", rej_area)
print("Rejected thin    :", rej_thick)
print("Rejected aspect  :", rej_aspect)
print("→", out_csv)
