import os
import cv2
from ultralytics import YOLO

# ================== CONFIG ==================
IMAGE_PATH = "pdf_images_visual/page_1_visual.png"
MODEL_PATH = "best.pt"

OUTPUT_DIR = "cropped_symbols_image_1"
CONF = 0.01
IMG_SIZE = 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== DETECTION ==================
model = YOLO(MODEL_PATH)

results = model.predict(
    source=IMAGE_PATH,
    imgsz=IMG_SIZE,
    conf=CONF,
    save=False
)

img = cv2.imread(IMAGE_PATH)

count = 0
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Safety clamp
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        continue

    count += 1

    # 🔥 BBOX EMBEDDED IN FILENAME
    out_path = os.path.join(
        OUTPUT_DIR,
        f"symbol_{count}_{x1}_{y1}_{x2}_{y2}.png"
    )

    cv2.imwrite(out_path, crop)

print(f"✅ Extracted {count} symbol crops WITH BBOX into {OUTPUT_DIR}")
