import cv2
import numpy as np
import pandas as pd
import os

# IMAGES = [
#     "pdf_images_visual/page_1_visual.png",
#     "pdf_images_visual/page_2_visual.png",
#     "pdf_images_visual/page_3_visual.png",
#     "pdf_images_visual/page_4_visual.png",
# ]
IMAGES = [
    "output_upscaled/pnid_cropped_upscaled.png"
    ]
BASE_OUT = "squares_output_pnid8"

MIN_SIZE = 25
MAX_SIZE = 260
AREA_MIN = 500
DEDUP_DIST = 80


def is_circle(cnt):
    area = cv2.contourArea(cnt)
    if area < 200:
        return False
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return False
    return (4 * np.pi * area / (peri * peri)) > 0.80


def dedupe(cx, cy, used):
    return any(abs(cx-x)<DEDUP_DIST and abs(cy-y)<DEDUP_DIST for x,y in used)


for img_path in IMAGES:
    name = os.path.splitext(os.path.basename(img_path))[0]

    OUT_DIR = os.path.join(BASE_OUT, name)
    CROP_DIR = os.path.join(OUT_DIR, "crops")
    CSV_OUT = os.path.join(OUT_DIR, "squares_raw.csv")
    DEBUG_IMG = os.path.join(OUT_DIR, "squares_raw_debug.png")

    os.makedirs(CROP_DIR, exist_ok=True)

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    edges = cv2.Canny(gray, 40, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    used = []
    debug = img.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) < AREA_MIN:
            continue
        if is_circle(cnt):
            continue

        x,y,w,h = cv2.boundingRect(cnt)
        if not (MIN_SIZE <= w <= MAX_SIZE and MIN_SIZE <= h <= MAX_SIZE):
            continue

        cx, cy = x+w//2, y+h//2
        if dedupe(cx, cy, used):
            continue

        used.append((cx,cy))
        boxes.append((x,y,x+w,y+h))

    records = []
    for i,(x1,y1,x2,y2) in enumerate(boxes):
        pad = int(max(x2-x1, y2-y1)*0.35)
        X1,Y1 = max(0,x1-pad), max(0,y1-pad)
        X2,Y2 = min(img.shape[1],x2+pad), min(img.shape[0],y2+pad)

        crop = img[Y1:Y2, X1:X2]
        path = os.path.join(CROP_DIR, f"square_{i}.png")
        cv2.imwrite(path, crop)

        cv2.rectangle(debug,(X1,Y1),(X2,Y2),(0,255,0),2)
        cv2.putText(debug,str(i),(X1,Y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        records.append({
            "id": i,
            "center_x": (X1+X2)//2,
            "center_y": (Y1+Y2)//2,
            "width": X2-X1,
            "height": Y2-Y1,
            "crop_path": path
        })

    pd.DataFrame(records).to_csv(CSV_OUT, index=False)
    cv2.imwrite(DEBUG_IMG, debug)

    print(f"✅ {name}: {len(records)} square candidates")



#++++++++++++++++



