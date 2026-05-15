# import cv2
# import numpy as np
# import pandas as pd
# import os
# import math

# # =========================================================
# # INPUT
# # =========================================================
# IMAGE = "output_upscaled/pnid_cropped_upscaled.png"
# OUT_DIR = "output/instruments_circle_validated"

# # =========================================================
# # PARAMETERS (STABLE)
# # =========================================================
# MIN_RADIUS = 16
# MAX_RADIUS = 65
# DEDUP_DIST = 40

# MIN_EDGE_SUPPORT = 0.35   # 🔑 critical

# # =========================================================
# # SETUP
# # =========================================================
# os.makedirs(os.path.join(OUT_DIR, "crops"), exist_ok=True)

# img = cv2.imread(IMAGE)
# if img is None:
#     raise RuntimeError("Image not found")

# H, W = img.shape[:2]

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
# blur = cv2.GaussianBlur(gray, (7, 7), 1.4)

# edges = cv2.Canny(gray, 60, 150)

# # =========================================================
# # HELPERS
# # =========================================================
# def dedupe(x, y, used):
#     for ux, uy in used:
#         if abs(x - ux) < DEDUP_DIST and abs(y - uy) < DEDUP_DIST:
#             return True
#     return False


# def circle_edge_support(edges, x, y, r):
#     """
#     Measure how much of the circle perimeter
#     is supported by real edge pixels.
#     """
#     mask = np.zeros_like(edges)
#     cv2.circle(mask, (x, y), r, 255, 2)

#     edge_hits = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))
#     circumference = 2 * math.pi * r

#     return edge_hits / circumference


# # =========================================================
# # HOUGH (PROPOSAL ONLY)
# # =========================================================
# circles = cv2.HoughCircles(
#     blur,
#     cv2.HOUGH_GRADIENT,
#     dp=1.1,
#     minDist=35,
#     param1=120,
#     param2=20,          # NOT too low
#     minRadius=MIN_RADIUS,
#     maxRadius=MAX_RADIUS
# )

# records = []
# used = []
# debug = img.copy()
# idx = 0

# if circles is not None:
#     circles = np.round(circles[0]).astype(int)

#     for x, y, r in circles:
#         if dedupe(x, y, used):
#             continue

#         support = circle_edge_support(edges, x, y, r)
#         if support < MIN_EDGE_SUPPORT:
#             continue

#         used.append((x, y))

#         pad = int(r * 0.35)
#         x1 = max(0, x - r - pad)
#         y1 = max(0, y - r - pad)
#         x2 = min(W, x + r + pad)
#         y2 = min(H, y + r + pad)

#         if x2 <= x1 or y2 <= y1:
#             continue

#         crop = img[y1:y2, x1:x2]
#         if crop.size == 0:
#             continue

#         cv2.imwrite(f"{OUT_DIR}/crops/inst_{idx}.png", crop)

#         cv2.circle(debug, (x, y), r, (0, 0, 255), 2)
#         cv2.putText(debug, str(idx), (x - 8, y - 8),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

#         records.append({
#             "id": idx,
#             "x": x,
#             "y": y,
#             "r": r,
#             "edge_support": round(support, 2)
#         })

#         idx += 1

# pd.DataFrame(records).to_csv(f"{OUT_DIR}/circles.csv", index=False)
# cv2.imwrite(f"{OUT_DIR}/debug.png", debug)

# print(f"✅ {idx} validated circle instruments detected")




#+++++++++
import os
import cv2
import pandas as pd
import pytesseract
from ultralytics import YOLO

# =========================================================
# INPUTS
# =========================================================
IMAGE_PATH = "output_upscaled/pnid_cropped_upscaled.png"
MODEL_PATH = "best.pt"
OUT_DIR = "output/instruments_yolo_ocr"

# =========================================================
# TESSERACT PATH (Windows)
# =========================================================
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\shital.ingole\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)
# =========================================================
# SETTINGS
# =========================================================
CONF_THRESHOLD = 0.25
PAD_RATIO = 0.15  # padding around each detected box

# =========================================================
# CREATE OUTPUT FOLDERS
# =========================================================
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "crops"), exist_ok=True)

# =========================================================
# LOAD IMAGE
# =========================================================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise RuntimeError(f"Image not found: {IMAGE_PATH}")

H, W = img.shape[:2]

# =========================================================
# LOAD YOLO MODEL
# =========================================================
model = YOLO(MODEL_PATH)

# =========================================================
# RUN DETECTION
# =========================================================
results = model.predict(
    source=IMAGE_PATH,
    conf=CONF_THRESHOLD,
    save=False,
    verbose=False
)

# =========================================================
# OCR FUNCTION
# =========================================================
def extract_text(crop):
    """
    Extract text from a cropped instrument image using Tesseract OCR.
    """
    if crop is None or crop.size == 0:
        return ""

    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Enlarge image for better OCR
    gray = cv2.resize(
        gray,
        None,
        fx=2.5,
        fy=2.5,
        interpolation=cv2.INTER_CUBIC
    )

    # Reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binary threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # OCR configuration
    config = (
        "--oem 3 "
        "--psm 7 "
        "-c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_/."
    )

    text = pytesseract.image_to_string(thresh, config=config)
    return text.strip()


# =========================================================
# PROCESS DETECTIONS
# =========================================================
debug = img.copy()
records = []
idx = 0

for result in results:
    if result.boxes is None:
        continue

    for box in result.boxes:
        conf = float(box.conf[0])

        if conf < CONF_THRESHOLD:
            continue

        # Class information
        cls_id = int(box.cls[0])
        class_name = model.names.get(cls_id, str(cls_id))

        # Bounding box coordinates
        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        # Add padding
        bw = x2 - x1
        bh = y2 - y1

        pad_x = int(bw * PAD_RATIO)
        pad_y = int(bh * PAD_RATIO)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(W, x2 + pad_x)
        y2 = min(H, y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            continue

        # Crop image
        crop = img[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            continue

        # Save crop
        crop_path = os.path.join(
            OUT_DIR,
            "crops",
            f"inst_{idx}.png"
        )
        cv2.imwrite(crop_path, crop)

        # OCR text extraction
        ocr_text = extract_text(crop)

        # Draw bounding box
        cv2.rectangle(
            debug,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        # Detection label
        label = f"{idx}: {class_name} ({conf:.2f})"
        cv2.putText(
            debug,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

        # OCR text below box
        if ocr_text:
            cv2.putText(
                debug,
                ocr_text,
                (x1, min(H - 10, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )

        # Save metadata
        records.append({
            "id": idx,
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": round(conf, 4),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": x2 - x1,
            "height": y2 - y1,
            "ocr_text": ocr_text,
            "crop_path": crop_path
        })

        idx += 1

# =========================================================
# SAVE OUTPUTS
# =========================================================
csv_path = os.path.join(OUT_DIR, "detections.csv")
debug_path = os.path.join(OUT_DIR, "debug.png")

pd.DataFrame(records).to_csv(csv_path, index=False)
cv2.imwrite(debug_path, debug)

# =========================================================
# FINAL OUTPUT
# =========================================================
print(f"✅ {idx} instruments detected using YOLO + Tesseract OCR")
print(f"📄 CSV saved to: {csv_path}")
print(f"🖼️ Debug image saved to: {debug_path}")
print(f"🗂️ Crops saved to: {os.path.join(OUT_DIR, 'crops')}")
