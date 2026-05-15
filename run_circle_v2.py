# import os
# import re
# import cv2
# import numpy as np
# import pandas as pd
# from ultralytics import YOLO
# import easyocr

# # =========================================================
# # CONFIGURATION
# # =========================================================
# MODEL_PATH = "best.pt"

# IMAGES = [
#     "output_upscaled/pnid_cropped_upscaled.png"
# ]

# BASE_OUT = "output/yolo_circle_ocr"

# CONF_THRESHOLD = 0.35
# PADDING = 20

# # Regex for valid instrument tags
# TAG_PATTERN = re.compile(
#     r'^[A-Z]{1,4}[- ]?\d{1,4}$|^[A-Z]{1,4}[- ]?[A-Z]{0,3}[- ]?\d{1,4}$'
# )

# # =========================================================
# # INITIALIZE MODELS
# # =========================================================
# model = YOLO(MODEL_PATH)
# reader = easyocr.Reader(['en'], gpu=False)

# # =========================================================
# # HELPERS
# # =========================================================
# def ensure_dirs(path):
#     os.makedirs(path, exist_ok=True)
#     os.makedirs(os.path.join(path, "crops"), exist_ok=True)


# def clean_text(text):
#     text = text.upper().strip()
#     text = re.sub(r'[^A-Z0-9\- ]', '', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text


# def is_valid_tag(text):
#     return bool(TAG_PATTERN.match(text))


# def extract_text(crop):
#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

#     # Upscale for better OCR
#     gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

#     # Threshold
#     _, thresh = cv2.threshold(
#         gray, 0, 255,
#         cv2.THRESH_BINARY + cv2.THRESH_OTSU
#     )

#     # EasyOCR
#     results = reader.readtext(
#         thresh,
#         detail=1,
#         paragraph=False,
#         allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
#     )

#     if not results:
#         return "", 0.0

#     # Highest-confidence result
#     best = max(results, key=lambda x: x[2])
#     text = clean_text(best[1])
#     conf = float(best[2])

#     return text, conf


# # =========================================================
# # MAIN PROCESS
# # =========================================================
# for img_path in IMAGES:
#     name = os.path.splitext(os.path.basename(img_path))[0]

#     out_dir = os.path.join(BASE_OUT, name)
#     crop_dir = os.path.join(out_dir, "crops")
#     csv_path = os.path.join(out_dir, "circles.csv")
#     debug_path = os.path.join(out_dir, "circles_debug.png")

#     ensure_dirs(out_dir)

#     print(f"\nProcessing: {name}")

#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"Image not found: {img_path}")
#         continue

#     h, w = img.shape[:2]
#     debug = img.copy()

#     # Run YOLO detection
#     results = model.predict(
#         source=img,
#         conf=CONF_THRESHOLD,
#         verbose=False
#     )

#     records = []
#     idx = 0

#     for result in results:
#         for box in result.boxes:
#             conf = float(box.conf[0])

#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             # Add padding
#             x1 = max(0, x1 - PADDING)
#             y1 = max(0, y1 - PADDING)
#             x2 = min(w, x2 + PADDING)
#             y2 = min(h, y2 + PADDING)

#             crop = img[y1:y2, x1:x2]
#             if crop.size == 0:
#                 continue

#             # OCR
#             text, ocr_conf = extract_text(crop)

#             # Save only valid tags with reasonable confidence
#             if text and ocr_conf >= 0.30:
#                 valid = is_valid_tag(text)
#             else:
#                 valid = False

#             # Save crop
#             crop_path = os.path.join(crop_dir, f"circle_{idx}.png")
#             cv2.imwrite(crop_path, crop)

#             # Draw box
#             color = (0, 255, 0) if valid else (0, 0, 255)
#             cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)

#             label = f"{idx}: {text} ({ocr_conf:.2f})"
#             cv2.putText(
#                 debug,
#                 label,
#                 (x1, max(20, y1 - 8)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 color,
#                 2
#             )

#             # Save all detections for review
#             records.append({
#                 "id": idx,
#                 "x1": x1,
#                 "y1": y1,
#                 "x2": x2,
#                 "y2": y2,
#                 "detection_confidence": round(conf, 4),
#                 "text": text,
#                 "ocr_confidence": round(ocr_conf, 4),
#                 "valid_tag": valid,
#                 "crop_path": crop_path
#             })

#             idx += 1

#     # Save outputs
#     pd.DataFrame(records).to_csv(csv_path, index=False)
#     cv2.imwrite(debug_path, debug)

#     print(f"Detections saved: {len(records)}")
#     print(f"CSV: {csv_path}")
#     print(f"Debug image: {debug_path}")


#++++++++++++++++++++++++++++++++++++++++++++
import os
import re
import cv2
import math
import numpy as np
import pandas as pd
import easyocr

# =========================================================
# CONFIGURATION
# =========================================================
IMAGES = [
    "output_upscaled/pnid_cropped_upscaled.png"
]

BASE_OUT = "output/demo_circle_ocr"

# Circle size range (adjust based on your P&ID)
MIN_RADIUS = 25
MAX_RADIUS = 120

# Detection parameters
MIN_TEXT_PIXELS = 30
MAX_FILL_RATIO = 0.45
DEDUP_DISTANCE = 50
PADDING_RATIO = 0.30

# OCR confidence threshold
OCR_CONF_THRESHOLD = 0.20

# Valid tag pattern examples:
# PT-101, FI002, TIC-205, LT301
TAG_PATTERN = re.compile(
    r'^[A-Z]{1,4}[- ]?[A-Z]{0,3}[- ]?\d{1,4}$'
)

# =========================================================
# INITIALIZE EASYOCR (Deep Learning OCR)
# =========================================================
reader = easyocr.Reader(['en'], gpu=False)

# =========================================================
# HELPERS
# =========================================================
def ensure_dirs(path):
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "crops"), exist_ok=True)


def clean_text(text):
    """Normalize OCR output."""
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9\- ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def is_valid_tag(text):
    """Validate instrument tag pattern."""
    return bool(TAG_PATTERN.match(text))


def dedupe(x, y, used):
    """Avoid duplicate detections."""
    for ux, uy in used:
        if abs(x - ux) < DEDUP_DISTANCE and abs(y - uy) < DEDUP_DISTANCE:
            return True
    return False


def fill_ratio(binary, x, y, r):
    """Measure how filled the circle is."""
    mask = np.zeros_like(binary)
    cv2.circle(mask, (x, y), r, 255, -1)

    filled = cv2.countNonZero(
        cv2.bitwise_and(binary, binary, mask=mask)
    )

    area = math.pi * r * r
    return filled / max(area, 1)


def text_pixels_inside(binary, x, y, r):
    """Count text-like pixels near the center."""
    mask = np.zeros_like(binary)
    cv2.circle(mask, (x, y), int(r * 0.6), 255, -1)

    return cv2.countNonZero(
        cv2.bitwise_and(binary, binary, mask=mask)
    )


def preprocess_for_ocr(crop):
    """Prepare crop for OCR."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Upscale for better OCR
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Increase contrast
    clahe = cv2.createCLAHE(2.0, (8, 8))
    gray = clahe.apply(gray)

    # Threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh


def extract_text(crop):
    """Extract text using EasyOCR."""
    processed = preprocess_for_ocr(crop)

    results = reader.readtext(
        processed,
        detail=1,
        paragraph=False,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
    )

    if not results:
        return "", 0.0

    # Choose highest confidence result
    best = max(results, key=lambda x: x[2])

    text = clean_text(best[1])
    conf = float(best[2])

    return text, conf


# =========================================================
# MAIN PROCESS
# =========================================================
for img_path in IMAGES:
    name = os.path.splitext(os.path.basename(img_path))[0]

    out_dir = os.path.join(BASE_OUT, name)
    crop_dir = os.path.join(out_dir, "crops")
    csv_path = os.path.join(out_dir, "circles.csv")
    debug_path = os.path.join(out_dir, "circles_debug.png")

    ensure_dirs(out_dir)

    print(f"\nProcessing: {name}")

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}")
        continue

    h, w = img.shape[:2]
    debug = img.copy()

    # =====================================================
    # PREPROCESSING
    # =====================================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    clahe = cv2.createCLAHE(2.0, (8, 8))
    gray_eq = clahe.apply(gray)

    # Blur
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 1.5)

    # Binary image for filtering
    binary = cv2.adaptiveThreshold(
        gray_eq,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5
    )

    # =====================================================
    # CIRCLE DETECTION (OpenCV)
    # =====================================================
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=60,
        param1=100,
        param2=18,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS
    )

    records = []
    used_centers = []
    idx = 0

    if circles is None:
        print("No circles detected.")
        continue

    circles = np.round(circles[0]).astype(int)
    print(f"Circles found: {len(circles)}")

    # =====================================================
    # PROCESS EACH CIRCLE
    # =====================================================
    for x, y, r in circles:
        # Skip circles touching image borders
        if x < r or y < r or x >= w - r or y >= h - r:
            continue

        # Must contain enough text pixels
        if text_pixels_inside(binary, x, y, r) < MIN_TEXT_PIXELS:
            continue

        # Skip very filled circles (likely noise)
        if fill_ratio(binary, x, y, r) > MAX_FILL_RATIO:
            continue

        # Remove duplicates
        if dedupe(x, y, used_centers):
            continue

        used_centers.append((x, y))

        # Crop with padding
        pad = int(r * PADDING_RATIO)

        x1 = max(0, x - r - pad)
        y1 = max(0, y - r - pad)
        x2 = min(w, x + r + pad)
        y2 = min(h, y + r + pad)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # OCR
        text, ocr_conf = extract_text(crop)

        # Validate tag
        valid_tag = (
            text != "" and
            ocr_conf >= OCR_CONF_THRESHOLD and
            is_valid_tag(text)
        )

        # Save crop
        crop_path = os.path.join(
            crop_dir,
            f"circle_{idx}.png"
        )
        cv2.imwrite(crop_path, crop)

        # Draw circle
        color = (0, 255, 0) if valid_tag else (0, 0, 255)

        cv2.circle(debug, (x, y), r, color, 2)

        label = f"{idx}: {text} ({ocr_conf:.2f})"

        cv2.putText(
            debug,
            label,
            (x - r, max(20, y - r - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        # Save record
        records.append({
            "id": idx,
            "center_x": x,
            "center_y": y,
            "radius": r,
            "text": text,
            "ocr_confidence": round(ocr_conf, 4),
            "valid_tag": valid_tag,
            "crop_path": crop_path
        })

        print(
            f"Circle {idx}: "
            f"text='{text}', "
            f"ocr_conf={ocr_conf:.3f}, "
            f"valid={valid_tag}"
        )

        idx += 1

    # =====================================================
    # SAVE RESULTS
    # =====================================================
    pd.DataFrame(records).to_csv(csv_path, index=False)
    cv2.imwrite(debug_path, debug)

    print(f"\nDetected circles saved: {len(records)}")
    print(f"CSV: {csv_path}")
    print(f"Debug image: {debug_path}")