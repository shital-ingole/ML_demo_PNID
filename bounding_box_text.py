import cv2
import numpy as np
import pandas as pd
import pytesseract
import re
import os

# =========================================================
# CONFIG
# =========================================================
IMAGE_PATH = "output_upscaled/pnid_cropped_upscaled.png"
BOXES_CSV = "output_final_v1/circle_boxes.csv"
OUT_CSV = "output_final_v1/circle_tags.csv"
DEBUG_DIR = "output_final_v1/tag_debug"

os.makedirs(DEBUG_DIR, exist_ok=True)
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\shital.ingole\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)
# =========================================================
# TAG NORMALIZATION
# =========================================================
VALID_PREFIXES = {
    "PI","PT","PIT",
    "TI","TT","TIT",
    "FI","FT","FIT",
    "LI","LT","LIT",
    "AI","AIT",
    "FC","FIC","LC","LIC","PC","PIC",
    "PSV","ESD",    "PI","PIT","PT",
    "TI","TIT","TT",
    "LI","LIT","LT",
    "FI","FIT",
    "AI","AIT",
    "FE","LG","WI","PT","FT","LT"
}

def normalize(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_tag(text):
    tokens = text.split()

    prefix = next((t for t in tokens if t in VALID_PREFIXES), None)
    number = next((t for t in tokens if re.fullmatch(r"\d{2,5}", t)), None)

    if not prefix or not number:
        return None

    return f"{prefix} {number}"

# =========================================================
# LOAD DATA
# =========================================================
img = cv2.imread(IMAGE_PATH)
df = pd.read_csv(BOXES_CSV)

records = []

# =========================================================
# PROCESS EACH BOX
# =========================================================
for _, r in df.iterrows():
    x1, y1, x2, y2 = map(int, [r.x1, r.y1, r.x3, r.y3])

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    h, w = crop.shape[:2]
    cx, cy = w // 2, h // 2
    radius = int(min(w, h) * 0.45)

    # --- circular mask ---
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=mask)

    # binarize
    bw = cv2.adaptiveThreshold(
        masked, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )

    # OCR
    text = pytesseract.image_to_string(
        bw,
        config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    norm = normalize(text)
    tag = extract_tag(norm)

    # debug save
    debug_path = os.path.join(DEBUG_DIR, f"{int(r.id)}.png")
    cv2.imwrite(debug_path, bw)

    records.append({
        "id": r.id,
        "cx": x1 + cx,
        "cy": y1 + cy,
        "radius": radius,
        "x1": x1, "y1": y1,
        "x2": x2, "y2": y2,
        "ocr_raw": text.strip(),
        "ocr_norm": norm,
        "tag": tag
    })

# =========================================================
# SAVE
# =========================================================
out_df = pd.DataFrame(records)
out_df.to_csv(OUT_CSV, index=False)

print(f"✅ Extracted {len(out_df)} symbols")
print("📄 Saved:", OUT_CSV)
print("🖼 Debug crops:", DEBUG_DIR)
