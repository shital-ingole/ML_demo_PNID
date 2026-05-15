import pytesseract
import cv2
import pandas as pd
import os
import re

VALVE_CSV  = "output_image_2_valves_MAX_RECALL/detected_valves_MAX_RECALL.csv"
CROP_DIR  = "output_image_2_valves_MAX_RECALL/valve_crops"
OUT_CSV   = "output_image_2_valves_MAX_RECALL/valves_ocr.csv"

pytesseract.pytesseract.tesseract_cmd = (
   r"C:\Users\shital.ingole\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

# common valve annotation patterns
VALID_REGEX = re.compile(
    r'(\d+(?:/\d+)?")\s*(NC|NO|LO)|'
    r'(HV|XV|MV|TV|LV|PV)[-\s]?\d+',
    re.IGNORECASE
)

# =========================================================
df = pd.read_csv(VALVE_CSV)
rows = []

for _, r in df.iterrows():
    img_path = os.path.join(CROP_DIR, r.valve_crop)
    img = cv2.imread(img_path)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    text = pytesseract.image_to_string(
        gray,
        config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./\"-"
    ).strip()

    match = VALID_REGEX.search(text)

    rows.append({
        **r.to_dict(),
        "ocr_raw": text,
        "valve_tag": match.group(0) if match else None
    })

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)

print("✅ Valve OCR complete")
print("→", OUT_CSV)
