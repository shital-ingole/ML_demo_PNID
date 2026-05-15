import os
import re
import base64
import time
import math
import requests
import pandas as pd
import cv2
import numpy as np
from dotenv import load_dotenv

# =========================================================
# CONFIG
# =========================================================
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CIRCLES_BASE = os.path.join(BASE_DIR, "output", "circles_output_pnid8")

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.2-90b-vision-instruct"

# =========================================================
# TAG PREFIXES
# =========================================================
VALID_PREFIXES = {
    "PI", "PIT", "PT", "PAH", "PAL", "PDT",
    "TI", "TIT", "TT", "TC", "TIC", "TAH", "TAL",
    "LI", "LIT", "LT", "LAH", "LAL",
    "FI", "FIT", "FT", "FAH", "FAL",
    "AI", "AIT", "AT", "AIC",
    "FE", "FG", "FR", "FRC",
    "WI", "WT", "WIC",
    "FC", "FCV", "FV",
    "PSV", "PCV", "XV", "XVT",
    "ESD", "SD", "HS", "SS", "SPT"
}

VALVE_PREFIXES = {"FCV", "PSV", "PCV", "XV", "XVT"}

MIN_RADIUS = 10
MAX_RADIUS = 120
DEDUP_DIST = 50

# =========================================================
# IMAGE PREPROCESS
# =========================================================
def preprocess_crop_for_ocr(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 11
    )

    return img, thr

def save_temp_preprocessed(img_path):
    img, thr = preprocess_crop_for_ocr(img_path)
    if img is None:
        return None

    temp_path = img_path.replace(".png", "_ocr.png").replace(".jpg", "_ocr.jpg").replace(".jpeg", "_ocr.jpeg")
    cv2.imwrite(temp_path, thr)
    return temp_path

# =========================================================
# OCR HELPERS
# =========================================================
def img_to_url(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return "data:image/png;base64," + b64

def ocr_circle(img_path, retries=5):
    if not NVIDIA_API_KEY:
        return "NONE"

    prompt = (
        "This is a cropped P&ID symbol tag.\n"
        "Read only the text inside the circle or bubble.\n"
        "Return only the visible tag text, nothing else.\n"
        "Use uppercase. Preserve spaces between prefix and number.\n"
        "If no clear text is visible, return exactly: NONE"
    )

    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": img_to_url(img_path)}}
            ]
        }],
        "temperature": 0.0,
        "max_tokens": 32
    }

    for i in range(retries):
        r = requests.post(
            INVOKE_URL,
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=90
        )

        if r.status_code == 429:
            time.sleep(2 ** i)
            continue

        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].strip()
        return out

    return "NONE"

def normalize(text):
    if not text:
        return "NONE"
    text = text.upper()
    text = text.replace("\n", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "NONE"

def extract_tag(norm):
    if not norm or norm == "NONE":
        return None

    candidates = []

    patterns = [
        r"\b([A-Z]{1,4})\s*([0-9]{1,5})\s*([0-9]{1,2})?\b",
        r"\b([A-Z]{1,4})\s*[-/]?\s*([0-9]{1,5})\s*([0-9]{1,2})?\b",
    ]

    for pat in patterns:
        for m in re.finditer(pat, norm):
            prefix = m.group(1)
            loop = m.group(2)
            suffix = m.group(3) if m.lastindex and m.lastindex >= 3 else None
            if prefix in VALID_PREFIXES and len(loop) >= 3:
                full = f"{prefix} {loop}" + (f" {suffix}" if suffix else "")
                tag_type = "valve" if prefix in VALVE_PREFIXES else "instrument"
                candidates.append((prefix, loop, suffix or "", full, tag_type))

    if candidates:
        return candidates[0]

    for prefix in VALID_PREFIXES:
        m = re.search(rf"\b{prefix}\s*([0-9]{{3,5}})(?:\s*([0-9]{{1,2}}))?\b", norm)
        if m:
            loop = m.group(1)
            suffix = m.group(2) or ""
            full = f"{prefix} {loop}" + (f" {suffix}" if suffix else "")
            tag_type = "valve" if prefix in VALVE_PREFIXES else "instrument"
            return prefix, loop, suffix, full, tag_type

    return None

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def best_crop_path(path):
    p1 = path
    p2 = path.replace(".png", "_ocr.png").replace(".jpg", "_ocr.jpg").replace(".jpeg", "_ocr.jpeg")
    return p2 if os.path.exists(p2) else p1

# =========================================================
# MAIN
# =========================================================
for image_name in sorted(os.listdir(CIRCLES_BASE)):
    img_dir = os.path.join(CIRCLES_BASE, image_name)
    circles_csv = os.path.join(img_dir, "circles.csv")
    out_csv = os.path.join(img_dir, "circles_valid_tags.csv")

    if not os.path.exists(circles_csv):
        continue

    df = pd.read_csv(circles_csv)
    print(f"Scanning {image_name}: {len(df)} circle candidates")

    df = df[(df["radius"] >= MIN_RADIUS) & (df["radius"] <= MAX_RADIUS)].copy()
    print(f"Radius-passed: {len(df)}")

    candidates = []

    for _, r in df.iterrows():
        crop = str(r["crop_path"])
        if not os.path.exists(crop):
            continue

        pre = save_temp_preprocessed(crop)
        use_path = pre if pre and os.path.exists(pre) else crop

        raw = ocr_circle(use_path)
        norm = normalize(raw)
        parsed = extract_tag(norm)

        if not parsed:
            continue

        prefix, loop, suffix, full_tag, tag_type = parsed

        candidates.append({
            "id": int(r["id"]),
            "center_x": int(r["center_x"]),
            "center_y": int(r["center_y"]),
            "radius": int(r["radius"]),
            "crop_path": crop,
            "ocr_path": use_path,
            "ocr_raw": raw,
            "ocr_norm": norm,
            "prefix": prefix,
            "loop": loop,
            "suffix": suffix,
            "full_tag": full_tag,
            "tag_type": tag_type
        })

        time.sleep(0.2)

    print(f"OCR parsed candidates: {len(candidates)}")

    if not candidates:
        pd.DataFrame([]).to_csv(out_csv, index=False)
        continue

    cand_df = pd.DataFrame(candidates)

    cand_df["radius_rank"] = cand_df["radius"]
    cand_df = cand_df.sort_values(["full_tag", "radius_rank"], ascending=[True, True])
    cand_df = cand_df.drop_duplicates(subset=["full_tag"], keep="first")

    kept = []
    used_centers = []

    for r in cand_df.itertuples():
        c = (r.center_x, r.center_y)
        if any(distance(c, u) < DEDUP_DIST for u in used_centers):
            continue
        used_centers.append(c)
        kept.append(r._asdict())

    final_df = pd.DataFrame(kept)
    final_df.to_csv(out_csv, index=False)

    n_inst = int((final_df["tag_type"] == "instrument").sum()) if not final_df.empty else 0
    n_valv = int((final_df["tag_type"] == "valve").sum()) if not final_df.empty else 0

    print(f"{image_name}: {len(final_df)} tags saved ({n_inst} instruments, {n_valv} valves)")

print("DONE")