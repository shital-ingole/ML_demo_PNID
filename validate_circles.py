import os
import re
import base64
import time
import math
import requests
import pandas as pd
from dotenv import load_dotenv

# =========================================================
# CONFIG
# =========================================================
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CIRCLES_BASE = os.path.join(BASE_DIR, "output", "circles_output_pnid8")

NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.2-90b-vision-instruct"

# =========================================================
# VALID CIRCULAR TAG PREFIXES (INSTRUMENTS + VALVES)
# =========================================================
VALID_PREFIXES = {
    # Instruments
    "PI","PIT","PT",
    "TI","TIT","TT",
    "LI","LIT","LT",
    "FI","FIT",
    "AI","AIT",
    "FE","LG","WI","PT","FT","LT"

    # Valves / functions
    "FCV","PSV","XVT"
}

VALVE_PREFIXES = {"FCV", "PSV", "XVT"}

# =========================================================
# RADIUS FILTERS (ONLY GEOMETRY AVAILABLE)
# Tune once per drawing set if needed
# =========================================================
MIN_RADIUS = 25
MAX_RADIUS = 120

DEDUP_DIST = 80  # pixels

# =========================================================
# OCR HELPERS
# =========================================================
def img_to_url(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

def ocr_circle(img_path, retries=5):
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Read ONLY the tag written inside the circle.\n"
                        "If there is no clear tag inside the circle, respond with EXACTLY: NONE\n"
                        "Do NOT guess. Do NOT infer."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": img_to_url(img_path)}
                }
            ]
        }],
        "temperature": 0.0,
        "max_tokens": 60
    }

    for i in range(retries):
        r = requests.post(
            INVOKE_URL,
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
            json=payload,
            timeout=90
        )

        if r.status_code == 429:
            time.sleep(2 ** i)
            continue

        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    return "NONE"

def normalize(text):
    text = text.upper().replace("-", " ")
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_tag(norm):
    if norm == "NONE":
        return None

    tokens = norm.split()

    prefix = next((t for t in tokens if t in VALID_PREFIXES), None)
    loop   = next((t for t in tokens if re.fullmatch(r"\d{4,5}", t)), None)
    suffix = next((t for t in tokens if re.fullmatch(r"\d{1,2}", t)), "")

    if not prefix or not loop:
        return None

    full = f"{prefix} {loop}" + (f" {suffix}" if suffix else "")
    tag_type = "valve" if prefix in VALVE_PREFIXES else "instrument"

    return prefix, loop, suffix, full, tag_type

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

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

    print(f"🔍 {image_name}: scanning {len(df)} circles")

    # -----------------------------------------------------
    # 1️⃣ RADIUS SANITY FILTER
    # -----------------------------------------------------
    df = df[
        (df["radius"] >= MIN_RADIUS) &
        (df["radius"] <= MAX_RADIUS)
    ]

    print(f"   radius-passed circles: {len(df)}")

    candidates = []

    # -----------------------------------------------------
    # 2️⃣ OCR
    # -----------------------------------------------------
    for _, r in df.iterrows():
        crop = r["crop_path"]
        if not os.path.exists(crop):
            continue

        raw = ocr_circle(crop)
        norm = normalize(raw)
        parsed = extract_tag(norm)

        if not parsed:
            continue

        prefix, loop, suffix, full_tag, tag_type = parsed

        candidates.append({
            "id": r["id"],
            "center_x": r["center_x"],
            "center_y": r["center_y"],
            "radius": r["radius"],
            "crop_path": r["crop_path"],
            "ocr_raw": raw,
            "ocr_norm": norm,
            "prefix": prefix,
            "loop": loop,
            "suffix": suffix,
            "full_tag": full_tag,
            "tag_type": tag_type
        })

        time.sleep(0.25)

    print(f"   OCR valid candidates: {len(candidates)}")

    if not candidates:
        continue

    cand_df = pd.DataFrame(candidates)

    # -----------------------------------------------------
    # 3️⃣ TAG DEDUPE (KEEP SMALLEST RADIUS)
    # -----------------------------------------------------
    before = len(cand_df)
    cand_df = cand_df.sort_values("radius", ascending=True)
    cand_df = cand_df.drop_duplicates(subset=["full_tag"], keep="first")
    print(f"   tag dedupe: {before} → {len(cand_df)}")

    # -----------------------------------------------------
    # 4️⃣ SPATIAL DEDUPE
    # -----------------------------------------------------
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

    n_inst = (final_df["tag_type"] == "instrument").sum()
    n_valv = (final_df["tag_type"] == "valve").sum()

    print(
        f"✅ {image_name}: {len(final_df)} total circles "
        f"({n_inst} instruments, {n_valv} valves)\n"
    )

print("🏁 DONE — CIRCULAR TAG EXTRACTION COMPLETE")
