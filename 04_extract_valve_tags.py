import pandas as pd
import re

IN_CSV  = "output_image_2_valves_MAX_RECALL/valves_ocr_clean.csv"
OUT_CSV = "output_image_2_valves_MAX_RECALL/valves_with_tags.csv"

df = pd.read_csv(IN_CSV)

# ---------------- REGEX ----------------
SIZE_RE = re.compile(r'(\d+(?:/\d+)?)\s*"', re.I)
COND_RE = re.compile(r'\b(NC|NO|LO)\b', re.I)
TAG_RE  = re.compile(r'\b(HV|XV|MV|TV|LV|PV)[-\s]?\d+', re.I)

# ---------------- EXTRACTOR ----------------
def extract(row):
    text = row.get("ocr_clean", "")

    # 🔥 critical fix
    if pd.isna(text):
        text = ""
    else:
        text = str(text)

    size = SIZE_RE.search(text)
    cond = COND_RE.search(text)
    tag  = TAG_RE.search(text)

    return pd.Series({
        "valve_size": size.group(1) + '"' if size else None,
        "valve_condition": cond.group(1).upper() if cond else None,
        "valve_tag": tag.group(0) if tag else None
    })

# ---------------- APPLY ----------------
tags = df.apply(extract, axis=1)
df = pd.concat([df, tags], axis=1)

df.to_csv(OUT_CSV, index=False)

print("✅ Valve tags extracted successfully")
print("→", OUT_CSV)
