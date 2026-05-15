import os
import re
import base64
import time
import requests
import pandas as pd
from dotenv import load_dotenv


load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQUARES_BASE = os.path.join(BASE_DIR, "squares_output")

NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.2-90b-vision-instruct"

VALID_PREFIXES = {
    "LSH","LSL","LSAL","LSDL","LAH","LAL",
    "PSH","PSL","PAH","PAL",
    "TSH","TSL","TSAH","TSAL","TSDH","TSDL",
    "ZSH","ZSL","TI","LI","FI","PI","DI","QI",
    "PSAH","PSAL","LSAH","LSAL","PSDH","PSDL","DPI",
    "XVI","XVSH","XVSL","XVAL","XVAH","AI","AIAH","AIAL",
}

# ================= HELPERS =================
def img_to_url(p):
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

def ocr_square(img_path):
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Read the alarm or switch tag inside the square. "
                        "Return ALL text you see, line by line."
                        "No explanation and no extra text."
                    )
                },
                {"type": "image_url", "image_url": {"url": img_to_url(img_path)}}
            ]
        }],
        "temperature": 0.0,
        "max_tokens": 80
    }
    r = requests.post(
        INVOKE_URL,
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
        json=payload,
        timeout=30
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def normalize(text):
    text = text.upper()
    text = text.replace("-", " ")
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_tag(norm_text):
    tokens = norm_text.split()

    prefix = None
    loop = None
    suffix = None

    # find prefix anywhere
    for t in tokens:
        if t in VALID_PREFIXES:
            prefix = t
            break

    # find loop (3–5 digits)
    for t in tokens:
        if re.fullmatch(r"\d{3,5}", t):
            loop = t
            break

    # find optional suffix (1–2 digits, not same as loop)
    for t in tokens:
        if loop and t != loop and re.fullmatch(r"\d{1,2}", t):
            suffix = t
            break

    if not prefix or not loop:
        return None

    full = f"{prefix} {loop}"
    if suffix:
        full += f" {suffix}"

    return prefix, loop, suffix or "", full

for image_name in sorted(os.listdir(SQUARES_BASE)):
    img_dir = os.path.join(SQUARES_BASE, image_name)
    csv_in  = os.path.join(img_dir, "squares_raw.csv")
    csv_out = os.path.join(img_dir, "squares_valid_tags.csv")

    if not os.path.exists(csv_in):
        continue

    print(f"🔍 Processing squares: {image_name}")

    df = pd.read_csv(csv_in)
    valid = []

    for _, row in df.iterrows():
        crop = row["crop_path"]
        if not os.path.exists(crop):
            continue

        raw = ocr_square(crop)
        norm = normalize(raw)

        result = extract_tag(norm)
        if not result:
            continue

        prefix, loop, suffix, full_tag = result

        out = row.to_dict()
        out.update({
            "ocr_raw": raw,
            "ocr_norm": norm,
            "tag": prefix,
            "loop": loop,
            "suffix": suffix,
            "full_tag": full_tag
        })

        valid.append(out)
        time.sleep(0.25)

    pd.DataFrame(valid).to_csv(csv_out, index=False)
    print(f"✅ {image_name}: {len(valid)} valid square tags")
