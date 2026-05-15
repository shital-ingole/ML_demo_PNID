import os
import requests
import base64
from PIL import Image
import io
import json
import re
import csv
from dotenv import load_dotenv
import sys

# =========================================================
# ENV & CONFIG
# =========================================================
load_dotenv()

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("❌ NVIDIA_API_KEY missing from .env")

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"

IMAGES_FOLDER = "cropped_symbols_image_1"
CSV_OUTPUT = "output_image_1/instrument_pipe_extraction_image_1.csv"

HEADERS = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# 🔥 MEMORY FIX: Max limits
MAX_PIXELS = 2_000_000  # 2MPix ~ good quality, <50KB base64
MAX_BATCH = 50          # Process in small batches
DOWNSCALE_IF = 4_000_000  # Downscale if >4MPix

CSV_FIELDS = [
    "image",
    "entity_type",
    "tag",
    "instrument_code",
    "loop_number",
    "element_number",
    "measured_variable",
    "function",
    "nominal_size",
    "service",
    "line_number",
    "line_type",
    "spec",
    "confidence",
    "x1", "y1", "x2", "y2"   # 🔥 ADDED
]

# =========================================================
# HELPERS (FIXED FOR MEMORY)
# =========================================================
def image_to_base64(path: str) -> str | None:
    """Memory-safe base64: downscale + JPEG 90% + skip huge images"""
    try:
        with Image.open(path) as img:
            w, h = img.size
            pixels = w * h
            print(f"   📏 {os.path.basename(path)}: {w}x{h} = {pixels:,} pix")
            
            # 🔥 SKIP TOO LARGE (common YOLO bbox error on page edges)
            if pixels > 8_000_000:
                print(f"   ❌ SKIP: Too massive {pixels:,}pix (YOLO bbox fail)")
                return None
            
            # 🔥 DOWNSCALE if needed
            if pixels > DOWNSCALE_IF:
                scale = (DOWNSCALE_IF / pixels)**0.5
                new_w, new_h = int(w * scale), int(h * scale)
                print(f"   🔽 Downscale {w}x{h} → {new_w}x{new_h}")
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # 🔥 Convert + save as JPEG 90% (5-10x smaller than PNG!)
            if img.mode != 'RGB':
                img = img.convert("RGB")
            
            buf = io.BytesIO()
            img.save(
                buf, 
                format="JPEG", 
                quality=90,     # High quality, tiny size
                optimize=True
            )
            buf.seek(0)
            b64_size = len(buf.getvalue()) / 1024  # KB
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            
            print(f"   ✅ Encoded: {b64_size:.1f}KB (was PNG ~{pixels/200:.1f}KB)")
            return b64_str
            
    except Exception as e:
        print(f"   ❌ FAILED {os.path.basename(path)}: {str(e)}")
        return None

def normalize_row(row: dict) -> dict:
    return {field: row.get(field, "") for field in CSV_FIELDS}

def get_image_stats():
    """Debug: show crop sizes"""
    total = skipped = processed = 0
    sizes = []
    for f in os.listdir(IMAGES_FOLDER):
        if f.lower().endswith(".png"):
            total += 1
            path = os.path.join(IMAGES_FOLDER, f)
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    sizes.append(w*h)
                    if w*h > MAX_PIXELS * 4:
                        skipped += 1
                    else:
                        processed += 1
            except:
                skipped += 1
    print(f"📊 STATS: {total} crops | Processed:{processed} | Skip:{skipped} | Max:{max(sizes):,}pix")
    if sizes:
        print(f"   Avg: {sum(sizes)/len(sizes):,.0f} | Top5: {sorted(sizes)[-5:]}")

# =========================================================
# MAIN (BATCHED + SAFE)
# =========================================================
def main():
    if not os.path.exists(IMAGES_FOLDER):
        raise FileNotFoundError(f"❌ {IMAGES_FOLDER} missing! Run pid_crop.py first")
    
    get_image_stats()  # Debug sizes
    
    rows = []
    png_files = [f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith(".png")]
    print(f"🔄 Processing {len(png_files)} crops...")
    
    for i, image_path in enumerate(png_files):
        if i % MAX_BATCH == 0 and i > 0:
            print(f"   ⏳ Batch {i//MAX_BATCH} complete")
        
        full_path = os.path.join(IMAGES_FOLDER, image_path)
        
        # -------------------------------------------------
        # 🔥 READ BBOX FROM FILENAME
        # symbol_12_x1_y1_x2_y2.png
        # -------------------------------------------------
        m = re.search(r'_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)\.png$', image_path)
        if not m:
            print(f"⚠️ SKIP {image_path}: No bbox in filename")
            continue

        x1, y1, x2, y2 = map(int, m.groups())
        bbox_area = (x2-x1) * (y2-y1)
        if bbox_area < 100:  # Tiny noise boxes
            print(f"⚠️ SKIP {image_path}: Tiny bbox {bbox_area}")
            continue

        print(f"🔍 [{i+1}/{len(png_files)}] {image_path} BBOX=({x1},{y1},{x2},{y2})")

        # 🔥 SAFE BASE64 (returns None if OOM/large)
        img_b64 = image_to_base64(full_path)
        if img_b64 is None:
            continue  # Skip this crop

        # API call
        payload = {
            "model": VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract ALL P&ID tags and line text. Return only text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }],
            "max_tokens": 800,
            "temperature": 0
        }

        try:
            res = requests.post(INVOKE_URL, headers=HEADERS, json=payload, timeout=60)
            res.raise_for_status()
            text = res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"   ❌ API ERROR {image_path}: {str(e)}")
            continue

        # -------------------------------------------------
        # SIMPLE TAG PARSING
        # -------------------------------------------------
        for line in text.splitlines():
            line = line.strip()
            if not line or len(line) < 3:
                continue

            parsed = None

            # Instrument (e.g. FT-4576-01, FCV3611-05)
            if re.match(r'^[A-Z]{1,4}[- ]?\d{4}[- ]?\d{2}', line.upper()):
                parsed = {
                    "entity_type": "instrument",
                    "tag": line.upper(),
                    "confidence": 1.0
                }

            # Pipe (e.g. * 2"-GL-4576-031440-X-PP)
            elif re.search(r'\d+"\s*[-A-Z]', line):
                parsed = {
                    "entity_type": "pipe",
                    "tag": line.upper(),
                    "confidence": 1.0
                }

            if parsed:
                # 🔥 ADD BBOX & IMAGE
                parsed["image"] = image_path
                parsed["x1"] = x1
                parsed["y1"] = y1
                parsed["x2"] = x2
                parsed["y2"] = y2
                rows.append(normalize_row(parsed))
                print(f"   → {parsed['entity_type']}: {parsed['tag']}")

    # SAVE
    os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("\n✅ CSV WITH BBOX CREATED (Memory-Safe!)")
    print(f"→ {CSV_OUTPUT}")
    print(f"Rows: {len(rows)} | Processed crops: {len(png_files)}")
    
    # Summary
    inst = len([r for r in rows if r['entity_type']=='instrument'])
    pipes = len(rows) - inst
    print(f"Instruments: {inst} | Pipes: {pipes}")


if __name__ == "__main__":
    main()