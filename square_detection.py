
from __future__ import annotations

import os
import re
import sys
import cv2
import numpy as np
import pandas as pd
import pytesseract
from ultralytics import YOLO

# ─────────────────────────── CONFIGURATION ───────────────────────────────────

IMAGES = [
    "output_upscaled/pnid_cropped_upscaled.png"
]

YOLO_MODEL_PATH = "best.pt"          
BASE_OUT        = "squares_output_pnid9"

TESSERACT_PATH  = r"C:\Users\shital.ingole\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Detection thresholds
YOLO_CONF       = 0.10               
MIN_SIZE        = 25                 
MAX_SIZE        = 260                
AREA_MIN        = 500                
DEDUP_DIST      = 80                 
CIRCULARITY_TH  = 0.80               
PAD_RATIO       = 0.35              

# Tesseract config — uniform block, alphanumeric + common P&ID chars
TESS_CONFIG     = r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz-_/. "


# ─────────────────────────── TESSERACT SETUP ─────────────────────────────────

def setup_tesseract() -> bool:
    """
    Point pytesseract at the Windows binary if needed.
    Returns True when Tesseract is usable, False otherwise.
    """
    # Already in PATH (Linux / Mac / correctly configured Windows)
    try:
        pytesseract.get_tesseract_version()
        print("✅ Tesseract found in PATH")
        return True
    except pytesseract.TesseractNotFoundError:
        pass

    # Try the configured Windows path
    if os.path.isfile(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        try:
            pytesseract.get_tesseract_version()
            print(f"✅ Tesseract found at: {TESSERACT_PATH}")
            return True
        except pytesseract.TesseractNotFoundError:
            pass

    # Auto-search common Windows install locations
    candidates = [
        r"C:\Users\shital.ingole\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            try:
                pytesseract.get_tesseract_version()
                print(f"✅ Tesseract auto-detected at: {path}")
                return True
            except pytesseract.TesseractNotFoundError:
                continue

    print("\n" + "="*60)
    print("❌  TESSERACT NOT FOUND")
    print("="*60)
    print("Install Tesseract on Windows:")
    print("  1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    print("  2. Run the installer (keep default path)")
    print("  3. OR set TESSERACT_PATH at the top of this script")
    print("="*60 + "\n")
    return False

# ─────────────────────────── HELPERS ─────────────────────────────────────────

def is_circle(cnt: np.ndarray) -> bool:
    """Return True when a contour is roughly circular (skip it)."""
    area = cv2.contourArea(cnt)
    if area < 200:
        return False
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return False
    return (4 * np.pi * area / (peri * peri)) > CIRCULARITY_TH


def is_duplicate(cx: int, cy: int, used: list[tuple[int, int]]) -> bool:
    """Return True when (cx, cy) is too close to an already-recorded centre."""
    return any(
        abs(cx - x) < DEDUP_DIST and abs(cy - y) < DEDUP_DIST
        for x, y in used
    )


def preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    """
    Clean up a crop so Tesseract reads it reliably:
      • upscale small crops
      • convert to greyscale
      • CLAHE contrast enhancement
      • Otsu threshold → clean black-on-white binary image
    """
    # Upscale if too small
    h, w = crop.shape[:2]
    scale = max(1, int(np.ceil(80 / min(h, w))))
    if scale > 1:
        crop = cv2.resize(crop, (w * scale, h * scale),
                          interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # CLAHE — helps with low-contrast scans
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=15)

    # Otsu binarise (inverted → black text on white)
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate slightly to reconnect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Invert back → black text on white (Tesseract default)
    return cv2.bitwise_not(binary)


def extract_text(crop: np.ndarray) -> str:
    """Run Tesseract on a pre-processed crop and return cleaned text.
    Returns empty string if Tesseract is not available."""
    try:
        processed = preprocess_for_ocr(crop)
        raw = pytesseract.image_to_string(processed, config=TESS_CONFIG)
        # Collapse whitespace, strip junk characters
        text = re.sub(r"[^\w\s\-_/.]", "", raw)
        text = " ".join(text.split())
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        return ""
    except Exception:
        return ""


def padded_crop(img: np.ndarray,
                x1: int, y1: int, x2: int, y2: int,
                pad_ratio: float = PAD_RATIO) -> tuple[np.ndarray, tuple]:
    """Return a padded crop and its absolute coordinates."""
    pad = int(max(x2 - x1, y2 - y1) * pad_ratio)
    X1 = max(0, x1 - pad)
    Y1 = max(0, y1 - pad)
    X2 = min(img.shape[1], x2 + pad)
    Y2 = min(img.shape[0], y2 + pad)
    return img[Y1:Y2, X1:X2].copy(), (X1, Y1, X2, Y2)


# ─────────────────────────── YOLO DETECTION ──────────────────────────────────

def detect_with_yolo(img: np.ndarray,
                     model: YOLO,
                     conf: float = YOLO_CONF
                     ) -> list[tuple[int, int, int, int, float, str]]:
    """
    Run YOLOv8 inference.
    Returns list of (x1, y1, x2, y2, confidence, class_name).
    Prints diagnostics if nothing is found.
    """
    results = model.predict(img, conf=conf, verbose=False)[0]
    detections = []

    # Debug: print all raw boxes before conf filter so you can tune threshold
    all_boxes = results.boxes
    if len(all_boxes) == 0:
        print(f"  ⚠️  YOLO: no boxes at all — check that best.pt was trained on P&ID data")
        print(f"       Image shape: {img.shape}  |  Model classes: {list(model.names.values())}")
    else:
        raw_confs = [float(b.conf[0]) for b in all_boxes]
        print(f"  ℹ️  YOLO raw boxes (before conf={conf} filter): {len(all_boxes)} "
              f"| max_conf={max(raw_confs):.3f} min_conf={min(raw_confs):.3f}")
        print(f"       If max_conf is low, lower YOLO_CONF in config (currently {conf})")

    for box in all_boxes:
        if float(box.conf[0]) < conf:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf_val  = float(box.conf[0])
        cls_name  = model.names[int(box.cls[0])]
        detections.append((x1, y1, x2, y2, conf_val, cls_name))

    return detections


# ─────────────────────────── GEOMETRY FALLBACK ───────────────────────────────

def detect_with_geometry(img: np.ndarray,
                         used: list[tuple[int, int]]
                         ) -> list[tuple[int, int, int, int]]:
    """
    Canny + contour fallback to catch instruments YOLO missed.
    Returns list of (x1, y1, x2, y2).
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray  = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    edges = cv2.Canny(gray, 40, 120)

    contours, _ = cv2.findContours(edges,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < AREA_MIN:
            continue
        if is_circle(cnt):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if not (MIN_SIZE <= w <= MAX_SIZE and MIN_SIZE <= h <= MAX_SIZE):
            continue

        cx, cy = x + w // 2, y + h // 2
        if is_duplicate(cx, cy, used):
            continue

        used.append((cx, cy))
        boxes.append((x, y, x + w, y + h))

    return boxes


# ─────────────────────────── MAIN PIPELINE ───────────────────────────────────

def process_image(img_path: str, model: YOLO) -> None:
    name     = os.path.splitext(os.path.basename(img_path))[0]
    out_dir  = os.path.join(BASE_OUT, name)
    crop_dir = os.path.join(out_dir, "crops")
    csv_out  = os.path.join(out_dir, "instruments.csv")
    debug_out= os.path.join(out_dir, "debug_detections.png")
    os.makedirs(crop_dir, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌  Cannot read image: {img_path}")
        return

    debug  = img.copy()
    used   = []   # centres already recorded
    records= []
    idx    = 0

    # ── Step 1: YOLO detections ───────────────────────────────────────────────
    yolo_detections = detect_with_yolo(img, model)
    print(f"  🔍 YOLO found {len(yolo_detections)} instrument(s)")

    for (x1, y1, x2, y2, conf_val, cls_name) in yolo_detections:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if is_duplicate(cx, cy, used):
            continue
        used.append((cx, cy))

        crop, (X1, Y1, X2, Y2) = padded_crop(img, x1, y1, x2, y2)
        text = extract_text(crop)

        crop_path = os.path.join(crop_dir, f"instrument_{idx}_yolo.png")
        cv2.imwrite(crop_path, crop)

        # Draw GREEN box for YOLO detections
        cv2.rectangle(debug, (X1, Y1), (X2, Y2), (0, 200, 0), 2)
        label = f"{idx}:{cls_name}({conf_val:.2f})"
        cv2.putText(debug, label, (X1, Y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        if text:
            cv2.putText(debug, text[:20], (X1, Y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 160, 0), 1)

        records.append({
            "id"          : idx,
            "source"      : "yolo",
            "class"       : cls_name,
            "confidence"  : round(conf_val, 4),
            "center_x"   : cx,
            "center_y"   : cy,
            "x1": X1, "y1": Y1, "x2": X2, "y2": Y2,
            "width"       : X2 - X1,
            "height"      : Y2 - Y1,
            "ocr_text"   : text,
            "crop_path"  : crop_path,
        })
        idx += 1

    # ── Step 2: Geometry fallback ─────────────────────────────────────────────
    geo_boxes = detect_with_geometry(img, used)
    print(f"  📐 Geometry fallback found {len(geo_boxes)} additional instrument(s)")

    for (x1, y1, x2, y2) in geo_boxes:
        crop, (X1, Y1, X2, Y2) = padded_crop(img, x1, y1, x2, y2)
        text = extract_text(crop)
        cx, cy = (X1 + X2) // 2, (Y1 + Y2) // 2

        crop_path = os.path.join(crop_dir, f"instrument_{idx}_geo.png")
        cv2.imwrite(crop_path, crop)

        # Draw BLUE box for geometry fallback
        cv2.rectangle(debug, (X1, Y1), (X2, Y2), (255, 100, 0), 2)
        cv2.putText(debug, str(idx), (X1, Y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)
        if text:
            cv2.putText(debug, text[:20], (X1, Y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 80, 0), 1)

        records.append({
            "id"          : idx,
            "source"      : "geometry",
            "class"       : "unknown",
            "confidence"  : None,
            "center_x"   : cx,
            "center_y"   : cy,
            "x1": X1, "y1": Y1, "x2": X2, "y2": Y2,
            "width"       : X2 - X1,
            "height"      : Y2 - Y1,
            "ocr_text"   : text,
            "crop_path"  : crop_path,
        })
        idx += 1

    # ── Step 3: Save outputs ──────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(csv_out, index=False)
    cv2.imwrite(debug_out, debug)

    # Print OCR summary
    detected_texts = df[df["ocr_text"].str.strip().astype(bool)]
    print(f"  ✅ {len(records)} total instruments | "
          f"{len(detected_texts)} with OCR text extracted")
    print(f"  📄 CSV  → {csv_out}")
    print(f"  🖼️  Debug → {debug_out}\n")

    if not detected_texts.empty:
        print("  Sample OCR results:")
        for _, row in detected_texts.head(10).iterrows():
            print(f"    [{row['id']:>3}] {row['source']:<8} "
                  f"class={row['class']:<15} text='{row['ocr_text']}'")


# ─────────────────────────── ENTRY POINT ─────────────────────────────────────

def main() -> None:
    print(f"\n{'='*60}")
    print("  P&ID Instrument Detector  (YOLO + Tesseract)")
    print(f"{'='*60}\n")

    # ── Tesseract check (non-fatal — OCR column will be empty if missing) ──
    tess_ok = setup_tesseract()
    if not tess_ok:
        print("⚠️  Continuing WITHOUT OCR — install Tesseract to enable text extraction.\n")

    # ── YOLO model ────────────────────────────────────────────────────────
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(
            f"\n❌ YOLO weights not found: '{YOLO_MODEL_PATH}'\n"
            "   Place best.pt in the same folder as this script "
            f"({os.path.abspath('.')})"
        )

    model = YOLO(YOLO_MODEL_PATH)
    print(f"✅ Loaded YOLO model : {YOLO_MODEL_PATH}")
    print(f"   Classes           : {list(model.names.values())}")
    print(f"   Conf threshold    : {YOLO_CONF}  (lower = more detections)\n")

    for img_path in IMAGES:
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found, skipping: {img_path}")
            continue
        print(f"Processing: {img_path}")
        process_image(img_path, model)

    print("\n✅ All done.")


if __name__ == "__main__":
    main()
