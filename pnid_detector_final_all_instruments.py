
# from __future__ import annotations

# import anthropic
# import base64
# import json
# import os
# import re
# import time
# import cv2
# import numpy as np
# import pandas as pd
# import pytesseract

# # ═══════════════════════════════ CONFIGURATION ═══════════════════════════════

# INPUT_IMAGE     = "output_upscaled/pnid_cropped_upscaled.png"
# OUTPUT_DIR      = "pnid_output_all_instruments"
# DEBUG_IMG_OUT   = os.path.join(OUTPUT_DIR, "detection_debug.png")
# CSV_OUT         = os.path.join(OUTPUT_DIR, "instruments.csv")
# CROPS_DIR       = os.path.join(OUTPUT_DIR, "crops")

# # Tesseract path — your Windows install
# TESSERACT_PATH  = r"C:\Users\shital.ingole\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# # Anthropic API key — set here OR as environment variable ANTHROPIC_API_KEY
# ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")

# # ── Geometry detection parameters ────────────────────────────────────────────
# MIN_W, MAX_W    = 18, 200      
# MIN_H, MAX_H    = 18, 200      
# AREA_MIN        = 300           
# CIRCULARITY_TH  = 0.78         
# DEDUP_DIST      = 40           
# ASPECT_MIN      = 0.35        
# ASPECT_MAX      = 2.80         
# PAD_PX          = 12            

# # ── Tesseract config ──────────────────────────────────────────────────────────
# TESS_CFG        = r"--oem 3 --psm 6"

# # ── Claude model ──────────────────────────────────────────────────────────────
# CLAUDE_MODEL    = "claude-opus-4-5"


# # ═══════════════════════════════ SETUP ═══════════════════════════════════════

# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(CROPS_DIR,  exist_ok=True)

# # Configure Tesseract
# if os.path.isfile(TESSERACT_PATH):
#     pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# # ═══════════════════════════════ HELPERS ═════════════════════════════════════

# def is_circle(cnt: np.ndarray) -> bool:
#     area = cv2.contourArea(cnt)
#     if area < 100:
#         return False
#     peri = cv2.arcLength(cnt, True)
#     return peri > 0 and (4 * np.pi * area / (peri ** 2)) > CIRCULARITY_TH


# def is_duplicate(cx: int, cy: int, used: list) -> bool:
#     return any(abs(cx - x) < DEDUP_DIST and abs(cy - y) < DEDUP_DIST
#                for x, y in used)


# def safe_crop(img: np.ndarray, x1: int, y1: int,
#               x2: int, y2: int, pad: int = PAD_PX):
#     H, W = img.shape[:2]
#     X1, Y1 = max(0, x1 - pad), max(0, y1 - pad)
#     X2, Y2 = min(W, x2 + pad), min(H, y2 + pad)
#     return img[Y1:Y2, X1:X2].copy(), (X1, Y1, X2, Y2)


# def img_to_b64(img: np.ndarray) -> str:
#     _, buf = cv2.imencode(".png", img)
#     return base64.b64encode(buf).decode()


# # ═══════════════════════════════ STAGE 1 — GEOMETRY DETECTION ════════════════

# def detect_instruments(img: np.ndarray) -> list[dict]:
#     """
#     Multi-pass geometry detection:
#       Pass A — Canny edges on full image
#       Pass B — adaptive threshold to catch faint/dashed borders
#     Returns list of raw box dicts.
#     """
#     H, W = img.shape[:2]
#     gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # ── enhance contrast ─────────────────────────────────────────────────────
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     enh   = clahe.apply(gray)

#     used  = []  
#     boxes = []   
#     def try_add(cnt):
#         if cv2.contourArea(cnt) < AREA_MIN:
#             return
#         if is_circle(cnt):
#             return
#         x, y, w, h = cv2.boundingRect(cnt)
#         if not (MIN_W <= w <= MAX_W and MIN_H <= h <= MAX_H):
#             return
#         ar = w / h
#         if not (ASPECT_MIN <= ar <= ASPECT_MAX):
#             return
#         cx, cy = x + w // 2, y + h // 2
#         if is_duplicate(cx, cy, used):
#             return
#         # Extra check: reject if region is mostly empty (no ink)
#         roi = gray[y:y+h, x:x+w]
#         _, bw = cv2.threshold(roi, 0, 255,
#                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#         ink_ratio = np.sum(bw > 0) / (w * h)
#         if ink_ratio < 0.04:          # < 4 % pixels are dark → skip
#             return
#         used.append((cx, cy))
#         boxes.append((x, y, x + w, y + h))

#     # Pass A — Canny
#     edges = cv2.Canny(enh, 30, 100)
#     cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
#                                 cv2.CHAIN_APPROX_SIMPLE)
#     for c in cnts:
#         try_add(c)

#     # Pass B — adaptive threshold (catches boxes Canny misses)
#     thr = cv2.adaptiveThreshold(enh, 255,
#                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                  cv2.THRESH_BINARY_INV, 15, 4)
#     cnts2, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
#                                  cv2.CHAIN_APPROX_SIMPLE)
#     for c in cnts2:
#         try_add(c)

#     print(f"  📐 Geometry: {len(boxes)} instrument candidates found")
#     return boxes


# # ═══════════════════════════════ STAGE 2 — TESSERACT OCR ═════════════════════

# def preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
#     """Upscale → CLAHE → denoise → binarise → dilate."""
#     h, w = crop.shape[:2]
#     scale = max(1, int(np.ceil(96 / min(h, w, 1))))
#     if scale > 1:
#         crop = cv2.resize(crop, (w * scale, h * scale),
#                           interpolation=cv2.INTER_CUBIC)
#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     gray = cv2.createCLAHE(3.0, (8, 8)).apply(gray)
#     gray = cv2.fastNlMeansDenoising(gray, h=12)
#     _, bw = cv2.threshold(gray, 0, 255,
#                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#     bw = cv2.dilate(bw, kernel, iterations=1)
#     return cv2.bitwise_not(bw)


# def tesseract_ocr(crop: np.ndarray) -> str:
#     try:
#         proc = preprocess_for_ocr(crop)
#         raw  = pytesseract.image_to_string(proc, config=TESS_CFG)
#         text = re.sub(r"[^\w\-/.]", " ", raw)
#         return " ".join(text.split()).strip()
#     except Exception:
#         return ""


# # ═══════════════════════════════ STAGE 3 — CLAUDE VISION ═════════════════════

# PNID_TAGS = [
#     "FC","FT","FI","FAL","FAH","FIC",
#     "PT","PI","PAH","PAL","PIC","PC",
#     "TC","TT","TI","TAH","TAL","TIC",
#     "LC","LT","LI","LAH","LAL","LIC",
#     "WI","WT",
#     "ESD","ESD",
#     "CV","FCV","PCV","TCV","LCV",
# ]
# TAG_HINT = ", ".join(sorted(set(PNID_TAGS)))


# def claude_extract_tag(client: anthropic.Anthropic,
#                        crop: np.ndarray,
#                        tess_text: str,
#                        crop_idx: int) -> dict:
#     """
#     Send the crop to Claude and ask for the exact instrument tag.
#     Returns dict with keys: tag, instrument_type, number, confidence, reasoning
#     """
#     b64 = img_to_b64(crop)

#     prompt = f"""You are a P&ID (Piping and Instrumentation Diagram) expert.

# I will show you a small cropped image of ONE instrument bubble/box from a P&ID diagram.
# Your task: read the EXACT tag text written inside the instrument box.

# Instrument tags follow this pattern:
#   - First line : instrument type abbreviation  e.g. FC, FT, PT, TC, LC, ESD, PAH, FAL …
#   - Second line: 3-digit number  e.g. 001, 002, 003 …
#   Combined tag example: FC-001, PT-001, ESD-007, PAH-006

# Common abbreviations seen in this diagram: {TAG_HINT}

# Tesseract (automated OCR) already attempted to read this crop and got: "{tess_text}"
# Use that as a hint but trust the image over Tesseract if they disagree.

# Respond ONLY with a JSON object — no extra text:
# {{
#   "tag": "XX-NNN",
#   "instrument_type": "FC",
#   "number": "001",
#   "confidence": "high|medium|low",
#   "reasoning": "short explanation"
# }}

# If the image is blank / not an instrument box, return:
# {{"tag": "", "instrument_type": "", "number": "", "confidence": "low", "reasoning": "not an instrument"}}
# """

#     try:
#         resp = client.messages.create(
#             model=CLAUDE_MODEL,
#             max_tokens=200,
#             messages=[{
#                 "role": "user",
#                 "content": [
#                     {
#                         "type" : "image",
#                         "source": {
#                             "type"       : "base64",
#                             "media_type" : "image/png",
#                             "data"       : b64,
#                         },
#                     },
#                     {"type": "text", "text": prompt},
#                 ],
#             }],
#         )
#         raw = resp.content[0].text.strip()
#         # Strip markdown fences if present
#         raw = re.sub(r"```(?:json)?|```", "", raw).strip()
#         return json.loads(raw)
#     except json.JSONDecodeError:
#         return {
#             "tag": tess_text, "instrument_type": "", "number": "",
#             "confidence": "low", "reasoning": "json parse error"
#         }
#     except Exception as e:
#         return {
#             "tag": tess_text, "instrument_type": "", "number": "",
#             "confidence": "low", "reasoning": str(e)
#         }


# def claude_verify_full_image(client: anthropic.Anthropic,
#                               img: np.ndarray,
#                               records: list[dict]) -> list[dict]:
#     """
#     Final pass: send the FULL annotated image to Claude and ask it to
#     verify / correct every tag in the CSV in one shot.
#     """
#     print("\n  🤖 Claude full-image verification pass …")
#     b64 = img_to_b64(img)

#     draft_table = "\n".join(
#         f"  id={r['id']}  tag={r['claude_tag'] or r['tess_text']}  "
#         f"pos=({r['cx']},{r['cy']})"
#         for r in records
#     )

#     prompt = f"""You are a P&ID expert reviewing an instrument detection result

# Below is the FULL P&ID image with numbered green rectangles drawn around detected instruments.

# I have already extracted these tags (may contain errors):
# {draft_table}

# Please:
# 1. Look at each numbered box in the image.
# 2. Read the EXACT tag text visible inside.
# 3. Return a corrected JSON array — one entry per instrument id:

# [
#   {{"id": 0, "tag": "FC-001", "instrument_type": "FC", "number": "001"}},
#   {{"id": 1, "tag": "PT-001", "instrument_type": "PT", "number": "001"}},
#   ...
# ]

# Rules:
# - Keep the same id numbers.
# - If a box is not an instrument, use "tag": "".
# - Format tags as TYPE-NNN  (e.g. FC-001, ESD-007).
# - Return ONLY the JSON array, no other text.
# """

#     try:
#         resp = client.messages.create(
#             model=CLAUDE_MODEL,
#             max_tokens=2000,
#             messages=[{
#                 "role": "user",
#                 "content": [
#                     {
#                         "type" : "image",
#                         "source": {
#                             "type"       : "base64",
#                             "media_type" : "image/png",
#                             "data"       : b64,
#                         },
#                     },
#                     {"type": "text", "text": prompt},
#                 ],
#             }],
#         )
#         raw = resp.content[0].text.strip()
#         raw = re.sub(r"```(?:json)?|```", "", raw).strip()
#         corrections = json.loads(raw)

#         # Apply corrections
#         correction_map = {c["id"]: c for c in corrections}
#         for r in records:
#             if r["id"] in correction_map:
#                 c = correction_map[r["id"]]
#                 r["final_tag"]         = c.get("tag", r["claude_tag"])
#                 r["final_type"]        = c.get("instrument_type", "")
#                 r["final_number"]      = c.get("number", "")
#                 r["verification"]      = "claude_full_pass"
#         print(f"  ✅ Full-image verification applied to {len(corrections)} instruments")
#     except Exception as e:
#         print(f"  ⚠️  Full-image verification failed: {e}")
#         for r in records:
#             if "final_tag" not in r:
#                 r["final_tag"]    = r.get("claude_tag", r["tess_text"])
#                 r["final_type"]   = r.get("claude_type", "")
#                 r["final_number"] = r.get("claude_number", "")
#                 r["verification"] = "claude_crop_only"

#     return records


# # ═══════════════════════════════ MAIN PIPELINE ═══════════════════════════════

# def run(image_path: str) -> None:
#     print(f"\n{'═'*62}")
#     print(f"  P&ID Instrument Detector — {os.path.basename(image_path)}")
#     print(f"{'═'*62}\n")

#     # ── Load image ────────────────────────────────────────────────────────────
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Cannot read: {image_path}")
#     print(f"  Image size: {img.shape[1]}×{img.shape[0]} px")

#     # ── Init Claude client ────────────────────────────────────────────────────
#     if ANTHROPIC_API_KEY == "YOUR_API_KEY_HERE":
#         raise ValueError(
#             "Set your Anthropic API key:\n"
#             "  export ANTHROPIC_API_KEY=sk-ant-...   (Linux/Mac)\n"
#             "  set ANTHROPIC_API_KEY=sk-ant-...      (Windows CMD)\n"
#             "  or edit ANTHROPIC_API_KEY in this script"
#         )
#     client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

#     # ── Stage 1: detect boxes ─────────────────────────────────────────────────
#     boxes = detect_instruments(img)

#     # ── Stage 2 + 3: OCR & Claude per crop ───────────────────────────────────
#     records      = []
#     debug        = img.copy()
#     annot_small  = img.copy()   # clean copy for full-image Claude pass

#     print(f"\n  Processing {len(boxes)} crops …\n")

#     for i, (x1, y1, x2, y2) in enumerate(boxes):
#         crop, (X1, Y1, X2, Y2) = safe_crop(img, x1, y1, x2, y2)
#         cx, cy = (X1 + X2) // 2, (Y1 + Y2) // 2

#         # Save crop
#         crop_path = os.path.join(CROPS_DIR, f"inst_{i:03d}.png")
#         cv2.imwrite(crop_path, crop)

#         # Tesseract
#         tess = tesseract_ocr(crop)

#         # Claude per-crop
#         llm = claude_extract_tag(client, crop, tess, i)
#         time.sleep(0.15)   # gentle rate-limiting

#         rec = {
#             "id"            : i,
#             "cx"            : cx,
#             "cy"            : cy,
#             "x1": X1, "y1": Y1, "x2": X2, "y2": Y2,
#             "tess_text"     : tess,
#             "claude_tag"    : llm.get("tag", ""),
#             "claude_type"   : llm.get("instrument_type", ""),
#             "claude_number" : llm.get("number", ""),
#             "llm_confidence": llm.get("confidence", ""),
#             "llm_reasoning" : llm.get("reasoning", ""),
#             "crop_path"     : crop_path,
#         }
#         records.append(rec)

#         tag_label = llm.get("tag") or tess or "?"
#         conf      = llm.get("confidence", "")
#         color     = (0, 200, 0) if conf == "high" else \
#                     (0, 165, 255) if conf == "medium" else \
#                     (0, 0, 220)

#         # Draw on debug image
#         cv2.rectangle(debug, (X1, Y1), (X2, Y2), color, 2)
#         cv2.putText(debug, f"{i}", (X1, Y1 - 14),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
#         cv2.putText(debug, tag_label, (X1, Y2 + 13),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

#         # Draw ID on annotation copy (for full-image pass)
#         cv2.rectangle(annot_small, (X1, Y1), (X2, Y2), (0, 200, 0), 1)
#         cv2.putText(annot_small, str(i), (X1, Y1 - 4),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)

#         print(f"    [{i:>3}] tess={tess!r:<18} claude={tag_label!r:<12} ({conf})")

#     # ── Stage 3b: full-image verification ────────────────────────────────────
#     records = claude_verify_full_image(client, annot_small, records)

#     # Re-draw final tags on debug image
#     for r in records:
#         ft = r.get("final_tag", "")
#         if ft:
#             cv2.putText(debug, ft,
#                         (r["x1"], r["y2"] + 26),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.40,
#                         (180, 0, 180), 1)

#     # ── Save outputs ──────────────────────────────────────────────────────────
#     cv2.imwrite(DEBUG_IMG_OUT, debug)

#     # Build final CSV
#     rows = []
#     for r in records:
#         rows.append({
#             "id"              : r["id"],
#             "final_tag"       : r.get("final_tag", r["claude_tag"]),
#             "instrument_type" : r.get("final_type",   r["claude_type"]),
#             "number"          : r.get("final_number", r["claude_number"]),
#             "tess_ocr"        : r["tess_text"],
#             "claude_crop_tag" : r["claude_tag"],
#             "llm_confidence"  : r["llm_confidence"],
#             "verification"    : r.get("verification", ""),
#             "center_x"        : r["cx"],
#             "center_y"        : r["cy"],
#             "x1": r["x1"], "y1": r["y1"],
#             "x2": r["x2"], "y2": r["y2"],
#             "crop_path"       : r["crop_path"],
#         })

#     df = pd.DataFrame(rows)
#     df.to_csv(CSV_OUT, index=False)

#     # ── Summary ───────────────────────────────────────────────────────────────
#     tagged = df[df["final_tag"].str.strip().astype(bool)]
#     print(f"\n{'═'*62}")
#     print(f"  ✅ Total instruments detected : {len(df)}")
#     print(f"  ✅ Tags successfully extracted: {len(tagged)}")
#     print(f"\n  📄 CSV         → {CSV_OUT}")
#     print(f"  🖼️  Debug image → {DEBUG_IMG_OUT}")
#     print(f"  📁 Crops       → {CROPS_DIR}/")
#     print(f"{'═'*62}\n")

#     print("  Final tag list:")
#     print(f"  {'ID':<5} {'Final Tag':<14} {'Type':<8} {'Number':<8} {'Confidence'}")
#     print("  " + "─" * 50)
#     for _, row in df.iterrows():
#         if row["final_tag"]:
#             print(f"  {int(row['id']):<5} {row['final_tag']:<14} "
#                   f"{row['instrument_type']:<8} {row['number']:<8} "
#                   f"{row['llm_confidence']}")


# # ═══════════════════════════════ ENTRY POINT ═════════════════════════════════

# if __name__ == "__main__":
#     run(INPUT_IMAGE)


import cv2
import numpy as np
from typing import List, Dict, Tuple
import pytesseract
from dataclasses import dataclass
import json

@dataclass
class Instrument:
    """Class to store instrument information"""
    tag: str
    type: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    center: Tuple[int, int]

class PnIDInstrumentDetector:
    """
    End-to-end P&ID Instrument Detection System
    Detects instruments like transmitters, controllers, valves, motors, etc.
    """
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = None
        self.gray = None
        self.instruments = []
        
    def load_image(self):
        """Load and preprocess the image"""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
    def detect_circles(self) -> List[Dict]:
        """Detect circular instruments (transmitters, controllers, etc.)"""
        circles_list = []
        
        # Apply different preprocessing for better circle detection
        blurred = cv2.GaussianBlur(self.gray, (9, 9), 2)
        
        # Enhanced parameter sets for better detection
        param_sets = [
            {'dp': 1, 'minDist': 20, 'param1': 50, 'param2': 25, 'minRadius': 8, 'maxRadius': 35},
            {'dp': 1, 'minDist': 15, 'param1': 80, 'param2': 20, 'minRadius': 10, 'maxRadius': 30},
            {'dp': 1.2, 'minDist': 25, 'param1': 60, 'param2': 30, 'minRadius': 12, 'maxRadius': 40},
        ]
        
        all_circles = []
        for params in param_sets:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=params['dp'],
                minDist=params['minDist'],
                param1=params['param1'],
                param2=params['param2'],
                minRadius=params['minRadius'],
                maxRadius=params['maxRadius']
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                all_circles.extend(circles)
        
        # Remove duplicate circles
        unique_circles = self._remove_duplicate_circles(all_circles, threshold=15)
        
        for (x, y, r) in unique_circles:
            bbox = (x - r, y - r, 2*r, 2*r)
            circles_list.append({
                'center': (x, y),
                'radius': r,
                'bbox': bbox,
                'type': 'circular_instrument'
            })
        
        return circles_list
    
    def _remove_duplicate_circles(self, circles: List, threshold: int = 15) -> List:
        """Remove duplicate/overlapping circles"""
        if len(circles) == 0:
            return []
        
        circles = np.array(circles)
        unique = []
        used = set()
        
        for i, circle in enumerate(circles):
            if i in used:
                continue
            unique.append(circle)
            x1, y1, r1 = circle
            
            for j, other in enumerate(circles[i+1:], i+1):
                x2, y2, r2 = other
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < threshold:
                    used.add(j)
        
        return unique
    
    def detect_rectangles(self) -> List[Dict]:
        """Detect rectangular instruments (motors, equipment boxes, etc.)"""
        rectangles = []
        
        # Edge detection with adjusted parameters
        edges = cv2.Canny(self.gray, 30, 100, apertureSize=3)
        
        # Dilate to connect broken edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for rectangles (4 vertices)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h if h > 0 else 0
                area = cv2.contourArea(contour)
                
                # Filter based on size and aspect ratio
                if (100 < area < 8000 and 
                    0.2 < aspect_ratio < 5.0 and 
                    w > 10 and h > 10):
                    rectangles.append({
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'type': 'rectangular_instrument',
                        'aspect_ratio': aspect_ratio
                    })
        
        return rectangles
    
    def detect_valves(self) -> List[Dict]:
        """Detect valve symbols (triangular shapes, butterfly, etc.)"""
        valves = []
        
        # Edge detection
        edges = cv2.Canny(self.gray, 20, 80)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            epsilon = 0.03 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Triangular valves (3 vertices)
            if len(approx) == 3:
                x, y, w, h = cv2.boundingRect(approx)
                area = cv2.contourArea(contour)
                
                if 50 < area < 2000 and w > 8 and h > 8:
                    valves.append({
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'type': 'valve',
                        'subtype': 'control_valve'
                    })
            
            # Diamond/Butterfly valves (4 vertices with specific orientation)
            elif len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h if h > 0 else 0
                area = cv2.contourArea(contour)
                
                if (30 < area < 800 and 
                    0.6 < aspect_ratio < 1.4 and 
                    w > 8 and h > 8 and w < 50 and h < 50):
                    valves.append({
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'type': 'valve',
                        'subtype': 'isolation_valve'
                    })
        
        return valves
    
    def detect_vessels_tanks(self) -> List[Dict]:
        """Detect vessels and tanks (large rectangles or cylinders)"""
        vessels = []
        
        # Find contours
        edges = cv2.Canny(self.gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Large vessels have significant area
            if area > 10000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(h) / w if w > 0 else 0
                
                # Vertical vessels/reactors
                if aspect_ratio > 1.2 and w > 80:
                    vessels.append({
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'type': 'vessel',
                        'subtype': 'vertical_vessel',
                        'area': area
                    })
                # Horizontal vessels
                elif aspect_ratio < 0.8 and h > 80:
                    vessels.append({
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'type': 'vessel',
                        'subtype': 'horizontal_vessel',
                        'area': area
                    })
        
        return vessels
    
    def detect_text_boxes(self) -> List[Dict]:
        """Detect text/label boxes using MSER or connected components"""
        text_boxes = []
        
        # MSER detector for text regions
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(self.gray)
        
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            aspect_ratio = float(w) / h if h > 0 else 0
            area = w * h
            
            # Filter text-like regions
            if (100 < area < 3000 and 
                1.5 < aspect_ratio < 8.0 and 
                10 < w < 200 and 5 < h < 50):
                text_boxes.append({
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2),
                    'type': 'text_label'
                })
        
        return text_boxes
    
    def extract_instrument_tags(self, detected_items: List[Dict]) -> List[Instrument]:
        """Extract instrument tags from detected regions using OCR"""
        instruments = []
        
        for idx, item in enumerate(detected_items):
            bbox = item['bbox']
            x, y, w, h = bbox
            
            # Expand ROI for better OCR
            margin = 15
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(self.image.shape[1], x + w + margin)
            y2 = min(self.image.shape[0], y + h + margin)
            
            roi = self.gray[y1:y2, x1:x2]
            
            # OCR to extract text
            text = self._ocr_region(roi)
            
            # Parse instrument tag
            tag = self._parse_instrument_tag(text)
            
            instrument = Instrument(
                tag=tag if tag else f"INST_{idx:03d}",
                type=item['type'],
                bbox=bbox,
                confidence=0.8,
                center=item.get('center', (x + w//2, y + h//2))
            )
            
            instruments.append(instrument)
        
        return instruments
    
    def _ocr_region(self, roi: np.ndarray) -> str:
        """Perform OCR on a region"""
        try:
            if roi.size == 0:
                return ""
            
            # Preprocess for better OCR
            _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
            
            # Configure tesseract
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_'
            text = pytesseract.image_to_string(denoised, config=custom_config)
            
            return text.strip()
        except Exception as e:
            return ""
    
    def _parse_instrument_tag(self, text: str) -> str:
        """Parse and clean instrument tag from OCR text"""
        import re
        
        # Common P&ID instrument tag patterns
        patterns = [
            r'[A-Z]{2,4}[-_]\d{3,4}[A-Z]?',  # FT-101, PAH-001
            r'[A-Z]{1,2}[-_]\d{2,4}',         # P-101, V-105
            r'[A-Z]+\d{2,4}',                 # M, FC001
            r'R[-_]\d{3,4}',                  # R-104 (reactor)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        # Clean and return text
        cleaned = re.sub(r'[^A-Z0-9-_]', '', text.upper())
        return cleaned[:15] if cleaned else ""
    
    def detect_all_instruments(self) -> List[Instrument]:
        """Main detection pipeline"""
        print("Loading image...")
        self.load_image()
        
        all_detected = []
        
        print("Detecting circular instruments...")
        circles = self.detect_circles()
        print(f"  Found {len(circles)} circular instruments")
        all_detected.extend(circles)
        
        print("Detecting rectangular instruments...")
        rectangles = self.detect_rectangles()
        print(f"  Found {len(rectangles)} rectangular instruments")
        all_detected.extend(rectangles)
        
        print("Detecting valves...")
        valves = self.detect_valves()
        print(f"  Found {len(valves)} valves")
        all_detected.extend(valves)
        
        print("Detecting vessels and tanks...")
        vessels = self.detect_vessels_tanks()
        print(f"  Found {len(vessels)} vessels")
        all_detected.extend(vessels)
        
        print("Detecting text labels...")
        text_boxes = self.detect_text_boxes()
        print(f"  Found {len(text_boxes)} text labels")
        all_detected.extend(text_boxes)
        
        print("Extracting instrument tags...")
        self.instruments = self.extract_instrument_tags(all_detected)
        
        return self.instruments
    
    def visualize_results(self, output_path: str = 'detected_instruments.jpg'):

        """Visualize detection results with bounding boxes"""
        result_image = self.image.copy()
        
        # Color scheme for different instrument types
        colors = {
            'circular_instrument': (0, 255, 0),       # Green
            'rectangular_instrument': (255, 0, 0),    # Blue
            'valve': (0, 0, 255),                     # Red
            'vessel': (255, 0, 255),                  # Magenta
            'text_label': (255, 255, 0),              # Cyan
        }
        
        for instrument in self.instruments:
            x, y, w, h = instrument.bbox
            color = colors.get(instrument.type, (255, 255, 255))
            
            # Draw bounding box with thicker lines
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)
            
            # Draw center point
            cv2.circle(result_image, instrument.center, 5, color, -1)
            
            # Draw label with background for better visibility
            label = f"{instrument.tag}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            
            # Get text size for background
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(result_image, 
                         (x, y - text_height - 8), 
                         (x + text_width + 4, y - 2), 
                         color, -1)
            
            # Draw text
            cv2.putText(result_image, label, (x + 2, y - 5),
                       font, font_scale, (255, 255, 255), thickness)
        
        cv2.imwrite(output_path, result_image)


        print(f"\nVisualization saved to {output_path}")
        
        return result_image
    
    def export_results(self, output_json: str = 'instruments.json'):
        """Export detection results to JSON"""
        results = []
        
        for instrument in self.instruments:
            results.append({
                'tag': instrument.tag,
                'type': instrument.type,
                'bbox': {
                    'x': int(instrument.bbox[0]),
                    'y': int(instrument.bbox[1]),
                    'width': int(instrument.bbox[2]),
                    'height': int(instrument.bbox[3])
                },
                'center': {
                    'x': int(instrument.center[0]),
                    'y': int(instrument.center[1])
                },
                'confidence': float(instrument.confidence)
            })
        
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results exported to {output_json}")
        
        return results
    
    def print_summary(self):
        """Print detection summary"""
        print("\n" + "="*70)
        print(" "*20 + "INSTRUMENT DETECTION SUMMARY")
        print("="*70)
        
        type_counts = {}
        for instrument in self.instruments:
            type_counts[instrument.type] = type_counts.get(instrument.type, 0) + 1
        
        print(f"\nTotal instruments detected: {len(self.instruments)}")
        print("\nBreakdown by type:")
        for inst_type, count in sorted(type_counts.items()):
            print(f"  {inst_type:.<30} {count:>3}")
        
        print("\n" + "="*70 + "\n")

def main():
    """Main execution function"""

    # Local image path (relative to your current folder)
    IMAGE_PATH = r"output_upscaled/pnid_cropped_upscaled.png"

    # If the above does not work, use full absolute path:
    # IMAGE_PATH = r"D:\PNID_28_2026_Backend\input\output_upscaled\pnid_cropped_upscaled.png"

    OUTPUT_IMAGE = "detected_instruments_annotated.jpg"
    OUTPUT_JSON = "detected_instruments.json"

    # Initialize detector
    detector = PnIDInstrumentDetector(IMAGE_PATH)

    # Run detection
    instruments = detector.detect_all_instruments()

    # Print summary
    detector.print_summary()

    # Visualize results
    detector.visualize_results(OUTPUT_IMAGE)

    # Export JSON
    detector.export_results(OUTPUT_JSON)

    print("Detection complete!")
    print(f"Found {len(instruments)} instruments")
    print(f"Annotated image saved to: {OUTPUT_IMAGE}")
    print(f"JSON results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
