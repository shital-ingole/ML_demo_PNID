import cv2
import numpy as np
import pandas as pd
import os
import math

# =========================================================
# INPUT IMAGES
# =========================================================
# IMAGES = [
#     "pdf_images_visual/page_1_visual.png",
#     "pdf_images_visual/page_2_visual.png",
#     "pdf_images_visual/page_3_visual.png",
#     "pdf_images_visual/page_4_visual.png",
# ]

IMAGES = [
    "output_upscaled/pnid_cropped_upscaled.png"
    ]
BASE_OUT = "output/circles_output_pnid8"

# =========================================================
# TUNED PARAMETERS (PNID-SAFE)
# =========================================================
MIN_RADIUS = 32
MAX_RADIUS = 115

MIN_TEXT_PIXELS = 35          # real instruments always have text
MAX_FILL_RATIO = 0.42         # reject filled / dark junk
PIPE_CENTER_RADIUS = 4        # pipe-at-center killer
DEDUP_DIST = 60               # remove overlapping detections

# =========================================================
# HELPERS
# =========================================================
def ensure_dirs(path):
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "crops"), exist_ok=True)


def build_pipe_mask(gray):
    edges = cv2.Canny(gray, 60, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=60,
        maxLineGap=10
    )

    mask = np.zeros_like(gray)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            # Only long vertical/horizontal pipes
            if abs(x1 - x2) < 6 or abs(y1 - y2) < 6:
                cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
    return mask


def has_pipe_at_center(pipe_mask, x, y):
    h, w = pipe_mask.shape
    x1 = max(0, x - PIPE_CENTER_RADIUS)
    y1 = max(0, y - PIPE_CENTER_RADIUS)
    x2 = min(w, x + PIPE_CENTER_RADIUS)
    y2 = min(h, y + PIPE_CENTER_RADIUS)
    return cv2.countNonZero(pipe_mask[y1:y2, x1:x2]) > 0


def text_pixels_inside(bw, x, y, r):
    mask = np.zeros_like(bw)
    cv2.circle(mask, (x, y), int(r * 0.6), 255, -1)
    return cv2.countNonZero(cv2.bitwise_and(bw, bw, mask=mask))


def fill_ratio(bw, x, y, r):
    mask = np.zeros_like(bw)
    cv2.circle(mask, (x, y), r, 255, -1)
    return cv2.countNonZero(cv2.bitwise_and(bw, bw, mask=mask)) / (math.pi * r * r)


def dedupe(cx, cy, used):
    for ux, uy in used:
        if abs(cx - ux) < DEDUP_DIST and abs(cy - uy) < DEDUP_DIST:
            return True
    return False


def is_circle_in_square(gray, x, y, r):
    """
    Detect if a circle is enclosed in a square box.
    Returns True if a square/rectangle is detected around the circle.
    """
    # Check region around the circle for rectangular edges
    padding = int(r * 0.4)
    x1 = max(0, x - r - padding)
    y1 = max(0, y - r - padding)
    x2 = min(gray.shape[1], x + r + padding)
    y2 = min(gray.shape[0], y + r + padding)
    
    roi = gray[y1:y2, x1:x2]
    edges = cv2.Canny(roi, 50, 150)
    
    # Detect lines (potential square edges)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=int(r * 0.8),  # Lines should be at least 80% of radius
        maxLineGap=5
    )
    
    if lines is None:
        return False
    
    # Count horizontal and vertical lines
    horizontal_lines = 0
    vertical_lines = 0
    
    for line in lines:
        x1_l, y1_l, x2_l, y2_l = line[0]
        
        # Check if line is horizontal
        if abs(y1_l - y2_l) < 5 and abs(x1_l - x2_l) > int(r * 0.8):
            horizontal_lines += 1
        
        # Check if line is vertical
        if abs(x1_l - x2_l) < 5 and abs(y1_l - y2_l) > int(r * 0.8):
            vertical_lines += 1
    
    # If we have at least 2 horizontal and 2 vertical lines, likely a square
    return horizontal_lines >= 2 and vertical_lines >= 2


# =========================================================
# MAIN LOOP
# =========================================================
for img_path in IMAGES:
    name = os.path.splitext(os.path.basename(img_path))[0]
    OUT_DIR = os.path.join(BASE_OUT, name)
    CROP_DIR = os.path.join(OUT_DIR, "crops")
    CSV_OUT = os.path.join(OUT_DIR, "circles.csv")
    DEBUG_IMG = os.path.join(OUT_DIR, "circles_debug.png")

    ensure_dirs(OUT_DIR)

    print(f"\n🔍 Processing {name}")

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Image not found")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )

    pipe_mask = build_pipe_mask(gray)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=70,
        param1=120,
        param2=20,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS
    )

    records = []
    used_centers = []
    debug = img.copy()
    idx = 0

    if circles is not None:
        circles = np.round(circles[0]).astype(int)

        for x, y, r in circles:
            if x < r or y < r or x >= W - r or y >= H - r:
                continue

            # ❌ Kill valves & fittings
            if has_pipe_at_center(pipe_mask, x, y):
                continue

            # ❌ Kill filled / dark junk
            if fill_ratio(bw, x, y, r) > MAX_FILL_RATIO:
                continue

            # ❌ Must have readable text
            if text_pixels_inside(bw, x, y, r) < MIN_TEXT_PIXELS:
                continue

            # ❌ Deduplicate
            if dedupe(x, y, used_centers):
                continue
            
            # ❌ Kill circles inside squares
            if is_circle_in_square(gray, x, y, r):
                continue

            used_centers.append((x, y))

            pad = int(r * 0.3)
            x1 = max(0, x - r - pad)
            y1 = max(0, y - r - pad)
            x2 = min(W, x + r + pad)
            y2 = min(H, y + r + pad)

            crop = img[y1:y2, x1:x2]
            crop_path = os.path.join(CROP_DIR, f"circle_{idx}.png")
            cv2.imwrite(crop_path, crop)

            cv2.circle(debug, (x, y), r, (0, 0, 255), 2)
            cv2.putText(debug, str(idx), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            records.append({
                "id": idx,
                "center_x": x,
                "center_y": y,
                "radius": r,
                "crop_path": crop_path
            })

            idx += 1

    pd.DataFrame(records).to_csv(CSV_OUT, index=False)
    cv2.imwrite(DEBUG_IMG, debug)

    print(f"✅ {name}: {len(records)} real instrument bubbles")
