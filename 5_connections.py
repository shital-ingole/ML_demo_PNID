
import cv2
import pandas as pd
import numpy as np
import os
import math

# =====================================================
# CONFIG
# =====================================================
IMAGE_PATH = "pdf_images_visual/page_1_visual.png"

PIPES_GEOM_CSV = "output_image_1/pipes_with_geometry_image_1.csv"
CIRCLES_VALID  = "circles_output/page_1_visual/circles_valid_tags.csv"

OUT_CSV   = "output_image_1/instrument_pipe_connections_image_1.csv"
DEBUG_IMG = "output_image_1/debug_impulse_connections_image_1.png"

# thresholds (pixel units)
IMPULSE_LEN_THRESHOLD = 140
INSTR_TO_IMPULSE_DIST = 25   # slightly increased tolerance
IMPULSE_TO_PIPE_DIST  = 12

# =====================================================
# LOAD IMAGE (just for debug drawing)
# =====================================================
img = cv2.imread(IMAGE_PATH)
if img is None:
    print("⚠️ Warning: Debug image not found, continuing without drawing...")
debug = img.copy() if img is not None else None

# =====================================================
# LOAD DATA
# =====================================================
pipes = pd.read_csv(PIPES_GEOM_CSV)
circles = pd.read_csv(CIRCLES_VALID)

# 🔥 Correctly map circle columns to usable ones
instruments = circles.rename(columns={
    "full_tag": "instrument_tag",
    "center_x": "cx",
    "center_y": "cy"
})[["instrument_tag", "cx", "cy"]]

# =====================================================
# HELPERS
# =====================================================
def line_length(p):
    return math.hypot(p.x2 - p.x1, p.y2 - p.y1)

def point_to_line(px, py, x1, y1, x2, y2):
    num = abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1)
    den = math.hypot(y2-y1, x2-x1) + 1e-6
    return num / den

# =====================================================
# CLASSIFY PIPES
# =====================================================
pipes["is_impulse"] = pipes["length"] < IMPULSE_LEN_THRESHOLD
impulses = pipes[pipes.is_impulse].copy()
mains    = pipes[~pipes.is_impulse].copy()

print(f"Main pipes   : {len(mains)}")
print(f"Impulse lines: {len(impulses)}")
print(f"Instruments  : {len(instruments)}")

# =====================================================
# MAP IMPULSE → MAIN PIPE
# =====================================================
impulse_to_main = {}

for imp_idx, imp in impulses.iterrows():
    impulse_to_main[imp_idx] = []
    for _, main in mains.iterrows():
        # check if impulse line endpoint touches main line
        if (imp.x1 == main.x1 and imp.y1 == main.y1) or (imp.x2 == main.x2 and imp.y2 == main.y2):
            impulse_to_main[imp_idx].append(main)

# =====================================================
# MAIN CONNECTION LOGIC
# =====================================================
rows = []
print("\n=== connecting instruments to impulse lines ===")

for _, instr in instruments.iterrows():
    cx, cy = int(instr.cx), int(instr.cy)
    tag = instr.instrument_tag

    for imp_idx, imp in impulses.iterrows():
        d_instr = point_to_line(cx, cy, imp.x1, imp.y1, imp.x2, imp.y2)

        if d_instr > INSTR_TO_IMPULSE_DIST:
            continue

        # direct impulse connection
        rows.append({
            "instrument_tag": tag,
            "connected_pipe": imp.pipe_tag,
            "connection_type": "DIRECT_IMPULSE"
        })

        # propagation to main pipe
        for main in impulse_to_main.get(imp_idx, []):
            rows.append({
                "instrument_tag": tag,
                "connected_pipe": main.pipe_tag,
                "connection_type": "IMPULSE_VIA_PIPE"
            })

# =====================================================
# SAVE
# =====================================================
df = pd.DataFrame(rows).drop_duplicates()
df.to_csv(OUT_CSV, index=False)

print("\n✅ CONNECTIONS GENERATED")
print("→", OUT_CSV)
print("rows:", len(df))
