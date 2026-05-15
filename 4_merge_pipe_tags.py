import pandas as pd
import numpy as np
import os
import math

# =====================================================
# CONFIG
# =====================================================
PIPE_LINES_CSV = "output_image_1/pipe_lines_image_1.csv"
OCR_CSV        = "output_image_1/instrument_pipe_extraction_image_1.csv"
OUT_CSV        = "output_image_1/pipes_with_geometry_image_1.csv"

MAX_DISTANCE = 25   # pixels (tune if needed)

os.makedirs("output", exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
pipes_geom = pd.read_csv(PIPE_LINES_CSV)
ocr = pd.read_csv(OCR_CSV)

pipe_tags = ocr[ocr.entity_type == "pipe"].copy()

print(f"Pipe geometries : {len(pipes_geom)}")
print(f"Pipe tags (OCR) : {len(pipe_tags)}")

# =====================================================
# HELPERS
# =====================================================
def bbox_center(r):
    return ((r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2)

def point_to_line(px, py, x1, y1, x2, y2):
    num = abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1)
    den = math.hypot(y2-y1, x2-x1) + 1e-6
    return num / den

# =====================================================
# MATCH PIPE TAG → PIPE LINE
# =====================================================
rows = []

for _, tag in pipe_tags.iterrows():

    cx, cy = bbox_center(tag)

    best = None
    best_dist = 1e9

    for _, line in pipes_geom.iterrows():
        d = point_to_line(cx, cy, line.x1, line.y1, line.x2, line.y2)
        if d < best_dist:
            best_dist = d
            best = line

    if best is not None and best_dist < MAX_DISTANCE:
        rows.append({
            "pipe_id": best.pipe_id,
            "pipe_tag": tag.tag,
            "x1": int(best.x1),
            "y1": int(best.y1),
            "x2": int(best.x2),
            "y2": int(best.y2),
            "length": round(best.length, 2),
            "match_distance": round(best_dist, 2)
        })
    else:
        print(f"⚠️ No match for pipe tag: {tag.tag}")

# =====================================================
# SAVE
# =====================================================
df = pd.DataFrame(rows).drop_duplicates(subset=["pipe_id"])
df.to_csv(OUT_CSV, index=False)

print("\n✅ PIPE TAGS MERGED WITH GEOMETRY")
print("→", OUT_CSV)
print("Matched pipes:", len(df))

