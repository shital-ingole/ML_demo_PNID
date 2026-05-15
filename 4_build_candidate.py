import os
import re
import math
import pandas as pd

OUT_DIR = "output_image_1"
VALVE_DIR = "cropped_valves_image_1_v1"

PIPES_CSV = f"{OUT_DIR}/pipes_with_geometry_image_1.csv"
BRANCH_CSV = f"{OUT_DIR}/valves_ocr_image_1_v1.csv"

OUT_CSV = f"{OUT_DIR}/candidate_valve_pipe_links.csv"

pipes = pd.read_csv(PIPES_CSV)
branches = pd.read_csv(BRANCH_CSV)

def point_to_line(px, py, x1, y1, x2, y2):
    return abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1) / (math.hypot(y2-y1, x2-x1) + 1e-6)

def valve_center(fname):
    m = re.search(r'_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)\.png$', fname)
    if not m:
        return None
    x1,y1,x2,y2 = map(int, m.groups())
    return (x1+x2)/2, (y1+y2)/2

rows = []

for _, b in branches.iterrows():
    if b["ocr_raw"] == "NONE":
        continue

    bx, by = b["center_x"], b["center_y"]

    for fname in os.listdir(VALVE_DIR):
        vc = valve_center(fname)
        if not vc:
            continue

        vx, vy = vc
        d_valve = math.hypot(bx-vx, by-vy)

        for _, p in pipes.iterrows():
            d_pipe = point_to_line(vx, vy, p.x1, p.y1, p.x2, p.y2)

            rows.append({
                "branch_text": b["ocr_raw"],
                "branch_x": bx,
                "branch_y": by,
                "valve_crop": fname,
                "valve_x": vx,
                "valve_y": vy,
                "pipe_id": p.get("pipe_id", ""),
                "pipe_tag": p.get("pipe_tag", ""),
                "pipe_x1": p.x1,
                "pipe_y1": p.y1,
                "pipe_x2": p.x2,
                "pipe_y2": p.y2,
                "distance_valve": round(d_valve, 2),
                "distance_pipe": round(d_pipe, 2),
            })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print(f"✅ Candidate graph built: {len(df)} rows")
print("→", OUT_CSV)
