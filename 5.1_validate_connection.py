import math
import pandas as pd
import re
from collections import defaultdict

IN  = "output_image_1/candidate_valve_pipe_links.csv"
OUT = "output_image_1/validated_valve_pipe_links_image_1_New.csv"

df = pd.read_csv(IN)

# =====================================================
# CONFIG
# =====================================================
MAIN_PIPE_LENGTH = 120
GRID_SIZE = 200   

# =====================================================
# GEOMETRY HELPERS
# =====================================================
def projection_ok(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return False
    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
    return 0.05 < t < 0.95

def pipe_length(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def bbox(x1,y1,x2,y2):
    return min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)

def bbox_intersect(b1, b2, tol=6):
    return not (
        b1[2] < b2[0]-tol or b2[2] < b1[0]-tol or
        b1[3] < b2[1]-tol or b2[3] < b1[1]-tol
    )

# =====================================================
# SIZE PARSING
# =====================================================
def size_to_float(size_str):
    if not size_str:
        return None
    m = re.search(r'(\d+(?:/\d+)?)\s*"', str(size_str))
    if not m:
        return None
    val = m.group(1)
    if "/" in val:
        a,b = val.split("/")
        return float(a)/float(b)
    return float(val)

# =====================================================
# BUILD SPATIAL INDEX OF MAIN PIPES
# =====================================================
main_pipes = []

for _, r in df.iterrows():
    if pipe_length(r.pipe_x1, r.pipe_y1, r.pipe_x2, r.pipe_y2) >= MAIN_PIPE_LENGTH:
        main_pipes.append({
            "bbox": bbox(r.pipe_x1, r.pipe_y1, r.pipe_x2, r.pipe_y2),
            "size": size_to_float(r.pipe_tag)
        })

grid = defaultdict(list)

def grid_keys(b):
    gx1 = int(b[0] // GRID_SIZE)
    gy1 = int(b[1] // GRID_SIZE)
    gx2 = int(b[2] // GRID_SIZE)
    gy2 = int(b[3] // GRID_SIZE)
    for gx in range(gx1, gx2+1):
        for gy in range(gy1, gy2+1):
            yield (gx, gy)

for mp in main_pipes:
    for k in grid_keys(mp["bbox"]):
        grid[k].append(mp)

# =====================================================
# VALIDATION LOOP (FAST)
# =====================================================
rows = []

for _, r in df.iterrows():

    branch_size = size_to_float(r.branch_text)
    pipe_size   = size_to_float(r.pipe_tag)
    if branch_size is None or pipe_size is None:
        continue

    p_len = pipe_length(r.pipe_x1, r.pipe_y1, r.pipe_x2, r.pipe_y2)
    confidence = 0.0
    valve_type = None

    # INLINE
    if p_len >= MAIN_PIPE_LENGTH:
        if not projection_ok(r.valve_x, r.valve_y,
                             r.pipe_x1, r.pipe_y1,
                             r.pipe_x2, r.pipe_y2):
            continue
        if branch_size > pipe_size:
            continue

        confidence = 0.6
        if r.distance_pipe < 20:
            confidence += 0.2
        if r.distance_valve < 40:
            confidence += 0.2

        valve_type = "INLINE"

    # BRANCH
    else:
        if not projection_ok(r.valve_x, r.valve_y,
                             r.pipe_x1, r.pipe_y1,
                             r.pipe_x2, r.pipe_y2):
            continue
        if branch_size > pipe_size:
            continue

        b_stub = bbox(r.pipe_x1, r.pipe_y1, r.pipe_x2, r.pipe_y2)
        connected = False

        for k in grid_keys(b_stub):
            for mp in grid.get(k, []):
                if mp["size"] and branch_size < mp["size"]:
                    if bbox_intersect(b_stub, mp["bbox"]):
                        connected = True
                        break
            if connected:
                break

        if not connected:
            continue

        confidence = 0.7
        if r.distance_pipe < 15:
            confidence += 0.15
        if r.distance_valve < 40:
            confidence += 0.15

        valve_type = "BRANCH"

    out = r.to_dict()
    out["confidence"] = round(confidence, 2)
    out["valve_type"] = valve_type

    if confidence >= 0.85:
        out["status"] = "ACCEPT"
    elif confidence >= 0.65:
        out["status"] = "REVIEW"
    else:
        out["status"] = "REJECT"

    rows.append(out)

pd.DataFrame(rows).to_csv(OUT, index=False)

print("✅ Validation complete (FAST, INLINE + BRANCH)")
print("→", OUT)
print("Rows:", len(rows))
