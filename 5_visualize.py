# import os
# import re
# import math
# import pandas as pd

# VALVE_CROPS_DIR = "cropped_valves_image_1_v1"

# # PIPES_CSV    = "output_image_1/pipes_clean_image_1.csv"
# PIPES_CSV    = "output_image_1/pipes_with_geometry_image_1.csv"
# # BRANCHES_CSV = "output_image_1/valve_branches_clean_image_1.csv"
# BRANCHES_CSV = "output_image_1/valve_branches_image_1.csv"

# OUT_CSV = "output_image_1/valve_branch_annotations_image_1.csv"

# MAX_PIPE_DISTANCE  = 30   
# MAX_VALVE_DISTANCE = 60   


# pipes    = pd.read_csv(PIPES_CSV)
# branches = pd.read_csv(BRANCHES_CSV)

# if "pipe_tag" in pipes.columns:
#     pass
# elif "tag" in pipes.columns:
#     pipes = pipes.rename(columns={"tag": "pipe_tag"})
# elif "line_number" in pipes.columns:
#     pipes = pipes.rename(columns={"line_number": "pipe_tag"})
# else:
#     raise RuntimeError(f"No pipe tag column found: {pipes.columns.tolist()}")

# # Normalize pipe tag text (OCR safety)
# pipes["pipe_tag"] = (
#     pipes["pipe_tag"]
#     .astype(str)
#     .str.strip()
#     .str.lstrip("*")
#     .str.replace(" ", "", regex=False)
# )


# if "pipe_id" not in pipes.columns:
#     pipes = pipes.reset_index(drop=True)
#     pipes["pipe_id"] = pipes.index.map(lambda i: f"P{i}")


# PIPE_REGEX = re.compile(
#     r'^\d+(?:/\d+)?["\']-'   
#     r'[A-Z]{1,4}-'           
#     r'\d{4}-'                
#     r'\d{6}-'               
#     r'[A-Z0-9\-]+$'          
# )

# pipes = pipes[pipes.pipe_tag.str.match(PIPE_REGEX, na=False)].copy()
# print(f"✅ Valid pipes loaded: {len(pipes)}")

# branch_text = branches[
#     branches.tag.str.match(
#         r'^\d+(?:/\d+)?["\']\s*(LO|NC|NO)$',
#         case=False,
#         na=False
#     )
# ].copy()

# print(f"✅ Branch annotations found: {len(branch_text)}")

# def center(x1, y1, x2, y2):
#     return ((x1 + x2) / 2, (y1 + y2) / 2)

# def point_to_line(px, py, x1, y1, x2, y2):
#     num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
#     den = math.hypot(y2 - y1, x2 - x1) + 1e-6
#     return num / den

# # =====================================================
# # MAP BRANCH → VALVE → PIPE  (CORRECT LOGIC)
# # =====================================================
# rows = []

# for _, t in branch_text.iterrows():
#     tx, ty = center(t.x1, t.y1, t.x2, t.y2)

#     # -------------------------------------------------
#     # 1️⃣ Find nearest VALVE (from crop centers)
#     # -------------------------------------------------
#     best_valve = None
#     best_valve_dist = float("inf")
#     best_valve_center = None

#     for fname in os.listdir(VALVE_CROPS_DIR):
#         m = re.search(r'_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)\.png$', fname)
#         if not m:
#             continue

#         vx, vy = center(*map(int, m.groups()))
#         d = math.hypot(tx - vx, ty - vy)

#         if d < best_valve_dist:
#             best_valve_dist = d
#             best_valve = fname
#             best_valve_center = (vx, vy)

#     if best_valve is None or best_valve_dist > MAX_VALVE_DISTANCE:
#         continue

#     vx, vy = best_valve_center

#     best_pipe = None
#     best_pipe_dist = float("inf")

#     for _, p in pipes.iterrows():
#         d = point_to_line(vx, vy, p.x1, p.y1, p.x2, p.y2)
#         if d < best_pipe_dist:
#             best_pipe_dist = d
#             best_pipe = p

#     if best_pipe is None or best_pipe_dist > MAX_PIPE_DISTANCE:
#         continue

#     m = re.match(r'(\d+(?:/\d+)?["\'])\s*(LO|NC|NO)', t.tag.upper())
#     if not m:
#         continue

#     rows.append({
#         "valve_crop": best_valve,
#         "pipe_id": best_pipe.pipe_id,
#         "pipe_line_number": best_pipe.pipe_tag,
#         "branch_size": m.group(1),
#         "condition": m.group(2),
#         "text": t.tag.upper(),
#         "distance_to_pipe": round(best_pipe_dist, 2),
#         "distance_to_valve": round(best_valve_dist, 2)
#     })

# df = (
#     pd.DataFrame(rows)
#     .sort_values("distance_to_pipe")
#     .drop_duplicates(
#         subset=[
#             "valve_crop",
#             "pipe_id",
#             "pipe_line_number",
#             "branch_size",
#             "condition"
#         ]
#     )
# )

# df.to_csv(OUT_CSV, index=False)

# print("\n✅ Valve branch annotations extracted (CORRECT)")
# print("→", OUT_CSV)
# print("Rows:", len(df))
















import os
import re
import math
import pandas as pd

# =====================================================
# CONFIG / PATHS
# =====================================================
OUT_DIR = "output_image_1"

RAW_PIPES_CSV    = os.path.join(OUT_DIR, "pipes_with_geometry_image_1.csv")
RAW_BRANCHES_CSV = os.path.join(OUT_DIR, "valves_ocr_image_1_v1.csv")

PIPES_CLEAN_CSV    = os.path.join(OUT_DIR, "pipes_clean_image_1.csv")
BRANCHES_CLEAN_CSV = os.path.join(OUT_DIR, "valve_branches_clean_image_1.csv")

FINAL_OUT_CSV = os.path.join(OUT_DIR, "valve_branch_annotations_image_1.csv")

VALVE_CROPS_DIR = "cropped_valves_image_1_v1"

MAX_PIPE_DISTANCE  = 30
MAX_VALVE_DISTANCE = 60

os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# HELPERS
# =====================================================
def point_to_line(px, py, x1, y1, x2, y2):
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = math.hypot(y2 - y1, x2 - x1) + 1e-6
    return num / den

def crop_center_from_filename(fname):
    """
    valve_12_x1_y1_x2_y2.png  ->  (cx, cy)
    """
    m = re.search(r'_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)\.png$', fname)
    if not m:
        return None
    x1, y1, x2, y2 = map(int, m.groups())
    return ((x1 + x2) / 2, (y1 + y2) / 2)

# =====================================================
# 1️⃣ CLEAN PIPES
# =====================================================
pipes = pd.read_csv(RAW_PIPES_CSV)

# normalize pipe tag
if "pipe_tag" not in pipes.columns:
    raise RuntimeError("❌ pipes_with_geometry CSV must contain pipe_tag")

pipes["pipe_tag"] = (
    pipes["pipe_tag"]
    .astype(str)
    .str.strip()
    .str.lstrip("*")
    .str.replace(" ", "", regex=False)
)

PIPE_REGEX = re.compile(
    r'^\d+(?:/\d+)?["\']-[A-Z]{1,4}-\d{4}-\d{6}-[A-Z0-9\-]+$'
)

pipes = pipes[pipes.pipe_tag.str.match(PIPE_REGEX, na=False)].copy()

if "pipe_id" not in pipes.columns:
    pipes = pipes.reset_index(drop=True)
    pipes["pipe_id"] = pipes.index.map(lambda i: f"P{i}")

pipes.to_csv(PIPES_CLEAN_CSV, index=False)
print(f"✅ pipes_clean_image_1.csv ({len(pipes)})")

# =====================================================
# 2️⃣ CLEAN VALVE BRANCH OCR (POINT-BASED)
# =====================================================
branches = pd.read_csv(RAW_BRANCHES_CSV)

required_cols = {"center_x", "center_y", "ocr_raw"}
missing = required_cols - set(branches.columns)
if missing:
    raise RuntimeError(f"❌ valves_ocr CSV missing columns: {missing}")

branches["tag"] = (
    branches["ocr_raw"]
    .astype(str)
    .str.upper()
    .str.strip()
)

BRANCH_REGEX = re.compile(r'^\d+(?:/\d+)?["\']\s*(LO|NC|NO)$')

branches = branches[branches.tag.str.match(BRANCH_REGEX, na=False)].copy()

branches = branches.rename(columns={
    "center_x": "cx",
    "center_y": "cy"
})[["tag", "cx", "cy"]]

branches.to_csv(BRANCHES_CLEAN_CSV, index=False)
print(f"✅ valve_branches_clean_image_1.csv ({len(branches)})")

# =====================================================
# 3️⃣ BRANCH → VALVE → PIPE ASSOCIATION
# =====================================================
pipes = pd.read_csv(PIPES_CLEAN_CSV)
branches = pd.read_csv(BRANCHES_CLEAN_CSV)

rows = []

valve_centers = []
for fname in os.listdir(VALVE_CROPS_DIR):
    c = crop_center_from_filename(fname)
    if c:
        valve_centers.append((fname, c[0], c[1]))

for _, t in branches.iterrows():
    tx, ty = t.cx, t.cy

    # ---- nearest valve ----
    best_valve = None
    best_valve_dist = float("inf")
    best_valve_center = None

    for fname, vx, vy in valve_centers:
        d = math.hypot(tx - vx, ty - vy)
        if d < best_valve_dist:
            best_valve_dist = d
            best_valve = fname
            best_valve_center = (vx, vy)

    if best_valve is None or best_valve_dist > MAX_VALVE_DISTANCE:
        continue

    vx, vy = best_valve_center

    # ---- nearest pipe ----
    best_pipe = None
    best_pipe_dist = float("inf")

    for _, p in pipes.iterrows():
        d = point_to_line(vx, vy, p.x1, p.y1, p.x2, p.y2)
        if d < best_pipe_dist:
            best_pipe_dist = d
            best_pipe = p

    if best_pipe is None or best_pipe_dist > MAX_PIPE_DISTANCE:
        continue

    m = re.match(r'(\d+(?:/\d+)?["\'])\s*(LO|NC|NO)', t.tag)
    if not m:
        continue

    rows.append({
        "valve_crop": best_valve,
        "pipe_id": best_pipe.pipe_id,
        "pipe_line_number": best_pipe.pipe_tag,
        "branch_size": m.group(1),
        "condition": m.group(2),
        "text": t.tag,
        "distance_to_pipe": round(best_pipe_dist, 2),
        "distance_to_valve": round(best_valve_dist, 2),
    })

df = (
    pd.DataFrame(rows)
    .sort_values("distance_to_pipe")
    .drop_duplicates(
        subset=[
            "valve_crop",
            "pipe_id",
            "pipe_line_number",
            "branch_size",
            "condition"
        ]
    )
)

df.to_csv(FINAL_OUT_CSV, index=False)

print(f"\n🎯 valve_branch_annotations_image_1.csv ({len(df)})")
print("→", FINAL_OUT_CSV)

