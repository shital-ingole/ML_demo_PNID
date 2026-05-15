# import math
# import pandas as pd
# import re

# IN  = "output_image_1/candidate_valve_pipe_links.csv"
# OUT = "output_image_1/validated_valve_pipe_links_image_1.csv"

# df = pd.read_csv(IN)

# # =====================================================
# # GEOMETRY HELPERS
# # =====================================================
# def projection_ok(px, py, x1, y1, x2, y2):
#     dx, dy = x2-x1, y2-y1
#     if dx == 0 and dy == 0:
#         return False
#     t = ((px-x1)*dx + (py-y1)*dy) / (dx*dx + dy*dy)
#     return 0.05 < t < 0.95

# def pipe_length(x1, y1, x2, y2):
#     return math.hypot(x2-x1, y2-y1)

# # =====================================================
# # SIZE PARSING (ROBUST)
# # =====================================================
# def size_to_float(size_str):
#     """
#     Extract pipe/branch size in inches as float.
#     Handles: * 2", 2", 3/4"
#     """
#     if not size_str:
#         return None

#     s = str(size_str)
#     m = re.search(r'(\d+(?:/\d+)?)\s*"', s)
#     if not m:
#         return None

#     val = m.group(1)
#     if "/" in val:
#         a, b = val.split("/")
#         return float(a) / float(b)
#     return float(val)

# # =====================================================
# # VALIDATION
# # =====================================================
# rows = []

# MAIN_PIPE_LENGTH = 120   # px threshold to separate main vs branch

# for _, r in df.iterrows():

#     branch_size = size_to_float(r.branch_text)
#     pipe_size   = size_to_float(r.pipe_tag)

#     if branch_size is None or pipe_size is None:
#         continue

#     p_len = pipe_length(r.pipe_x1, r.pipe_y1, r.pipe_x2, r.pipe_y2)

#     confidence = 0.0
#     valve_type = None

#     # =================================================
#     # INLINE VALVE (MAIN PIPE)
#     # =================================================
#     if p_len >= MAIN_PIPE_LENGTH:

#         if not projection_ok(
#             r.valve_x, r.valve_y,
#             r.pipe_x1, r.pipe_y1,
#             r.pipe_x2, r.pipe_y2
#         ):
#             continue

#         if branch_size > pipe_size:
#             continue  # impossible

#         # scoring
#         confidence += 0.6
#         if r.distance_pipe < 20:
#             confidence += 0.2
#         if r.distance_valve < 40:
#             confidence += 0.2

#         valve_type = "INLINE"

#     # =================================================
#     # BRANCH VALVE (SHORT STUB)
#     # =================================================
#     else:
#         if not projection_ok(
#             r.valve_x, r.valve_y,
#             r.pipe_x1, r.pipe_y1,
#             r.pipe_x2, r.pipe_y2
#         ):
#             continue

#         if branch_size > pipe_size:
#             continue

#         # scoring
#         confidence += 0.5
#         if r.distance_pipe < 15:
#             confidence += 0.2
#         if r.distance_valve < 40:
#             confidence += 0.2

#         valve_type = "BRANCH"

#     # =================================================
#     # FINAL CLASSIFICATION
#     # =================================================
#     out = r.to_dict()
#     out["confidence"] = round(confidence, 2)
#     out["valve_type"] = valve_type

#     if confidence >= 0.85:
#         out["status"] = "ACCEPT"
#     elif confidence >= 0.65:
#         out["status"] = "REVIEW"
#     else:
#         out["status"] = "REJECT"

#     rows.append(out)

# pd.DataFrame(rows).to_csv(OUT, index=False)

# print("✅ Validation complete (INLINE + BRANCH valves)")
# print("→", OUT)
# print("Rows:", len(rows))













import math
import pandas as pd
import re

IN  = "output_image_1/candidate_valve_pipe_links.csv"
OUT = "output_image_1/validated_valve_pipe_links_image_1.csv"

df = pd.read_csv(IN)

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

def pipes_intersect(x1,y1,x2,y2, a1,b1,a2,b2, tol=6):
    minx1, maxx1 = min(x1,x2)-tol, max(x1,x2)+tol
    miny1, maxy1 = min(y1,y2)-tol, max(y1,y2)+tol
    minx2, maxx2 = min(a1,a2)-tol, max(a1,a2)+tol
    miny2, maxy2 = min(b1,b2)-tol, max(b1,b2)+tol

    return not (
        maxx1 < minx2 or maxx2 < minx1 or
        maxy1 < miny2 or maxy2 < miny1
    )

# =====================================================
# SIZE PARSING (ROBUST)
# =====================================================
def size_to_float(size_str):
    """
    Extract pipe/branch size in inches as float.
    Handles: * 2", 2", 3/4"
    """
    if not size_str:
        return None

    s = str(size_str)
    m = re.search(r'(\d+(?:/\d+)?)\s*"', s)
    if not m:
        return None

    val = m.group(1)
    if "/" in val:
        a, b = val.split("/")
        return float(a) / float(b)
    return float(val)

# =====================================================
# PRE-COMPUTE MAIN PIPES (IMPORTANT)
# =====================================================
MAIN_PIPE_LENGTH = 120  # px

main_pipes = []

for _, r in df.iterrows():
    plen = pipe_length(r.pipe_x1, r.pipe_y1, r.pipe_x2, r.pipe_y2)
    if plen >= MAIN_PIPE_LENGTH:
        main_pipes.append(r)

# =====================================================
# VALIDATION
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

    # =================================================
    # INLINE VALVE (ON MAIN PIPE)
    # =================================================
    if p_len >= MAIN_PIPE_LENGTH:

        if not projection_ok(
            r.valve_x, r.valve_y,
            r.pipe_x1, r.pipe_y1,
            r.pipe_x2, r.pipe_y2
        ):
            continue

        if branch_size > pipe_size:
            continue

        confidence += 0.6
        if r.distance_pipe < 20:
            confidence += 0.2
        if r.distance_valve < 40:
            confidence += 0.2

        valve_type = "INLINE"

    # =================================================
    # BRANCH VALVE (SHORT STUB CONNECTED TO MAIN)
    # =================================================
    else:
        if not projection_ok(
            r.valve_x, r.valve_y,
            r.pipe_x1, r.pipe_y1,
            r.pipe_x2, r.pipe_y2
        ):
            continue

        if branch_size > pipe_size:
            continue

        # 🔑 CRITICAL FIX: stub must connect to a main pipe
        connected_to_main = False

        for mp in main_pipes:
            if pipes_intersect(
                r.pipe_x1, r.pipe_y1, r.pipe_x2, r.pipe_y2,
                mp.pipe_x1, mp.pipe_y1, mp.pipe_x2, mp.pipe_y2
            ):
                main_size = size_to_float(mp.pipe_tag)
                if main_size and branch_size < main_size:
                    connected_to_main = True
                    break

        if not connected_to_main:
            continue

        confidence += 0.7
        if r.distance_pipe < 15:
            confidence += 0.15
        if r.distance_valve < 40:
            confidence += 0.15

        valve_type = "BRANCH"

    # =================================================
    # FINAL CLASSIFICATION
    # =================================================
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

print("✅ Validation complete (INLINE + BRANCH valves)")
print("→", OUT)
print("Rows:", len(rows))
