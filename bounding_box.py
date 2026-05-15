# # import cv2
# # import numpy as np
# # import pandas as pd
# # import os
# # import math

# # # =========================================================
# # # CONFIG
# # # =========================================================
# # IMAGE_PATH = "output_upscaled/pnid_cropped_upscaled.png"
# # OUT_DIR = "./outputv2"

# # OUT_CSV = os.path.join(OUT_DIR, "circle_bboxes.csv")
# # DEBUG_IMG = os.path.join(OUT_DIR, "circle_bboxes_debug.png")

# # MIN_RADIUS = 28
# # MAX_RADIUS = 120
# # MIN_DIST = 70
# # PADDING = 10

# # CLUSTER_DIST_FACTOR = 0.6  # <-- key fix

# # os.makedirs(OUT_DIR, exist_ok=True)

# # # =========================================================
# # # HELPERS
# # # =========================================================
# # def is_circle_in_square(gray, x, y, r):
# #     pad = int(r * 0.5)
# #     x1 = max(0, x - r - pad)
# #     y1 = max(0, y - r - pad)
# #     x2 = min(gray.shape[1], x + r + pad)
# #     y2 = min(gray.shape[0], y + r + pad)

# #     roi = gray[y1:y2, x1:x2]
# #     edges = cv2.Canny(roi, 60, 150)

# #     lines = cv2.HoughLinesP(
# #         edges,
# #         rho=1,
# #         theta=np.pi / 180,
# #         threshold=40,
# #         minLineLength=int(r * 1.2),
# #         maxLineGap=6
# #     )

# #     if lines is None:
# #         return False

# #     h_lines = v_lines = 0
# #     for l in lines:
# #         x1_l, y1_l, x2_l, y2_l = l[0]
# #         if abs(y1_l - y2_l) < 6:
# #             h_lines += 1
# #         if abs(x1_l - x2_l) < 6:
# #             v_lines += 1

# #     return h_lines >= 2 and v_lines >= 2


# # def center(box):
# #     x1, y1, x2, y2 = box
# #     return ((x1 + x2) / 2, (y1 + y2) / 2)


# # def merge_boxes_by_distance(boxes):
# #     clusters = []

# #     for box in boxes:
# #         cx, cy = center(box)
# #         bw = box[2] - box[0]
# #         bh = box[3] - box[1]
# #         max_dim = max(bw, bh)

# #         assigned = False

# #         for cluster in clusters:
# #             ccx, ccy = center(cluster[0])
# #             if math.hypot(cx - ccx, cy - ccy) < CLUSTER_DIST_FACTOR * max_dim:
# #                 cluster.append(box)
# #                 assigned = True
# #                 break

# #         if not assigned:
# #             clusters.append([box])

# #     merged = []
# #     for cluster in clusters:
# #         xs1 = [b[0] for b in cluster]
# #         ys1 = [b[1] for b in cluster]
# #         xs2 = [b[2] for b in cluster]
# #         ys2 = [b[3] for b in cluster]
# #         merged.append((min(xs1), min(ys1), max(xs2), max(ys2)))

# #     return merged

# # # =========================================================
# # # LOAD IMAGE
# # # =========================================================
# # img = cv2.imread(IMAGE_PATH)
# # if img is None:
# #     raise RuntimeError("❌ Image not found")

# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

# # # =========================================================
# # # DETECT CIRCULAR CANDIDATES (LOOSE)
# # # =========================================================
# # circles = cv2.HoughCircles(
# #     gray,
# #     cv2.HOUGH_GRADIENT,
# #     dp=1.2,
# #     minDist=MIN_DIST,
# #     param1=120,
# #     param2=22,
# #     minRadius=MIN_RADIUS,
# #     maxRadius=MAX_RADIUS
# # )

# # if circles is None:
# #     raise RuntimeError("❌ No circle candidates detected")

# # circles = np.round(circles[0]).astype(int)

# # # =========================================================
# # # BUILD RAW BOXES
# # # =========================================================
# # raw_boxes = []

# # for cx, cy, r in circles:
# #     if is_circle_in_square(gray, cx, cy, r):
# #         continue

# #     x1 = max(0, cx - r - PADDING)
# #     y1 = max(0, cy - r - PADDING)
# #     x2 = min(img.shape[1], cx + r + PADDING)
# #     y2 = min(img.shape[0], cy + r + PADDING)

# #     raw_boxes.append((x1, y1, x2, y2))

# # print(f"Raw candidate boxes: {len(raw_boxes)}")

# # # =========================================================
# # # CLUSTER + MERGE (THE REAL FIX)
# # # =========================================================
# # final_boxes = merge_boxes_by_distance(raw_boxes)

# # print(f"Final merged boxes: {len(final_boxes)}")

# # # =========================================================
# # # SAVE CSV + DEBUG IMAGE
# # # =========================================================
# # records = []
# # debug = img.copy()

# # for idx, (x1, y1, x2, y2) in enumerate(final_boxes):
# #     records.append({
# #         "id": idx,
# #         "x1": x1, "y1": y1,
# #         "x2": x2, "y2": y1,
# #         "x3": x2, "y3": y2,
# #         "x4": x1, "y4": y2
# #     })

# #     cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
# #     cv2.putText(
# #         debug,
# #         str(idx),
# #         (x1 + 4, y1 - 6),
# #         cv2.FONT_HERSHEY_SIMPLEX,
# #         0.5,
# #         (255, 0, 0),
# #         1
# #     )

# # df = pd.DataFrame(records)
# # df.to_csv(OUT_CSV, index=False)
# # cv2.imwrite(DEBUG_IMG, debug)

# # print("✅ DONE")
# # print("📄 CSV:", OUT_CSV)
# # print("🖼 Debug image:", DEBUG_IMG)











# import cv2
# import numpy as np
# import pandas as pd
# import os
# import math

# # =========================================================
# # CONFIG
# # =========================================================
# IMAGE_PATH = "output_upscaled/pnid_cropped_upscaled.png"
# OUT_DIR = "./output_final"

# OUT_CSV = os.path.join(OUT_DIR, "circle_bboxes_tight.csv")
# DEBUG_IMG = os.path.join(OUT_DIR, "circle_bboxes_tight_debug.png")

# MIN_RADIUS = 28
# MAX_RADIUS = 120
# MIN_DIST = 70
# PADDING = 10

# CLUSTER_DIST_FACTOR = 0.6
# TIGHT_PAD = 6   # final padding after tightening

# os.makedirs(OUT_DIR, exist_ok=True)

# # =========================================================
# # HELPERS
# # =========================================================
# def is_circle_in_square(gray, x, y, r):
#     pad = int(r * 0.5)
#     x1 = max(0, x - r - pad)
#     y1 = max(0, y - r - pad)
#     x2 = min(gray.shape[1], x + r + pad)
#     y2 = min(gray.shape[0], y + r + pad)

#     roi = gray[y1:y2, x1:x2]
#     edges = cv2.Canny(roi, 60, 150)

#     lines = cv2.HoughLinesP(
#         edges,
#         rho=1,
#         theta=np.pi / 180,
#         threshold=40,
#         minLineLength=int(r * 1.2),
#         maxLineGap=6
#     )

#     if lines is None:
#         return False

#     h = v = 0
#     for l in lines:
#         x1l, y1l, x2l, y2l = l[0]
#         if abs(y1l - y2l) < 6:
#             h += 1
#         if abs(x1l - x2l) < 6:
#             v += 1

#     return h >= 2 and v >= 2


# def center(box):
#     x1, y1, x2, y2 = box
#     return ((x1 + x2) / 2, (y1 + y2) / 2)


# def merge_boxes_by_distance(boxes):
#     clusters = []

#     for box in boxes:
#         cx, cy = center(box)
#         bw = box[2] - box[0]
#         bh = box[3] - box[1]
#         max_dim = max(bw, bh)

#         placed = False
#         for cluster in clusters:
#             ccx, ccy = center(cluster[0])
#             if math.hypot(cx - ccx, cy - ccy) < CLUSTER_DIST_FACTOR * max_dim:
#                 cluster.append(box)
#                 placed = True
#                 break

#         if not placed:
#             clusters.append([box])

#     merged = []
#     for cl in clusters:
#         xs1 = [b[0] for b in cl]
#         ys1 = [b[1] for b in cl]
#         xs2 = [b[2] for b in cl]
#         ys2 = [b[3] for b in cl]
#         merged.append((min(xs1), min(ys1), max(xs2), max(ys2)))

#     return merged


# def tighten_box_to_circle(img, box, pad=TIGHT_PAD):
#     x1, y1, x2, y2 = map(int, box)
#     crop = img[y1:y2, x1:x2]

#     if crop.size == 0:
#         return box

#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 80, 200)

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

#     contours, _ = cv2.findContours(
#         edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )

#     best = None
#     best_score = 0

#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area < 200:
#             continue

#         peri = cv2.arcLength(cnt, True)
#         if peri == 0:
#             continue

#         circularity = 4 * np.pi * area / (peri * peri)

#         if circularity > best_score:
#             best_score = circularity
#             best = cnt

#     if best is None or best_score < 0.4:
#         return box

#     bx, by, bw, bh = cv2.boundingRect(best)

#     nx1 = max(0, x1 + bx - pad)
#     ny1 = max(0, y1 + by - pad)
#     nx2 = min(img.shape[1], x1 + bx + bw + pad)
#     ny2 = min(img.shape[0], y1 + by + bh + pad)

#     return (nx1, ny1, nx2, ny2)

# # =========================================================
# # LOAD IMAGE
# # =========================================================
# img = cv2.imread(IMAGE_PATH)
# if img is None:
#     raise RuntimeError("❌ Image not found")

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

# # =========================================================
# # DETECT CIRCULAR CANDIDATES
# # =========================================================
# circles = cv2.HoughCircles(
#     gray,
#     cv2.HOUGH_GRADIENT,
#     dp=1.2,
#     minDist=MIN_DIST,
#     param1=120,
#     param2=22,
#     minRadius=MIN_RADIUS,
#     maxRadius=MAX_RADIUS
# )

# if circles is None:
#     raise RuntimeError("❌ No circle candidates detected")

# circles = np.round(circles[0]).astype(int)

# # =========================================================
# # BUILD RAW BOXES
# # =========================================================
# raw_boxes = []
# for cx, cy, r in circles:
#     if is_circle_in_square(gray, cx, cy, r):
#         continue

#     raw_boxes.append((
#         max(0, cx - r - PADDING),
#         max(0, cy - r - PADDING),
#         min(img.shape[1], cx + r + PADDING),
#         min(img.shape[0], cy + r + PADDING)
#     ))

# print("Raw boxes:", len(raw_boxes))

# # =========================================================
# # CLUSTER + TIGHTEN
# # =========================================================
# merged = merge_boxes_by_distance(raw_boxes)

# final_boxes = []
# for b in merged:
#     final_boxes.append(tighten_box_to_circle(img, b))

# print("Final tight boxes:", len(final_boxes))

# # =========================================================
# # SAVE CSV + DEBUG
# # =========================================================
# records = []
# debug = img.copy()

# for i, (x1, y1, x2, y2) in enumerate(final_boxes):
#     records.append({
#         "id": i,
#         "x1": x1, "y1": y1,
#         "x2": x2, "y2": y1,
#         "x3": x2, "y3": y2,
#         "x4": x1, "y4": y2
#     })

#     cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     cv2.putText(debug, str(i), (x1 + 3, y1 - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

# df = pd.DataFrame(records)
# df.to_csv(OUT_CSV, index=False)
# cv2.imwrite(DEBUG_IMG, debug)

# print("✅ DONE")
# print("📄 CSV:", OUT_CSV)
# print("🖼 Debug image:", DEBUG_IMG)

















import cv2
import numpy as np
import pandas as pd
import os
import math

# =========================================================
# CONFIG
# =========================================================
IMAGE_PATH = "output_upscaled/pnid_cropped_upscaled.png"
OUT_DIR = "./output_final_v1"

CSV_PATH = os.path.join(OUT_DIR, "circle_boxes.csv")
DEBUG_IMG = os.path.join(OUT_DIR, "circle_boxes_debug.png")

MIN_RADIUS = 28
MAX_RADIUS = 110
MIN_DIST = 70

EDGE_MIN = 80        # minimum edge pixels
R_STD_RATIO = 0.35   # ring thickness tolerance
ASPECT_TOL = 0.30    # width-height similarity
PAD = 6

os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# HELPERS
# =========================================================
def validate_and_tighten_circle(img, cx, cy, r):
    """
    Validates that the detected circle is a true instrument bubble
    using radial edge statistics and isotropy.
    Returns tight bounding box or None.
    """
    h, w = img.shape[:2]

    x1 = max(0, cx - r - 10)
    y1 = max(0, cy - r - 10)
    x2 = min(w, cx + r + 10)
    y2 = min(h, cy + r + 10)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)

    ys, xs = np.where(edges > 0)
    if len(xs) < EDGE_MIN:
        return None

    ccx = xs.mean()
    ccy = ys.mean()

    rs = np.sqrt((xs - ccx) ** 2 + (ys - ccy) ** 2)

    r_med = np.median(rs)
    r_std = np.std(rs)

    # ❌ reject arcs, dashed loops, vessel edges
    if r_std > R_STD_RATIO * r_med:
        return None

    # build tight box from radius
    nx1 = int(x1 + ccx - r_med - PAD)
    ny1 = int(y1 + ccy - r_med - PAD)
    nx2 = int(x1 + ccx + r_med + PAD)
    ny2 = int(y1 + ccy + r_med + PAD)

    bw = nx2 - nx1
    bh = ny2 - ny1

    # ❌ reject non-isotropic shapes
    if abs(bw - bh) / max(bw, bh) > ASPECT_TOL:
        return None

    return nx1, ny1, nx2, ny2


def dedupe_boxes(boxes, dist_thresh=60):
    final = []
    centers = []

    for b in boxes:
        cx = (b[0] + b[2]) / 2
        cy = (b[1] + b[3]) / 2

        if any(math.hypot(cx - px, cy - py) < dist_thresh for px, py in centers):
            continue

        centers.append((cx, cy))
        final.append(b)

    return final


# =========================================================
# LOAD IMAGE
# =========================================================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise RuntimeError("❌ Image not found")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

# =========================================================
# HOUGH CANDIDATES (LOOSE)
# =========================================================
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=MIN_DIST,
    param1=120,
    param2=22,
    minRadius=MIN_RADIUS,
    maxRadius=MAX_RADIUS
)

if circles is None:
    raise RuntimeError("❌ No circles detected")

circles = np.round(circles[0]).astype(int)

# =========================================================
# VALIDATE + TIGHTEN
# =========================================================
valid_boxes = []

for cx, cy, r in circles:
    box = validate_and_tighten_circle(img, cx, cy, r)
    if box:
        valid_boxes.append(box)

print(f"Validated circles: {len(valid_boxes)}")

# =========================================================
# DEDUPE
# =========================================================
final_boxes = dedupe_boxes(valid_boxes)
print(f"Final circles: {len(final_boxes)}")

# =========================================================
# SAVE OUTPUTS
# =========================================================
records = []
debug = img.copy()

for i, (x1, y1, x2, y2) in enumerate(final_boxes):
    records.append({
        "id": i,
        "x1": x1, "y1": y1,
        "x2": x2, "y2": y1,
        "x3": x2, "y3": y2,
        "x4": x1, "y4": y2
    })

    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(debug, str(i), (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

pd.DataFrame(records).to_csv(CSV_PATH, index=False)
cv2.imwrite(DEBUG_IMG, debug)

print("✅ DONE")
print("📄 CSV:", CSV_PATH)
print("🖼 Debug:", DEBUG_IMG)
