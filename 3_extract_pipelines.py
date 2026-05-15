
import cv2
import numpy as np
import pandas as pd
import os

IMAGE_PATH = "pdf_images_visual/page_1_visual.png"
OUT_CSV = "output_image_1/pipe_lines_image_1.csv"
DEBUG_IMG = "output_image_1/debug_pipe_lines_image_1.png"

os.makedirs("output_image_1", exist_ok=True)

img = cv2.imread(IMAGE_PATH)
assert img is not None, "❌ Image not found"

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# remove noise, keep lines
blur = cv2.GaussianBlur(gray, (3,3), 0)

# adaptive threshold (NOT fixed!)
bw = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    21, 5
)

# horizontal + vertical line kernels
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))

h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

line_mask = cv2.bitwise_or(h_lines, v_lines)

# edges only AFTER line isolation
edges = cv2.Canny(line_mask, 50, 150, apertureSize=3)

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=80,
    minLineLength=60,
    maxLineGap=20
)

rows = []
debug = img.copy()
pipe_id = 1

if lines is None:
    raise RuntimeError("❌ NO LINES DETECTED — check input image")

for l in lines:
    x1, y1, x2, y2 = l[0]
    length = np.hypot(x2-x1, y2-y1)

    if length < 50:
        continue

    rows.append({
        "pipe_id": f"P{pipe_id}",
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "length": round(float(length), 2)
    })

    cv2.line(debug, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(debug, f"P{pipe_id}", (x1,y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

    pipe_id += 1

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
cv2.imwrite(DEBUG_IMG, debug)

print("✅ PIPE GEOMETRY EXTRACTED")
print("→", OUT_CSV)
print("Lines:", len(df))
