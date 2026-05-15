import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ---------------- CONFIG ----------------
IMAGE_PATH = "./input/pnid8.png"
OUTPUT_FOLDER = "./output"
UPSCALED_FOLDER = "./output_upscaled"

TILE = 1024
UPSCALE = 2

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(UPSCALED_FOLDER, exist_ok=True)

# ---------------- UPSCALE HELPERS ----------------
def enhance_tile(tile):
    pil = Image.fromarray(tile)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    w, h = pil.size
    pil = pil.resize((w * UPSCALE, h * UPSCALE), Image.LANCZOS)
    pil = ImageEnhance.Contrast(pil).enhance(1.4)
    pil = pil.filter(ImageFilter.UnsharpMask(radius=1, percent=160, threshold=3))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def upscale_image_tiled(img):
    H, W = img.shape[:2]
    out = np.zeros((H * UPSCALE, W * UPSCALE, 3), dtype=np.uint8)

    for y in range(0, H, TILE):
        for x in range(0, W, TILE):
            tile = img[y:y + TILE, x:x + TILE]
            enhanced = enhance_tile(tile)
            out[y*UPSCALE:y*UPSCALE+enhanced.shape[0],
                x*UPSCALE:x*UPSCALE+enhanced.shape[1]] = enhanced

    return out

def crop_main_drawing(img, pad=20):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Binary ink mask
    _, bw = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    h, w = bw.shape

    # 2) AGGRESSIVELY REMOVE PAGE FRAME
    # (thicker than frame line width)
    frame_thickness = int(min(h, w) * 0.03)

    bw[:frame_thickness, :] = 0
    bw[-frame_thickness:, :] = 0
    bw[:, :frame_thickness] = 0
    bw[:, -frame_thickness:] = 0

    # 3) Close drawing lines so it becomes one component
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    # 4) Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    if num_labels <= 1:
        print("⚠️ No components found")
        return img

    # 5) Pick LARGEST component (actual drawing)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, wc, hc, _ = stats[largest]

    # 6) Final padded crop
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + wc + pad)
    y1 = min(h, y + hc + pad)

    print(f"✅ Final crop: ({x0},{y0}) → ({x1},{y1})")
    return img[y0:y1, x0:x1]

# ---------------- MAIN ----------------
if __name__ == "__main__":
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise RuntimeError("Failed to load image")

    cropped = crop_main_drawing(img)

    cropped_path = os.path.join(OUTPUT_FOLDER, "pnid_cropped.png")
    cv2.imwrite(cropped_path, cropped)

    upscaled = upscale_image_tiled(cropped)
    upscaled_path = os.path.join(UPSCALED_FOLDER, "pnid_cropped_upscaled.png")
    cv2.imwrite(upscaled_path, upscaled)

    print("✅ Done")
