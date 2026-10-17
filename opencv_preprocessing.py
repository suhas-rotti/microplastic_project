
from pathlib import Path
import cv2
import numpy as np

IMAGE_PATH = Path("test(1).jpeg")
OUTPUT_DIR = Path("opencv_trials_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise FileNotFoundError(f"Cannot read {IMAGE_PATH}")

# Work with brightness rather than the blue-purple colour cast.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Emphasize small bright details.
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (21, 21)
)
tophat = cv2.morphologyEx(
    gray, cv2.MORPH_TOPHAT, kernel
)

# Reduce pixel-level noise.
tophat = cv2.GaussianBlur(tophat, (3, 3), 0)

# Threshold the enhanced details.
_, raw_mask = cv2.threshold(
    tophat, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Remove tiny isolated regions; keep larger candidate regions.
count, labels, stats, _ = cv2.connectedComponentsWithStats(
    raw_mask, connectivity=8
)

clean_mask = np.zeros_like(raw_mask)
MIN_AREA = 5

for i in range(1, count):
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= MIN_AREA:
        clean_mask[labels == i] = 255

# Visualize candidates in yellow on black.
yellow = np.zeros_like(image)
yellow[clean_mask > 0] = (0, 255, 255)

cv2.imwrite(str(OUTPUT_DIR / "01_original.jpg"), image)
cv2.imwrite(str(OUTPUT_DIR / "02_tophat.jpg"), tophat)
cv2.imwrite(str(OUTPUT_DIR / "03_raw_mask.jpg"), raw_mask)
cv2.imwrite(str(OUTPUT_DIR / "04_clean_mask.jpg"), clean_mask)
cv2.imwrite(str(OUTPUT_DIR / "05_yellow_candidates.jpg"), yellow)

print("Saved outputs to:", OUTPUT_DIR.resolve())
print("These are candidate features, not confirmed microplastics.")
