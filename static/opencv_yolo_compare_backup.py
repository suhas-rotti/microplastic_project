
from pathlib import Path
import cv2
from ultralytics import YOLO


# 1. Paths
MODEL_PATH = Path(
    r"C:\Users\Srushti\OneDrive\Desktop\final-year-project"
    r"\runs\detect\pe_v5s_50\weights\best.pt"
)

IMAGE_PATH = Path(
    r"C:\Users\Srushti\OneDrive\Desktop\microplastic_project"
    r"\IMG_20261013_145715_SR[1].jpg"
)

OUTPUT_DIR = Path("opencv_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load the original image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Cannot read image: {IMAGE_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find model: {MODEL_PATH}"
    )



# Experimental bright-particle enhancement
# Work on the original image; do not overwrite it.

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Estimate uneven background
background = cv2.GaussianBlur(
    gray, (0, 0), sigmaX=15, sigmaY=15
)

# Highlight local bright features
bright_features = cv2.subtract(gray, background)

# Improve contrast for visual inspection
bright_features = cv2.normalize(
    bright_features, None, 0, 255, cv2.NORM_MINMAX
)

# Convert to 3 channels for YOLO
bright_particle_image = cv2.cvtColor(
    bright_features, cv2.COLOR_GRAY2BGR
)

cv2.imwrite(
    str(OUTPUT_DIR / "bright_particle_preview.jpg"),
    bright_particle_image
)

# Run the same model on the enhanced image
bright_result = model.predict(
    source=bright_particle_image,
    imgsz=960,
    conf=0.001,
    save=False,
    verbose=False
)[0]

cv2.imwrite(
    str(OUTPUT_DIR / "bright_particle_yolo.jpg"),
    bright_result.plot()
)

print("Bright-particle detections:", len(bright_result.boxes))



# 4. Load the SAME trained YOLO model
model = YOLO(str(MODEL_PATH))


# 5. Red/pink colour-selective experiment
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Red hue range near zero
lower_red1 = cv2.inRange(
    hsv,
    (0, 30, 50),
    (10, 255, 255)
)

# Red/magenta hue range near the end of OpenCV's hue scale
upper_red2 = cv2.inRange(
    hsv,
    (160, 30, 50),
    (179, 255, 255)
)

# Combine both hue ranges
red_pink_mask = cv2.bitwise_or(
    lower_red1, upper_red2
)

# Retain selected pixels; other pixels become black
red_pink_enhanced_image = cv2.bitwise_and(
    image,
    image,
    mask=red_pink_mask
)

cv2.imwrite(
    str(OUTPUT_DIR / "red_pink_selective.jpg"),
    red_pink_enhanced_image
)


# 6. Run YOLO on the original image
original_result = model.predict(
    source=image,
    imgsz=960,
    conf=0.001,
    save=False,
    verbose=False
)[0]


# 7. Run YOLO on the background-removed image
filtered_result = model.predict(
    source=filtered_image,
    imgsz=960,
    conf=0.001,
    save=False,
    verbose=False
)[0]


# 8. Run YOLO on the red/pink selective image
red_pink_result = model.predict(
    source=red_pink_enhanced_image,
    imgsz=960,
    conf=0.001,
    save=False,
    verbose=False
)[0]


# 9. Save annotated outputs
cv2.imwrite(
    str(OUTPUT_DIR / "original_yolo.jpg"),
    original_result.plot()
)

cv2.imwrite(
    str(OUTPUT_DIR / "filtered_yolo.jpg"),
    filtered_result.plot()
)

cv2.imwrite(
    str(OUTPUT_DIR / "red_pink_yolo.jpg"),
    red_pink_result.plot()
)


# 10. Print detection counts
print("\n--- YOLO + OPENCV COMPARISON ---")
print("Model classes:", model.names)

print(
    "Original detections:",
    len(original_result.boxes)
)

print(
    "Background-removed detections:",
    len(filtered_result.boxes)
)

print(
    "Red/pink detections:",
    len(red_pink_result.boxes)
)


# 11. Print confidence scores for comparison
for label, result in [
    ("Original", original_result),
    ("Background-removed", filtered_result),
    ("Red/pink selective", red_pink_result),
]:
    print(f"\n{label} image:")
    print("Detections:", len(result.boxes))

    if result.boxes is not None and len(result.boxes) > 0:
        scores = result.boxes.conf.cpu().tolist()
        print("Confidence scores:", scores)
    else:
        print("No detections returned.")


# 12. List saved files
print("\nSaved files:")
print(OUTPUT_DIR.resolve())
print(" - background_removed_preview.jpg")
print(" - red_pink_selective.jpg")
print(" - original_yolo.jpg")
print(" - filtered_yolo.jpg")
print(" - red_pink_yolo.jpg")
