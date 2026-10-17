
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

# 2. Check paths and load the original image
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not IMAGE_PATH.is_file():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise ValueError(f"OpenCV could not read: {IMAGE_PATH}")

print("Original image size:", image.shape)
print("Loading trained YOLO model...")

# 3. Load the model BEFORE making any predictions
model = YOLO(str(MODEL_PATH))
print("Model classes:", model.names)

# 4. Experimental background subtraction
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

background = cv2.GaussianBlur(
    gray, (0, 0), sigmaX=15, sigmaY=15
)

bright_features = cv2.subtract(gray, background)

bright_features = cv2.normalize(
    bright_features, None, 0, 255, cv2.NORM_MINMAX
)

bright_particle_image = cv2.cvtColor(
    bright_features, cv2.COLOR_GRAY2BGR
)

cv2.imwrite(
    str(OUTPUT_DIR / "bright_particle_preview.jpg"),
    bright_particle_image
)

# 5. Experimental red/pink colour selection
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red1 = cv2.inRange(
    hsv, (0, 30, 50), (10, 255, 255)
)

upper_red2 = cv2.inRange(
    hsv, (160, 30, 50), (179, 255, 255)
)

red_pink_mask = cv2.bitwise_or(lower_red1, upper_red2)

red_pink_image = cv2.bitwise_and(
    image, image, mask=red_pink_mask
)

cv2.imwrite(
    str(OUTPUT_DIR / "red_pink_selective.jpg"),
    red_pink_image
)

# 6. Run the SAME model on all three images
# A low threshold is used for exploratory comparison only.
CONFIDENCE_THRESHOLD = 0.01

test_images = [
    ("Original", image, "original_yolo.jpg"),
    (
        "Background-subtracted",
        bright_particle_image,
        "bright_particle_yolo.jpg"
    ),
    (
        "Red/pink-selective",
        red_pink_image,
        "red_pink_yolo.jpg"
    ),
]

print("\n--- YOLO + OPENCV COMPARISON ---")

for label, test_image, output_name in test_images:
    result = model.predict(
        source=test_image,
        imgsz=960,
        conf=CONFIDENCE_THRESHOLD,
        save=False,
        verbose=False
    )[0]

    cv2.imwrite(
        str(OUTPUT_DIR / output_name),
        result.plot()
    )

    scores = (
        result.boxes.conf.cpu().tolist()
        if result.boxes is not None and len(result.boxes) > 0
        else []
    )

    print(f"\n{label}:")
    print("Detection count:", len(scores))
    print("Confidence scores:", [round(s, 4) for s in scores])

print("\nSaved outputs in:", OUTPUT_DIR.resolve())
