
from pathlib import Path
from ultralytics import YOLO

model_path = (
    r"C:\Users\Srushti\OneDrive\Desktop\final-year-project"
    r"\runs\detect\pe_v5s_50\weights\best.pt"
)

model = YOLO(model_path)

images = [
    Path("test(1).jpeg"),
    Path("test(2).jpeg"),
    Path("test(3).jpeg"),
]

for image in images:
    result = model.predict(
        source=str(image),
        conf=0.001,
        save=True,
        project="baseline_low_conf",
        name=image.stem,
        exist_ok=True,
        verbose=False,
    )[0]

    count = len(result.boxes)

    if count:
        highest = float(result.boxes.conf.max())
        print(f"{image.name}: {count} candidate boxes; "
              f"highest confidence: {highest:.4f}")
    else:
        print(f"{image.name}: 0 candidate boxes")
