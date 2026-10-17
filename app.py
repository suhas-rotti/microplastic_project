
from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
ESP32_CAPTURE_URL = "http://10.27.5.135/capture"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -------------------------------------------------
# Dashboard
# -------------------------------------------------

@app.route("/")
def home():

    images = []

    for file in os.listdir(UPLOAD_FOLDER):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(file)

    images.sort(
        key=lambda image: os.path.getmtime(
            os.path.join(UPLOAD_FOLDER, image)
        ),
        reverse=True
    )

    latest_image = images[0] if images else None

    return render_template(
        "index.html",
        latest_image=latest_image,
        images=images
    )


# -------------------------------------------------
# Capture one image directly from ESP32-CAM
# -------------------------------------------------

@app.route("/save_image", methods=["POST"])
def save_image():

    try:
        with urlopen(ESP32_CAPTURE_URL, timeout=15) as response:
            image_data = response.read()

        # Basic check that the response is a JPEG image
        if not image_data.startswith(b"\xff\xd8"):
            return jsonify({
                "success": False,
                "message": "ESP32 did not return a valid JPEG image."
            }), 500

        filename = datetime.now().strftime(
            "image_%Y%m%d_%H%M%S_%f.jpg"
        )

        image_path = os.path.join(UPLOAD_FOLDER, filename)

        with open(image_path, "wb") as image_file:
            image_file.write(image_data)

        print(f"Image saved: {filename}")

        return jsonify({
            "success": True,
            "message": "Image saved successfully.",
            "filename": filename
        })

    except (URLError, HTTPError, TimeoutError, OSError) as error:

        print(f"Camera capture error: {error}")

        return jsonify({
            "success": False,
            "message": "Could not capture image from ESP32-CAM. Check the camera connection."
        }), 500


# -------------------------------------------------
# Run Flask
# -------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
