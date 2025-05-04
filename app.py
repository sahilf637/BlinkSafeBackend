from flask import Flask, request, jsonify
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from flask_cors import CORS
import logging
import cv2
import os
from inference_sdk import InferenceHTTPClient

# Initialize Flask app
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Roboflow Client for Redness and Dryness Detection
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="B2qR7Mx5VzokQdgQU5xQ"
)

# Load Haar Cascades from OpenCV’s built-in data
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def decode_base64_image(encoded_string):
    """Decode base64 string (with or without data URL header) into an OpenCV image."""
    # Strip off any header (e.g. "data:image/jpeg;base64,")
    if "," in encoded_string:
        encoded_string = encoded_string.split(",", 1)[1]
    try:
        img_data = base64.b64decode(encoded_string)
    except Exception as e:
        raise ValueError(f"Base64 decoding failed: {e}")
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("OpenCV could not decode the image buffer")
    return img


def extract_eyes(image):
    """Detects and extracts both eyes from a given image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]  # Take the first detected face
    face_roi_color = image[y:y+h, x:x+w]
    face_roi_gray = gray[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(face_roi_gray)
    if len(eyes) < 2:
        return None, None

    eye_images = []
    for (ex, ey, ew, eh) in eyes[:2]:  # First two detected eyes
        eye_crop = face_roi_color[ey:ey+eh, ex:ex+ew]
        eye_resized = cv2.resize(eye_crop, (224, 224))
        eye_images.append(eye_resized)

    return eye_images[0], eye_images[1]

@app.route("/predict", methods=["POST"])
def predict():
    logging.debug("🔍 Received request for eye redness and dryness detection")
    try:
        data = request.get_json(force=True)
        logging.debug(f"Payload keys: {list(data.keys())}")
        if "image" not in data:
            return jsonify({"error": "No image field in JSON"}), 400

        image = decode_base64_image(data["image"])
        logging.debug("✅ Decoded image, shape: %s", image.shape)

        left_eye, right_eye = extract_eyes(image)
        if left_eye is None or right_eye is None:
            return jsonify({"error": "Could not detect two eyes"}), 400
        logging.debug("✅ Extracted both eyes")

        # Convert to PIL and save
        left_pil = Image.fromarray(cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB))
        right_pil = Image.fromarray(cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB))
        left_pil.save("left_eye.jpg")
        right_pil.save("right_eye.jpg")
        logging.debug("✅ Saved eye crops to disk")

        # Redness Detection
        left_red = CLIENT.infer("left_eye.jpg", model_id="red-eye-bj8nk/1")
        right_red = CLIENT.infer("right_eye.jpg", model_id="red-eye-bj8nk/1")
        lr = left_red["predictions"][0]["confidence"] if left_red["predictions"] else 0
        rr = right_red["predictions"][0]["confidence"] if right_red["predictions"] else 0
        avg_redness = round(((lr + rr) / 2) * 100, 2)

        # Dryness Detection
        left_dry = CLIENT.infer("left_eye.jpg", model_id="dry-eye-disease-l9jt2/4")
        right_dry = CLIENT.infer("right_eye.jpg", model_id="dry-eye-disease-l9jt2/4")
        ld = left_dry["predictions"][0]["confidence"] if left_dry["predictions"] else 0
        rd = right_dry["predictions"][0]["confidence"] if right_dry["predictions"] else 0
        avg_dryness = round(((ld + rd) / 2) * 100, 2)

        # Cleanup
        for f in ("left_eye.jpg", "right_eye.jpg"):
            try: os.remove(f)
            except FileNotFoundError: logging.warning(f"{f} not found for deletion.")

        return jsonify({
            "average_redness": avg_redness,
            "average_dryness": avg_dryness
        })

    except Exception as e:
        logging.exception("❌ Error in /predict")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)