from flask import Flask, request, jsonify
import torch
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from flask_cors import CORS
import logging
import cv2
from inference_sdk import InferenceHTTPClient

# Initialize Flask app
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Load YOLO model (Ensure best.pt is in the same directory)
model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", force_reload=True)

# Initialize Roboflow Client for Dryness Detection
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="B2qR7Mx5VzokQdgQU5xQ"
)

# Load Haar Cascade for eye detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def decode_base64_image(encoded_string):
    """Decode base64 string into OpenCV image."""
    img_data = base64.b64decode(encoded_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
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

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(face_roi_gray)
    if len(eyes) < 2:
        return None, None

    eye_images = []
    for (ex, ey, ew, eh) in eyes[:2]:  # Get first two detected eyes
        eye_crop = face_roi_color[ey:ey+eh, ex:ex+ew]
        eye_resized = cv2.resize(eye_crop, (224, 224))  # Resize for ML model
        eye_images.append(eye_resized)

    return eye_images[0], eye_images[1]

@app.route("/predict", methods=["POST"])
def predict():
    logging.debug("Received request for eye redness and dryness detection")
    try:
        data = request.get_json()
        if "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode Base64 Image
        image = decode_base64_image(data["image"])
        left_eye, right_eye = extract_eyes(image)

        if left_eye is None or right_eye is None:
            return jsonify({"error": "Both eyes not detected"}), 400

        # Convert eyes to PIL format for YOLO and Roboflow
        left_pil = Image.fromarray(cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB))
        right_pil = Image.fromarray(cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB))
        
        left_pil.save("left_eye.jpg")
        right_pil.save("right_eye.jpg")
        
        # Run YOLO inference for Redness Detection
        left_results = model(left_pil)
        right_results = model(right_pil)

        left_redness = max([row["confidence"] for _, row in left_results.pandas().xyxy[0].iterrows()], default=0)
        right_redness = max([row["confidence"] for _, row in right_results.pandas().xyxy[0].iterrows()], default=0)

        avg_redness = round(((left_redness + right_redness) / 2) * 100, 2)

        # Run Roboflow API for Dryness Detection
        left_dryness_result = CLIENT.infer("left_eye.jpg", model_id="dry-eye-disease-l9jt2/4")
        right_dryness_result = CLIENT.infer("right_eye.jpg", model_id="dry-eye-disease-l9jt2/4")
        
        left_dryness = left_dryness_result["predictions"][0]["confidence"] if left_dryness_result["predictions"] else 0
        right_dryness = right_dryness_result["predictions"][0]["confidence"] if right_dryness_result["predictions"] else 0

        avg_dryness = round(((left_dryness + right_dryness) / 2) * 100, 2)

        return jsonify({
            "average_redness": avg_redness,
            "average_dryness": avg_dryness
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
