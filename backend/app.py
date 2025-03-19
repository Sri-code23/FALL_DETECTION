import os
import cv2
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__, template_folder="templates")

# ESP32-CAM Stream URL
ESP32_URL = "http://192.168.1.3/capture"  # Replace with your ESP32-CAM IP

# Load YOLOv8 Fall Detection Model
model = YOLO("../best.pt")  # Ensure this file is in the correct directory

# Define folders for uploaded and processed images
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

@app.route("/")
def index():
    """Serve the frontend page."""
    return render_template("index.html")

def capture_image():
    """Fetches an image from the ESP32-CAM."""
    try:
        response = requests.get(ESP32_URL, timeout=5)
        if response.status_code == 200:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique timestamp
            filename = f"frame_{timestamp}.jpg"
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            with open(image_path, "wb") as file:
                file.write(response.content)
            return filename
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def process_image(image_path, filename):
    """Runs YOLOv8 fall detection on the captured image."""
    results = model.predict(image_path, save=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique timestamp
    processed_filename = f"processed_{timestamp}.jpg"
    processed_path = os.path.join(app.config["PROCESSED_FOLDER"], processed_filename)

    # Move YOLO's processed image to our processed folder
    yolo_output_path = os.path.join("runs", "detect", "predict", filename)

    if os.path.exists(yolo_output_path):
        if os.path.exists(processed_path):  
            os.remove(processed_path)  # Delete existing file if necessary
        os.rename(yolo_output_path, processed_path)  # Move the new processed image

    fall_detected = False
    confidence = 0.0

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        conf = round(float(box.conf[0]) * 100, 2)

        if class_id == 0:  # Assuming class ID 0 is "Fall"
            fall_detected = True
            confidence = max(confidence, conf)  # Use highest confidence

    return processed_filename, fall_detected, confidence

@app.route("/process", methods=["GET"])
def process_image_from_esp32():
    """Fetch an image from ESP32-CAM, process it, and return results."""
    filename = capture_image()
    if not filename:
        return jsonify({"error": "Failed to capture image from ESP32-CAM"}), 500

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    processed_filename, fall_detected, confidence = process_image(image_path, filename)

    processed_image_url = url_for("get_processed_image", filename=processed_filename, _external=True)

    return jsonify({
        "fall_detected": fall_detected,
        "confidence": confidence,
        "processed_image": processed_image_url
    })

@app.route("/processed/<filename>")
def get_processed_image(filename):
    """Serve processed images."""
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)




































# from flask import Flask, render_template, request, jsonify
# import os
# import requests
# import time
# from datetime import datetime
# from ultralytics import YOLO

# app = Flask(__name__)

# # ESP32-CAM Stream URL
# ESP32_URL = "http://192.168.1.3/capture"  # Replace with your ESP32-CAM IP

# # Load YOLOv8 Fall Detection Model
# model = YOLO("../best.pt")

# # Paths
# OUTPUT_DIR = "static/output"
# PROCESSED_DIR = "static/processed"
# REPORT_FILE = "fall_detection_report.txt"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(PROCESSED_DIR, exist_ok=True)

# def capture_image():
#     """Fetches an image from the ESP32-CAM."""
#     try:
#         response = requests.get(ESP32_URL, timeout=5)
#         if response.status_code == 200:
#             image_name = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#             image_path = os.path.join(OUTPUT_DIR, image_name)
#             with open(image_path, "wb") as file:
#                 file.write(response.content)
#             return image_name
#         else:
#             return None
#     except requests.exceptions.RequestException:
#         return None

# def detect_fall(image_name):
#     """Runs YOLOv8 fall detection on the captured image."""
#     image_path = os.path.join(OUTPUT_DIR, image_name)
#     results = model.predict(image_path, save=True)

#     processed_image = os.path.join(PROCESSED_DIR, f"processed_{image_name}")
    
#     # YOLO saves results in 'runs/detect/predict', move processed image to `static/processed`
#     yolo_output_path = os.path.join("runs", "detect", "predict", image_name)
#     if os.path.exists(yolo_output_path):
#         os.rename(yolo_output_path, processed_image)

#     fall_detected = False
#     detection_report = ""

#     for box in results[0].boxes:
#         class_id = int(box.cls[0])  
#         confidence = round(float(box.conf[0]) * 100, 2)  

#         if class_id == 0:  # Assuming "0" is the class ID for "Fallen"
#             fall_detected = True
#             detection_report += f"⚠️ Fall detected! Confidence: {confidence}%\n"
#         else:
#             detection_report += f"✅ No fall detected. Confidence: {confidence}%\n"

#     return fall_detected, processed_image, detection_report

# @app.route("/")
# def index():
#     """Renders the frontend."""
#     return render_template("index.html")

# @app.route("/process", methods=["GET"])
# def process_image():
#     """Fetches an image, runs fall detection, and returns results."""
#     image_name = capture_image()
#     if image_name:
#         fall_detected, processed_image, report = detect_fall(image_name)
        
#         # Save report
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         with open(REPORT_FILE, "a", encoding="utf-8") as f:
#             f.write(f"\n[{timestamp}] - {report}")

#         return jsonify({
#             "input_image": f"/static/output/{image_name}",
#             "output_image": f"/{processed_image}",
#             "response": report
#         })
#     else:
#         return jsonify({"error": "Failed to capture image"}), 500

# if __name__ == "__main__":
#     app.run(debug=True)























# from flask import Flask, render_template, request, jsonify, send_from_directory
# import os
# import requests
# import time
# from datetime import datetime
# from ultralytics import YOLO

# app = Flask(__name__)

# # ESP32-CAM Stream URL
# ESP32_URL = "http://192.168.1.3/capture"  # Replace with your ESP32-CAM IP

# # Load YOLOv8 Fall Detection Model
# model = YOLO("../best.pt")

# # Paths
# OUTPUT_DIR = "static/output"
# REPORT_FILE = "fall_detection_report.txt"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# def capture_image():
#     """Fetches an image from the ESP32-CAM."""
#     try:
#         response = requests.get(ESP32_URL, timeout=5)
#         if response.status_code == 200:
#             image_name = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#             image_path = os.path.join(OUTPUT_DIR, image_name)
#             with open(image_path, "wb") as file:
#                 file.write(response.content)
#             return image_name
#         else:
#             return None
#     except requests.exceptions.RequestException:
#         return None

# def detect_fall(image_name):
#     """Runs YOLOv8 fall detection on the captured image."""
#     image_path = os.path.join(OUTPUT_DIR, image_name)
#     results = model.predict(image_path, save=True)
#     output_image = os.path.join("runs", "detect", "predict", image_name)

#     fall_detected = False
#     detection_report = ""

#     for box in results[0].boxes:
#         class_id = int(box.cls[0])  
#         confidence = round(float(box.conf[0]) * 100, 2)  

#         if class_id == 0:  # Assuming "0" is the class ID for "Fallen"
#             fall_detected = True
#             detection_report += f"⚠️ Fall detected! Confidence: {confidence}%\n"
#         else:
#             detection_report += f"✅ No fall detected. Confidence: {confidence}%\n"

#     return fall_detected, output_image, detection_report

# @app.route("/")
# def index():
#     """Renders the frontend."""
#     return render_template("index.html")

# @app.route("/process", methods=["GET"])
# def process_image():
#     """Fetches an image, runs fall detection, and returns results."""
#     image_name = capture_image()
#     if image_name:
#         fall_detected, output_image, report = detect_fall(image_name)
        
#         # Save report
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         with open(REPORT_FILE, "a") as f:
#             f.write(f"\n[{timestamp}] - {report}")

#         return jsonify({
#             "input_image": f"/static/output/{image_name}",
#             "output_image": f"/{output_image}",
#             "response": report
#         })
#     else:
#         return jsonify({"error": "Failed to capture image"}), 500

# if __name__ == "__main__":
#     app.run(debug=True)
