import cv2
import requests
import time
import os
from ultralytics import YOLO
from datetime import datetime

# ESP32-CAM URL (Replace with your actual ESP32-CAM IP)
ESP32_URL = "http://192.168.1.3/capture"  # Change this to your ESP32-CAM URL

# Load YOLOv8 Fall Detection Model
model = YOLO("best.pt")  # Ensure "best.pt" is in the project folder

# Output folders
output_dir = "output"
report_file = "fall_detection_report.txt"

os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

def capture_image():
    """Fetches an image from the ESP32-CAM stream."""
    try:
        response = requests.get(ESP32_URL, timeout=5)
        if response.status_code == 200:
            image_path = os.path.join(output_dir, f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            with open(image_path, "wb") as file:
                file.write(response.content)
            return image_path
        else:
            print("❌ Failed to capture image. Status Code:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print("❌ Error capturing image:", e)
        return None

def detect_fall(image_path):
    """Runs YOLOv8 fall detection on the captured image."""
    results = model.predict(image_path, save=True)
    
    # Get processed image path
    output_image_path = os.path.join("runs", "detect", "predict", os.path.basename(image_path))

    # Extract detection results
    detections = results[0].boxes
    fall_detected = False
    detection_report = ""

    for box in detections:
        class_id = int(box.cls[0])  # Get detected class
        confidence = round(float(box.conf[0]) * 100, 2)  # Confidence score

        if class_id == 0:  # Assuming class "0" corresponds to "Fallen"
            fall_detected = True
            detection_report += f"⚠️ Fall detected! Confidence: {confidence}%\n"
        else:
            detection_report += f"✅ No fall detected. Confidence: {confidence}%\n"

    return fall_detected, output_image_path, detection_report

def log_report(report_text):
    """Logs the detection report to a text file."""
    with open(report_file, "a") as f:
        f.write(report_text + "\n")
    print(report_text)

if __name__ == "__main__":
    while True:
        print("\n📸 Capturing image from ESP32-CAM...")
        image_path = capture_image()

        if image_path:
            print("🔍 Running fall detection...")
            fall_detected, output_image_path, report = detect_fall(image_path)

            # Create a log entry
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_text = f"\n[{timestamp}] - Image: {image_path}\n{report}"
            log_report(report_text)

            print(f"✅ Detection complete! Processed image saved at: {output_image_path}")

        # Wait for the next capture
        time.sleep(5)  # Capture an image every 5 seconds
