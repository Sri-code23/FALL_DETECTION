import cv2
from ultralytics import YOLO

# Load the pre-trained fall detection model
model = YOLO("best.pt")  # Use the downloaded model

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform fall detection
    results = model.predict(frame)  
    frame = results[0].plot()  # Draw detections

    cv2.imshow("Fall Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
