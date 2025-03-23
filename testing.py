from ultralytics import YOLO

# Load the trained model
model = YOLO("best.pt")  # Ensure "best.pt" is in the project folder

# Perform fall detection on an image
results = model.predict("sample2.jpg", save=True)

# Output the results
print("✅ Detection complete! Check the 'runs/detect/predict/' folder.")
