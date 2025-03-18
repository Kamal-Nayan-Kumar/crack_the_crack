from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("scripts/train10/weights/best.pt")  # Update the path if needed

# Run validation on the dataset
metrics = model.val()

# Print key performance metrics
print(f"mAP@50: {metrics['metrics/mAP_50']:.4f}")
print(f"mAP@50-95: {metrics['metrics/mAP_50-95']:.4f}")
