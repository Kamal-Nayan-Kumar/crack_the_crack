from ultralytics import YOLO

# Load YOLOv8 model (nano version for fast training)
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data="scripts/data.yml",
    epochs=50,
    batch=8,
    imgsz=640
)
