from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Load pretrained model

# Train the model

results = model.train(
    data="dataset/dataset_cracked/data.yaml",
    epochs=100,
    batch=4,
    imgsz=640
)

print("✅ YOLOv8 Training Completed!")
