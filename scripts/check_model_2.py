from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO("scripts/train/weights/best.pt")  # Update with your trained model path

# Path to input video
video_path = "input_video.mp4"  # Replace with your video file
output_path = "output_video.mp4"  # Where the processed video will be saved

# Open video
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when video ends

    # Run YOLO inference
    results = model(frame)

    # Draw detections on frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class ID

            label = f"{model.names[cls]}: {conf:.2f}"
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # Green for bottles, Red for cracks

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)  # Save the frame

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete! Output saved as:", output_path)
