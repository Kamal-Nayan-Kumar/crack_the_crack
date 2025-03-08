from ultralytics import YOLO
import cv2

# Load your trained model (replace 'best.pt' with your trained model file)
model = YOLO("runs/detect/train4/weights/best.pt")  

# Load the video file
video_path = "dataset/video.mp4"  # Replace with your actual video file path
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video

# Create VideoWriter to save output
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if no more frames

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Draw detections on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            conf = box.conf[0].item()  # Get confidence score
            cls = int(box.cls[0].item())  # Get class ID
            
            # Draw bounding box with label
            label = f"bottle {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to stop
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
