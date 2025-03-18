from ultralytics import YOLO
import cv2

# Load trained YOLO model
model = YOLO("scripts/train10/weights/best.pt")  # Update with your trained model path

# Load video
video_path = "dataset/naya_video.mp4"  # Update with your video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Unable to open video file!")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create output video writer
output_video_path = "output_detected_8.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Extract detections
    bottles = []
    cracks = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0])  # Class ID

            if conf < 0.3:  # Confidence threshold (adjust if needed)
                continue

            if cls == 0:  # Assuming class 0 = Bottle
                bottles.append((x1, y1, x2, y2))
            elif cls == 1:  # Assuming class 1 = Crack
                cracks.append((x1, y1, x2, y2))

    # Check if crack is inside bottle
    valid_detections = []
    for bx1, by1, bx2, by2 in bottles:
        crack_inside = False

        for cx1, cy1, cx2, cy2 in cracks:
            if bx1 <= cx1 and bx2 >= cx2 and by1 <= cy1 and by2 >= cy2:  # Crack inside bottle
                crack_inside = True
                valid_detections.append(("crack", (cx1, cy1, cx2, cy2)))

        if crack_inside:
            valid_detections.append(("bottle_with_crack", (bx1, by1, bx2, by2)))
        else:
            valid_detections.append(("bottle", (bx1, by1, bx2, by2)))

    # Draw valid detections
    for obj, (x1, y1, x2, y2) in valid_detections:
        if obj == "bottle":
            color = (0, 255, 0)  # Green for normal bottle
            label = "Bottle"
        elif obj == "bottle_with_crack":
            color = (0, 165, 255)  # Orange for bottle with crack
            label = "Bottle (Cracked)"
        else:
            color = (0, 0, 255)  # Red for crack
            label = "Crack"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Write frame to output video
    out.write(frame)
    cv2.imshow("Video Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"🎉 Video detection complete! Output saved as: {output_video_path}")
