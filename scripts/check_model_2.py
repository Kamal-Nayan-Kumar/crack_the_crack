from ultralytics import YOLO
import cv2

# Load trained YOLO model
model = YOLO("scripts/train10/weights/best.pt")  # Update with your trained model path

# Load image
image_path = "dataset/naya_image.png"  # Update with your image path
image = cv2.imread(image_path)

if image is None:
    print("❌ Error: Unable to load image!")
    exit()

# Run YOLO detection
results = model(image)

# Extract detections
bottles = []
cracks = []

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0])  # Class ID

        if conf < 0.3:  # Confidence threshold (adjust if needed)
            continue

        if cls == 0:  # Assuming class 0 = Bottle
            bottles.append((x1, y1, x2, y2))
        elif cls == 1:  # Assuming class 1 = Crack
            cracks.append((x1, y1, x2, y2))

# Check if crack is inside a bottle
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

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Show the image
cv2.imshow("Image Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output image
output_image_path = "output_detected.jpg"
cv2.imwrite(output_image_path, image)
print(f"🎉 Image detection complete! Output saved as: {output_image_path}")
