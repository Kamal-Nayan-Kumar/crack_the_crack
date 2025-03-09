from ultralytics import YOLO
import cv2

# Load trained YOLO model
model = YOLO("runs/detect/train4/weights/best.pt")

def detect_bottles(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    bottle_boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
    return image, bottle_boxes

# Example usage
if __name__ == "__main__":
    image, boxes = detect_bottles("dataset/frames/frame_813.jpg")
    print("Detected Bottles:", boxes)
