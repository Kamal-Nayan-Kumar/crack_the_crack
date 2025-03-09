import cv2
import numpy as np
from detect_bottles import detect_bottles

def generate_crack_pattern(image, bbox):
    """
    Generates a realistic thin crack inside the detected bottle region.
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure the bounding box is large enough for crack generation
    if (x2 - x1) < 50 or (y2 - y1) < 50:
        return image  # Skip small boxes

    overlay = image.copy()
    
    # Define crack starting point inside the bounding box
    start_x = np.random.randint(x1 + 10, x2 - 10)  # Adjusted to avoid invalid range
    start_y = np.random.randint(y1 + 10, y2 - 10)

    # Generate a curved crack shape (Bezier-like path)
    crack_points = [(start_x, start_y)]
    num_points = np.random.randint(8, 12)
    
    for _ in range(num_points):
        next_x = np.clip(crack_points[-1][0] + np.random.randint(-15, 15), x1, x2)
        next_y = np.clip(crack_points[-1][1] + np.random.randint(10, 20), y1, y2)
        crack_points.append((next_x, next_y))
    
    # Draw the main crack line (thin and curved)
    for i in range(len(crack_points) - 1):
        cv2.line(overlay, crack_points[i], crack_points[i + 1], (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    
    # Add small branching cracks
    num_branches = np.random.randint(2, 4)
    for _ in range(num_branches):
        branch_start = crack_points[np.random.randint(2, len(crack_points) - 2)]
        branch_points = [branch_start]

        for _ in range(np.random.randint(3, 5)):
            next_x = np.clip(branch_points[-1][0] + np.random.randint(-10, 10), x1, x2)
            next_y = np.clip(branch_points[-1][1] + np.random.randint(-10, 10), y1, y2)
            branch_points.append((next_x, next_y))
        
        for i in range(len(branch_points) - 1):
            cv2.line(overlay, branch_points[i], branch_points[i + 1], (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    
    return cv2.addWeighted(overlay, 0.8, image, 0.2, 0)

def generate_synthetic_data(image_path, output_path):
    """
    Detects bottles and adds realistic cracks inside them.
    """
    image, boxes = detect_bottles(image_path)
    for bbox in boxes:
        image = generate_crack_pattern(image, bbox)
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    generate_synthetic_data("sample_image.jpg", "output_cracked.jpg")