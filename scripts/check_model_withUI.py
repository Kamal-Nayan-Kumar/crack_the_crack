import cv2
import os
import tkinter as tk
from tkinter import Label
from ultralytics import YOLO
from PIL import Image, ImageTk

# Load YOLO model
model = YOLO("scripts/train10/weights/best.pt")  # Update model path
video_path = "dataset/final_video.mp4"  # Update video path

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Unable to open video file!")
    exit()

# Create folder to save crack frames
crack_frame_folder = "dataset/crack_frames_3"
os.makedirs(crack_frame_folder, exist_ok=True)

# Tkinter UI setup
root = tk.Tk()
root.title("Bottle & Crack Detection")
root.geometry("800x600")

# Labels
video_label = Label(root)
video_label.pack()

bottle_count_label = Label(root, text="Bottles: 0", font=("Arial", 14), fg="green")
bottle_count_label.pack()
crack_count_label = Label(root, text="Cracks: 0", font=("Arial", 14), fg="red")
crack_count_label.pack()

def update_video():
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    results = model(frame)
    bottles, cracks = 0, 0
    frame_saved = False

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])

            if conf < 0.3:
                continue

            color, label = (0, 255, 0), "Bottle"
            if cls == 1:  # Assuming class 1 = Crack
                color, label, cracks = (0, 0, 255), "Crack", cracks + 1
                if not frame_saved:  # Save only one frame per detection cycle
                    frame_filename = os.path.join(crack_frame_folder, f"crack_frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    frame_saved = True
            elif cls == 0:  # Assuming class 0 = Bottle
                bottles += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Update UI counters
    bottle_count_label.config(text=f"Bottles: {bottles}")
    crack_count_label.config(text=f"Cracks: {cracks}")

    # Convert frame to Tkinter format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_video)  # Update every 10ms

update_video()
root.mainloop()
