import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
import cvzone
import math
import os

# Define constants
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 320
CONFIDENCE_THRESHOLD = 0.5
OBJECT_TRACKING_DICT = {}
IMAGE_FOLDER = "images"
PREVIOUS_PERSON_COUNT = 0
PREVIOUS_COUNTS = []  # Add this line

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define class names
CLASS_NAMES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

process = subprocess.Popen(
    ["pylwdrone", "stream", "start", "--out-file", "-"],
    stdout=subprocess.PIPE
)

ffmpeg_process = subprocess.Popen(
    ["ffmpeg", "-i", "-", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
    stdin=process.stdout,
    stdout=subprocess.PIPE
)

person_count = 0
image_counter = 0

while True:
    raw_frame = ffmpeg_process.stdout.read(2048 * 1152 * 3)
    if not raw_frame:
        break
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((1152, 2048, 3))

    # Resize frame
    frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

    results = model(frame, stream=True)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Calculate width and height
            w, h = x2 - x1, y2 - y1

            # Check if class name is person
            class_id = int(box.cls[0])
            class_name = CLASS_NAMES[class_id]
            confidence = box.conf[0]

            # Check if confidence is greater than 0.5
            if confidence > 0.5:
                if class_name == "person":
                    # Draw bounding box
                    cvzone.cornerRect(frame, (x1, y1, w, h), l=15)

                    # Display class name and confidence
                    cvzone.putTextRect(frame, f'{class_name} {confidence:.2f}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

                    # Increment person count
                    person_count += 1

    # Display object counts
    cvzone.putTextRect(frame, f'Total Persons: {person_count}', (10, 20), scale=0.7, thickness=1)

    if person_count > 2 and person_count not in PREVIOUS_COUNTS:
        # Capture 2 images
        for i in range(2):
            image_path = os.path.join(IMAGE_FOLDER, f"image_{image_counter}.jpg")
            cv2.imwrite(image_path, frame)
            image_counter += 1
        PREVIOUS_COUNTS.append(person_count)

    # Reset person count
    person_count = 0

    # Display the frame
    cv2.imshow('Drone Camera Stream', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

process.terminate()
cv2.destroyAllWindows()