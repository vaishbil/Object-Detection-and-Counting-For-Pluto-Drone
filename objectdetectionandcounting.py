import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
import cvzone
import math

# Define constants
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 320
CONFIDENCE_THRESHOLD = 0.5
OBJECT_TRACKING_DICT = {}

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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

while True:
    raw_frame = ffmpeg_process.stdout.read(2048 * 1152 * 3)
    if not raw_frame:
        break
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((1152, 2048, 3))

    # Resize frame
    frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

    results = model(frame, stream=True)
    object_counts = {}

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Calculate width and height
            w, h = x2 - x1, y2 - y1

            # Draw bounding box
            cvzone.cornerRect(frame, (x1, y1, w, h), l=15)

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Class name
            class_id = int(box.cls[0])
            class_name = classNames[class_id]

            # Update object count
            if class_name in object_counts:
                object_counts[class_name] += 1
            else:
                object_counts[class_name] = 1

            # Display class name and confidence
            cvzone.putTextRect(frame, f'{class_name} {confidence}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

    # Display object counts
    total_objects = sum(object_counts.values())
    cvzone.putTextRect(frame, f'Total Objects: {total_objects}', (10, 20), scale=0.7, thickness=1)
    for i, (class_name, count) in enumerate(object_counts.items()):
        cvzone.putTextRect(frame, f'{class_name}: {count}', (10, 40 + i * 20), scale=0.7, thickness=1)

    # Display the frame
    cv2.imshow('Camera Stream', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

process.terminate()
cv2.destroyAllWindows()
