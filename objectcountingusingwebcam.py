from ultralytics import YOLO
import cv2
import cvzone
import math

# Define constants
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 320
CONFIDENCE_THRESHOLD = 0.5
OBJECT_TRACKING_DICT = {}

# Initialize video capture
cap = cv2.VideoCapture(0)  # for webcam
cap.set(3, VIDEO_WIDTH)
cap.set(4, VIDEO_HEIGHT)

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

while True:
    success, img = cap.read()
    results = model(img, stream=True)
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
            cvzone.cornerRect(img, (x1, y1, w, h), l=15)

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
            cvzone.putTextRect(img, f'{class_name} {confidence}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

    # Display object counts
    total_objects = sum(object_counts.values())
    cvzone.putTextRect(img, f'Total Objects: {total_objects}', (10, 20), scale=0.7, thickness=1)
    for i, (class_name, count) in enumerate(object_counts.items()):
        cvzone.putTextRect(img, f'{class_name}: {count}', (10, 40 + i * 20), scale=0.7, thickness=1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)