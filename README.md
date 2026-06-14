# Object-Detection-and-Counting-For-Pluto-Drone

Real-time object detection and counting system built for the **Pluto drone** by [Drona Aviation](https://dronaaviation.com), developed during an internship at Drona Aviation in 2024. This project integrates the Pluto drone's WiFi camera module with **YOLOv8** to perform live object detection and counting directly from the drone's aerial feed.

---

## About the Project

During my internship at **Drona Aviation** (2024), I worked with the Pluto drone platform — India's leading DIY educational nano drone — and its WiFi camera module to build a vision-based object detection and counting pipeline.

The system streams live video from the drone camera, decodes it using FFmpeg, and runs YOLOv8 inference on each frame to detect and count objects in real time, displaying bounding boxes, class labels, confidence scores, and a live object count overlay on screen.

---

## Hardware

- **Drone**: Pluto / Pluto X by [Drona Aviation](https://dronaaviation.com)
- **Camera Module**: Drona Aviation WiFi Camera Module
  - Resolution: 720p / 1080p video, 1MP / 2MP photo
  - Live transmission: ~18 FPS
  - Weight: 8g (with casing)
  - Range: 30m
  - SD card supported

---

## File Structure

| File | Description |
|---|---|
| `camerastream.py` | Streams raw camera feed from the Pluto drone using `pylwdrone` and displays it with OpenCV |
| `imageclick.py` | Captures a single image from the drone camera |
| `imageclickadvanced.py` | Advanced image capture with additional configuration |
| `objectdetectionandcounting.py` | **Main script** — live object detection and counting on the drone's camera stream using YOLOv8 |
| `detectpersonsonly.py` | Filtered detection that identifies and counts only persons in the frame |
| `objectcountingusingwebcam.py` | Standalone version of the detection pipeline for testing with a local webcam (no drone required) |
| `yolov8n.pt` | Pre-trained YOLOv8 nano model weights |

---

## How It Works

1. **Stream**: The `pylwdrone` CLI tool connects to the Pluto drone over WiFi and pipes raw video to stdout.
2. **Decode**: FFmpeg decodes the stream into raw BGR frames (2048×1152 resolution).
3. **Resize**: Frames are resized to 640×320 for efficient inference.
4. **Detect**: YOLOv8n runs inference on each frame, returning bounding boxes, class IDs, and confidence scores.
5. **Count**: Detected objects are grouped by class and counted per frame.
6. **Display**: `cvzone` overlays bounding boxes, labels, confidence scores, and a live object count summary on the video window.

---

## Tech Stack

- **Python 3**
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection model
- **OpenCV** (`cv2`) — frame display and image processing
- **cvzone** — bounding box and text overlay utilities
- **NumPy** — raw frame buffer handling
- **FFmpeg** — video stream decoding
- **pylwdrone** — Pluto drone SDK / CLI for camera streaming

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- FFmpeg installed and available in your PATH
- `pylwdrone` installed (Drona Aviation's drone control package)
- Pluto drone connected via WiFi

### Install Python Dependencies

```bash
pip install ultralytics opencv-python cvzone numpy
```

### Install pylwdrone

Refer to [Drona Aviation's GitHub](https://github.com/DronaAviation) for the latest installation instructions for `pylwdrone`.

---

## 🚀 Usage

### 1. Test your camera stream

Before running detection, verify the drone camera feed is working:

```bash
python camerastream.py
```

Press `q` to quit.

### 2. Run object detection on the drone feed

```bash
python objectdetectionandcounting.py
```

This will connect to the Pluto drone's camera, stream video, and display real-time detection with object counts overlaid on the frame.

### 3. Detect persons only

```bash
python detectpersonsonly.py
```

### 4. Test without a drone (webcam)

If you don't have a Pluto drone handy, you can test the detection pipeline using your system webcam:

```bash
python objectcountingusingwebcam.py
```

---

## Detected Object Classes

The model detects all 80 COCO classes including people, vehicles, animals, and everyday objects — for example: `person`, `car`, `bicycle`, `motorbike`, `bus`, `truck`, `bird`, `cat`, `dog`, `bottle`, `chair`, `laptop`, `cell phone`, and more.


---

## Author

Developed as an intern project at **Drona Aviation** (2024).

---

## 📄 License

This project is open for educational and research use. Please refer to Drona Aviation's resources and the Ultralytics YOLOv8 license for usage of their respective tools and models.
