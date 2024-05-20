import cv2
import subprocess
import numpy as np

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
    # Display the frame
    cv2.imshow('Camera Stream', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

process.terminate()
cv2.destroyAllWindows()