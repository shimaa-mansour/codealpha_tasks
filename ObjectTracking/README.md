# CodeAlpha Object Detection and Tracking

Real-time object detection and tracking from a webcam or a video file.

## How it works
1. **Video input:** OpenCV reads frames from a webcam or a video file.
2. **Detection:** each frame goes through a pre-trained **YOLOv8** model (COCO, 80 classes) to get bounding boxes, labels and confidence scores.
3. **Tracking:** detections are passed to **SORT** (`sort.py`), implemented from scratch:
   - a **Kalman filter** predicts where each tracked object will be in the next frame,
   - the **Hungarian algorithm** matches detections to tracks using IoU,
   - unmatched detections start new tracks, and tracks that disappear are removed after a few frames.
4. **Display:** bounding boxes, class labels, **tracking IDs**, confidence and motion trails are drawn live, together with FPS and the number of unique objects seen.

## Technologies
Python, OpenCV, Ultralytics YOLOv8, NumPy, SciPy

## Installation
```bash
git clone https://github.com/shimaa-mansour/codealpha_tasks.git
cd codealpha_tasks/ObjectTracking
pip install -r requirements.txt
```
The YOLO weights (`yolov8n.pt`) are downloaded automatically on the first run.

## Usage
```bash
python main.py                                  # webcam
python main.py --source video.mp4               # video file
python main.py --source video.mp4 --save out.mp4 --no-display
python main.py --classes 0 2                    # only people and cars
```
Press **q** in the window to quit.

| Option | Description |
|---|---|
| `--source` | webcam index (default `0`) or video path |
| `--model` | YOLO weights, e.g. `yolov8s.pt` for higher accuracy |
| `--conf` | confidence threshold (default `0.4`) |
| `--classes` | COCO class ids to keep |
| `--save` | save the annotated video |
| `--no-display` | run without opening a window |

## Project structure
```
main.py            # video loop: detect -> track -> draw
sort.py            # SORT tracker (Kalman filter + Hungarian matching)
requirements.txt
```

## Author
Shimaa Mansour
