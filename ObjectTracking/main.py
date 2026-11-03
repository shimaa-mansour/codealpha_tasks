"""
Real-time object detection and tracking.

  Detection : YOLOv8 (pre-trained on COCO, via ultralytics)
  Tracking  : SORT (Kalman filter + Hungarian algorithm), implemented in sort.py
  Video I/O : OpenCV

Examples:
  python main.py                         # webcam
  python main.py --source video.mp4      # video file
  python main.py --source video.mp4 --save output.mp4 --no-display
"""
import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

from sort import Sort


def parse_args():
    p = argparse.ArgumentParser(description="Object detection + SORT tracking")
    p.add_argument("--source", default="0", help="webcam index (0) or path to a video file")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO weights (downloaded automatically if missing)")
    p.add_argument("--conf", type=float, default=0.4, help="detection confidence threshold")
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="only track these COCO class ids, e.g. --classes 0 2 (person, car)")
    p.add_argument("--save", default=None, help="write the annotated video to this path")
    p.add_argument("--no-display", action="store_true", help="do not open a window (useful on servers)")
    return p.parse_args()


def color_for(track_id: int):
    """A stable, distinct color per track id."""
    rng = np.random.default_rng(track_id * 9973)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def draw(frame, tracks, trails):
    for x1, y1, x2, y2, tid, label, score in tracks:
        x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)
        color = color_for(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"ID {tid} {label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # motion trail of the object's center
        trail = trails.setdefault(tid, [])
        trail.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
        del trail[:-30]
        for a, b in zip(trail, trail[1:]):
            cv2.line(frame, a, b, color, 2)


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.model)
    names = model.names
    tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, size)

    trails, seen_ids, frames, t0 = {}, set(), 0, time.time()
    print("Running... press 'q' in the window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) detect
        result = model.predict(frame, conf=args.conf, classes=args.classes, verbose=False)[0]
        detections = [
            (*box.xyxy[0].tolist(), float(box.conf[0]), names[int(box.cls[0])])
            for box in result.boxes
        ]

        # 2) track
        tracks = tracker.update(detections)
        seen_ids.update(int(t[4]) for t in tracks)

        # 3) draw + show
        draw(frame, tracks, trails)
        frames += 1
        fps = frames / (time.time() - t0)
        info = f"FPS: {fps:.1f}  Tracked now: {len(tracks)}  Unique IDs: {len(seen_ids)}"
        (iw, ih), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (0, 0), (iw + 20, ih + 20), (0, 0, 0), -1)
        cv2.putText(frame, info, (10, ih + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

        if writer:
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("Object Detection & Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved annotated video to {args.save}")
    cv2.destroyAllWindows()
    print(f"Processed {frames} frames, {len(seen_ids)} unique objects tracked.")


if __name__ == "__main__":
    main()
