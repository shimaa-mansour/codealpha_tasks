"""
SORT: Simple Online and Realtime Tracking (Bewley et al., 2016).

Each tracked object has a Kalman filter (constant-velocity model) over its
bounding box. On every frame:
  1. Predict the new position of every existing track.
  2. Match detections to predicted tracks with the Hungarian algorithm (IoU cost).
  3. Update matched tracks, create tracks for unmatched detections,
     and delete tracks that have not been seen for `max_age` frames.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4) in [x1,y1,x2,y2] format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def box_to_z(box):
    """[x1,y1,x2,y2] -> [cx, cy, area, aspect_ratio]."""
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / max(h, 1e-6)])


def x_to_box(x):
    """Kalman state -> [x1,y1,x2,y2]."""
    area, ratio = max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(area * ratio)
    h = area / w
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class KalmanBoxTracker:
    """One tracked object. State: [cx, cy, s, r, vx, vy, vs]."""
    _next_id = 1

    def __init__(self, box, label=None, score=0.0):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0   # position += velocity
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)                           # we only observe the box
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])             # measurement noise
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])  # initial uncertainty
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4])   # process noise
        self.x = np.zeros(7)
        self.x[:4] = box_to_z(box)

        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1
        self.label, self.score = label, score
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0

    def predict(self):
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return x_to_box(self.x)

    def update(self, box, label=None, score=0.0):
        z = box_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        if label is not None:
            self.label, self.score = label, score

    @property
    def box(self):
        return x_to_box(self.x)


class Sort:
    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """
        detections: list of (x1, y1, x2, y2, score, label).
        Returns a list of (x1, y1, x2, y2, track_id, label, score) for confirmed tracks.
        """
        self.frame_count += 1
        det_boxes = np.array([d[:4] for d in detections], dtype=float).reshape(-1, 4)

        # 1) predict existing tracks
        predicted = []
        for t in self.trackers:
            box = t.predict()
            predicted.append(box if np.all(np.isfinite(box)) else np.zeros(4))
        predicted = np.array(predicted).reshape(-1, 4)

        # 2) associate detections with tracks (Hungarian on IoU)
        iou = iou_matrix(det_boxes, predicted)
        matches, unmatched_dets = [], list(range(len(detections)))
        if iou.size:
            rows, cols = linear_sum_assignment(-iou)
            unmatched_dets = [d for d in range(len(detections)) if d not in rows]
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_threshold:
                    matches.append((r, c))
                else:
                    unmatched_dets.append(r)

        # 3) update matched, create new tracks
        for d, t in matches:
            det = detections[d]
            self.trackers[t].update(det[:4], det[5], det[4])
        for d in unmatched_dets:
            det = detections[d]
            self.trackers.append(KalmanBoxTracker(det[:4], det[5], det[4]))

        # 4) output confirmed tracks, drop dead ones
        results = []
        for t in self.trackers:
            confirmed = t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            if t.time_since_update < 1 and confirmed:
                b = t.box
                results.append((*b, t.id, t.label, t.score))
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return results
