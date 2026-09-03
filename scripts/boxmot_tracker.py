import numpy as np
from collections import deque
from pathlib import Path

# Use BoxMOT's official factory builder to remove manual class parsing boilerplate
from boxmot.trackers.registry import create_tracker


class BoxMOTTracker:
    """
    Wraps any BoxMOT tracker behind the interface expected by classify_track.py.

    Usage:
        tracker = BoxMOTTracker(
            tracker_type='ocsort',
            class_names=['fish'],
            frame_rate=fps / (frame_skip + 1),
            det_thresh=0.25,
            max_age=45,
            min_hits=3,
            iou_threshold=0.2
        )
        assignment = tracker.update(detections, frame)
    """

    def __init__(
        self,
        tracker_type="ocsort",
        class_names=None,
        frame_rate=30.0,
        det_thresh=0.25,
        max_age=45,
        min_hits=3,
        iou_threshold=0.2,
        device="cpu",
        half=False,
        reid_weights="osnet_x0_25_msmt17.pt",
        velocity_window=5,
    ):
        self.tracker_type = tracker_type.lower()
        self.class_names = list(class_names) if class_names else []
        self._name_to_idx = {n: i for i, n in enumerate(self.class_names)}
        self.velocity_window = max(2, int(velocity_window))
        self.max_age = max_age
        self.min_hits = min_hits
        self._history = {}  # tid -> deque of (frame, cx, cy)
        self.tracks = {}  # tid -> track info dict
        self._frame_idx = 0

        # Create the tracker instance securely using BoxMOT's factory framework
        # If your tracker is motion-only, reid_weights will safely ignore itself
        self._tracker = create_tracker(
            tracker_type=self.tracker_type,
            reid_weights=Path(reid_weights) if reid_weights else None,
            device=device,
            half=half,
        )

        # Overwrite internal tracker attributes to respect your custom hyperparameter arguments
        # Handling structural name variations between traditional trackers and ByteTrack parameters
        if hasattr(self._tracker, "det_thresh"):
            self._tracker.det_thresh = det_thresh
        if hasattr(self._tracker, "iou_threshold"):
            self._tracker.iou_threshold = iou_threshold
        elif hasattr(self._tracker, "match_thresh"):
            self._tracker.match_thresh = 1.0 - iou_threshold

        # Handle max_age and track frames constraints
        if self.tracker_type == "bytetrack" and hasattr(self._tracker, "track_buffer"):
            if frame_rate > 0:
                self._tracker.track_buffer = max(
                    1, int(round(max_age * 30.0 / float(frame_rate)))
                )
            else:
                self._tracker.track_buffer = max_age
        elif hasattr(self._tracker, "max_age"):
            self._tracker.max_age = max_age

        # Dynamically determine if we need to emulate min_hits downstream
        if hasattr(self._tracker, "min_hits"):
            self._tracker.min_hits = min_hits
            self._emulate_min_hits = False
        else:
            self._emulate_min_hits = True

    # ---- Helpers for class name ↔ index ----
    def _cls_idx(self, name):
        if name is None:
            return -1
        if name not in self._name_to_idx:
            self._name_to_idx[name] = len(self.class_names)
            self.class_names.append(name)
        return self._name_to_idx[name]

    def _cls_name(self, idx):
        idx = int(idx)
        if 0 <= idx < len(self.class_names):
            return self.class_names[idx]
        return ""

    # ---- Main update step ----
    def update(self, detections, frame, frame_idx=None):
        """
        Process new detections.

        detections : list of dicts with keys 'coords' (x1,y1,x2,y2),
                     'primary_conf', 'primary_class'.
        frame      : BGR image numpy array.
        frame_idx  : optional frame number; auto-incremented if not given.

        Returns    : dict {detection_index: track_id}
        """
        self._frame_idx = self._frame_idx + 1 if frame_idx is None else int(frame_idx)

        # Build the (N,6) array expected by BoxMOT
        rows, orig_index = [], []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["coords"]
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append(
                [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(det.get("primary_conf", 1.0) or 0.0),
                    float(self._cls_idx(det.get("primary_class", ""))),
                ]
            )
            orig_index.append(i)

        dets = (
            np.asarray(rows, dtype=np.float32)
            if rows
            else np.empty((0, 6), dtype=np.float32)
        )

        # Update tracker safely
        out = self._tracker.update(dets, frame)
        out = (
            np.empty((0, 8))
            if out is None or len(out) == 0
            else np.asarray(out, dtype=float)
        )

        assignment = {}
        alive = set()

        for row in out:
            x1, y1, x2, y2 = row[:4]
            tid = int(row[4])
            conf = float(row[5]) if len(row) > 5 else 0.0
            cls_i = int(row[6]) if len(row) > 6 else -1
            det_ind = int(row[7]) if len(row) > 7 else -1

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            alive.add(tid)

            prev = self.tracks.get(tid, {})
            hits = prev.get("hits", 0) + 1
            confirmed = (not self._emulate_min_hits) or (hits >= self.min_hits)

            self.tracks[tid] = {
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "centroid": (cx, cy),
                "cls_name": self._cls_name(cls_i),
                "conf": conf,
                "last_frame": self._frame_idx,
                "hits": hits,
                "confirmed": confirmed,
            }

            # Only assign if confirmed and we have a detection index
            if confirmed and 0 <= det_ind < len(orig_index):
                assignment[orig_index[det_ind]] = tid

            # Store centroid history for velocity estimation
            hist = self._history.setdefault(tid, deque(maxlen=self.velocity_window))
            hist.append((self._frame_idx, cx, cy))

        # Remove stale tracks from our mirror cache
        stale = [
            tid
            for tid, v in self.tracks.items()
            if tid not in alive and self._frame_idx - v["last_frame"] > self.max_age
        ]
        for tid in stale:
            self.tracks.pop(tid, None)
            self._history.pop(tid, None)

        return assignment

    # ---- Accessors used by drawing / CSV code ----
    def state(self, tid):
        """
        Return (x, y, vx, vy) for a track.
        Velocity is in pixels per processed frame.
        """
        hist = self._history.get(tid)
        if not hist:
            return None
        f_last, x_last, y_last = hist[-1]
        if len(hist) < 2:
            return (x_last, y_last, 0.0, 0.0)
        f_first, x_first, y_first = hist[0]
        dt = max(1, f_last - f_first)
        return (x_last, y_last, (x_last - x_first) / dt, (y_last - y_first) / dt)

    def box(self, tid):
        """Tracker‑smoothed bounding box (x1,y1,x2,y2)."""
        tr = self.tracks.get(tid)
        return tr["box"] if tr else None

    def __contains__(self, tid):
        return tid in self.tracks
