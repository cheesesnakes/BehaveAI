"""
boxmot_tracker.py
=================

Drop-in replacement for the hand-rolled `KalmanTracker` in classify_track.py,
backed by BoxMOT (https://github.com/mikel-brostrom/boxmot).

Design goals
------------
1. **Same public contract.** `update()` returns `{detection_index: track_id}`
   and the object exposes a `.tracks` dict, so the `tid not in tracker.tracks`
   guard in process_video() keeps working unchanged.

2. **Boxes, not centroids.** The old tracker threw away the bounding boxes at
   the tracker boundary. BoxMOT associates on IoU + box geometry (+ appearance
   for the ReID trackers), which is strictly more information.

3. **Version tolerance.** BoxMOT's import path and class names have moved
   between major versions (`boxmot.OCSORT` -> `boxmot.trackers.OcSort`) and
   constructor kwargs differ per tracker. Both are resolved at runtime by
   introspection rather than hard-coded, so a `pip install -U boxmot` doesn't
   break the pipeline.

4. **Velocity preserved.** The old code drew a motion vector by reaching into
   `tracker.tracks[tid]['kf'].statePost`. BoxMOT trackers don't expose their
   internal filters uniformly, so this adapter keeps its own short centroid
   history and derives velocity from it. `state(tid)` returns the same
   `(x, y, vx, vy)` tuple the drawing code expects.

Install
-------
    pip install boxmot

For the appearance-based trackers you also need ReID weights; BoxMOT will
auto-download them on first use if you pass a bare filename such as
`osnet_x0_25_msmt17.pt`. For fish this is usually NOT worth it — see the
note under `TRACKER_ALIASES`.
"""

import inspect
from collections import deque
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Tracker resolution
# ---------------------------------------------------------------------------
# Maps a lowercase config string to the candidate class names BoxMOT has used
# across versions. First one that imports wins.
#
# Which to pick for fish:
#   ocsort      - RECOMMENDED DEFAULT. Motion-only, and its observation-centric
#                 re-update repairs the Kalman state along a virtual trajectory
#                 when a target reappears after a gap. Built for exactly the
#                 non-linear, brief-disappearance case.
#   bytetrack   - Fastest. Recovers low-confidence detections instead of
#                 dropping them. Use if OC-SORT is too slow on a Pi.
#   botsort     - Adds camera-motion compensation. Worth it if your rig
#                 vibrates or the water surface shifts the whole frame.
#   deepocsort  - OC-SORT + appearance. Appearance embeddings are trained on
#                 pedestrians and near-useless for identical-looking fish;
#                 costs a lot of compute for little gain. Try it last, not
#                 first.
#   occluboost  - Best on the MOT17 ablation in the current README. Uses ReID,
#                 same caveat as above.
TRACKER_ALIASES = {
    "bytetrack": ("ByteTrack", "BYTETracker", "BYTETrack"),
    "botsort": ("BotSort", "BoTSORT", "BoTSort"),
    "strongsort": ("StrongSort", "StrongSORT"),
    "ocsort": ("OcSort", "OCSORT", "OCSort"),
    "deepocsort": ("DeepOcSort", "DeepOCSORT", "DeepOCSort"),
    "hybridsort": ("HybridSort", "HybridSORT"),
    "boosttrack": ("BoostTrack",),
    "occluboost": ("OccluBoost",),
    "sfsort": ("SFSORT", "SFSort"),
}

# Trackers that need ReID weights to construct.
REID_TRACKERS = {
    "botsort",
    "strongsort",
    "deepocsort",
    "hybridsort",
    "boosttrack",
    "occluboost",
}


def _resolve_tracker_class(tracker_type):
    """
    Find the BoxMOT class for `tracker_type`, trying both the modern
    `boxmot.trackers` namespace and the legacy top-level `boxmot` one.
    Raises ImportError with a useful message if nothing matches.
    """
    key = tracker_type.lower().replace("-", "").replace("_", "")
    if key not in TRACKER_ALIASES:
        raise ValueError(
            f"Unknown tracker_type '{tracker_type}'. "
            f"Choose one of: {', '.join(sorted(TRACKER_ALIASES))}"
        )

    modules = []
    try:
        import boxmot.trackers as _t

        modules.append(_t)
    except Exception:
        pass
    try:
        import boxmot as _b

        modules.append(_b)
    except Exception:
        pass

    if not modules:
        raise ImportError(
            "BoxMOT is not installed. Run:  pip install boxmot\n"
            "(If you are on the Raspberry Pi / NCNN path, note that BoxMOT "
            "pulls in torch. Keep using the built-in tracker there.)"
        )

    for mod in modules:
        for name in TRACKER_ALIASES[key]:
            cls = getattr(mod, name, None)
            if cls is not None:
                return key, cls

    tried = ", ".join(TRACKER_ALIASES[key])
    raise ImportError(
        f"Could not find a class for '{tracker_type}' in your BoxMOT install "
        f'(tried: {tried}). Check `python -c "import boxmot.trackers as t; '
        f'print(dir(t))"` and add the correct name to TRACKER_ALIASES.'
    )


def _filter_kwargs(cls, candidate_kwargs):
    """
    Keep only the kwargs this tracker's __init__ actually accepts. BoxMOT
    constructors vary a lot between trackers (and versions); passing an
    unexpected kwarg is a hard TypeError, so filter rather than guess.
    Returns (accepted, dropped) for logging.
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return dict(candidate_kwargs), {}

    # If the constructor takes **kwargs, everything is fair game.
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(candidate_kwargs), {}

    accepted, dropped = {}, {}
    for k, v in candidate_kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
        else:
            dropped[k] = v
    return accepted, dropped


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class BoxMOTTracker:
    """
    Wraps any BoxMOT tracker behind the interface classify_track.py expects.

    Usage in process_video():

        tracker = BoxMOTTracker(
            tracker_type=params.get("tracker_type", "ocsort"),
            class_names=params["primary_classes"],
            frame_rate=fps / (params["frame_skip"] + 1),
        )
        ...
        assignment = tracker.update(processed_detections, frame)
    """

    def __init__(
        self,
        tracker_type="ocsort",
        class_names=None,
        frame_rate=30.0,
        det_thresh=0.25,
        max_age=45,
        min_hits=3,
        iou_threshold=0.20,
        per_class=False,
        device="cpu",
        half=False,
        reid_weights="osnet_x0_25_msmt17.pt",
        velocity_window=5,
        verbose=True,
        _tracker_override=None,
    ):
        """
        Parameters
        ----------
        tracker_type : key from TRACKER_ALIASES.
        class_names  : list of primary class-name strings. BoxMOT wants integer
                       class ids, so names are mapped to indices via this list;
                       unseen names are appended.
        frame_rate   : frames per second AS SEEN BY THE TRACKER. If you process
                       every Nth frame, pass fps / N — otherwise ByteTrack-style
                       buffers are wrong by a factor of N.
        max_age      : frames a track survives unmatched. In PROCESSED frames.
                       This is your "brief disappearance" knob. At 30 fps with
                       frame_skip=0, 45 ~= 1.5 s of occlusion.
        min_hits     : detections required before a track is reported. Stops a
                       single flickering false positive from becoming a
                       permanent individual in your CSV.
        iou_threshold: association gate. Fish are small and fast, so this wants
                       to be looser than the pedestrian default of 0.3.
        per_class    : keep separate track pools per class. Leave False unless
                       your primary classes are genuinely different animals
                       that never get confused for one another.
        _tracker_override : inject a fake tracker for unit tests. Not for
                       production use.
        """
        self.tracker_type = tracker_type
        self.velocity_window = max(2, int(velocity_window))
        self.verbose = verbose

        # ---- class name <-> index mapping -----------------------------
        self.class_names = list(class_names) if class_names else []
        self._name_to_idx = {n: i for i, n in enumerate(self.class_names)}

        # ---- construct the underlying tracker --------------------------
        if _tracker_override is not None:
            self._key = tracker_type.lower()
            self._tracker = _tracker_override
        else:
            self._key, cls = _resolve_tracker_class(tracker_type)

            candidates = {
                "det_thresh": det_thresh,
                "min_conf": det_thresh,
                "track_thresh": det_thresh,
                "max_age": max_age,
                "track_buffer": max_age,
                "min_hits": min_hits,
                "iou_threshold": iou_threshold,
                "match_thresh": 1.0 - iou_threshold,
                "frame_rate": frame_rate,
                "per_class": per_class,
                "device": device,
                "half": half,
                "fp16": half,
            }
            if self._key in REID_TRACKERS:
                w = Path(reid_weights)
                candidates["reid_weights"] = w
                candidates["model_weights"] = w  # legacy kwarg name

            accepted, dropped = _filter_kwargs(cls, candidates)
            self._tracker = cls(**accepted)

            if self.verbose:
                print(f"[tracker] {cls.__name__} ({', '.join(sorted(accepted))})")
                if dropped:
                    print(
                        f"[tracker] not accepted by this version, ignored: "
                        f"{', '.join(sorted(dropped))}"
                    )

        # ---- bookkeeping the pipeline reads ---------------------------
        # tid -> {'box', 'centroid', 'cls_name', 'conf', 'last_frame', 'hits'}
        self.tracks = {}
        self._history = {}  # tid -> deque[(frame_idx, cx, cy)]
        self._frame_idx = 0
        self.max_age = max_age

    # -- class id helpers ------------------------------------------------

    def _cls_idx(self, name):
        if name is None or name == "":
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

    # -- main step -------------------------------------------------------

    def update(self, detections, frame, frame_idx=None):
        """
        detections : list of the pipeline's merged detection dicts. Each needs
                     'coords' as (x1, y1, x2, y2); 'primary_conf' and
                     'primary_class' are used if present.
        frame      : the BGR image for this step. Motion-only trackers ignore
                     it; ReID trackers crop appearance patches from it, so pass
                     the STATIC frame, never the false-colour motion image.

        Returns    : {index into `detections` -> track_id}, same as the old
                     KalmanTracker.
        """
        self._frame_idx = self._frame_idx + 1 if frame_idx is None else int(frame_idx)

        # Build the (N, 6) array, dropping degenerate boxes. BoxMOT will happily
        # accept a zero-area box and then produce NaNs downstream.
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

        # Always call update, even with zero detections — that is how the
        # trackers age out and coast their existing tracks.
        out = self._tracker.update(dets, frame)
        out = (
            np.empty((0, 8))
            if out is None or len(out) == 0
            else np.asarray(out, dtype=float)
        )

        assignment = {}
        alive = set()

        for row in out:
            x1, y1, x2, y2 = row[0:4]
            tid = int(row[4])
            conf = float(row[5]) if len(row) > 5 else 0.0
            cls_i = int(row[6]) if len(row) > 6 else -1
            det_ind = int(row[7]) if len(row) > 7 else -1

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            alive.add(tid)

            if 0 <= det_ind < len(orig_index):
                assignment[orig_index[det_ind]] = tid

            prev = self.tracks.get(tid, {})
            self.tracks[tid] = {
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "centroid": (cx, cy),
                "cls_name": self._cls_name(cls_i),
                "conf": conf,
                "last_frame": self._frame_idx,
                "hits": prev.get("hits", 0) + 1,
            }

            hist = self._history.setdefault(tid, deque(maxlen=self.velocity_window))
            hist.append((self._frame_idx, cx, cy))

        # Retire stale entries from our own mirror. BoxMOT manages its real
        # track pool internally; this just stops the dict growing forever.
        for tid in [
            t
            for t, v in self.tracks.items()
            if t not in alive and self._frame_idx - v["last_frame"] > self.max_age
        ]:
            self.tracks.pop(tid, None)
            self._history.pop(tid, None)

        return assignment

    # -- accessors used by the drawing / CSV code ------------------------

    def state(self, tid):
        """
        (x, y, vx, vy) for a track, replacing the old
        `tracker.tracks[tid]['kf'].statePost` access.

        Velocity is a finite difference over the centroid history and is
        expressed in pixels PER PROCESSED FRAME, matching the old behaviour.
        Multiply by fps/(frame_skip+1) for px/second.
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
        """Tracker-smoothed box. Steadier than the raw detection box, which
        matters if you measure size or position from the overlay."""
        tr = self.tracks.get(tid)
        return tr["box"] if tr else None

    def __contains__(self, tid):
        return tid in self.tracks
