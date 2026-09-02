"""
pseudo_label.py
---------------
Multi-stream pseudo-labeller.

Walks every video in `clips_dir` and writes BehaveAI-compatible annotations for
whichever of the four streams have a usable model on disk:

  PRIMARY STATIC   ->  <annot>/annot_static/{images,labels}/{train,val}/
  PRIMARY MOTION   ->  <annot>/annot_motion/{images,labels}/{train,val}/
  SECONDARY STATIC ->  <annot>/annot_static_crop/<primary_class>/[train|val/]<secondary_class>/
  SECONDARY MOTION ->  <annot>/annot_motion_crop/<primary_class>/[train|val/]<secondary_class>/

Model selection (mirrors what classify_track.py does at inference):

  primary static    external model if `primary_static_external_model` is set,
                    otherwise models/model_primary_static/train/weights/best.pt.
                    The external model WINS — the two are never merged.
  primary motion    models/model_primary_motion/train/weights/best.pt
  secondary static  external bundle if `secondary_static_external_model` is set,
                    otherwise one YOLO-cls model per primary class under models/
  secondary motion  one YOLO-cls model per primary class under models/

Any stream whose model is missing is silently skipped. Running with nothing but
the external model reproduces the behaviour of the previous version of this
script.

Label format matches scripts/annotation.py:
    <cls> <xc> <yc> <w> <h>   (all normalized, 6 decimal places)

Class-ID assumptions
--------------------
Static labels index `primary_static_classes`; motion labels index
`primary_motion_classes`. Locally trained models satisfy this by construction.
The EXTERNAL model is assumed to already match `primary_static_classes` order —
no remap is performed, and out-of-range IDs are dropped and counted.

Frame cadence
-------------
The motion image is built by advancing a 3-frame history, so it only means
anything if the history advances at the same cadence inference uses. This
script therefore decodes every frame, applies `frame_skip` exactly as
classify_track.py does, and advances the motion history on every processed
frame. `--sample-every` then selects which of those PROCESSED frames get
labelled and written. The first processed frame of each video primes the
history and is never written.

A caveat worth stating plainly: pseudo-labelling a stream with the model that
was trained on that same stream's dataset is self-reinforcing. It will happily
reproduce its own errors, and reviewing the output in the annotation GUI is not
optional. The script prints a warning whenever it does this.

Usage:
    python scripts/pseudo_label.py <project_dir_or_ini>
        [--sample-every N]     # 1 in N PROCESSED frames (default 30)
        [--max-per-video K]    # cap frames per video (0 = unlimited)
        [--val-frequency F]    # override config's val_frequency
        [--conf C]             # detection confidence (default 0.6)
        [--skip-static] [--skip-motion] [--skip-secondary]
        [--dry-run]            # don't write anything, just report

Config keys read beyond what load_params already provides:
    primary_static_external_model
    secondary_static_external_model
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np
from load_configs import load_params
from motion import create_motion_image, new_history

# GLOBALS

DEFAULT_CONF = 0.6
IMG_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

# Candidate directory names for the per-primary-class secondary models. The
# static and motion halves of train_models() do not use the same naming
# convention, and older projects may have either — probe rather than assume.
SECONDARY_STATIC_DIRS = ("model_static_static_{c}", "model_secondary_static_{c}")
SECONDARY_MOTION_DIRS = ("model_secondary_motion_{c}", "model_motion_motion_{c}")


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------
def _iter_videos(clips_dir):
    for root, _, files in os.walk(clips_dir):
        for f in sorted(files):
            if f.lower().endswith(IMG_EXTS):
                yield os.path.join(root, f)


def _video_label(path, clips_dir):
    """Mirror annotation.py's '<video_label>_<frame>' filename convention."""
    rel = os.path.relpath(path, clips_dir)
    stem, _ = os.path.splitext(rel)
    return stem.replace(os.sep, "__").replace(" ", "_")


def _load_yolo(weights, task, label):
    """
    Load a .pt with ultralytics, returning None (with a printed reason) on any
    failure so a broken model degrades to a skipped stream instead of a crash.

    NCNN is deliberately not used here. It exists for the edge-inference path;
    pseudo-labelling is a one-off batch job where the .pt is both available and
    more accurate.
    """
    from ultralytics import YOLO

    if not weights:
        return None
    if not (os.path.isfile(weights) or os.path.isdir(weights)):
        print(f"  [{label}] not found: {weights}")
        return None
    try:
        return YOLO(weights, task=task)
    except Exception as e:  # noqa: BLE001 — a bad model should not kill the run
        print(f"  [{label}] failed to load ({weights}): {e}")
        return None


def _secondary_class_candidates(params):
    """
    The primary classes that are eligible for secondary classification, using
    the same two exclusions train_models() applies: a primary class whose
    hotkey is also a secondary hotkey, and anything in ignore_secondary.
    """
    out = []
    for i, primary_class in enumerate(params["primary_classes"]):
        hotkey = params["primary_hotkeys"][i]
        if hotkey in params["secondary_hotkeys"]:
            continue
        if primary_class in params["ignore_secondary"]:
            continue
        out.append(primary_class)
    return out


def _find_secondary_models(params, dir_patterns, label):
    """Return {primary_class: YOLO} for every per-class model found on disk."""
    models = {}
    for primary_class in _secondary_class_candidates(params):
        for pattern in dir_patterns:
            model_dir = os.path.join(
                params["model_folder"], pattern.format(c=primary_class)
            )
            weights = os.path.join(model_dir, "train", "weights", "best.pt")
            if not os.path.isfile(weights):
                continue
            m = _load_yolo(weights, "classify", f"{label}:{primary_class}")
            if m is not None:
                models[primary_class] = m
            break
    return models


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def iou(box1, box2):
    """
    Proportional overlap relative to the SMALLER box — the same non-standard
    measure classify_track.py merges detections with. Copied rather than
    imported for the same reason motion.py exists.
    """
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    if area1 <= 0 or area2 <= 0:
        return 0.0
    return max(0.0, max(inter / area1, inter / area2))


def _to_yolo_line(cls_id, box, w, h):
    x1, y1, x2, y2 = box
    xc = min(max(((x1 + x2) / 2.0) / w, 0.0), 1.0)
    yc = min(max(((y1 + y2) / 2.0) / h, 0.0), 1.0)
    bw = min(max(abs(x2 - x1) / w, 0.0), 1.0)
    bh = min(max(abs(y2 - y1) / h, 0.0), 1.0)
    return f"{int(cls_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def _detect(model, image, class_names, conf, imgsz, stats, stream):
    """
    Run one detector and return (yolo_label_lines, detection_dicts).

    Detections whose class ID falls outside `class_names` are dropped and
    counted — that is the signal that an external model's class order does not
    actually match the project's.
    """
    h, w = image.shape[:2]
    lines, dets = [], []
    results = model.predict(image, conf=conf, imgsz=imgsz, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return lines, dets

    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), cls_id, c in zip(xyxy, cls, confs):
        if c < conf:
            continue
        if not (0 <= cls_id < len(class_names)):
            stats["dropped_oor"] += 1
            continue
        lines.append(_to_yolo_line(cls_id, (x1, y1, x2, y2), w, h))
        dets.append(
            {
                "coords": tuple(map(int, (x1, y1, x2, y2))),
                "centroid": (int((x1 + x2) // 2), int((y1 + y2) // 2)),
                "primary_class": class_names[cls_id],
                "primary_conf": float(c),
                "source": stream,
            }
        )
    return lines, dets


def merge_detections(all_dets, params):
    """
    Collapse static and motion detections of the same animal into one box,
    using the same proximity/overlap rule and dominant_source policy as
    classify_track.py. Only used to decide which crops to cut — the primary
    datasets are written from the unmerged, per-stream detections.
    """
    merged = []
    for det in all_dets:
        cx, cy = det["centroid"]
        matched = False
        for md in merged:
            md_cx, md_cy = md["centroid"]
            dist = np.hypot(cx - md_cx, cy - md_cy)
            overlap = iou(det["coords"], md["coords"])
            if (
                dist >= params["centroid_merge_thresh"]
                and overlap <= params["iou_thresh"]
            ):
                continue

            if (
                det["source"] == md["source"]
                or params["dominant_source"] == "confidence"
            ):
                take = det["primary_conf"] > md["primary_conf"]
            else:
                take = det["source"] == params["dominant_source"]

            if take:
                md.update(det)
            matched = True
            break

        if not matched:
            merged.append(dict(det))
    return merged


# ---------------------------------------------------------------------------
# Secondary crop output
# ---------------------------------------------------------------------------
class CropWriter:
    """
    Routes a crop to <root>/<primary_class>/[train|val/]<secondary_class>/.

    The train/val level is included only if the primary class's folder already
    has a `train` subdirectory, so this matches whatever layout the annotation
    GUI produced rather than imposing one.
    """

    def __init__(self, root, dry_run):
        self.root = root
        self.dry_run = dry_run
        self._split = {}

    def _uses_split(self, primary_class):
        if primary_class not in self._split:
            self._split[primary_class] = os.path.isdir(
                os.path.join(self.root, primary_class, "train")
            )
        return self._split[primary_class]

    def write(self, crop, primary_class, secondary_class, basename, is_val):
        if crop is None or crop.size == 0:
            return False
        parts = [self.root, primary_class]
        if self._uses_split(primary_class):
            parts.append("val" if is_val else "train")
        parts.append(secondary_class)
        out_dir = os.path.join(*parts)
        if not self.dry_run:
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"{basename}.jpg"), crop)
        return True


def _classify_crop(models, crop, primary_class, params):
    """
    Run the secondary classifier for `primary_class`. `models` is either a dict
    of per-class YOLO-cls models or a single external engine exposing
    predict_single(). Returns (class_name, conf) or (None, None).
    """
    if models is None or crop is None or crop.size == 0:
        return None, None

    if not isinstance(models, dict):
        # External bundle (FishInferenceEngine).
        try:
            best = models.predict_single(crop).best
            cls, conf = best.name, best.accuracy
        except Exception as e:  # noqa: BLE001
            print(f"  [secondary external] prediction failed: {e}")
            return None, None
    else:
        m = models.get(primary_class)
        if m is None:
            return None, None
        res = m.predict(crop, imgsz=params["secondary_imgsz"], verbose=False)
        if res[0].probs is None:
            return None, None
        conf = res[0].probs.top1conf.item()
        cls = m.names[res[0].probs.top1]

    if cls is None or conf is None or conf < params["secondary_conf_thresh"]:
        return None, None
    return cls, conf


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def build_models(params, args):
    """Load every model that exists. Returns a dict of loaded components."""
    m = {
        "static": None,
        "motion": None,
        "sec_static": None,
        "sec_motion": None,
        "static_is_local": False,
    }

    # ---- primary static ------------------------------------------------
    if not args.skip_static and params["primary_static_classes"][0] != "0":
        external = params["primary_static_external_model"]
        if external:
            m["static"] = _load_yolo(external, "detect", "primary static (external)")
        else:
            m["static"] = _load_yolo(
                params["primary_static_model_path"], "detect", "primary static (local)"
            )
            m["static_is_local"] = m["static"] is not None

    # ---- primary motion ------------------------------------------------
    if not args.skip_motion and params["primary_motion_classes"][0] != "0":
        m["motion"] = _load_yolo(
            params["primary_motion_model_path"], "detect", "primary motion"
        )

    # ---- secondaries ---------------------------------------------------
    if not args.skip_secondary and params["hierarchical_mode"]:
        external_sec = params["secondary_static_external_model"]
        if external_sec:
            try:
                from fishial_inference import FishInferenceEngine

                m["sec_static"] = FishInferenceEngine.from_bundle(external_sec)
                print(f"  [secondary static] external bundle: {external_sec}")
            except Exception as e:  # noqa: BLE001
                print(f"  [secondary static] external bundle failed to load: {e}")
        elif len(params["secondary_static_classes"]) >= 2:
            m["sec_static"] = (
                _find_secondary_models(
                    params, SECONDARY_STATIC_DIRS, "secondary static"
                )
                or None
            )

        if len(params["secondary_motion_classes"]) >= 2:
            m["sec_motion"] = (
                _find_secondary_models(
                    params, SECONDARY_MOTION_DIRS, "secondary motion"
                )
                or None
            )

    return m


# ---------------------------------------------------------------------------
# Background generation (temporal median) – separate module
# ---------------------------------------------------------------------------
def generate_backgrounds(params, args):
    """
    Standalone background frame generator.

    For each video, sample `args.generate_backgrounds` temporal windows uniformly
    across the video, compute the pixel‑wise median of each window, and write
    the result as a background frame (empty .txt label) into the primary
    static/motion train/val folders.

    The function respects --bg-static and --bg-motion flags. If neither is given,
    it writes to both streams (provided the stream is configured in the project).

    Naming convention: <video_label>_bg_<six_digit_index>.jpg/.txt

    This function is called from pseudo_label() and causes an immediate exit
    after completion, making it act as a separate sub‑command.
    """
    import cv2
    import numpy as np
    import random

    target_per_video = args.generate_backgrounds
    if target_per_video <= 0:
        return

    window = args.bg_window
    if window < 2:
        print("[bg] ERROR: --bg-window must be at least 2.")
        sys.exit(1)

    # Determine which streams are configured.
    static_configured = (
        len(params["primary_static_classes"]) > 0
        and params["primary_static_classes"][0] != "0"
    )
    motion_configured = (
        len(params["primary_motion_classes"]) > 0
        and params["primary_motion_classes"][0] != "0"
    )

    do_static = args.bg_static
    do_motion = args.bg_motion
    # If user specified neither, default to both configured streams.
    if not do_static and not do_motion:
        do_static = static_configured
        do_motion = motion_configured

    # Clamp to what actually exists.
    do_static = do_static and static_configured
    do_motion = do_motion and motion_configured

    if not do_static and not do_motion:
        print(
            "[bg] No stream selected (or configured). "
            "Use --bg-static, --bg-motion, or configure primary classes."
        )
        return

    # Output directories.
    static_dirs = (
        params["static_train_images_dir"],
        params["static_train_labels_dir"],
        params["static_val_images_dir"],
        params["static_val_labels_dir"],
    )
    motion_dirs = (
        params["motion_train_images_dir"],
        params["motion_train_labels_dir"],
        params["motion_val_images_dir"],
        params["motion_val_labels_dir"],
    )

    if not args.dry_run:
        for d in static_dirs + motion_dirs:
            os.makedirs(d, exist_ok=True)

    # Split seed – deterministic but can be overridden via INI if we add that key.
    val_freq = (
        args.val_frequency
        if args.val_frequency is not None
        else float(params.get("val_frequency", 0.1))
    )
    rng = random.Random(0)

    print("[bg] Starting background generation (temporal median)")
    print(f"[bg]   Target frames per video: {target_per_video}")
    print(f"[bg]   Window size: {window}")
    print(f"[bg]   Static stream: {'yes' if do_static else 'no'}")
    print(f"[bg]   Motion stream: {'yes' if do_motion else 'no'}")
    print(f"[bg]   Val fraction: {val_freq}")
    print(f"[bg]   Dry run: {args.dry_run}")
    print()

    total_bg_written = 0

    for video_path in _iter_videos(params["clips_dir"]):
        vlabel = _video_label(video_path, params["clips_dir"])
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [bg] skip: cannot open {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < window:
            print(
                f"  [bg] skip: {os.path.basename(video_path)} too short ({total_frames} < {window})"
            )
            cap.release()
            continue

        # Determine step size to uniformly sample `target_per_video` windows.
        # We want start indices s such that 0 <= s <= total_frames - window.
        max_start = total_frames - window
        if target_per_video >= max_start + 1:
            # If target >= number of possible windows, just take one every `window` frames.
            step = window
            # But clamp target to avoid infinite loops.
            actual_target = min(target_per_video, (total_frames // window))
        else:
            # Uniformly spaced windows.
            step = max(1, (max_start + 1) // target_per_video)
            actual_target = target_per_video

        # Read the whole video into memory? For large videos, better to seek.
        # We'll sample by seeking to start positions and reading `window` frames.
        # However, seeking backwards/forwards with H.264 can be slow.
        # Simpler: read sequentially, compute median at each step.
        # We'll store only the current window buffer.
        frames_buffer = []
        frame_idx = 0
        generated = 0
        next_write_idx = 0  # we want roughly evenly spaced writes

        # We'll compute the target step in terms of frame indices.
        # If we want `actual_target` windows, we write at positions:
        # start_positions = [0, step, 2*step, ...] capped.
        # We'll generate `actual_target` windows.
        # But we can just read sequentially and when frame_idx equals next_start, compute median.
        start_positions = [min(i * step, max_start) for i in range(actual_target)]

        # Remove duplicates
        start_positions = sorted(set(start_positions))

        for start in start_positions:
            # Seek to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames_buffer = []
            for _ in range(window):
                ret, frame = cap.read()
                if not ret:
                    break
                if params["scale_factor"] != 1.0:
                    frame = cv2.resize(
                        frame,
                        None,
                        fx=params["scale_factor"],
                        fy=params["scale_factor"],
                    )
                frames_buffer.append(frame)
            if len(frames_buffer) < window:
                continue  # should not happen if start <= max_start

            median_img = np.median(frames_buffer, axis=0).astype(np.uint8)

            # Train/val assignment
            is_val = rng.random() < val_freq
            base = f"{vlabel}_bg_{generated:06d}"
            generated += 1

            # Write static
            if do_static:
                img_dir = static_dirs[2] if is_val else static_dirs[0]
                lbl_dir = static_dirs[3] if is_val else static_dirs[1]
                if not args.dry_run:
                    cv2.imwrite(os.path.join(img_dir, f"{base}.jpg"), median_img)
                    with open(os.path.join(lbl_dir, f"{base}.txt"), "w") as f:
                        pass  # empty -> background
                total_bg_written += 1

            # Write motion (same median image)
            if do_motion:
                img_dir = motion_dirs[2] if is_val else motion_dirs[0]
                lbl_dir = motion_dirs[3] if is_val else motion_dirs[1]
                if not args.dry_run:
                    cv2.imwrite(os.path.join(img_dir, f"{base}.jpg"), median_img)
                    with open(os.path.join(lbl_dir, f"{base}.txt"), "w") as f:
                        pass
                total_bg_written += 1

            # Stop if we've generated enough for this video (avoid overrun if duplicates removed)
            if generated >= target_per_video:
                break

        cap.release()
        print(f"  [bg] {os.path.basename(video_path)}: generated {generated} windows")

    print()
    print(
        f"[bg] Done. Total background images written (across all streams): {total_bg_written}"
    )
    if args.dry_run:
        print("[bg] DRY RUN — no files were written.")


def pseudo_label(params, args):
    # ---- BACKGROUND GENERATION MODULE (separate) ----
    if args.generate_backgrounds > 0:
        generate_backgrounds(params, args)
        # Exit immediately so the pseudo-labeller does not run.
        print("[bg] Background generation complete. Exiting.")
        return

    conf = args.conf if args.conf is not None else DEFAULT_CONF
    val_freq = (
        args.val_frequency
        if args.val_frequency is not None
        else float(params.get("val_frequency", 0.1))
    )
    save_empty = str(params.get("save_empty_frames", "false")).lower() == "true"

    print("[pseudo-label] loading models")
    models = build_models(params, args)

    if models["static"] is None and models["motion"] is None:
        sys.exit(
            "No primary model available (no external model, and no trained "
            "primary static/motion weights under models/). Nothing to do."
        )

    need_motion = models["motion"] is not None or models["sec_motion"] is not None

    # Dataset roots.
    static_dirs = (
        params["static_train_images_dir"],
        params["static_train_labels_dir"],
        params["static_val_images_dir"],
        params["static_val_labels_dir"],
    )
    motion_dirs = (
        params["motion_train_images_dir"],
        params["motion_train_labels_dir"],
        params["motion_val_images_dir"],
        params["motion_val_labels_dir"],
    )
    if not args.dry_run:
        if models["static"] is not None:
            for d in static_dirs:
                os.makedirs(d, exist_ok=True)
        if models["motion"] is not None:
            for d in motion_dirs:
                os.makedirs(d, exist_ok=True)

    crop_static = (
        CropWriter(params["static_cropped_base_dir"], args.dry_run)
        if models["sec_static"] is not None
        else None
    )
    crop_motion = (
        CropWriter(params["motion_cropped_base_dir"], args.dry_run)
        if models["sec_motion"] is not None
        else None
    )

    def _n(x):
        return len(x) if isinstance(x, dict) else ("external" if x else 0)

    print()
    print(f"[pseudo-label] primary static:    {'yes' if models['static'] else 'no'}")
    print(f"[pseudo-label] primary motion:    {'yes' if models['motion'] else 'no'}")
    print(f"[pseudo-label] secondary static:  {_n(models['sec_static'])}")
    print(f"[pseudo-label] secondary motion:  {_n(models['sec_motion'])}")
    print(f"[pseudo-label] conf:              {conf}")
    print(f"[pseudo-label] static classes:    {params['primary_static_classes']}")
    print(f"[pseudo-label] motion classes:    {params['primary_motion_classes']}")
    print(f"[pseudo-label] frame_skip:        {params['frame_skip']}")
    print(f"[pseudo-label] sample:            1 in {args.sample_every} processed")
    print(f"[pseudo-label] val_freq:          {val_freq}")
    print(f"[pseudo-label] save_empty:        {save_empty}")
    print(f"[pseudo-label] dry_run:           {args.dry_run}")

    self_labelling = [
        name
        for name, present in (
            ("primary static", models["static_is_local"]),
            ("primary motion", models["motion"] is not None),
            ("secondary static", isinstance(models["sec_static"], dict)),
            ("secondary motion", isinstance(models["sec_motion"], dict)),
        )
        if present
    ]
    if self_labelling:
        print()
        print(
            "[pseudo-label] WARNING: these streams are being labelled by a model\n"
            "               trained on their own dataset: "
            + ", ".join(self_labelling)
            + ".\n"
            "               The output will reproduce that model's existing\n"
            "               mistakes. Review it in the annotation GUI before\n"
            "               retraining on it."
        )
    print()

    stats = {
        "dropped_oor": 0,
        "videos": 0,
        "static_frames": 0,
        "static_dets": 0,
        "motion_frames": 0,
        "motion_dets": 0,
        "static_crops": 0,
        "motion_crops": 0,
    }
    rng = random.Random(0)  # deterministic train/val split

    for video_path in _iter_videos(params["clips_dir"]):
        stats["videos"] += 1
        vlabel = _video_label(video_path, params["clips_dir"])
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [skip] cannot open {video_path}")
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        prev_frames = None
        kept = 0
        frame_idx = -1
        proc_idx = -1
        skip_count = 0

        while True:
            ok, raw = cap.read()
            if not ok:
                break
            frame_idx += 1

            # Honour frame_skip exactly as classify_track.py does, so the
            # motion history advances at the cadence the model was trained at.
            if skip_count != 0:
                skip_count = (skip_count + 1) % (params["frame_skip"] + 1)
                continue
            skip_count = (skip_count + 1) % (params["frame_skip"] + 1)

            if params["scale_factor"] != 1.0:
                raw = cv2.resize(
                    raw, None, fx=params["scale_factor"], fy=params["scale_factor"]
                )
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

            # First processed frame only primes the history.
            if prev_frames is None:
                prev_frames = new_history(gray)
                continue
            proc_idx += 1

            # MUST run every processed frame — it mutates prev_frames.
            motion_image = (
                create_motion_image(prev_frames, gray, params) if need_motion else None
            )

            if proc_idx % args.sample_every != 0:
                continue
            if args.max_per_video and kept >= args.max_per_video:
                break

            base = f"{vlabel}_{frame_idx}"
            is_val = rng.random() < val_freq
            wrote_anything = False
            all_dets = []

            # ---- primary static ---------------------------------------
            if models["static"] is not None:
                lines, dets = _detect(
                    models["static"],
                    raw,
                    params["primary_static_classes"],
                    conf,
                    params["inference_imgsz"],
                    stats,
                    "static",
                )
                all_dets.extend(dets)
                if lines or save_empty:
                    img_dir = static_dirs[2] if is_val else static_dirs[0]
                    lbl_dir = static_dirs[3] if is_val else static_dirs[1]
                    if not args.dry_run:
                        cv2.imwrite(os.path.join(img_dir, f"{base}.jpg"), raw)
                        with open(os.path.join(lbl_dir, f"{base}.txt"), "w") as f:
                            if lines:
                                f.write("\n".join(lines) + "\n")
                            # An empty .txt is valid YOLO (background frame)
                    stats["static_frames"] += 1
                    stats["static_dets"] += len(lines)
                    wrote_anything = True

            # ---- primary motion ---------------------------------------
            if models["motion"] is not None and motion_image is not None:
                lines, dets = _detect(
                    models["motion"],
                    motion_image,
                    params["primary_motion_classes"],
                    conf,
                    params["inference_imgsz"],
                    stats,
                    "motion",
                )
                all_dets.extend(dets)
                if lines or save_empty:
                    img_dir = motion_dirs[2] if is_val else motion_dirs[0]
                    lbl_dir = motion_dirs[3] if is_val else motion_dirs[1]
                    if not args.dry_run:
                        cv2.imwrite(os.path.join(img_dir, f"{base}.jpg"), motion_image)
                        with open(os.path.join(lbl_dir, f"{base}.txt"), "w") as f:
                            if lines:
                                f.write("\n".join(lines) + "\n")
                    stats["motion_frames"] += 1
                    stats["motion_dets"] += len(lines)
                    wrote_anything = True

            # ---- secondary crops --------------------------------------
            if (crop_static or crop_motion) and all_dets:
                for i, det in enumerate(merge_detections(all_dets, params)):
                    if det["primary_conf"] < params["primary_conf_thresh"]:
                        continue
                    x1, y1, x2, y2 = det["coords"]
                    x1, y1 = max(0, x1), max(0, y1)
                    pc = det["primary_class"]
                    crop_base = f"{base}_{x1}_{y1}"

                    if crop_static is not None:
                        sc = raw[y1:y2, x1:x2]
                        cls, _ = _classify_crop(models["sec_static"], sc, pc, params)
                        if cls and crop_static.write(sc, pc, cls, crop_base, is_val):
                            stats["static_crops"] += 1
                            wrote_anything = True

                    if crop_motion is not None and motion_image is not None:
                        mc = motion_image[y1:y2, x1:x2]
                        cls, _ = _classify_crop(models["sec_motion"], mc, pc, params)
                        if cls and crop_motion.write(mc, pc, cls, crop_base, is_val):
                            stats["motion_crops"] += 1
                            wrote_anything = True

            if wrote_anything:
                kept += 1

        cap.release()
        print(f"  {os.path.basename(video_path)}: {kept} frames kept ({total} total)")

    print()
    print(f"[pseudo-label] videos processed:     {stats['videos']}")
    print(
        f"[pseudo-label] static frames/dets:   "
        f"{stats['static_frames']} / {stats['static_dets']}"
    )
    print(
        f"[pseudo-label] motion frames/dets:   "
        f"{stats['motion_frames']} / {stats['motion_dets']}"
    )
    print(f"[pseudo-label] static crops:        {stats['static_crops']}")
    print(f"[pseudo-label] motion crops:        {stats['motion_crops']}")
    if stats["dropped_oor"]:
        print(
            f"[pseudo-label] detections dropped (cls out of range): "
            f"{stats['dropped_oor']}\n"
            f"                -> a model emitted class IDs outside its stream's\n"
            f"                   class list; check the external model really\n"
            f"                   matches primary_static_classes."
        )
    if args.dry_run:
        print("[pseudo-label] DRY RUN — no files were written.")
    else:
        print(
            "[pseudo-label] Done. Open the annotation GUI to review and correct "
            "before retraining."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "project",
        nargs="?",
        default=None,
        help="Project directory or path to BehaveAI_settings.ini. "
        "Omit to get the file-picker dialog.",
    )
    ap.add_argument(
        "--sample-every",
        type=int,
        default=30,
        help="Label 1 in N PROCESSED frames (default: 30)",
    )
    ap.add_argument(
        "--max-per-video",
        type=int,
        default=0,
        help="Cap frames kept per video (0 = unlimited)",
    )
    ap.add_argument(
        "--val-frequency",
        type=float,
        default=None,
        help="Override config's val_frequency",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"Detection confidence floor (default: {DEFAULT_CONF})",
    )
    ap.add_argument("--skip-static", action="store_true", help="Skip primary static")
    ap.add_argument("--skip-motion", action="store_true", help="Skip primary motion")
    ap.add_argument(
        "--skip-secondary", action="store_true", help="Skip all secondary crops"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write files, just print what would happen",
    )
    # ---------- Background generation flags ----------
    ap.add_argument(
        "--generate-backgrounds",
        type=int,
        default=0,
        help="Generate N background frames per video using temporal median, then exit. "
        "0 = disabled.",
    )
    ap.add_argument(
        "--bg-window",
        type=int,
        default=30,
        help="Number of consecutive frames to median over (default: 30).",
    )
    ap.add_argument(
        "--bg-static",
        action="store_true",
        help="Write backgrounds to the static stream dataset.",
    )
    ap.add_argument(
        "--bg-motion",
        action="store_true",
        help="Write backgrounds to the motion stream dataset.",
    )
    # If neither --bg-static nor --bg-motion is given, we default to both (if the stream exists).
    args = ap.parse_args()

    # load_params() reads sys.argv[1] itself (project dir OR .ini path) and
    # chdirs into the project. Hand it exactly one argument so our own flags
    # are never mistaken for a config path, and let it fall through to the
    # file dialog when `project` was omitted.
    sys.argv = [sys.argv[0]] + ([args.project] if args.project else [])

    params = load_params()
    pseudo_label(params, args)


if __name__ == "__main__":
    main()
