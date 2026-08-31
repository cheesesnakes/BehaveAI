#!/usr/bin/env python3
"""
Regenerate motion annotation images and secondary-classifier crops for a
BehaveAI project, and quarantine anything left inconsistent.

Usage:
        python regenerate_annotations.py <path/to/BehaveAI_settings.ini>
or:
        python regenerate_annotations.py        # will prompt for INI via file dialog

This script:
 - reads the settings INI (and resolves relative paths relative to the INI's directory)
 - rebuilds motion images (annot_motion/images/{train,val}) using the same processing
   as the annotation tool (sampling a small window of frames, computing diffs, chromatic tail, etc.)
 - applies masks and blocking boxes in the same way as your annotator
 - re-cuts the secondary-classifier crops under annot_{static,motion}_crop/ from the
   regenerated frames, so the classifier and the detector stay in sync
 - moves anything it could not regenerate into annotations/_stale/

Crop regeneration
-----------------
The annotation tool writes crops as

    annotations/annot_<stream>_crop/<primary>/<secondary>/<video>_<frame>_<x1>_<y1>.jpg

The path carries the class assignment; the filename carries only the top-left
corner. The crop rectangle itself lives in the YOLO label file for that frame.
So regeneration indexes the existing crops, then recovers each box by matching
its corner against the denormalised label boxes, and overwrites the crop in
place — every hand-assigned class survives untouched.

Quarantine
----------
Nothing is deleted. Files that can no longer be regenerated are MOVED to

    annotations/_stale/<original relative path>

preserving their directory structure, so a crop that was in
annot_motion_crop/fish/feeding/ lands in _stale/annot_motion_crop/fish/feeding/
and can be moved straight back. An orphan usually means an annotation was
edited after the crop was written, not that the file is genuinely dead — so
inspect _stale/ before emptying it.

Four things get quarantined:
  * crops whose top-left corner matches no box in the current label file
  * crops whose source frame has a label file with no boxes in it
  * crops whose source frame is absent from annot_*/labels entirely
  * images in annot_*/images whose label file is gone, or whose frame could
    not be regenerated this run (missing video, unreadable clip)

Set QUARANTINE_STALE = False to run in report-only mode: the same warnings are
printed, but nothing is moved.
"""

import configparser
import glob
import os
import shutil
import sys
import time

import cv2
from motion import advance_history, compose_motion_image

# optional GUI prompt if INI not supplied
try:
    import tkinter as tk
    from tkinter import filedialog

    _HAS_TK = True
except Exception:
    _HAS_TK = False

# Corner-match tolerance in pixels. Crop filenames store integer display
# coordinates; label files store normalised floats. Round-tripping through both
# introduces a pixel or two of drift.
CORNER_TOL = 4

# Move inconsistent files into annotations/_stale/ instead of leaving them in
# the training set. Set False to report without touching anything.
QUARANTINE_STALE = True
STALE_ROOT = os.path.join("annotations", "_stale")

ANNOT_ROOT = "annotations"

# -----------------------
# Helpers: path resolve / config loader
# -----------------------


def resolve_project_path(project_dir, value, fallback):
    """Resolve a path specified in the INI: absolute or relative to project_dir."""
    if value is None or str(value).strip() == "":
        value = fallback
    value = str(value)
    if os.path.isabs(value):
        return os.path.normpath(value)
    return os.path.normpath(os.path.join(project_dir, value))


def load_config(config_path):
    """
    Read configuration from config_path and return (params_dict, clips_dir_resolved).
    params contains numeric / strategy settings used by the image generation pipeline,
    and also the crop base directories for secondary classifiers.
    """
    config = configparser.ConfigParser()
    config.optionxform = str  # preserve case
    config.read(config_path)

    project_dir = os.path.dirname(os.path.abspath(config_path))

    params = {}
    try:
        # Read parameters (same names as your previous implementation)
        params["scale_factor"] = float(config["DEFAULT"].get("scale_factor", "1.0"))
        params["expA"] = float(config["DEFAULT"].get("expA", "0.5"))
        params["expB"] = float(config["DEFAULT"].get("expB", "0.8"))
        params["strategy"] = config["DEFAULT"].get("strategy", "exponential")
        params["chromatic_tail_only"] = (
            config["DEFAULT"].get("chromatic_tail_only", "false").lower()
        )
        params["lum_weight"] = float(config["DEFAULT"].get("lum_weight", "0.7"))
        params["rgb_multipliers"] = [
            float(x)
            for x in config["DEFAULT"].get("rgb_multipliers", "2,2,2").split(",")
        ]
        params["frame_skip"] = int(config["DEFAULT"].get("frame_skip", "0"))
        params["motion_threshold"] = -1 * int(
            config["DEFAULT"].get("motion_threshold", "0")
        )
        params["motion_blocks_static"] = (
            config["DEFAULT"].get("motion_blocks_static", "false").lower()
        )
        params["static_blocks_motion"] = (
            config["DEFAULT"].get("static_blocks_motion", "false").lower()
        )
        params["save_empty_frames"] = (
            config["DEFAULT"].get("save_empty_frames", "false").lower()
        )

        # --- Read crop base directories (new) ---
        params["static_cropped_base_dir"] = resolve_project_path(
            project_dir,
            config["DEFAULT"].get("static_cropped_base_dir", ""),
            "annotations/annot_static_crop",
        )
        params["motion_cropped_base_dir"] = resolve_project_path(
            project_dir,
            config["DEFAULT"].get("motion_cropped_base_dir", ""),
            "annotations/annot_motion_crop",
        )
        # ---------------------------------------------------------------

        # Compute base frame window size (number of sampled frames)
        base_window = 4
        if params["strategy"] == "exponential":
            if params["expA"] > 0.2 or params["expB"] > 0.2:
                base_window = 5
            if params["expA"] > 0.5 or params["expB"] > 0.5:
                base_window = 10
            if params["expA"] > 0.7 or params["expB"] > 0.7:
                base_window = 15
            if params["expA"] > 0.8 or params["expB"] > 0.8:
                base_window = 20
            if params["expA"] > 0.9 or params["expB"] > 0.9:
                base_window = 45

        params["base_frame_window"] = base_window
        params["frame_window"] = base_window * (params["frame_skip"] + 1)

    except KeyError as e:
        raise KeyError(f"Missing configuration parameter: {e}")

    # Resolve clips_dir relative to project_dir (fallback 'clips')
    clips_dir_ini = config["DEFAULT"].get("clips_dir", "clips")
    clips_dir = resolve_project_path(project_dir, clips_dir_ini, "clips")

    return params, clips_dir


# -----------------------
# Quarantine
# -----------------------


def quarantine(path, reason, stats):
    """
    Move one stale file into annotations/_stale/, preserving its path below
    annotations/. Never deletes; on name collision, appends a counter.
    """
    stats["count"] += 1
    if not os.path.exists(path):
        return

    if not QUARANTINE_STALE:
        print(f"  STALE (not moved): {path} — {reason}")
        return

    rel = os.path.relpath(path, ANNOT_ROOT)
    if rel.startswith(os.pardir):  # outside annotations/, keep the full path
        rel = os.path.relpath(path, ".")
    dest = os.path.join(STALE_ROOT, rel)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        stem, ext = os.path.splitext(dest)
        n = 1
        while os.path.exists(f"{stem}.{n}{ext}"):
            n += 1
        dest = f"{stem}.{n}{ext}"

    shutil.move(path, dest)
    stats["moved"] += 1
    print(f"  Quarantined: {path} -> {dest} ({reason})")


# -----------------------
# Image processing helpers
# -----------------------


def generate_base_images(video_path, frame_num, params):
    """
    Generate static and motion images for a specific video frame.
    frame_num is interpreted as the LAST frame of the motion window to mimic the annotator.
    Returns (static_img_bgr, motion_img_bgr) or (None, None) on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"Video appears empty or unreadable: {video_path}")
        cap.release()
        return None, None

    step = params["frame_skip"] + 1
    base_N = params.get("base_frame_window", 4)

    # compute start so last appended index should equal frame_num
    start_frame = int(frame_num - (base_N - 1) * step)
    start_frame = max(0, start_frame)
    if start_frame > total_frames - 1:
        print(
            f"Start frame {start_frame} beyond video length ({total_frames}) for {video_path}"
        )
        cap.release()
        return None, None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    collected = []
    read_count = 0
    idx = start_frame
    # safety limit: don't try more than frame_window + some slack
    max_reads = params["frame_window"] + 10

    while (
        len(collected) < base_N and idx <= total_frames - 1 and read_count <= max_reads
    ):
        ret, frame = cap.read()
        if not ret:
            break
        if (read_count % step) == 0:
            if params["scale_factor"] != 1.0:
                frame = cv2.resize(
                    frame, None, fx=params["scale_factor"], fy=params["scale_factor"]
                )
            collected.append(frame.copy())
        read_count += 1
        idx += 1

    if not collected:
        cap.release()
        print(
            f"Could not collect frames for target {frame_num} (start {start_frame}) in {video_path}"
        )
        return None, None

    # Process collected frames to produce diffs for the last frame
    prev_frames = [None] * 3
    static_img = None
    diffs = None
    gray = None

    for i, f in enumerate(collected):
        if f is None:
            continue
        frame_bgr = f
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if static_img is None:
            static_img = frame_bgr.copy()
            prev_frames = [gray.copy()] * 3
            continue

        current_diffs = advance_history(prev_frames, gray, params)

        if params["strategy"] == "exponential":
            prev_frames[0] = gray
            prev_frames[1] = cv2.addWeighted(
                prev_frames[1], params["expA"], gray, 1 - params["expA"], 0
            )
            prev_frames[2] = cv2.addWeighted(
                prev_frames[2], params["expB"], gray, 1 - params["expB"], 0
            )
        elif params["strategy"] == "sequential":
            prev_frames[2] = prev_frames[1]
            prev_frames[1] = prev_frames[0]
            prev_frames[0] = gray

        static_img = frame_bgr.copy()
        diffs = current_diffs

    if diffs is None or gray is None:
        cap.release()
        print(
            f"Insufficient frames to compute diffs for {frame_num} (collected {len(collected)} frames)"
        )
        return None, None

    # Build motion image (chromatic tail or normal)
    motion_img = compose_motion_image(gray, diffs, params)

    cap.release()
    return static_img, motion_img


def read_mask_file(mask_path):
    boxes = []
    if os.path.exists(mask_path):
        with open(mask_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    try:
                        boxes.append(tuple(map(int, parts)))
                    except Exception:
                        pass
    return boxes


def apply_grey_boxes(image, boxes):
    result = image.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(result, (x1, y1), (x2, y2), (128, 128, 128), -1)
    return result


def apply_blocking_boxes(image, boxes):
    result = image.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(result, (x1, y1), (x2, y2), (128, 128, 128), -1)
    return result


def get_blocking_boxes(label_path, img_w, img_h):
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except Exception:
                    continue
                x1 = int((xc - w / 2) * img_w)
                y1 = int((yc - h / 2) * img_h)
                x2 = int((xc + w / 2) * img_w)
                y2 = int((yc + h / 2) * img_h)
                boxes.append((x1, y1, x2, y2))
    return boxes


# -----------------------
# Secondary-classifier crops
# -----------------------


def index_crops(crop_base):
    """
    Walk a crop tree and index it by source frame.

    Returns {(video_name, frame_num): [(crop_path, x1, y1), ...]}
    """
    index = {}
    if not crop_base or not os.path.isdir(crop_base):
        return index

    pattern = os.path.join(crop_base, "**", "*.jpg")
    matches = [
        p for p in glob.glob(pattern, recursive=True) if "_stale" not in p.split(os.sep)
    ]
    if not matches:
        print(f"  No crops found under {crop_base}")
        return index

    for crop_path in matches:
        stem = os.path.splitext(os.path.basename(crop_path))[0]
        # video_label may itself contain underscores, so split from the right:
        # the last three tokens are frame, x1, y1.
        parts = stem.rsplit("_", 3)
        if len(parts) != 4:
            print(f"  Skipping unparseable crop name: {crop_path}")
            continue
        video_name, frame_s, x1_s, y1_s = parts
        try:
            key = (video_name, int(frame_s))
            index.setdefault(key, []).append((crop_path, int(x1_s), int(y1_s)))
        except ValueError:
            print(f"  Skipping unparseable crop name: {crop_path}")

    return index


def regenerate_crops_for_frame(
    video_name,
    frame_num,
    final_img,
    label_path,
    crop_index,
    img_w,
    img_h,
    stream_name,
):
    """
    Re-cut every indexed crop for one frame from a freshly generated image.
    """
    entries = crop_index.get((video_name, frame_num), [])
    if not entries or final_img is None:
        return 0, []

    boxes = get_blocking_boxes(label_path, img_w, img_h)
    if not boxes:
        print(
            f"  {stream_name}: {len(entries)} crop(s) for {video_name} frame "
            f"{frame_num} but no boxes in {label_path}"
        )
        return 0, [(p, "label file has no boxes") for p, _, _ in entries]

    regenerated, stale = 0, []
    for crop_path, cx1, cy1 in entries:
        # nearest label box by top-left corner
        best, best_d = None, None
        for bx1, by1, bx2, by2 in boxes:
            d = abs(bx1 - cx1) + abs(by1 - cy1)
            if best_d is None or d < best_d:
                best, best_d = (bx1, by1, bx2, by2), d

        if best_d > CORNER_TOL:
            stale.append((crop_path, f"no box within {CORNER_TOL}px of ({cx1},{cy1})"))
            continue

        x1, y1, x2, y2 = best
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            stale.append((crop_path, "matched box is empty after clipping"))
            continue

        crop = final_img[y1:y2, x1:x2]
        if crop.size == 0:
            stale.append((crop_path, "crop region is empty"))
            continue

        cv2.imwrite(crop_path, crop)
        regenerated += 1

    return regenerated, stale


# -----------------------
# Main regeneration function
# -----------------------


def regenerate_annotations(config_path):
    """Regenerate images and crops, then quarantine what could not be rebuilt."""
    params, clips_dir = load_config(config_path)

    # Ensure we operate with project_dir as cwd to keep relative paths consistent
    project_dir = os.path.dirname(os.path.abspath(config_path))
    os.chdir(project_dir)

    # Extract crop base directories from params (now available)
    static_crop_base = params["static_cropped_base_dir"]
    motion_crop_base = params["motion_cropped_base_dir"]

    print(f"Regenerating using INI: {config_path}")
    print(f"Using clips directory: {clips_dir}")
    print(f"Static crop base: {static_crop_base}")
    print(f"Motion crop base: {motion_crop_base}")
    if not QUARANTINE_STALE:
        print("Quarantine disabled — stale files will be reported, not moved.")

    stats = {"count": 0, "moved": 0}

    # collect annotated frames from both motion and static label dirs
    base_dirs = [("annot_motion", ["train", "val"]), ("annot_static", ["train", "val"])]

    # collect unique base names from these directories
    base_names = set()
    for base_dir, splits in base_dirs:
        for split in splits:
            label_dir = os.path.join(ANNOT_ROOT, base_dir, "labels", split)
            if not os.path.exists(label_dir):
                continue
            for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
                if label_file.endswith(".mask.txt"):
                    continue
                base_name = os.path.splitext(os.path.basename(label_file))[0]
                base_names.add((base_name, split, base_dir))

    print(f"Found {len(base_names)} annotated frames to process (motion + static).")

    # Index existing secondary-classifier crops using the config‑provided base dirs
    static_crop_index = index_crops(static_crop_base)
    motion_crop_index = index_crops(motion_crop_base)
    n_static_crops = sum(len(v) for v in static_crop_index.values())
    n_motion_crops = sum(len(v) for v in motion_crop_index.values())
    print(f"Indexed {n_static_crops} static and {n_motion_crops} motion crops.")

    crops_done = 0
    static_seen, motion_seen = set(), set()

    # extensions to search for video files
    exts = [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]

    # process each unique frame
    for base_name, split, base_dir in sorted(base_names):
        parts = base_name.split("_")
        try:
            frame_num = int(parts[-1])
        except ValueError:
            print(f"Skipping {base_name}: trailing token is not an integer")
            continue
        video_name = "_".join(parts[:-1])

        static_img_path = os.path.join(
            ANNOT_ROOT, "annot_static", "images", split, f"{base_name}.jpg"
        )
        motion_img_path = os.path.join(
            ANNOT_ROOT, "annot_motion", "images", split, f"{base_name}.jpg"
        )

        # find video in clips_dir
        video_path = None
        for ext in exts:
            test_path = os.path.join(clips_dir, video_name + ext)
            if os.path.exists(test_path):
                video_path = test_path
                break

        if not video_path:
            print(
                f"Video not found for {base_name}: looking in {clips_dir} for files named {video_name}.*"
            )
            quarantine(static_img_path, "source video not found", stats)
            quarantine(motion_img_path, "source video not found", stats)
            continue

        static_img, motion_img = generate_base_images(video_path, frame_num, params)
        if static_img is None and motion_img is None:
            print(f"  Could not generate images for {base_name}")
            quarantine(static_img_path, "frame could not be regenerated", stats)
            quarantine(motion_img_path, "frame could not be regenerated", stats)
            continue

        # image dims (prefer static_img if available else motion_img)
        ref_img = static_img if static_img is not None else motion_img
        img_h, img_w = ref_img.shape[:2]

        # mask & label paths for both static and motion (may or may not exist)
        static_mask_path = os.path.join(
            ANNOT_ROOT, "annot_static", "masks", split, f"{base_name}.mask.txt"
        )
        motion_mask_path = os.path.join(
            ANNOT_ROOT, "annot_motion", "masks", split, f"{base_name}.mask.txt"
        )

        static_mask_boxes = read_mask_file(static_mask_path)
        motion_mask_boxes = read_mask_file(motion_mask_path)

        static_label_path = os.path.join(
            ANNOT_ROOT, "annot_static", "labels", split, f"{base_name}.txt"
        )
        motion_label_path = os.path.join(
            ANNOT_ROOT, "annot_motion", "labels", split, f"{base_name}.txt"
        )

        # Build both processed frames (same as before)
        static_final = None
        if static_img is not None:
            static_final = apply_grey_boxes(static_img, static_mask_boxes)
            if params.get("motion_blocks_static", "false") == "true":
                static_block_boxes = get_blocking_boxes(motion_label_path, img_w, img_h)
                static_final = apply_blocking_boxes(static_final, static_block_boxes)

        motion_final = None
        if motion_img is not None:
            motion_final = apply_grey_boxes(motion_img, motion_mask_boxes)
            if params.get("static_blocks_motion", "false") == "true":
                static_boxes = get_blocking_boxes(static_label_path, img_w, img_h)
                motion_final = apply_blocking_boxes(motion_final, static_boxes)

        # Write full frames
        if base_dir == "annot_static" or params["save_empty_frames"] == "true":
            if static_final is None:
                print(f"  No static image for {base_name}")
                quarantine(static_img_path, "static frame could not be built", stats)
            else:
                os.makedirs(os.path.dirname(static_img_path), exist_ok=True)
                cv2.imwrite(static_img_path, static_final)
                print(f"Regenerated static: {static_img_path}")

        if base_dir == "annot_motion" or params["save_empty_frames"] == "true":
            if motion_final is None:
                print(f"  No motion image for {base_name}")
                quarantine(motion_img_path, "motion frame could not be built", stats)
            else:
                os.makedirs(os.path.dirname(motion_img_path), exist_ok=True)
                cv2.imwrite(motion_img_path, motion_final)
                print(f"Regenerated motion: {motion_img_path}")

        # Re-cut secondary-classifier crops from the same processed frames
        key = (video_name, frame_num)
        for stream, final_img, lbl, idx, seen in (
            ("static", static_final, static_label_path, static_crop_index, static_seen),
            ("motion", motion_final, motion_label_path, motion_crop_index, motion_seen),
        ):
            if final_img is None:
                continue
            seen.add(key)
            r, stale = regenerate_crops_for_frame(
                video_name, frame_num, final_img, lbl, idx, img_w, img_h, stream
            )
            crops_done += r
            for path, reason in stale:
                quarantine(path, reason, stats)

    print("Regeneration loop complete.")

    # Sweep 1: crops whose source frame was never processed
    for idx, seen, name in (
        (static_crop_index, static_seen, "static"),
        (motion_crop_index, motion_seen, "motion"),
    ):
        for key, entries in idx.items():
            if key in seen:
                continue
            for crop_path, _, _ in entries:
                quarantine(
                    crop_path,
                    f"no regenerated {name} frame for {key[0]} frame {key[1]}",
                    stats,
                )

    # Sweep 2: images with no surviving label file (unchanged)
    for base_dir in ("annot_static", "annot_motion"):
        for split in ("train", "val"):
            img_dir = os.path.join(ANNOT_ROOT, base_dir, "images", split)
            for img_path in glob.glob(os.path.join(img_dir, "*.jpg")):
                stem = os.path.splitext(os.path.basename(img_path))[0]
                lbl = os.path.join(ANNOT_ROOT, base_dir, "labels", split, f"{stem}.txt")
                if not os.path.exists(lbl):
                    quarantine(img_path, "no matching label file", stats)

    print(f"Crops: {crops_done} regenerated.")
    verb = "quarantined" if QUARANTINE_STALE else "flagged (not moved)"
    print(
        f"Stale files {verb}: {stats['moved'] if QUARANTINE_STALE else stats['count']}"
    )
    if stats["count"]:
        print(
            f"  Review {STALE_ROOT} before deleting anything — an orphan usually "
            "means an annotation was edited, not that the file is dead."
        )


# -----------------------
# CLI & prompt logic
# -----------------------


def choose_ini_path_via_dialog():
    if not _HAS_TK:
        return None
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select BehaveAI settings INI",
        filetypes=[("INI files", "*.ini"), ("All files", "*.*")],
    )
    root.destroy()
    return path


if __name__ == "__main__":
    # Determine config_path from command-line or prompt
    if len(sys.argv) > 1:
        arg = os.path.abspath(sys.argv[1])
        if os.path.isdir(arg):
            config_path = os.path.join(arg, "BehaveAI_settings.ini")
        else:
            config_path = arg
    else:
        config_path = choose_ini_path_via_dialog()
        if not config_path:
            # no selection: report and exit
            print("No settings INI selected — exiting.")
            sys.exit(0)

    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    # Run regeneration
    start_t = time.time()
    regenerate_annotations(config_path)
    elapsed = time.time() - start_t
    print(f"Regeneration complete! Elapsed {elapsed:.1f} s")
