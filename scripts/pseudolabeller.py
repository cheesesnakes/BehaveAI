"""
pseudo_label.py
---------------
One-shot pseudo-labeller for the primary STATIC stream.

Walks every video in `clips_dir`, runs the external YOLO detector
specified by `primary_static_external_model`, confidence-filters the
detections, and writes BehaveAI-compatible annotations:

  <project>/annot_static/images/{train,val}/<video>_<frame>.jpg
  <project>/annot_static/labels/{train,val}/<video>_<frame>.txt

Label format is identical to what scripts/annotation.py writes:
    <cls> <xc> <yc> <w> <h>   (all normalized, 6 decimal places)

ASSUMPTION: the external model's class IDs already match this project's
`primary_static_classes` order. No remap is performed. If the external
model emits a class ID outside the valid range it is dropped.

After this runs, open the annotation GUI — every pseudo-labelled frame
shows up ready for correction.

Usage:
    python scripts/pseudo_label.py
        [--sample-every N]     # take 1 frame every N frames (default 30)
        [--max-per-video K]    # cap frames per video (0 = unlimited)
        [--val-frequency F]    # override config's val_frequency
        [--dry-run]            # don't write anything, just report

Config keys it reads (in addition to what load_configs already loads):
    primary_static_external_model = /path/to/best.pt
    pseudo_label_conf = 0.6
"""

import argparse
import os
import random
import sys

import cv2

# GLOBALS

DEFAULT_CONF = 0.6

from load_configs import load_params  # noqa: E402


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def _iter_videos(clips_dir):
    exts = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
    for root, _, files in os.walk(clips_dir):
        for f in sorted(files):
            if f.lower().endswith(exts):
                yield os.path.join(root, f)


def _video_label(path, clips_dir):
    """Mirror annotation.py's '<video_label>_<frame>' filename convention."""
    rel = os.path.relpath(path, clips_dir)
    stem, _ = os.path.splitext(rel)
    return stem.replace(os.sep, "__").replace(" ", "_")


def pseudo_label(params, args):
    from ultralytics import YOLO  # lazy import so --help stays fast

    weights = params["primary_static_external_model"]
    if not weights:
        sys.exit(
            "primary_static_external_model is empty in config.ini — nothing to do."
        )
    if not os.path.isfile(weights) and not os.path.isdir(weights):
        sys.exit(f"primary_static_external_model path does not exist: {weights}")

    conf = args.conf if args.conf is not None else DEFAULT_CONF

    n_project_classes = len(params["primary_static_classes"])

    val_freq = (
        args.val_frequency
        if args.val_frequency is not None
        else float(params.get("val_frequency", 0.1))
    )
    save_empty = str(params.get("save_empty_frames", "false")).lower() == "true"

    clips_dir = params["clips_dir"]
    train_img = params["static_train_images_dir"]
    train_lbl = params["static_train_labels_dir"]
    val_img = params["static_val_images_dir"]
    val_lbl = params["static_val_labels_dir"]
    for d in (train_img, train_lbl, val_img, val_lbl):
        os.makedirs(d, exist_ok=True)

    print(f"[pseudo-label] weights:       {weights}")
    print(f"[pseudo-label] conf:          {conf}")
    print(f"[pseudo-label] project cls:   {params['primary_static_classes']}")
    print(f"[pseudo-label] sample:        every {args.sample_every} frames")
    print(f"[pseudo-label] val_freq:      {val_freq}")
    print(f"[pseudo-label] save_empty:    {save_empty}")
    print(f"[pseudo-label] dry_run:       {args.dry_run}")

    model = YOLO(weights)
    rng = random.Random(0)  # deterministic train/val split

    n_frames_written = 0
    n_detections = 0
    n_videos = 0
    n_dropped_oor = 0

    for video_path in _iter_videos(clips_dir):
        n_videos += 1
        vlabel = _video_label(video_path, clips_dir)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [skip] cannot open {video_path}")
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        kept_for_video = 0
        frame_idx = -1
        while True:
            frame_idx += 1
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.sample_every != 0:
                continue
            if args.max_per_video and kept_for_video >= args.max_per_video:
                break

            h, w = frame.shape[:2]
            results = model.predict(frame, conf=conf, verbose=False)
            r = results[0]
            lines = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), ext_cls, c in zip(xyxy, cls, confs):
                    if c < conf:
                        continue
                    # External class ID must be a valid index into
                    # primary_static_classes (e.g. 0 for a fish-only model).
                    if not (0 <= ext_cls < n_project_classes):
                        n_dropped_oor += 1
                        continue
                    xc = ((x1 + x2) / 2.0) / w
                    yc = ((y1 + y2) / 2.0) / h
                    bw = abs(x2 - x1) / w
                    bh = abs(y2 - y1) / h
                    xc = min(max(xc, 0.0), 1.0)
                    yc = min(max(yc, 0.0), 1.0)
                    bw = min(max(bw, 0.0), 1.0)
                    bh = min(max(bh, 0.0), 1.0)
                    lines.append(f"{int(ext_cls)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            if not lines and not save_empty:
                continue

            is_val = rng.random() < val_freq
            target_img_dir = val_img if is_val else train_img
            target_lbl_dir = val_lbl if is_val else train_lbl

            base = f"{vlabel}_{frame_idx}"
            img_path = os.path.join(target_img_dir, f"{base}.jpg")
            lbl_path = os.path.join(target_lbl_dir, f"{base}.txt")

            if not args.dry_run:
                cv2.imwrite(img_path, frame)
                with open(lbl_path, "w") as f:
                    if lines:
                        f.write("\n".join(lines) + "\n")
                    # An empty .txt is valid YOLO (background frame)

            kept_for_video += 1
            n_frames_written += 1
            n_detections += len(lines)

        cap.release()
        print(
            f"  {os.path.basename(video_path)}: "
            f"{kept_for_video} frames kept ({total} total)"
        )

    print("")
    print(f"[pseudo-label] videos processed:        {n_videos}")
    print(f"[pseudo-label] frames written:          {n_frames_written}")
    print(f"[pseudo-label] detections written:      {n_detections}")
    if n_dropped_oor:
        print(
            f"[pseudo-label] detections dropped (cls out of range): {n_dropped_oor}\n"
            f"                -> external model emitted class IDs outside "
            f"0..{n_project_classes - 1}; check it really matches primary_static_classes."
        )
    if args.dry_run:
        print("[pseudo-label] DRY RUN — no files were written.")
    else:
        print(
            "[pseudo-label] Done. Run training and then open the annotation GUI to review and correct."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project_name", help="Name of the project (must already exist)")
    ap.add_argument(
        "--sample-every",
        type=int,
        default=30,
        help="Take 1 frame every N frames (default: 30)",
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
        "--dry-run",
        action="store_true",
        help="Don't write files, just print what would happen",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help="Override config's pseudo_label_conf",
    )

    args = ap.parse_args()

    # set up project path and load config
    project_path = os.path.join(os.getcwd(), "projects", args.project_name)
    if not os.path.isdir(project_path):
        sys.exit(f"Project directory does not exist: {project_path}")
    os.chdir(project_path)
    params = load_params()
    pseudo_label(params, args)


if __name__ == "__main__":
    main()
