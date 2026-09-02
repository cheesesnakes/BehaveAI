"""
classify_track.py
=================

Batch pipeline that takes every video in the project's `input/` folder and
produces two artefacts per video into `output/`:

    <n>_detected.mp4   — annotated video with boxes, labels, tracks
    <n>_tracking.csv   — per-frame, per-track class + confidence dump

The pipeline has five conceptual stages:

    1. Train-or-verify models     (train_models)
    2. Load primary models        (inside process_video, per video)
    3. Per-frame detection loop   (motion image → YOLO → merge)
    4. Secondary classification   (hierarchical crops → classifier YOLO)
    5. Tracking + rendering       (tracker → overlays → CSV/MP4)

Missing-model handling
----------------------
Primary and secondary model weights are only produced if training has enough
annotations to run. When weights are absent:

    * primary static/motion:   the stream is skipped for every video; other
                               streams keep running; a video is skipped only
                               if BOTH primary models are missing.
    * secondary classifiers:   the per-class key is absent from the dict, so
                               the runtime lookup falls through to primary
                               class only (secondary_* columns default to
                               primary class with conf=1.0).

This lets classification run as soon as any one model trains, instead of
requiring the full stack.

Class imbalance in the secondary classifiers
--------------------------------------------
The behaviour subclasses are heavily skewed (a "moving"-dominated dataset is
typical). Two mechanisms address that here:

    * BalancedClassificationTrainer replaces the training sampler with a
      WeightedRandomSampler so every subclass contributes roughly equally
      per epoch, without deleting any images from disk. The reweighting
      exponent is `secondary_sampler_power` in the INI (0.0 = off / natural
      frequencies, 0.5 = sqrt-inverse frequency [default], 1.0 = full inverse
      frequency). Full inverse is usually too aggressive on a rare class with
      only a couple of hundred originals — it overfits.

    * Training hyperparameters are chosen per TASK rather than shared with
      the detectors. See DET_TRAIN_ARGS / CLS_TRAIN_ARGS_* below; in
      particular the motion classifier gets hue/saturation augmentation
      disabled, because in a motion image the colour IS the signal.

Tracking
--------
Two backends, selected by the `tracker_type` key in the [tracker] section
of the project INI. All other tracker settings live in that same section
(det_thresh, det_conf_floor, max_age, min_hits, iou_threshold, ...) and
are read by load_configs.read_tracker_params() into params["tracker"].

    "builtin"  -> KalmanTracker below. No torch dependency, so this is the
                  one to use on the NCNN / Raspberry Pi path.
    anything   -> BoxMOTTracker (ocsort, bytetrack, botsort, deepocsort...).
       else       Requires `pip install boxmot`.

Both expose the same interface:
    update(detections, frame, frame_idx) -> {det_index: track_id}
    state(tid) -> (x, y, vx, vy)  |  box(tid) -> (x1, y1, x2, y2)
    `tid in tracker.tracks`
"""

import csv
import glob
import os
import random
import re
import shutil
import time
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from load_configs import load_params
from motion import create_motion_image
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# Optional — only needed when tracker_type != "builtin".
try:
    from boxmot_tracker import BoxMOTTracker
except Exception:  # noqa: BLE001 - boxmot is an optional heavyweight dep
    BoxMOTTracker = None

# Load all config into a single dict. See load_configs.py for keys.
params = load_params()

# Image extensions recognised when counting dataset contents.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ============================================================================
# STAGE 0 — Training hyperparameters, split by task
# ----------------------------------------------------------------------------
# maybe_retrain() is shared between the primary DETECTORS and the secondary
# CLASSIFIERS. Previously it passed one argument block to both, which meant
# the classifiers were being trained with detector settings:
#
#   * box / cls / dfl and mosaic are detection-only losses and augmentations.
#     Inert for a classification task, but it also meant nobody had ever
#     actually chosen classification hyperparameters.
#
#   * More seriously, Ultralytics' classification defaults include
#     hsv_h=0.015 and hsv_s=0.7. The motion image encodes three temporally
#     offset frame differences into the B/G/R channels — the hue and
#     saturation ARE the motion cue. Jittering them scrambles exactly the
#     signal the motion classifier needs to separate a directed swim from a
#     stationary scan. Disabled below for the motion stream only; the static
#     stream sees real appearance, so mild colour jitter is fine (and helps).
# ============================================================================

DET_TRAIN_ARGS = dict(
    # ----- Regularization (Adjusted for Detection) -----
    weight_decay=0.0005,
    dropout=0.0,
    label_smoothing=0.1,
    batch=16,
    # ----- Optimizer -----
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    # ----- Object-Specific Augmentations (The Magic) -----
    copy_paste=0.3,
    mixup=0.1,
    # ----- Standard Augmentations -----
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.2,
    scale=0.5,
    degrees=10,
    erasing=0.2,
    fliplr=0.5,
    # --- Loss Weights ---
    box=7.5,
    cls=0.5,
    dfl=1.5,
)

# Secondary classifier on MOTION crops. Colour is the motion encoding, so
# hue/saturation augmentation is off. Geometry jitter is kept small because a
# behaviour crop is already tightly framed on the animal.
CLS_TRAIN_ARGS_MOTION = dict(
    dropout=0.2,
    weight_decay=0.005,
    batch=128,
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    hsv_h=0.0,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.15,
    scale=0.3,
    degrees=10,
    erasing=0,
    fliplr=0.5,
)

# Secondary classifier on STATIC crops. Colour here is genuine appearance, so
# normal photometric augmentation applies.
CLS_TRAIN_ARGS_STATIC = dict(
    dropout=0.2,
    weight_decay=0.005,
    batch=128,
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    hsv_h=0.0,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.15,
    scale=0.3,
    degrees=10,
    erasing=0,
    fliplr=0.5,
)


def train_args_for(task, stream=None):
    """
    Return a fresh copy of the training-argument block for a given task.

    task   : "detect" (primary models) or "classify" (secondary models)
    stream : "static" or "motion" — only consulted for classification.
    """
    if task == "classify":
        base = CLS_TRAIN_ARGS_MOTION if stream == "motion" else CLS_TRAIN_ARGS_STATIC
        return dict(base)
    return dict(DET_TRAIN_ARGS)


# ============================================================================
# STAGE 0c — Class-balanced sampling for the secondary classifiers
# ----------------------------------------------------------------------------
# With a subclass distribution like moving=2500, vigilance=698, bite=327,
# scanning=194, a classifier trained on natural frequencies collapses toward
# the majority class: it can score ~67% top-1 by answering "moving" every
# time, and the gradient signal is dominated by moving regardless.
#
# Rather than deleting images from disk, this trainer swaps the training
# dataloader's sampler for a WeightedRandomSampler with per-sample weight
#
#       w_i = 1 / count(class_i) ** power
#
# so each class's expected contribution per epoch is equalised (at power=1)
# or partially equalised (0 < power < 1). num_samples is kept at len(dataset)
# so epoch length — and therefore the LR schedule and epoch budget — is
# unchanged.
#
# Why the default power is 0.5 and not 1.0: at full inverse frequency each
# scanning image would be shown roughly 13x as often as each moving image.
# With only ~194 scanning originals that is a direct route to memorising
# them. sqrt-inverse is the usual compromise; sweep 0.25 / 0.5 / 0.75 on
# val macro-F1.
#
# Validation is deliberately left UNTOUCHED (mode != "train" returns early):
# val must reflect the real distribution for the metrics to mean anything.
# Note however that Ultralytics selects best.pt by val top-1 accuracy, which
# on a skewed val set still rewards collapsing into the majority class. If
# checkpoint selection looks wrong, equalise the val split on disk as well.
#
# This subclass reaches into Ultralytics internals (dataset.samples, the
# InfiniteDataLoader type). Those move between releases, so every step is
# guarded: on any failure it logs and falls back to the stock loader, and
# training proceeds unbalanced rather than crashing.
# ============================================================================

try:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from ultralytics.models.yolo.classify import ClassificationTrainer

    class BalancedClassificationTrainer(ClassificationTrainer):
        """ClassificationTrainer with inverse-frequency training sampler."""

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            loader = super().get_dataloader(dataset_path, batch_size, rank, mode)

            # Never touch the validation loader.
            if mode != "train":
                return loader

            try:
                dataset = loader.dataset

                # torchvision-style ImageFolder tree: samples is a list of
                # (path, class_index) pairs.
                samples = getattr(dataset, "samples", None)
                if not samples:
                    print(
                        "  [balanced-sampler] dataset exposes no .samples — "
                        "falling back to the default sampler."
                    )
                    return loader

                targets = np.asarray([int(s[1]) for s in samples])
                n_classes = int(targets.max()) + 1
                counts = np.bincount(targets, minlength=n_classes).astype(np.float64)

                power = float(params.get("secondary_sampler_power", 0.5))
                if power <= 0.0:
                    print("  [balanced-sampler] power <= 0, sampler disabled.")
                    return loader

                weights_per_class = 1.0 / np.maximum(counts, 1.0) ** power
                sample_weights = weights_per_class[targets]

                # Report what the sampler will actually do, per class.
                # Ultralytics' ClassificationDataset keeps the ImageFolder on
                # .base, so .classes is not on the dataset itself.
                names = (
                    getattr(dataset, "classes", None)
                    or getattr(getattr(dataset, "base", None), "classes", None)
                    or [str(i) for i in range(n_classes)]
                )
                share = (weights_per_class * counts) / (
                    weights_per_class * counts
                ).sum()
                print(f"  [balanced-sampler] power={power}")
                for i, name in enumerate(names[:n_classes]):
                    print(
                        f"      {name:<12} n={int(counts[i]):>5}  "
                        f"natural={counts[i] / counts.sum():.3f}  "
                        f"sampled={share[i]:.3f}"
                    )

                sampler = WeightedRandomSampler(
                    torch.as_tensor(sample_weights, dtype=torch.double),
                    num_samples=len(targets),  # keep epoch length unchanged
                    replacement=True,
                )

                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    num_workers=getattr(loader, "num_workers", 0),
                    pin_memory=getattr(loader, "pin_memory", True),
                    collate_fn=getattr(loader, "collate_fn", None),
                    drop_last=getattr(loader, "drop_last", False),
                    persistent_workers=False,
                )
            except Exception as e:  # noqa: BLE001 - never fail training over this
                print(
                    f"  Warning: balanced sampler unavailable ({e}); "
                    "training with the default sampler."
                )
                return loader

except Exception as e:  # noqa: BLE001 - torch/ultralytics layout changed
    print(f"Note: BalancedClassificationTrainer unavailable ({e}).")
    BalancedClassificationTrainer = None


def trainer_kwarg_for(task):
    """
    Return {"trainer": ...} for classification when the balanced trainer is
    importable, else {} so model.train() uses the stock trainer.
    """
    if task == "classify" and BalancedClassificationTrainer is not None:
        return {"trainer": BalancedClassificationTrainer}
    return {}


# ============================================================================
# STAGE 0a — NCNN export/load helpers
# ----------------------------------------------------------------------------
# NCNN is a portable inference runtime optimised for CPU / edge devices (e.g.
# Raspberry Pi). Ultralytics can export a .pt model to an NCNN folder of
# .param + .bin files. These helpers (a) check/produce that folder, and
# (b) load it, with graceful fallback to the original .pt on any failure.
# ============================================================================


def ncnn_dir_for_weights(weights_path):
    """Return the expected NCNN export directory for a given .pt path."""
    base, _ext = os.path.splitext(weights_path)
    # Ultralytics export creates a folder named "<base>_ncnn_model".
    return base + "_ncnn_model"


def ncnn_files_exist(ncnn_dir):
    """Return True if NCNN .param and .bin files appear to exist in the dir."""
    if not os.path.isdir(ncnn_dir):
        return False
    has_param = any(f.endswith(".param") for f in os.listdir(ncnn_dir))
    has_bin = any(f.endswith(".bin") for f in os.listdir(ncnn_dir))
    return has_param and has_bin


def ensure_ncnn_export(weights_path, task, timeout=300):
    """
    Ensure an NCNN conversion exists for `weights_path`.
    Returns the ncnn_dir on success, or None on failure (caller falls back to .pt).
    Skips conversion if the NCNN folder already exists.
    """
    ncnn_dir = ncnn_dir_for_weights(weights_path)
    if ncnn_files_exist(ncnn_dir):
        return ncnn_dir

    try:
        print(f"Exporting {weights_path} -> NCNN (this may take a while)...")
        model = YOLO(weights_path, task=task)
        # Triggers "<base>_ncnn_model" folder creation.
        model.export(format="ncnn")
        # Poll for the output folder — export is usually synchronous but
        # we guard against slow filesystems.
        start = time.time()
        while time.time() - start < timeout:
            if ncnn_files_exist(ncnn_dir):
                print(f"NCNN export complete: {ncnn_dir}")
                return ncnn_dir
            time.sleep(0.5)
        print(f"NCNN export timeout for {weights_path}")
        return None
    except Exception as e:
        # Don't crash the whole run — caller will fall back to .pt.
        print(f"Warning: NCNN export failed for {weights_path}: {e}")
        return None


def load_model_with_ncnn_preference(weights_path, task):
    """
    Prefer NCNN if available (or convert once). On any failure, fall back
    to the original PyTorch .pt path. Returns a YOLO instance.
    """
    # If caller passed a folder instead of a .pt, just try loading directly.
    if not weights_path.endswith(".pt"):
        try:
            return YOLO(weights_path, task=task)
        except Exception as e:
            print(f"Error loading model {weights_path}: {e}")
            raise

    ncnn_dir = ncnn_dir_for_weights(weights_path)

    # Path 1: NCNN folder already exists — use it.
    if ncnn_files_exist(ncnn_dir):
        try:
            print(f"Loading NCNN model from {ncnn_dir}")
            return YOLO(ncnn_dir, task=task)
        except Exception as e:
            print(f"Failed to load NCNN model at {ncnn_dir}: {e} (falling back to .pt)")

    # Path 2: convert .pt -> NCNN once, then load.
    exported = ensure_ncnn_export(weights_path, task)
    if exported:
        try:
            return YOLO(exported, task=task)
        except Exception as e:
            print(
                f"Failed to load NCNN-exported model {exported}: {e} (falling back to .pt)"
            )

    # Path 3: original PyTorch weights.
    print(f"Using original weights (PyTorch) at {weights_path}")
    return YOLO(weights_path, task=task)


# ============================================================================
# STAGE 0b — Training output relocation
# ----------------------------------------------------------------------------
# Ultralytics sometimes writes training outputs to its own `runs/detect/...`
# folder regardless of the `project=` argument. This helper finds the most
# recent best.pt anywhere plausible and moves its containing run directory
# into the canonical location the rest of the code expects:
#
#     <project_path>/<run_name>/weights/best.pt
# ============================================================================


def move_to_expected(project_path, run_name="train", runs_root="runs"):
    """
    Locate the most recently written best.pt anywhere Ultralytics might have
    put it, and relocate its parent run directory into project_path/run_name.
    """
    # Fast path: weights already in the expected location.
    expected_weights = os.path.join(project_path, run_name, "weights", "best.pt")
    if os.path.exists(expected_weights):
        return os.path.join(project_path, run_name)

    # Broad search — exclude project_path itself (previous partial move) and
    # any *_backup dirs.
    search_roots = [runs_root, "../../runs", "."]
    candidates = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for path in glob.glob(
            os.path.join(root, "**", "weights", "best.pt"), recursive=True
        ):
            abs_path = os.path.abspath(path)
            if os.path.abspath(project_path) in abs_path:
                continue
            if "_backup" in abs_path:
                continue
            candidates.append(abs_path)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find 'best.pt' after training. Searched in '{search_roots}' "
            f"and current directory, excluding '{project_path}' and backups."
        )

    # Most recently modified wins.
    candidates.sort(key=os.path.getmtime, reverse=True)
    best_pt = candidates[0]

    # The run dir is two levels up from best.pt: <run_dir>/weights/best.pt
    src_run_dir = os.path.dirname(os.path.dirname(best_pt))
    dst_run_dir = os.path.join(project_path, run_name)

    try:
        os.makedirs(project_path, exist_ok=True)
        if os.path.exists(dst_run_dir):
            shutil.rmtree(dst_run_dir)
        shutil.move(src_run_dir, dst_run_dir)
        print(f"Moved YOLO training output: '{src_run_dir}' -> '{dst_run_dir}'")

        # Cleanup: remove now-empty ancestor dirs under runs_root.
        runs_root_abs = os.path.abspath(runs_root)
        parent = os.path.abspath(os.path.dirname(src_run_dir))
        while parent.startswith(runs_root_abs) and parent != runs_root_abs:
            try:
                if os.path.isdir(parent) and not os.listdir(parent):
                    os.rmdir(parent)
                    parent = os.path.dirname(parent)
                else:
                    break
            except Exception:
                break

        return dst_run_dir
    except Exception as e:
        print(f"Warning: failed to move '{src_run_dir}' -> '{dst_run_dir}': {e}")
        return None


# Global used by interactive retrain prompts (legacy; kept for compatibility).
global_response = 0


# ============================================================================
# STAGE 1 — Training or verifying models
# ----------------------------------------------------------------------------
# Before any video processing, walk every configured model and make sure its
# weights are on disk. count_images_in_dataset() answers "do we have enough
# training data?", maybe_retrain() either trains from scratch, fine-tunes from
# existing weights, or silently skips if data is insufficient.
#
# Minimum-image policy:
#   * Primary models:   at least  2 images required  (in maybe_retrain).
#   * Secondary models: at least  2 images required  (in train_models).
# A model that fails to train leaves its best.pt absent — downstream code
# detects this at load time and skips that stream for inference.
# ============================================================================


# Split directory names, in preference order. Ultralytics classification uses
# train/ and val/; other exporters emit valid/, validation/ or test/.
TRAIN_SPLIT_NAMES = ("train", "training")
VAL_SPLIT_NAMES = ("val", "valid", "validation", "test")


def count_images_under(root):
    """Count image files anywhere under `root`, at any depth. 0 if absent."""
    if not root or not os.path.isdir(root):
        return 0
    n = 0
    for _dirpath, _dirnames, files in os.walk(root):
        n += sum(1 for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    return n


def find_split_dir(path, names):
    """
    Locate a split directory under `path` by name.

    Checks <path>/<name> first, then one level deeper (<path>/*/<name>) to
    cover layouts that wrap the splits, e.g. <path>/images/train/<class>.
    Returns the absolute-ish path or None.
    """
    for n in names:
        d = os.path.join(path, n)
        if os.path.isdir(d):
            return d
    try:
        mids = sorted(os.listdir(path))
    except OSError:
        return None
    for mid in mids:
        mid_dir = os.path.join(path, mid)
        if not os.path.isdir(mid_dir):
            continue
        for n in names:
            d = os.path.join(mid_dir, n)
            if os.path.isdir(d):
                return d
    return None


def count_images_in_dataset(path):
    """
    Count images in a training dataset. Returns (train_count, val_count).

      * YAML path   -> read the `train:` / `val:` keys, count files in each
      * directory   -> classification tree; counts <path>/train/<class>/* and
                       <path>/val/<class>/* SEPARATELY
    Returns (0, 0) on any error.

    NOTE ON THE DIRECTORY BRANCH
    ----------------------------
    This used to os.walk() the whole tree, sum every leaf directory, and
    return that single total as BOTH the train and the val count. Three
    things went wrong as a result:

      1. The readiness summary reported train+val as the training size, so
         the dataset looked roughly twice as large as it was.
      2. The `train != last_count` retrain trigger compared a number that
         conflated both splits, so a change confined to val forced a
         needless retrain (and a change that moved N images from val to
         train looked like no change at all).
      3. The `val_count < 2` guard could never fire independently: a class
         with plenty of train images but ZERO val images still passed, and
         training then ran without a usable validation split.

    Because the recorded train_count.txt values were written under the old
    (train+val) semantics, the first run after this change will see a
    "changed" count for every secondary model and trigger one retrain. That
    is expected and self-correcting — delete the train_count.txt files
    beforehand if you would rather skip the fine-tune pass.
    """
    # Primary-model case: YAML descriptor pointing at images/train
    if path.endswith(".yaml"):
        try:
            import yaml

            with open(path, "r") as f:
                data = yaml.safe_load(f)

            train_path = data["train"]
            val_path = data.get("val", None)
            base_dir = os.path.dirname(path)
            abs_train_path = os.path.join(base_dir, train_path) if train_path else None
            abs_val_path = os.path.join(base_dir, val_path) if val_path else None

            if abs_train_path.endswith(".txt"):
                # List-of-paths format.
                with open(abs_train_path, "r") as f:
                    train_count = len(f.readlines()) if abs_train_path else 0
                with open(abs_val_path, "r") as f:
                    val_count = len(f.readlines()) if abs_val_path else 0
            else:
                # Directory full of image files.
                train_count = (
                    len(
                        [
                            f
                            for f in os.listdir(abs_train_path)
                            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                        ]
                    )
                    if abs_train_path
                    else 0
                )
                val_count = (
                    len(
                        [
                            f
                            for f in os.listdir(abs_val_path)
                            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                        ]
                    )
                    if abs_val_path
                    else 0
                )
            return train_count, val_count
        except Exception as e:
            print(f"Error counting images: {e}")
            return 0, 0

    # Secondary-model case: classification tree, <path>/<split>/<class>/*.jpg
    elif os.path.isdir(path):
        train_dir = find_split_dir(path, TRAIN_SPLIT_NAMES)

        if train_dir is not None:
            val_dir = find_split_dir(path, VAL_SPLIT_NAMES)
            return count_images_under(train_dir), count_images_under(val_dir)

        # No split directory. The tree is flat class folders
        # (<path>/<subclass>/*.jpg) — Ultralytics cannot train from this
        # layout at all. Report the total as both counts (the pre-fix
        # behaviour) so the caller's >= 2 guard still passes and training
        # proceeds to the point where Ultralytics gives a real error,
        # rather than this function silently claiming the dataset is empty.
        total = count_images_under(path)
        return total, total

    else:
        print(f"Unsupported dataset format: {path}")
        return 0, 0


def count_subclasses_in_dataset(path, split="train"):
    """
    Per-subclass image counts for one split of a classification tree.
    Returns {subclass_name: n}.

    Falls back to the flat layout (<path>/<subclass>/*) when no split
    directory exists, so the breakdown still prints on an unsplit tree.
    """
    names = TRAIN_SPLIT_NAMES if split.startswith("train") else VAL_SPLIT_NAMES
    root = find_split_dir(path, names)
    if root is None:
        # Unsplit tree: the class folders are directly under `path`. Only
        # meaningful for the train side; report nothing for val.
        if split.startswith("train") and os.path.isdir(path):
            root = path
        else:
            return {}

    counts = {}
    for sub in sorted(os.listdir(root)):
        class_dir = os.path.join(root, sub)
        if not os.path.isdir(class_dir) or sub in TRAIN_SPLIT_NAMES + VAL_SPLIT_NAMES:
            continue
        counts[sub] = count_images_under(class_dir)
    return counts


# ============================================================================
# STAGE 1b — Staging a train/val split from a flat class tree
# ----------------------------------------------------------------------------
# The annotation tree is kept FLAT on purpose:
#
#     annot_motion_crop/fish/{bite,chase,escape,moving,scanning,vigilance}/*.jpg
#
# Ultralytics >= 8.4 tolerates that by auto-splitting into a sibling
# "<name>_split" directory, 80/20, sampled per image. Two problems with
# letting it:
#
#   1. A class with fewer images than the split ratio can land entirely on
#      one side. `escape` (1 image) went wholly to val, so train saw 5
#      classes and val 6, and the run aborted with
#          "found 3576 images in 5 classes (requires 6 classes, not 5)"
#      before any dataloader — and therefore the balanced sampler — existed.
#
#   2. The split is per IMAGE. Consecutive crops from one track are near
#      duplicates, so the same animal in the same second appears in both
#      train and val. Val accuracy measured that way is inflated and says
#      nothing about held-out video.
#
# So the split is staged here instead, into the model directory, as a build
# artefact. The annotation tree is never written to. Files are SYMLINKED, so
# staging costs no disk and no copy time.
#
# Grouping: crops are grouped by source video (see `secondary_video_regex`)
# and whole videos are assigned to one side or the other. If the pattern
# matches nothing useful the code says so and falls back to a per-image
# split for that class, which is no worse than the Ultralytics default.
#
# Classes too small to split, or listed in `secondary_ignore_subclasses`,
# are excluded from BOTH sides — which is what keeps the class sets
# identical and the run alive.
# ============================================================================

# Everything before the first _id / _frame / _track marker is treated as the
# source video name. Override with `secondary_video_regex` in the INI.
DEFAULT_VIDEO_REGEX = r"^(.*?)(?:_id\d+|_frame\d+|_track\d+)"


def _video_key(filename, pattern):
    """Source-video identifier for a crop filename."""
    if pattern is not None:
        m = pattern.match(filename)
        if m and m.group(1):
            return m.group(1)
    # No match: treat the file as its own group (=> per-image split).
    return os.path.splitext(filename)[0]


def stage_split_dataset(
    src_root,
    dst_root,
    val_fraction=0.2,
    seed=0,
    ignore=(),
    min_train=2,
    min_val=2,
    video_regex=DEFAULT_VIDEO_REGEX,
):
    """
    Build <dst_root>/{train,val}/<class>/ as symlinks into a flat <src_root>.

    Rebuilt from scratch on every call, so it always reflects the current
    annotations. Returns dst_root, or None if fewer than two classes survive.
    """
    if not os.path.isdir(src_root):
        print(f"  Cannot stage dataset: '{src_root}' does not exist.")
        return None

    try:
        pattern = re.compile(video_regex) if video_regex else None
    except re.error as e:
        print(f"  Warning: bad secondary_video_regex ({e}); grouping disabled.")
        pattern = None

    reserved = set(TRAIN_SPLIT_NAMES) | set(VAL_SPLIT_NAMES)
    ignore = {c.strip() for c in ignore if c and c.strip()}

    classes = [
        d
        for d in sorted(os.listdir(src_root))
        if os.path.isdir(os.path.join(src_root, d)) and d not in reserved
    ]

    rng = random.Random(seed)
    plan = {}  # class -> (train_files, val_files)
    dropped = []

    for cls in classes:
        cdir = os.path.join(src_root, cls)
        files = sorted(
            f for f in os.listdir(cdir) if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        if cls in ignore:
            dropped.append((cls, len(files), "listed in secondary_ignore_subclasses"))
            continue
        if len(files) < min_train + min_val:
            dropped.append(
                (cls, len(files), f"needs >= {min_train + min_val} images to split")
            )
            continue

        groups = defaultdict(list)
        for f in files:
            groups[_video_key(f, pattern)].append(f)

        keys = sorted(groups)
        rng.shuffle(keys)
        target = len(files) * val_fraction

        val_files, train_files = [], []
        for k in keys:
            take_val = (
                len(val_files) < target
                and len(train_files) + len(groups[k]) <= len(files) - min_val
            )
            (val_files if take_val else train_files).extend(groups[k])

        grouped_ok = len(train_files) >= min_train and len(val_files) >= min_val

        if not grouped_ok:
            # One dominant group swallowed the class. Fall back to a
            # per-image split so the class is not lost entirely.
            print(
                f"  Note: '{cls}' could not be split by video "
                f"({len(keys)} group(s) for {len(files)} images) — "
                "falling back to a per-image split for this class. "
                "Its val score will be optimistic."
            )
            shuffled = list(files)
            rng.shuffle(shuffled)
            n_val = max(min_val, int(round(len(shuffled) * val_fraction)))
            n_val = min(n_val, len(shuffled) - min_train)
            val_files = shuffled[:n_val]
            train_files = shuffled[n_val:]

        plan[cls] = (train_files, val_files)

    if len(plan) < 2:
        print(
            f"  Cannot stage dataset from '{src_root}': only {len(plan)} "
            "usable class(es) after filtering; a classifier needs at least 2."
        )
        return None

    # Rebuild from scratch — stale staged trees are worse than none.
    if os.path.isdir(dst_root):
        shutil.rmtree(dst_root)

    for cls, (train_files, val_files) in plan.items():
        for split, group in (("train", train_files), ("val", val_files)):
            out_dir = os.path.join(dst_root, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            for f in group:
                src = os.path.abspath(os.path.join(src_root, cls, f))
                dst = os.path.join(out_dir, f)
                try:
                    os.symlink(src, dst)
                except OSError:
                    # Filesystem without symlink support (or Windows without
                    # developer mode) — fall back to copying.
                    shutil.copy2(src, dst)

    print(f"  Staged split -> {dst_root}")
    for cls in sorted(plan):
        tr, va = plan[cls]
        print(f"      {cls:<12} train={len(tr):>5}  val={len(va):>5}")
    for cls, n, why in dropped:
        print(f"      {cls:<12} EXCLUDED ({n} images: {why})")

    return dst_root


def maybe_retrain(
    model_type,
    yaml_path,
    project_path,
    model_path,
    classifier,
    epochs,
    imgsz,
    task="detect",
    stream=None,
):
    """
    Decide whether to (re)train a model based on existence and image counts.

      * model_path exists + count changed -> backup old model_dir to
        <project>_backup<N>, fine-tune from the backup's best.pt, and move
        the new run into place.
      * model_path missing                -> first-time train, but only if
        the dataset has at least 2 images.
      * count unchanged                   -> no-op.

    task   : "detect" for the primary detectors, "classify" for the
             secondary classifiers. Selects the hyperparameter block and,
             for classification, the class-balanced trainer.
    stream : "static" or "motion" — only used to pick the classification
             augmentation profile (motion crops must not get hue/saturation
             jitter).

    Returns True if training ran, False otherwise.
    """
    extra = train_args_for(task, stream)
    extra.update(trainer_kwarg_for(task))

    # ---- branch A: model already exists -------------------------------
    if os.path.exists(model_path):
        # Load the last-trained image count (or -1 if unknown).
        if os.path.exists(os.path.join(project_path, "train_count.txt")):
            try:
                with open(os.path.join(project_path, "train_count.txt"), "r") as f:
                    last_count = int(f.read().strip())
            except Exception:
                last_count = -1
        else:
            last_count = -1

        train, val = count_images_in_dataset(yaml_path)

        # If the count changed, retrain.
        if train != last_count:
            print(
                f"New annotations detected for '{model_type}' model.\n"
                f"Training image count changed from {last_count} to {train}.\n\n"
                "Retraining the model."
            )

            # Backup the whole model dir so we never lose old weights.
            backup_dir = project_path + "_backup"
            i = 1
            while os.path.exists(f"{backup_dir}{i}"):
                i += 1
            final_backup = f"{backup_dir}{i}"
            try:
                shutil.copytree(project_path, final_backup)
                print(f"Existing model copied to {final_backup}")
            except Exception as e:
                print(f"Warning: failed to backup {project_path}: {e}")

            # Fine-tune from the backed-up weights.
            start_weights = os.path.join(final_backup, "train", "weights", "best.pt")
            print(f"Training new {model_type} model using existing weights...")
            model = YOLO(start_weights)
            model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                project=project_path,
                name="train",
                exist_ok=True,
                # --- Core Training ---
                device=params["train_device"],
                patience=60,
                # --- Task-specific optimizer / augmentation / loss block ---
                **extra,
            )
            try:
                move_to_expected(project_path, run_name="train", runs_root="runs")
            except Exception as e:
                print(f"Error: {e}")

            print(f"Done training {model_type} model")

            # Record count + snapshot of settings used.
            with open(os.path.join(project_path, "train_count.txt"), "w") as f:
                f.write(str(train))
            os.makedirs(project_path, exist_ok=True)
            dst = os.path.join(project_path, "saved_settings.ini")
            try:
                shutil.copy2(params["config_path"], dst)
                print(f"Saved settings snapshot to {dst}")
            except Exception as e:
                print(f"Warning: could not copy settings to model dir: {e}")
            return True

        # Counts match -> nothing to do.
        return False

    # ---- branch B: first-time training --------------------------------
    else:
        print(f"{model_type} model not found, building it...")
        train, val = count_images_in_dataset(yaml_path)
        if train < 2 or val < 2:
            # Not enough data. Leave best.pt absent; caller handles skip.
            print(
                f"Error: Not enough images to train {model_type} model "
                f"(found {train} training images and {val} validation images, need at least 2 of each)."
            )
            return False

        model = YOLO(classifier)
        model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            project=project_path,
            name="train",
            exist_ok=True,
            # --- Core Training ---
            device=params["train_device"],
            patience=60,
            # --- Task-specific optimizer / augmentation / loss block ---
            **extra,
        )
        try:
            move_to_expected(project_path, run_name="train", runs_root="runs")
        except Exception as e:
            print(f"Error: {e}")

        print(f"Done training {model_type} model")

        # Record count + snapshot of settings used.
        train, val = count_images_in_dataset(yaml_path)
        os.makedirs(project_path, exist_ok=True)
        with open(os.path.join(project_path, "train_count.txt"), "w") as f:
            f.write(str(train))
        dst = os.path.join(project_path, "saved_settings.ini")
        try:
            shutil.copy2(params["config_path"], dst)
            print(f"Saved settings snapshot to {dst}")
        except Exception as e:
            print(f"Warning: could not copy settings to model dir: {e}")

        return True


def train_models():
    """
    Walk the configured model hierarchy and ensure each has weights on disk.
    This function now validates that secondary crop directories exist before
    attempting to train classifiers; if a directory is missing, the entire
    secondary model for that primary class is skipped.
    """
    global secondary_static_models, secondary_motion_models
    global static_class_map, motion_class_map

    secondary_static_models = None
    secondary_motion_models = None

    # ---- hierarchical (secondary) models ------------------------------
    if params["hierarchical_mode"]:
        # train if no external model is specified
        if params["secondary_static_external_model"] == "":
            # Secondary STATIC classifiers — one YOLO-cls model per primary class.
            secondary_static_models = {}
            static_class_map = [
                [None] * len(params["secondary_classes"])
                for _ in range(len(params["primary_classes"]))
            ]
            if len(params["secondary_static_classes"]) >= 2:
                # Check that the base crop directory exists before attempting any static secondary training
                static_data_root = params.get("secondary_static_data_path", "")
                if not static_data_root or not os.path.isdir(static_data_root):
                    print(
                        f"Warning: secondary_static_data_path '{static_data_root}' does not exist; "
                        "skipping all static secondary models."
                    )
                else:
                    # dict.fromkeys deduplicates while preserving order. A
                    # repeated entry in primary_classes would otherwise train
                    # the same model_dir twice, because .index() below always
                    # returns the FIRST match — so every duplicate iteration
                    # recomputes an identical idx, data_dir and weights_path,
                    # then overwrites the weights it just produced.
                    for primary_class in dict.fromkeys(params["primary_classes"]):
                        idx = params["primary_classes"].index(primary_class)
                        hotkey = params["primary_hotkeys"][idx]

                        if hotkey in params["secondary_hotkeys"]:
                            continue
                        if primary_class in params["ignore_secondary"]:
                            continue
                        data_dir = os.path.join(static_data_root, primary_class)
                        if not os.path.isdir(data_dir):
                            print(
                                f"  Skipping static secondary for '{primary_class}': {data_dir} does not exist"
                            )
                            continue

                        model_dir = f"models/model_static_static_{primary_class}"
                        weights_path = os.path.join(
                            model_dir, "train", "weights", "best.pt"
                        )

                        # train/val are now counted separately (see
                        # count_images_in_dataset) so a class with no val
                        # split is caught here instead of silently training.
                        train_count, val_count = count_images_in_dataset(data_dir)
                        if train_count < 2 or val_count < 2:
                            print(
                                f"Error: Not enough images to train secondary static model "
                                f"for primary class '{primary_class}' (found {train_count} "
                                f"training images and {val_count} validation images, "
                                f"need at least 2 of each). Skipping this secondary model."
                            )
                            continue

                        sub_counts = count_subclasses_in_dataset(data_dir, "train")
                        if sub_counts:
                            print(
                                f"  [{primary_class}] static subclasses: "
                                + ", ".join(f"{k}={v}" for k, v in sub_counts.items())
                            )

                        # Stage a deterministic, video-grouped split rather
                        # than letting Ultralytics improvise one. Training
                        # reads from here; the annotation tree stays flat.
                        train_root = stage_split_dataset(
                            data_dir,
                            os.path.join(model_dir, "dataset"),
                            val_fraction=float(
                                params.get("secondary_val_fraction", 0.2)
                            ),
                            seed=int(params.get("secondary_split_seed", 0)),
                            ignore=params.get("secondary_ignore_subclasses", []),
                            video_regex=params.get(
                                "secondary_video_regex", DEFAULT_VIDEO_REGEX
                            ),
                        )
                        if train_root is None:
                            print(
                                f"  Skipping static secondary for "
                                f"'{primary_class}': could not stage a split."
                            )
                            continue

                        maybe_retrain(
                            model_dir,
                            train_root,
                            model_dir,
                            weights_path,
                            params["secondary_classifier"],
                            params["secondary_epochs"],
                            params["secondary_imgsz"],
                            task="classify",
                            stream="static",
                        )

                        if os.path.isfile(weights_path):
                            try:
                                if params["use_ncnn"] == "true":
                                    secondary_static_models[primary_class] = (
                                        load_model_with_ncnn_preference(
                                            weights_path, "classify"
                                        )
                                    )
                                else:
                                    secondary_static_models[primary_class] = YOLO(
                                        weights_path
                                    )
                            except Exception as e:
                                print(
                                    f"Warning: failed to load secondary static model for "
                                    f"'{primary_class}': {e} — skipping at inference."
                                )
                        else:
                            print(
                                f"Secondary static model for '{primary_class}' has no "
                                f"weights at {weights_path} — skipping at inference."
                            )
        else:
            print(
                "Using external secondary static model:",
                params["secondary_static_external_model"],
            )
            from fishial_inference import FishInferenceEngine

            secondary_static_models = FishInferenceEngine.from_bundle(
                params["secondary_static_external_model"]
            )

        # Secondary MOTION classifiers — mirror of static block with same checks
        secondary_motion_models = {}
        motion_class_map = [
            [None] * len(params["secondary_classes"])
            for _ in range(len(params["primary_classes"]))
        ]
        if len(params["secondary_motion_classes"]) >= 2:
            motion_data_root = params.get("secondary_motion_data_path", "")
            if not motion_data_root or not os.path.isdir(motion_data_root):
                print(
                    f"Warning: secondary_motion_data_path '{motion_data_root}' does not exist; "
                    "skipping all motion secondary models."
                )
            else:
                # See the static block: deduplicated for the same reason.
                for primary_class in dict.fromkeys(params["primary_classes"]):
                    idx = params["primary_classes"].index(primary_class)
                    hotkey = params["primary_hotkeys"][idx]

                    if hotkey in params["secondary_hotkeys"]:
                        continue
                    if primary_class in params["ignore_secondary"]:
                        continue
                    data_dir = os.path.join(motion_data_root, primary_class)
                    if not os.path.isdir(data_dir):
                        print(
                            f"  Skipping motion secondary for '{primary_class}': {data_dir} does not exist"
                        )
                        continue

                    model_dir = f"models/model_secondary_motion_{primary_class}"
                    weights_path = os.path.join(
                        model_dir, "train", "weights", "best.pt"
                    )

                    train_count, val_count = count_images_in_dataset(data_dir)
                    if train_count < 2 or val_count < 2:
                        print(
                            f"Error: Not enough images to train secondary motion model "
                            f"for primary class '{primary_class}' (found {train_count} training images and {val_count} validation images, "
                            f"need at least 2 of each). Skipping this secondary model."
                        )
                        continue

                    sub_counts = count_subclasses_in_dataset(data_dir, "train")
                    if sub_counts:
                        print(
                            f"  [{primary_class}] motion subclasses: "
                            + ", ".join(f"{k}={v}" for k, v in sub_counts.items())
                        )

                    # See the static block above.
                    train_root = stage_split_dataset(
                        data_dir,
                        os.path.join(model_dir, "dataset"),
                        val_fraction=float(params.get("secondary_val_fraction", 0.2)),
                        seed=int(params.get("secondary_split_seed", 0)),
                        ignore=params.get("secondary_ignore_subclasses", []),
                        video_regex=params.get(
                            "secondary_video_regex", DEFAULT_VIDEO_REGEX
                        ),
                    )
                    if train_root is None:
                        print(
                            f"  Skipping motion secondary for '{primary_class}': "
                            "could not stage a split."
                        )
                        continue

                    maybe_retrain(
                        model_dir,
                        train_root,
                        model_dir,
                        weights_path,
                        params["secondary_classifier"],
                        params["secondary_epochs"],
                        params["secondary_imgsz"],
                        task="classify",
                        stream="motion",
                    )

                    if os.path.isfile(weights_path):
                        try:
                            if params["use_ncnn"] == "true":
                                secondary_motion_models[primary_class] = (
                                    load_model_with_ncnn_preference(
                                        weights_path, "classify"
                                    )
                                )
                            else:
                                secondary_motion_models[primary_class] = YOLO(
                                    weights_path
                                )
                        except Exception as e:
                            print(
                                f"Warning: failed to load secondary motion model for "
                                f"'{primary_class}': {e} — skipping at inference."
                            )
                    else:
                        print(
                            f"Secondary motion model for '{primary_class}' has no "
                            f"weights at {weights_path} — skipping at inference."
                        )

    # ---- primary detectors (unchanged) ---------------------------------
    if params["use_local_static_model"]:
        print("Training primary static model")
        if params["primary_static_classes"][0] != "0":
            maybe_retrain(
                "models/model_primary_static",
                params["primary_static_yaml_path"],
                params["primary_static_project_path"],
                params["primary_static_model_path"],
                params["primary_classifier"],
                params["primary_epochs"],
                params["primary_imgsz"],
                task="detect",
            )

    if params["primary_motion_classes"][0] != "0":
        maybe_retrain(
            "models/model_primary_motion",
            params["primary_motion_yaml_path"],
            params["primary_motion_project_path"],
            params["primary_motion_model_path"],
            params["primary_classifier"],
            params["primary_epochs"],
            params["primary_imgsz"],
            task="detect",
        )


# ============================================================================
# STAGE 5a — Overlap helper
# ----------------------------------------------------------------------------
# NOT traditional IoU. Returns the *larger* proportional overlap relative to
# each box's own area, so if one box is fully inside another the score is 1.0.
# Used when merging detections from the static and motion streams.
# (True IoU, for the tracker's association cost, is `true_iou` below.)
# ============================================================================


def iou(box1, box2):
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    if area1 <= 0 or area2 <= 0:
        return 0.0
    prop1 = inter / area1
    prop2 = inter / area2
    return max(0.0, max(prop1, prop2))


def true_iou(box1, box2):
    """Standard intersection-over-union. Used in the tracker's cost matrix."""
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter <= 0:
        return 0.0
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ============================================================================
# STAGE 4a — Crop extraction
# ----------------------------------------------------------------------------
# Pulling the secondary-classifier crop out of a frame used to be a bare
# slice: frame[y1:y2, x1:x2]. Two problems with that.
#
#   1. NO BOUNDS CHECK. Merged detection boxes are taken wholesale from
#      whichever stream won the merge, and nothing guarantees they lie inside
#      the frame. A negative x1 makes numpy index from the END of the axis,
#      so instead of erroring you silently classify a thin strip from the
#      opposite side of the image. x1 > x2 or an off-frame box yields an
#      empty array, which _run() then rejects — so the detection quietly
#      loses its secondary label with no diagnostic.
#
#   2. NO CONTEXT. A tight box crops away the very evidence some behaviour
#      classes are defined by: what the animal is oriented toward (substrate,
#      a conspecific, shelter). On the MOTION image it is worse — a moving
#      animal's coloured difference tail extends behind it, outside the
#      detection box, so a tight crop discards the clearest motion cue
#      available.
#
# `secondary_crop_margin` in the INI expands the box by that fraction of its
# own width/height on each side before clamping. It defaults to 0.0, i.e.
# exactly the previous framing.
#
# IMPORTANT: if you change the margin you MUST regenerate the training crops
# with the same value. Train-time and inference-time crop geometry have to
# match, or the classifier sees a different framing than it learned on. Treat
# 0.0 / 0.15 / 0.3 as an ablation, not a free knob.
# ============================================================================


def expand_and_clamp_box(coords, width, height, margin=0.0):
    """
    Expand a box by `margin` (fraction of its own w/h, each side) and clamp
    it to the image. Returns (x1, y1, x2, y2) or None if the result is empty.
    """
    x1, y1, x2, y2 = coords

    # Normalise ordering — merged boxes are not guaranteed to be sorted.
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)

    if margin:
        bw = x2 - x1
        bh = y2 - y1
        x1 -= margin * bw
        x2 += margin * bw
        y1 -= margin * bh
        y2 += margin * bh

    cx1 = max(0, int(round(x1)))
    cy1 = max(0, int(round(y1)))
    cx2 = min(int(width), int(round(x2)))
    cy2 = min(int(height), int(round(y2)))

    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2


def crop_region(image, box):
    """Slice `box` out of `image`, or return None if either is unusable."""
    if image is None or box is None:
        return None
    cx1, cy1, cx2, cy2 = box
    crop = image[cy1:cy2, cx1:cx2]
    return crop if crop.size else None


# ============================================================================
# STAGE 5b — Kalman-filter multi-object tracker
# ----------------------------------------------------------------------------
# Each track keeps a 4D state (x, y, vx, vy). Detections are matched to tracks
# by Hungarian assignment on a combined cost of Mahalanobis distance (which
# accounts for how uncertain each track currently is) and box IoU.
#
# Changes from the original implementation, and why each matters:
#
#   1. errorCovPost is initialised. cv2.KalmanFilter zero-fills the covariance
#      matrices, so with P0 = 0 the Kalman gain starts near zero and the state
#      creeps toward the measurements over dozens of frames. Every track was
#      effectively blind for its first second of life.
#
#   2. Process noise is SET from a stored baseline on each miss, not multiplied
#      into itself. The old `Q *= scale` compounded (~12x after five misses)
#      and never reset on re-acquisition, so any track that briefly vanished
#      permanently degenerated into a random walk.
#
#   3. The greedy nearest-track fallback is gone. It ran over ALL tracks
#      including ones Hungarian had already matched, so a single track could
#      receive two corrections in one frame and two detections could be handed
#      the same track id — silently corrupting per-individual analysis.
#
#   4. Gating happens BEFORE assignment (impossible pairs are masked to
#      infinity), not after. Post-hoc filtering let Hungarian waste a good
#      track on a far detection and leave the correct pairing unmatched.
#
#   5. Mahalanobis distance replaces raw Euclidean. The gate widens
#      automatically for tracks that have been unobserved and stays tight for
#      well-observed ones — the principled version of what the old process-
#      noise hack approximated.
#
#   6. Tracks must be confirmed (min_hits) before they are reported. One
#      flickering false positive no longer becomes a permanent individual in
#      the CSV.
#
#   7. Boxes and class labels participate in association, not just centroids.
# ============================================================================


# Chi-square 95th percentile, 2 degrees of freedom. A Mahalanobis distance
# above this means the detection is inconsistent with the track's predicted
# position given its current uncertainty.
CHI2_95_2DOF = 5.991


class KalmanTracker:
    """
    Interface-compatible with BoxMOTTracker:
        update(detections, frame=None, frame_idx=None) -> {det_index: track_id}
        state(tid) -> (x, y, vx, vy)
        box(tid)   -> (x1, y1, x2, y2)
        `tid in tracker.tracks`
    """

    def __init__(
        self,
        dist_thresh,
        max_missed,
        min_hits=3,
        iou_weight=0.4,
        class_penalty=2.0,
        process_noise_pos=None,
        process_noise_vel=None,
        measurement_noise=None,
    ):
        self.next_id = 1
        self.tracks = {}  # tid -> track dict (CONFIRMED tracks only, see below)
        self._all = {}  # tid -> track dict (including tentative)
        self.dist_thresh = float(dist_thresh)
        self.max_missed = int(max_missed)
        self.min_hits = int(min_hits)
        self.iou_weight = float(iou_weight)
        self.class_penalty = float(class_penalty)
        self._frame_idx = 0

        # Noise parameters. Fall back to the config values if not passed.
        self.q_pos = float(
            process_noise_pos
            if process_noise_pos is not None
            else params["process_noise_pos"]
        )
        self.q_vel = float(
            process_noise_vel
            if process_noise_vel is not None
            else params["process_noise_vel"]
        )
        self.r_meas = float(
            measurement_noise
            if measurement_noise is not None
            else params["measurement_noise"]
        )

    # -- filter construction --------------------------------------------

    def _create_kf(self, initial_pt):
        """State: [x, y, vx, vy]; measurement: [x, y]. dt = 1 PROCESSED frame."""
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kf.processNoiseCov = np.diag(
            [self.q_pos, self.q_pos, self.q_vel, self.q_vel]
        ).astype(np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.r_meas

        # FIX 1: seed the posterior covariance. OpenCV leaves this as zeros,
        # which makes the filter refuse to believe its own measurements for
        # the first ~30 frames. Position uncertainty starts at roughly the
        # association gate; velocity is completely unknown, so start it wide.
        kf.errorCovPost = np.diag(
            [
                self.dist_thresh**2,
                self.dist_thresh**2,
                (self.dist_thresh / 2.0) ** 2,
                (self.dist_thresh / 2.0) ** 2,
            ]
        ).astype(np.float32)

        kf.statePre = np.array(
            [[initial_pt[0]], [initial_pt[1]], [0.0], [0.0]], dtype=np.float32
        )
        kf.statePost = kf.statePre.copy()
        return kf

    # -- association helpers ---------------------------------------------

    @staticmethod
    def _mahalanobis(kf, meas_xy):
        """
        Squared Mahalanobis distance between a measurement and a track's
        predicted position, in units of the track's own uncertainty.

            S = H P- H^T + R          (innovation covariance)
            d^2 = (z - H x-)^T S^-1 (z - H x-)

        Falls back to scaled Euclidean if S is singular.
        """
        H = kf.measurementMatrix
        P = kf.errorCovPre
        R = kf.measurementNoiseCov
        S = H @ P @ H.T + R
        innov = np.array(
            [[meas_xy[0] - kf.statePre[0, 0]], [meas_xy[1] - kf.statePre[1, 0]]],
            dtype=np.float64,
        )
        try:
            # .item() rather than float(): numpy >= 2 refuses to coerce a
            # 1x1 array to a Python scalar implicitly.
            return (innov.T @ np.linalg.inv(S.astype(np.float64)) @ innov).item()
        except np.linalg.LinAlgError:
            return (innov.T @ innov).item() / max(1e-6, float(np.trace(S)))

    def predict_all(self):
        """Run KF predict() for every track. Returns list of (tid, (x, y))."""
        preds = []
        for tid, tr in self._all.items():
            pred = tr["kf"].predict()
            preds.append((tid, (float(pred[0, 0]), float(pred[1, 0]))))
        return preds

    # -- main step --------------------------------------------------------

    def update(self, detections, frame=None, frame_idx=None):
        """
        detections : list of the pipeline's merged detection dicts, each with
                     'centroid', 'coords', and optionally 'primary_class'.
        frame      : ignored (signature parity with BoxMOTTracker).

        Returns    : {index into `detections` -> track_id} for CONFIRMED
                     tracks only. Tentative tracks are maintained internally
                     but not reported, so they never reach the CSV.
        """
        self._frame_idx = self._frame_idx + 1 if frame_idx is None else int(frame_idx)

        # 1) Predict every existing track forward one step.
        preds = self.predict_all()
        track_ids = [t[0] for t in preds]

        n_t, n_d = len(track_ids), len(detections)

        assigned = {}
        matched_tracks = set()
        matched_dets = set()

        # 2) Cost matrix. Mahalanobis distance for motion, (1 - IoU) for
        #    geometry, plus a flat penalty for class disagreement. Pairs that
        #    fail the chi-square gate are masked BEFORE the solve (FIX 4).
        if n_t and n_d:
            BIG = 1e6
            cost = np.full((n_t, n_d), BIG, dtype=np.float64)

            for i, tid in enumerate(track_ids):
                tr = self._all[tid]
                kf = tr["kf"]
                for j, det in enumerate(detections):
                    cx, cy = det["centroid"]

                    # FIX 5: gate in units of the track's own uncertainty.
                    d2 = self._mahalanobis(kf, (cx, cy))
                    if d2 > CHI2_95_2DOF:
                        continue

                    # Hard cap in pixels as well — Mahalanobis alone can let a
                    # very uncertain track reach implausibly far.
                    if np.hypot(cx - kf.statePre[0, 0], cy - kf.statePre[1, 0]) > (
                        self.dist_thresh * 2.0
                    ):
                        continue

                    c = d2 / CHI2_95_2DOF  # normalised to [0, 1] inside the gate

                    # FIX 7a: boxes participate. Overlap is strong evidence.
                    if tr.get("box") is not None and "coords" in det:
                        c += self.iou_weight * (
                            1.0 - true_iou(tr["box"], tuple(det["coords"]))
                        )

                    # FIX 7b: penalise class disagreement rather than ignoring it.
                    dcls = det.get("primary_class", None)
                    if dcls and tr.get("cls_name") and dcls != tr["cls_name"]:
                        c += self.class_penalty

                    cost[i, j] = c

            row_idx, col_idx = linear_sum_assignment(cost)

            for r, c_ in zip(row_idx, col_idx):
                if cost[r, c_] >= BIG:
                    continue  # gated-out pair the solver was forced to take
                tid = track_ids[r]
                det = detections[c_]
                cx, cy = det["centroid"]

                kf = self._all[tid]["kf"]
                kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))

                tr = self._all[tid]
                tr["missed"] = 0
                tr["hits"] += 1
                tr["last_frame"] = self._frame_idx
                tr["box"] = tuple(det["coords"]) if "coords" in det else tr.get("box")
                tr["cls_name"] = det.get("primary_class", tr.get("cls_name"))

                # FIX 2 (part 2): restore the baseline process noise on
                # re-acquisition. The old code never reset it.
                kf.processNoiseCov = tr["Q0"].copy()

                matched_tracks.add(tid)
                matched_dets.add(int(c_))
                if tr["hits"] >= self.min_hits:
                    assigned[int(c_)] = tid

        # 3) Unmatched detections become NEW tracks.
        #    FIX 3: no greedy nearest-track fallback. Hungarian already found
        #    the optimal assignment; re-adding rejected pairs greedily both
        #    undoes that optimality and allows double-correction.
        for j, det in enumerate(detections):
            if j in matched_dets:
                continue
            cx, cy = det["centroid"]
            tid = self.next_id
            self.next_id += 1
            kf = self._create_kf((cx, cy))
            self._all[tid] = {
                "kf": kf,
                "Q0": kf.processNoiseCov.copy(),  # FIX 2 (part 1): baseline Q
                "missed": 0,
                "hits": 1,
                "last_frame": self._frame_idx,
                "box": tuple(det["coords"]) if "coords" in det else None,
                "cls_name": det.get("primary_class", None),
            }
            # FIX 6: a brand-new track is TENTATIVE. It is not reported until
            # it has been seen min_hits times, so a one-frame false positive
            # never reaches the CSV or triggers a snapshot.
            if self.min_hits <= 1:
                assigned[j] = tid

        # 4) Age unmatched tracks; widen uncertainty; delete if too old.
        for tid in list(self._all.keys()):
            if tid in matched_tracks:
                continue
            tr = self._all[tid]
            tr["missed"] += 1

            # FIX 2 (part 3): SET from the stored baseline, never multiply the
            # live matrix into itself.
            scale = min(4.0, 1.0 + 0.3 * tr["missed"])
            tr["kf"].processNoiseCov = (tr["Q0"] * scale).astype(np.float32)

            # A tentative track that misses even once is almost certainly a
            # false positive — drop it immediately rather than after max_missed.
            expired = tr["missed"] > self.max_missed
            tentative_and_lost = tr["hits"] < self.min_hits and tr["missed"] > 1
            if expired or tentative_and_lost:
                del self._all[tid]

        # 5) Publish the confirmed subset. `self.tracks` is what the rest of
        #    the pipeline sees, so tentative tracks stay invisible.
        self.tracks = {
            tid: tr for tid, tr in self._all.items() if tr["hits"] >= self.min_hits
        }

        return assigned

    # -- accessors used by the drawing / CSV code ------------------------

    def state(self, tid):
        """(x, y, vx, vy) in pixels and pixels-per-PROCESSED-frame."""
        tr = self._all.get(tid)
        if tr is None:
            return None
        s = tr["kf"].statePost
        return (float(s[0, 0]), float(s[1, 0]), float(s[2, 0]), float(s[3, 0]))

    def box(self, tid):
        tr = self._all.get(tid)
        return tr.get("box") if tr else None


# ============================================================================
# STAGE 2–5 — Per-video pipeline
# ----------------------------------------------------------------------------
# process_video() runs the end-to-end pipeline on one file:
#   * opens video + output writers (MP4 + CSV)
#   * loads whichever primary models have trained weights on disk
#   * for each frame: build motion image, run detection(s), merge, run
#     secondary classification, track, draw, and write a CSV row per track.
# ============================================================================


def build_tracker(fps):
    """
    Construct whichever tracker backend the config asks for.

    `frame_rate` and `max_missed` are expressed in PROCESSED frames. With
    frame_skip = N the tracker only ever sees every (N+1)th frame, so a buffer
    meant to represent one second must be fps / (N + 1), not fps.
    """
    eff_fps = (fps or 30.0) / (params["frame_skip"] + 1)
    tracker_type = params.get("tracker_type", "builtin")

    if tracker_type != "builtin" and BoxMOTTracker is not None:
        print(f"Using {tracker_type} tracker.")
        # Every knob comes from the [tracker] INI section via
        # params["tracker"]. Note that max_age is NOT delete_after_missed
        # any more: that key belongs to the builtin tracker and was far too
        # short for BoxMOT once the frame_rate/30 buffer scaling was
        # applied on top of it.
        return BoxMOTTracker.from_params(params, frame_rate=eff_fps)

    if tracker_type != "builtin" and BoxMOTTracker is None:
        print(
            f"Warning: tracker_type='{tracker_type}' requested but boxmot is not "
            "installed (pip install boxmot). Falling back to the built-in tracker."
        )

    return KalmanTracker(
        dist_thresh=params["match_distance_thresh"],
        max_missed=params["delete_after_missed"],
        # min_hits lives in the [tracker] block alongside the BoxMOT
        # settings, so both backends honour the same value.
        min_hits=params["tracker"]["min_hits"],
        iou_weight=params["tracker_iou_weight"],
        class_penalty=params["tracker_class_penalty"],
    )


def process_video(file):
    # ---- STAGE 2a: open inputs and outputs ----------------------------
    # Preserve any subfolder structure from input/ under output/.
    #   output/annotated_videos/<rel_dir>/<base>_detected.mp4
    #   output/annotated_frames/<rel_dir>/<base>_id<TID>.jpg
    # The tracking CSV still sits alongside the video.
    rel = os.path.relpath(file, params["input_folder"])
    rel_dir = os.path.dirname(rel)  # "site_A/day_2" or ""
    base = os.path.splitext(os.path.basename(rel))[0]

    video_out_dir = os.path.join(params["output_folder"], "annotated_videos", rel_dir)
    frames_out_dir = os.path.join(params["output_folder"], "annotated_frames", rel_dir)
    os.makedirs(video_out_dir, exist_ok=True)
    os.makedirs(frames_out_dir, exist_ok=True)

    cap = cv2.VideoCapture(file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened():
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * params["scale_factor"])
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * params["scale_factor"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(
        os.path.join(video_out_dir, base + "_detected.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    # Crop margin for the secondary classifiers. Must match whatever was used
    # to generate the training crops — see expand_and_clamp_box().
    crop_margin = float(params.get("secondary_crop_margin", 0.0))

    # ---- STAGE 2b: defensively load primary models --------------------
    # For each stream, three things must be true to load:
    #   (1) the stream is configured in the INI (classes[0] != "0")
    #   (2) the best.pt file exists on disk
    #   (3) YOLO/NCNN can actually open it
    model_static = None
    model_motion = None

    # Primary STATIC
    if params["primary_static_classes"][0] != "0" and params["use_local_static_model"]:
        weights = params["primary_static_model_path"]  # already ends in best.pt
    else:
        weights = params["primary_static_external_model"]

    if os.path.isfile(weights):
        try:
            if params["use_ncnn"] == "true":
                model_static = load_model_with_ncnn_preference(weights, "detect")
            else:
                model_static = YOLO(weights)
        except Exception as e:
            print(f"Warning: failed to load primary static model ({weights}): {e}")
            print("  -> skipping primary static stream for this video")
            model_static = None
    else:
        print(f"Primary static model not trained (no {weights})")
        print("  -> skipping primary static stream for this video")

    # Primary MOTION
    if params["primary_motion_classes"][0] != "0":
        weights = params["primary_motion_model_path"]  # already ends in best.pt
        if os.path.isfile(weights):
            try:
                if params["use_ncnn"] == "true":
                    model_motion = load_model_with_ncnn_preference(weights, "detect")
                else:
                    model_motion = YOLO(weights)
            except Exception as e:
                print(f"Warning: failed to load primary motion model ({weights}): {e}")
                print("  -> skipping primary motion stream for this video")
                model_motion = None
        else:
            print(f"Primary motion model not trained (no {weights})")
            print("  -> skipping primary motion stream for this video")

    # If neither primary is available, there's nothing to detect.
    if model_static is None and model_motion is None:
        print(f"Skipping {file}: no trained primary models available")
        cap.release()
        writer.release()
        try:
            os.remove(os.path.join(video_out_dir, base + "_detected.mp4"))
        except OSError:
            pass
        return

    # ---- STAGE 2c: initialise tracker + CSV ---------------------------
    tracker = build_tracker(fps)

    # Do we need a motion image at all this video? Either the motion detector
    # is loaded, or hierarchical mode wants motion crops for the secondaries.
    need_motion_image = model_motion is not None or (
        params["hierarchical_mode"] and len(params["secondary_motion_classes"]) >= 2
    )

    # Track IDs we've already exported a snapshot for. Each new tid triggers
    # one JPEG save of the current annotated frame. Note that with min_hits > 1
    # this fires at CONFIRMATION, a few frames after the fish first appears.
    seen_track_ids = set()

    prev_frames, frame_idx = None, 0
    proc_idx = 0  # index in PROCESSED frames — the tracker's clock
    csv_file = open(
        os.path.join(video_out_dir, base + "_tracking.csv"),
        "w",
        newline="",
    )
    csv_writer = csv.writer(csv_file)
    # One row per frame per tracked object. Empty string / 0.0 indicates the
    # corresponding model was not available or did not fire.
    #
    # `frame` is the raw video frame number; `proc_frame` counts only frames
    # the pipeline actually processed. Velocities from the tracker are per
    # PROCESSED frame, so use proc_frame (not frame) as the time axis for any
    # speed or duration calculation, or you will be off by (frame_skip + 1).
    csv_writer.writerow(
        [
            "frame",
            "proc_frame",
            "id",
            "x1",
            "y1",
            "x2",
            "y2",
            "vx",
            "vy",
            "primary_static_class",
            "primary_static_conf",
            "primary_motion_class",
            "primary_motion_conf",
            "secondary_static_class",
            "secondary_static_conf",
            "secondary_motion_class",
            "secondary_motion_conf",
        ]
    )

    print(f"Processing video: {file}")
    print("Initialising")
    current_frame = 0
    print_tick = 0
    start_time = time.time()
    current_fps = 0.0  # so final print is safe even if the video is empty

    frame_count = 0

    # ======================================================================
    # STAGE 3 — Per-frame loop
    # ======================================================================
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Only process every (frame_skip+1)th frame. Counter reset at bottom.
        if frame_count == 0:
            # ---- 3a: downscale + grayscale ----------------------------
            if params["scale_factor"] != 1.0:
                raw_frame = cv2.resize(
                    raw_frame,
                    None,
                    fx=params["scale_factor"],
                    fy=params["scale_factor"],
                )
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            frame = raw_frame.copy()
            frame_h, frame_w = frame.shape[:2]

            # Prime the 3-frame history on the first iteration.
            if prev_frames is None:
                prev_frames = [gray.copy() for _ in range(3)]
                continue

            proc_idx += 1

            # ---- 3b: build the false-colour motion image --------------
            # Three temporally-offset frame differences are mapped to B/G/R
            # channels so a moving object leaves a coloured "tail" that the
            # motion detector can learn from.
            #
            # BUG FIX: computed exactly ONCE per frame and reused everywhere
            # downstream. create_motion_image() mutates prev_frames, so the
            # previous per-detection call in Stage 4 was advancing the frame
            # history once per detection and corrupting the motion tails.
            motion_image = (
                create_motion_image(prev_frames, gray, params)
                if need_motion_image
                else None
            )

            # ---- 3c: primary detections -------------------------------
            # Predicate is "did a model actually load?" rather than "is a
            # stream configured?". A configured stream with no weights falls
            # through to the merge step with zero detections.
            all_detections = []

            # Primary STATIC detection
            if model_static is not None:
                # Run at detector_conf_thresh, NOT primary_conf_thresh. For a
                # two-stage tracker these differ: the extra low-confidence
                # boxes are what the second association pass recovers tracks
                # from. Anything that fails to join a track is never
                # confirmed and never reaches the CSV.
                results_static = model_static.predict(
                    frame,
                    conf=params["detector_conf_thresh"],
                    imgsz=params["inference_imgsz"],
                    verbose=False,
                )
                for box in results_static[0].boxes:
                    coords = tuple(map(int, box.xyxy[0].tolist()))
                    class_idx = int(box.cls[0])
                    class_name = params["primary_static_classes"][class_idx]
                    conf = float(box.conf[0])
                    all_detections.append(
                        {
                            "coords": coords,
                            "primary_class": class_name,
                            "primary_conf": conf,
                            "source": "static",
                            "primary_class_combined": "",
                            "primary_conf_combined": 0.0,
                        }
                    )

            # Primary MOTION detection
            if model_motion is not None and motion_image is not None:
                results_motion = model_motion.predict(
                    motion_image,
                    conf=params["detector_conf_thresh"],
                    imgsz=params["inference_imgsz"],
                    verbose=False,
                )
                for box in results_motion[0].boxes:
                    coords = tuple(map(int, box.xyxy[0].tolist()))
                    class_idx = int(box.cls[0])
                    class_name = params["primary_motion_classes"][class_idx]
                    conf = float(box.conf[0])
                    all_detections.append(
                        {
                            "coords": coords,
                            "primary_class": class_name,
                            "primary_conf": conf,
                            "source": "motion",
                            "primary_class_combined": "",
                            "primary_conf_combined": 0.0,
                        }
                    )

            # ---- 3d: merge overlapping detections ---------------------
            # Two detections for the "same object" may come from both streams.
            # Merge by proximity (centroid distance) or overlap. The
            # dominant_source setting decides which stream's class wins; the
            # losing stream's label is kept in the *_combined fields.
            merged_detections = []
            for det in all_detections:
                x1, y1, x2, y2 = det["coords"]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                matched = False
                for md in merged_detections:
                    md_cx, md_cy = md["centroid"]
                    dist = np.hypot(cx - md_cx, cy - md_cy)

                    md_x1, md_y1, md_x2, md_y2 = md["coords"]
                    overlap = iou((x1, y1, x2, y2), (md_x1, md_y1, md_x2, md_y2))
                    ms_source = md["source"]

                    if (
                        dist < params["centroid_merge_thresh"]
                        or overlap > params["iou_thresh"]
                    ):
                        take = False
                        if (
                            det["source"] == ms_source
                            or params["dominant_source"] == "confidence"
                        ):
                            # Higher confidence wins.
                            take = (
                                "primary_conf" not in md
                                or det["primary_conf"] > md["primary_conf"]
                            )
                        elif (
                            det["source"] == "static"
                            and params["dominant_source"] == "static"
                        ) or (
                            det["source"] == "motion"
                            and params["dominant_source"] == "motion"
                        ):
                            # Configured dominant stream always wins.
                            take = True

                        if take:
                            md["primary_class_combined"] = md["primary_class"]
                            md["primary_conf_combined"] = md["primary_conf"]
                            md["primary_class"] = det["primary_class"]
                            md["primary_conf"] = det["primary_conf"]
                            md["coords"] = det["coords"]
                            md["centroid"] = (cx, cy)
                            md["source"] = det["source"]

                        matched = True
                        break

                if not matched:
                    # New unique detection.
                    merged_detections.append(
                        {
                            "coords": det["coords"],
                            "centroid": (cx, cy),
                            "source": det["source"],
                            "primary_class": det["primary_class"],
                            "primary_conf": det["primary_conf"],
                            "primary_class_combined": "",
                            "primary_conf_combined": 0.0,
                        }
                    )

            # ================================================================
            # STAGE 4 — Secondary (hierarchical) classification
            # ----------------------------------------------------------------
            # For each merged detection, optionally crop the box out and feed
            # it to a per-class YOLO classifier. Missing secondary models for
            # a primary class fall through to the default (secondary_class =
            # primary_class).
            # ================================================================
            processed_detections = []
            for det in merged_detections:
                coords = det["coords"]
                primary_class = det["primary_class"]
                primary_conf = det["primary_conf"]
                source = det["source"]
                primary_class_combined = det["primary_class_combined"]
                primary_conf_combined = det["primary_conf_combined"]

                # Route primary labels into per-stream columns for the CSV.
                if source == "static":
                    det["primary_static_class"] = primary_class
                    det["primary_static_conf"] = primary_conf
                    det["primary_motion_class"] = primary_class_combined
                    det["primary_motion_conf"] = primary_conf_combined
                else:
                    det["primary_motion_class"] = primary_class
                    det["primary_motion_conf"] = primary_conf
                    det["primary_static_class"] = primary_class_combined
                    det["primary_static_conf"] = primary_conf_combined

                # Lowering the detector floor multiplies the number of crops
                # reaching this stage, and the secondary classifier is the
                # most expensive part of the loop. Only classify detections
                # that clear primary_conf_thresh; the rest exist purely to
                # feed the tracker's recovery pass. If one is later promoted
                # into a track its secondary columns stay blank, which is
                # honest — we never actually classified it.
                if params["hierarchical_mode"] and (
                    primary_conf >= params["primary_conf_thresh"]
                ):
                    # Expand by secondary_crop_margin and clamp to the frame.
                    # The clamp is not optional hygiene: an unclamped negative
                    # coordinate makes numpy wrap to the far side of the axis,
                    # so the classifier silently receives a strip of the wrong
                    # part of the image rather than raising.
                    crop_box = expand_and_clamp_box(
                        coords, frame_w, frame_h, margin=crop_margin
                    )
                    if crop_box is None:
                        # Box lies entirely outside the frame — nothing to
                        # classify. Say so rather than failing quietly.
                        print(
                            f"  Warning: detection box {coords} is outside the "
                            f"{frame_w}x{frame_h} frame at frame {frame_idx}; "
                            "skipping secondary classification for it."
                        )

                    # Uses the motion image built once in 3b — does NOT rebuild
                    # it (that was the frame-history corruption bug).
                    static_crop = crop_region(frame, crop_box)
                    motion_crop = crop_region(motion_image, crop_box)

                    def _run(model_dict, crop):
                        if model_dict is None or crop is None or crop.size == 0:
                            return None, None
                        if not isinstance(model_dict, dict):
                            m = model_dict
                        else:
                            m = model_dict.get(primary_class)
                        if m is None:
                            return None, None
                        res = m.predict(
                            crop,
                            imgsz=params["secondary_imgsz"],
                            verbose=False,
                        )
                        if res[0].probs is None:
                            return None, None
                        conf = res[0].probs.top1conf.item()
                        # Below the threshold the top-1 label is noise. Returning it
                        # anyway is how a single track ends up cycling through four
                        # species across four consecutive frames.
                        if conf < params["secondary_conf_thresh"]:
                            return None, None
                        return m.names[res[0].probs.top1], conf

                    # Static secondary — only if configured
                    if len(params["secondary_static_classes"]) >= 2:
                        cls, conf = _run(secondary_static_models, static_crop)
                        if cls is not None:
                            det["secondary_static_class"] = cls
                            det["secondary_static_conf"] = conf
                    # External static secondary — needs a static_crop
                    elif (
                        params["secondary_static_external_model"] != ""
                        and static_crop is not None
                        and static_crop.size > 0
                    ):
                        res = secondary_static_models.predict_single(static_crop)
                        best_prediction = res.best
                        cls = best_prediction.name
                        conf = best_prediction.accuracy

                        if cls is not None and conf >= params["secondary_conf_thresh"]:
                            det["secondary_static_class"] = cls
                            det["secondary_static_conf"] = conf

                    # Motion secondary — needs a motion_image
                    if len(params["secondary_motion_classes"]) >= 2:
                        cls, conf = _run(secondary_motion_models, motion_crop)
                        if cls is not None:
                            det["secondary_motion_class"] = cls
                            det["secondary_motion_conf"] = conf

                processed_detections.append(det)

            # ================================================================
            # STAGE 5 — Tracking, rendering, CSV output
            # ================================================================

            LABEL_TYPE = "external"  # "primary", "external"

            # Both backends take the full detection dicts (boxes AND centroids)
            # and return {det_index: track_id}.
            #
            # raw_frame, not frame: `frame` accumulates drawn overlays as this
            # loop proceeds, and the ReID-based BoxMOT trackers crop appearance
            # patches from whatever image they are handed. Never pass
            # motion_image here — a false-colour difference image is
            # meaningless as an appearance cue.
            assignment = tracker.update(
                processed_detections, raw_frame, frame_idx=proc_idx
            )

            # Collect new track IDs appearing in this frame so we can save a
            # single annotated snapshot per individual *after* drawing is done.
            new_ids_this_frame = []

            # Draw each tracked detection on the output frame and log it.
            for idx, det in enumerate(processed_detections):
                tid = assignment.get(idx, None)
                # tid is None for tentative (unconfirmed) tracks — they are
                # deliberately not drawn or logged.
                if tid is None or tid not in tracker.tracks:
                    continue

                # First time we see this track id -> queue a snapshot.
                if tid not in seen_track_ids:
                    seen_track_ids.add(tid)
                    new_ids_this_frame.append(tid)

                x1, y1, x2, y2 = det["coords"]
                cx, cy = det["centroid"]

                # Pull all four stream results with safe defaults.
                ps_class = det.get("primary_static_class", "")
                ps_conf = det.get("primary_static_conf", 0)
                pm_class = det.get("primary_motion_class", "")
                pm_conf = det.get("primary_motion_conf", 0)
                ss_class = det.get("secondary_static_class", "")
                ss_conf = det.get("secondary_static_conf", 0)
                sm_class = det.get("secondary_motion_class", "")
                sm_conf = det.get("secondary_motion_conf", 0)
                p_source = det.get("source", "")

                # Label text uses whichever stream produced this detection.
                label_parts = []
                if p_source == "static":
                    label_parts.append(f"{ps_class.upper()}")
                    primary_cls = ps_class
                else:
                    label_parts.append(f"{pm_class.upper()}")
                    primary_cls = pm_class

                if params["hierarchical_mode"]:
                    if ss_class != "" and ss_class != primary_cls:
                        label_parts.append(f"{ss_class}")
                    if sm_class != "" and sm_class != primary_cls:
                        label_parts.append(f"{sm_class}")

                if primary_cls in params["primary_classes"]:
                    primary_col = params["primary_colors"][
                        params["primary_classes"].index(primary_cls)
                    ]
                else:
                    primary_col = (200, 200, 200)

                secondary_col = (255, 255, 255)

                # ---- 5a: draw bounding box + label ---------------------
                def _make_label(label, color=primary_col, position="top"):
                    label_size, _ = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        params["font_size"],
                        params["line_thickness"],
                    )
                    label_w, label_h = label_size

                    if position == "top":
                        l_x1 = x1
                        l_y1 = y1 - label_h - params["line_thickness"] * 2
                    else:
                        # bottom of rectangle, plus a small gap
                        l_x1 = x1
                        l_y1 = y2 + label_h * 2 + params["line_thickness"] * 2

                    # center large labels if they would overflow the box
                    if label_w > (x2 - x1):
                        l_x1 = x1 + (x2 - x1 - label_w) // 2

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        params["line_thickness"],
                    )

                    cv2.putText(
                        frame,
                        label,
                        (l_x1, l_y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        params["font_size"],
                        color,
                        params["line_thickness"],
                        cv2.LINE_AA,
                    )

                if params["hierarchical_mode"]:
                    # Pick secondary colour from whichever secondary fired.
                    if sm_class != "" and sm_class != primary_cls:
                        if sm_class in params["secondary_classes"]:
                            secondary_col = params["secondary_colors"][
                                params["secondary_classes"].index(sm_class)
                            ]
                    if ss_class != "" and ss_class != primary_cls:
                        if (
                            params["secondary_static_external_model"] == ""
                            and ss_class in params["secondary_classes"]
                        ):
                            secondary_col = params["secondary_colors"][
                                params["secondary_classes"].index(ss_class)
                            ]

                    if (
                        LABEL_TYPE == "primary"
                        and primary_cls in params["primary_classes"]
                    ):
                        _make_label(f"{tid} {primary_cls.upper()}")
                    elif LABEL_TYPE == "external":
                        _make_label(f"{tid} {ss_class}")

                    if sm_class != "" and sm_class != primary_cls:
                        # Nested boxes: outer = primary, inner = secondary.
                        outer_thickness = params["line_thickness"] + 2
                        cv2.rectangle(
                            frame,
                            (x1 - outer_thickness, y1 - outer_thickness),
                            (x2 + outer_thickness, y2 + outer_thickness),
                            primary_col,
                            outer_thickness,
                        )
                        _make_label(
                            f"{sm_class.upper()}",
                            color=secondary_col,
                            position="bottom",
                        )
                else:
                    # Flat mode — single box, primary label only.
                    _make_label(f"{tid} {primary_cls}")

                # ---- 5b: draw the motion vector ------------------------
                # Both backends expose state(tid) -> (x, y, vx, vy), so this
                # no longer reaches into a backend-specific Kalman filter.
                st = tracker.state(tid)
                if st is None:
                    st = (float(cx), float(cy), 0.0, 0.0)
                sx, sy, vx, vy = st
                next_x, next_y = sx + vx, sy + vy

                light_color = tuple(int(0.8 * ch + 0.2 * 255) for ch in primary_col)
                cv2.line(
                    frame,
                    (int(sx), int(sy)),
                    (int(next_x), int(next_y)),
                    primary_col,
                    params["line_thickness"],
                )
                cv2.circle(
                    frame,
                    (int(next_x), int(next_y)),
                    3,
                    light_color,
                    -params["line_thickness"],
                )
                cv2.circle(
                    frame,
                    (int(cx), int(cy)),
                    3,
                    primary_col,
                    -params["line_thickness"],
                )

                # ---- 5c: CSV row ---------------------------------------
                csv_writer.writerow(
                    [
                        frame_idx,
                        proc_idx,
                        tid,
                        x1,
                        y1,
                        x2,
                        y2,
                        f"{vx:.3f}",
                        f"{vy:.3f}",
                        ps_class,
                        f"{ps_conf:.3f}",
                        pm_class,
                        f"{pm_conf:.3f}",
                        ss_class,
                        f"{ss_conf:.3f}",
                        sm_class,
                        f"{sm_conf:.3f}",
                    ]
                )

            # ---- 5d: frame counter HUD + progress ---------------------
            text_color = (255, 255, 255)
            label = str(current_frame)
            label_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                params["font_size"],
                params["line_thickness"],
            )
            label_w, label_h = label_size
            cv2.rectangle(
                frame,
                (0, 0),
                (
                    label_w + params["line_thickness"] * 4,
                    label_h + params["line_thickness"] * 4,
                ),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                label,
                (params["line_thickness"] * 2, label_h + params["line_thickness"] * 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                params["font_size"],
                text_color,
                params["line_thickness"],
            )

            # write the annotated frame to the output video
            writer.write(frame)

            # ---- 5d-bis: save one annotated frame per newly-seen ID ---
            # Done *after* all drawing + HUD so the exported frame is the
            # same image that gets written to the video.
            for tid in new_ids_this_frame:
                snap_path = os.path.join(
                    frames_out_dir,
                    f"{base}_id{tid:04d}_frame{frame_idx:06d}.jpg",
                )
                cv2.imwrite(snap_path, frame)

            if print_tick > params["progress_update"]:
                elapsed = time.time() - start_time
                current_fps = current_frame / elapsed if elapsed > 0 else 0
                pc_done = (
                    100 * (params["frame_skip"] + 1) * current_frame / total_frames
                    if total_frames
                    else 0
                )
                print(
                    f"Progress: {pc_done:.2f}% | {current_fps:.1f} FPS",
                    end="\r",
                    flush=True,
                )
                print_tick = 0
            current_frame += 1
            print_tick += 1

        # Frame-skip counter — rolls over at frame_skip+1 so we process
        # every (frame_skip+1)th frame.
        frame_count += 1
        if frame_count > params["frame_skip"]:
            frame_count = 0

    # ---- STAGE 5e: close outputs -------------------------------------
    cap.release()
    writer.release()
    csv_file.close()

    data = pd.read_csv(os.path.join(video_out_dir, base + "_tracking.csv"))

    n_tracks = data["id"].nunique() if len(data) else 0
    print(f"Done processing {base} | {current_fps:.1f} FPS | {n_tracks} tracks")

    return data


# ============================================================================
# Entry point
# ----------------------------------------------------------------------------
# Train (or verify) models once, then batch-process every file in input/.
# ============================================================================
if __name__ == "__main__":
    train_models()

    input_root = params["input_folder"]
    video_exts = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".m4v",
        ".wmv",
        ".flv",
        ".mpg",
        ".mpeg",
    }

    data = pd.DataFrame()

    for vid in sorted(glob.glob(os.path.join(input_root, "**", "*"), recursive=True)):
        if not os.path.isfile(vid):
            continue
        if os.path.splitext(vid)[1].lower() not in video_exts:
            continue
        temp = process_video(vid)

        # process_video returns None when a video is skipped (no models, or
        # the file would not open). Guard before touching the frame.
        if temp is None or len(temp) == 0:
            continue

        temp["video"] = os.path.relpath(vid, input_root)
        data = pd.concat([data, temp], ignore_index=True)

    if len(data):
        data.to_csv(
            os.path.join(params["output_folder"], "tracking_data.csv"), index=False
        )
    else:
        print("No tracking data produced.")
