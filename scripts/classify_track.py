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
    5. Tracking + rendering       (Kalman tracker → overlays → CSV/MP4)

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
"""

import csv
import glob
import os
import shutil
import time
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
import pandas as pd
from load_configs import load_params
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# Load all config into a single dict. See load_configs.py for keys.
params = load_params()


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
#   * Primary models:   at least  5 images required  (in maybe_retrain).
#   * Secondary models: at least  2 images required  (in train_models).
# A model that fails to train leaves its best.pt absent — downstream code
# detects this at load time and skips that stream for inference.
# ============================================================================


def count_images_in_dataset(path):
    """
    Count images in a training dataset.
      * YAML path   -> read the `train:` key, count files there
      * directory   -> recursive, count image files in leaf dirs only
    Returns 0 on any error.
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
                image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
                train_count = (
                    len(
                        [
                            f
                            for f in os.listdir(abs_train_path)
                            if os.path.splitext(f)[1].lower() in image_exts
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
                            if os.path.splitext(f)[1].lower() in image_exts
                        ]
                    )
                    if abs_val_path
                    else 0
                )
            return train_count, val_count
        except Exception as e:
            print(f"Error counting images: {e}")
            return 0, 0

    # Secondary-model case: class-folder tree.
    elif os.path.isdir(path):
        total_count = 0
        image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        for root, dirs, files in os.walk(path):
            if not dirs:  # leaf class directory
                total_count += sum(
                    1 for f in files if os.path.splitext(f)[1].lower() in image_exts
                )
        return total_count

    else:
        print(f"Unsupported dataset format: {path}")
        return 0


def maybe_retrain(
    model_type, yaml_path, project_path, model_path, classifier, epochs, imgsz
):
    """
    Decide whether to (re)train a model based on existence and image counts.

      * model_path exists + count changed -> prompt user; on 'yes', backup
        old model_dir to <project>_backup<N>, fine-tune from the backup's
        best.pt, and move the new run into place.
      * model_path missing                -> first-time train, but only if
        the dataset has at least 5 images.
      * count unchanged                   -> no-op.

    Returns True if training ran, False otherwise.
    """
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

        # If the count changed, ask the user whether to retrain.
        if train != last_count:
            root = tk.Tk()
            root.withdraw()
            msg = (
                f"New annotations detected for '{model_type}' model.\n"
                f"Training image count changed from {last_count} to {train}.\n\n"
                "Do you want to re-train this model?"
            )
            response = messagebox.askyesno("Retrain model?", msg)
            root.destroy()

            if response:
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
                start_weights = os.path.join(
                    final_backup, "train", "weights", "best.pt"
                )
                print(f"Training new {model_type} model using existing weights...")
                model = YOLO(start_weights)
                model.train(
                    data=yaml_path,
                    epochs=epochs,
                    imgsz=imgsz,
                    project=project_path,
                    name="train",
                    exist_ok=True,
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
        os.makedirs(project_path, exist_ok=True)
        dst = os.path.join(project_path, "saved_settings.ini")
        try:
            shutil.copy2(params["config_path"], dst)
            print(f"Saved settings snapshot to {dst}")
        except Exception as e:
            print(f"Warning: could not copy settings to model dir: {e}")

        return True


def train_models():
    """
    Walk the configured model hierarchy and ensure each has weights on disk
    (or leave it absent if there isn't enough annotation data). Populates two
    global dicts used later at inference time:

        secondary_static_models[primary_class] = YOLO instance
        secondary_motion_models[primary_class] = YOLO instance

    Keys are only set for classes whose secondary model was successfully
    trained AND loaded. Missing keys are handled naturally by `.get(..., None)`
    at inference time.
    """
    global secondary_static_models, secondary_motion_models
    global static_class_map, motion_class_map

    secondary_static_models = None
    secondary_motion_models = None

    # ---- hierarchical (secondary) models ------------------------------
    if params["hierarchical_mode"]:
        # Secondary STATIC classifiers — one YOLO-cls model per primary class.
        secondary_static_models = {}
        static_class_map = [
            [None] * len(params["secondary_classes"])
            for _ in range(len(params["primary_classes"]))
        ]
        if len(params["secondary_static_classes"]) >= 2:
            for primary_class in params["primary_classes"]:
                idx = params["primary_classes"].index(primary_class)
                hotkey = params["primary_hotkeys"][idx]

                # Skip primaries that are themselves in the secondary set, or
                # in the ignore list, or have no annotation data on disk.
                if hotkey in params["secondary_hotkeys"]:
                    continue
                if primary_class in params["ignore_secondary"]:
                    continue
                data_dir = os.path.join(
                    params["secondary_static_data_path"], primary_class
                )
                if not os.path.isdir(data_dir):
                    continue

                model_dir = f"models/model_static_static_{primary_class}"
                weights_path = os.path.join(model_dir, "train", "weights", "best.pt")

                # Not enough annotations -> leave weights absent, skip load.
                n_image = count_images_in_dataset(data_dir)
                if n_image < 2:
                    print(
                        f"Error: Not enough images to train secondary static model "
                        f"for primary class '{primary_class}' (found {n_image}, "
                        f"need at least 2). Skipping this secondary model."
                    )
                    continue

                maybe_retrain(
                    model_dir,
                    data_dir,
                    model_dir,
                    weights_path,
                    params["secondary_classifier"],
                    params["secondary_epochs"],
                    224,
                )

                # Load only if weights actually exist. maybe_retrain can
                # silently skip when first-time training isn't allowed.
                if os.path.isfile(weights_path):
                    try:
                        if params["use_ncnn"] == "true":
                            secondary_static_models[primary_class] = (
                                load_model_with_ncnn_preference(
                                    weights_path, "classify"
                                )
                            )
                        else:
                            secondary_static_models[primary_class] = YOLO(weights_path)
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

        # Secondary MOTION classifiers — mirror of the static block.
        secondary_motion_models = {}
        motion_class_map = [
            [None] * len(params["secondary_classes"])
            for _ in range(len(params["primary_classes"]))
        ]
        if len(params["secondary_motion_classes"]) >= 2:
            for primary_class in params["primary_classes"]:
                idx = params["primary_classes"].index(primary_class)
                hotkey = params["primary_hotkeys"][idx]

                if hotkey in params["secondary_hotkeys"]:
                    continue
                if primary_class in params["ignore_secondary"]:
                    continue
                data_dir = os.path.join(
                    params["secondary_motion_data_path"], primary_class
                )
                if not os.path.isdir(data_dir):
                    continue

                model_dir = f"models/model_secondary_motion_{primary_class}"
                weights_path = os.path.join(model_dir, "train", "weights", "best.pt")

                train_count, val_count = count_images_in_dataset(data_dir)
                if train_count < 2 or val_count < 2:
                    print(
                        f"Error: Not enough images to train secondary motion model "
                        f"for primary class '{primary_class}' (found {train_count} training images and {val_count} validation images, "
                        f"need at least 2 of each). Skipping this secondary model."
                    )
                    continue

                maybe_retrain(
                    model_dir,
                    data_dir,
                    model_dir,
                    weights_path,
                    params["secondary_classifier"],
                    params["secondary_epochs"],
                    224,
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
                            secondary_motion_models[primary_class] = YOLO(weights_path)
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

    # ---- primary detectors --------------------------------------------
    # These are trained AFTER the secondaries because in hierarchical mode
    # the secondary annotations share source frames with primary annotations.
    if params["primary_static_classes"][0] != "0":
        maybe_retrain(
            "models/model_primary_static",
            params["primary_static_yaml_path"],
            params["primary_static_project_path"],
            params["primary_static_model_path"],
            params["primary_classifier"],
            params["primary_epochs"],
            640,
        )

    if params["primary_motion_classes"][0] != "0":
        maybe_retrain(
            "models/model_primary_motion",
            params["primary_motion_yaml_path"],
            params["primary_motion_project_path"],
            params["primary_motion_model_path"],
            params["primary_classifier"],
            params["primary_epochs"],
            640,
        )


# ============================================================================
# STAGE 5a — IoU helper
# ----------------------------------------------------------------------------
# Not traditional IoU. Returns the *larger* proportional overlap relative to
# each box's own area, so if one box is fully inside another the score is 1.0.
# Used when merging detections from the static and motion streams.
# ============================================================================


def iou(box1, box2):
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    prop1 = inter / area1
    prop2 = inter / area2
    if prop1 > prop2:
        return prop1 if prop1 > 0 else 0
    else:
        return prop2 if prop2 > 0 else 0


# ============================================================================
# STAGE 5b — Kalman-filter multi-object tracker
# ----------------------------------------------------------------------------
# Each track keeps a 4D state (x, y, vx, vy) that predicts next position.
# Detections are matched to tracks via Hungarian assignment on Euclidean
# distance. Unmatched detections become new tracks; unmatched tracks age and
# eventually die.
# ============================================================================


class KalmanTracker:
    def __init__(self, dist_thresh, max_missed):
        self.next_id = 1
        self.tracks = {}  # tid -> {'kf': KalmanFilter, 'missed': int}
        self.prev_positions = {}  # tid -> last-observed (x, y)
        self.dist_thresh = dist_thresh
        self.max_missed = max_missed

    def _create_kf(self, initial_pt):
        # State: [x, y, vx, vy]; measurement: [x, y].
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kf.processNoiseCov = np.diag(
            [
                params["process_noise_pos"],
                params["process_noise_pos"],
                params["process_noise_vel"],
                params["process_noise_vel"],
            ]
        ).astype(np.float32)
        kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * params["measurement_noise"]
        )
        kf.statePre = np.array(
            [[initial_pt[0]], [initial_pt[1]], [0.0], [0.0]], dtype=np.float32
        )
        kf.statePost = kf.statePre.copy()
        return kf

    def predict_all(self):
        """Run KF predict() for every track. Returns list of (tid, (x, y))."""
        preds = []
        for tid, tr in self.tracks.items():
            pred = tr["kf"].predict()
            preds.append((tid, (float(pred[0, 0]), float(pred[1, 0]))))
        return preds

    def _prune_duplicate_tracks(self):
        """Merge any two tracks whose posteriors are very close."""
        tids = list(self.tracks.keys())
        posts = {
            tid: (
                float(self.tracks[tid]["kf"].statePost[0, 0]),
                float(self.tracks[tid]["kf"].statePost[1, 0]),
            )
            for tid in tids
        }
        to_drop = set()
        for i, t1 in enumerate(tids):
            x1, y1 = posts[t1]
            for t2 in tids[i + 1 :]:
                x2, y2 = posts[t2]
                if np.hypot(x1 - x2, y1 - y2) < self.dist_thresh * 0.5:
                    to_drop.add(max(t1, t2))
        for tid in to_drop:
            del self.tracks[tid]

    def update(self, detections):
        """
        Main tracker step.
          detections: list of (x, y) centroids for this frame.
          Returns:    dict mapping detection-index -> track-id.
        """
        # 1) Predict every existing track forward one step.
        preds = self.predict_all()
        track_ids = [t[0] for t in preds]
        pred_pts = [t[1] for t in preds]

        # 2) Build cost matrix (Euclidean distance) and solve assignment.
        if pred_pts and detections:
            cost = np.zeros((len(pred_pts), len(detections)), dtype=np.float32)
            for i, p in enumerate(pred_pts):
                for j, d in enumerate(detections):
                    cost[i, j] = np.hypot(p[0] - d[0], p[1] - d[1])
            row_idx, col_idx = linear_sum_assignment(cost)
        else:
            row_idx = np.array([], dtype=int)
            col_idx = np.array([], dtype=int)

        assigned_detects = {}
        matched_tracks = set()
        matched_dets = set()

        # 3) Associate tracks <-> detections that are within threshold.
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < self.dist_thresh:
                tid = track_ids[r]
                matched_tracks.add(tid)
                matched_dets.add(c)
                assigned_detects[c] = tid

                dpt = detections[c]
                meas = np.array([[np.float32(dpt[0])], [np.float32(dpt[1])]])

                self.tracks[tid]["kf"].correct(meas)
                self.tracks[tid]["missed"] = 0
                self.prev_positions[tid] = (dpt[0], dpt[1])

        # 4) Unassigned detections: try nearest-track fallback, else new track.
        for i, dpt in enumerate(detections):
            if i in matched_dets:
                continue

            best_tid, best_dist = None, float("inf")
            for tid, (px, py) in preds:
                d = np.hypot(dpt[0] - px, dpt[1] - py)
                if d < best_dist:
                    best_dist, best_tid = d, tid

            if best_dist < self.dist_thresh:
                # Claim this detection for the nearest track.
                assigned_detects[i] = best_tid
                self.tracks[best_tid]["missed"] = 0
                meas = np.array([[np.float32(dpt[0])], [np.float32(dpt[1])]])
                self.tracks[best_tid]["kf"].correct(meas)
                self.prev_positions[best_tid] = (dpt[0], dpt[1])
                matched_tracks.add(best_tid)
            else:
                # Birth of a new track.
                tid = self.next_id
                kf = self._create_kf(dpt)
                self.tracks[tid] = {"kf": kf, "missed": 0}
                assigned_detects[i] = tid
                self.prev_positions[tid] = (dpt[0], dpt[1])
                matched_tracks.add(tid)
                self.next_id += 1

        # 5) Age unmatched tracks; inflate uncertainty; delete if too old.
        for tid in list(self.tracks.keys()):
            if tid not in matched_tracks:
                self.tracks[tid]["missed"] += 1
                noise_scale = min(2.0, 1.0 + self.tracks[tid]["missed"] * 0.2)

                kf = self.tracks[tid]["kf"]
                new_noise = kf.processNoiseCov.copy()
                new_noise *= noise_scale
                kf.processNoiseCov = new_noise

                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]
                    if tid in self.prev_positions:
                        del self.prev_positions[tid]

        return assigned_detects


# ============================================================================
# STAGE 2–5 — Per-video pipeline
# ----------------------------------------------------------------------------
# process_video() runs the end-to-end pipeline on one file:
#   * opens video + output writers (MP4 + CSV)
#   * loads whichever primary models have trained weights on disk
#   * for each frame: build motion image, run detection(s), merge, run
#     secondary classification, track, draw, and write a CSV row per track.
#
# The model-load section below is defensive: a missing best.pt just means the
# corresponding stream is skipped for this video. If both primary models are
# missing the whole video is skipped cleanly.
# ============================================================================


def process_video(file):
    # ---- STAGE 2a: open inputs and outputs ----------------------------
    # Preserve any subfolder structure from input/ under output/.
    # e.g.  input/site_A/day_2/clip.mp4
    #   ->  output/site_A/day_2/clip_detected.mp4
    #       output/site_A/day_2/clip_tracking.csv
    rel = os.path.relpath(file, params["input_folder"])
    rel_dir = os.path.dirname(rel)  # "site_A/day_2" or ""
    base = os.path.splitext(os.path.basename(rel))[0]

    out_dir = os.path.join(params["output_folder"], rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened():
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * params["scale_factor"])
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * params["scale_factor"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(
        os.path.join(out_dir, base + "_detected.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    # ---- STAGE 2b: defensively load primary models --------------------
    # For each stream, three things must be true to load:
    #   (1) the stream is configured in the INI (classes[0] != "0")
    #   (2) the best.pt file exists on disk
    #   (3) YOLO/NCNN can actually open it
    # If any fails, the model stays None and the per-frame detection block
    # below is silently skipped for that stream.
    model_static = None
    model_motion = None

    # Primary STATIC
    if params["primary_static_classes"][0] != "0":
        weights = params["primary_static_model_path"]  # already ends in best.pt
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

    # If neither primary is available, there's nothing to detect. Clean up
    # and skip this video instead of producing an empty output.
    if model_static is None and model_motion is None:
        print(f"Skipping {file}: no trained primary models available")
        cap.release()
        writer.release()
        try:
            os.remove(os.path.join(out_dir, base + "_detected.mp4"))
        except OSError:
            pass
        return
    # ---- STAGE 2c: initialise tracker + CSV ---------------------------
    tracker = KalmanTracker(
        params["match_distance_thresh"], params["delete_after_missed"]
    )

    prev_frames, frame_idx = None, 0
    csv_file = open(
        os.path.join(out_dir, base + "_tracking.csv"),
        "w",
        newline="",
    )
    csv_writer = csv.writer(csv_file)
    # One row per frame per tracked object. Empty string / 0.0 indicates the
    # corresponding model was not available or did not fire.
    csv_writer.writerow(
        [
            "frame",
            "id",
            "x",
            "y",
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

            # Prime the 3-frame history on the first iteration.
            if prev_frames is None:
                prev_frames = [gray.copy() for _ in range(3)]
                continue

            # ---- 3b: build the false-colour motion image --------------
            # Three temporally-offset frame differences are mapped to B/G/R
            # channels so a moving object leaves a coloured "tail" that the
            # motion detector can learn from.
            if params["primary_motion_classes"][0] != "0":
                diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]

                if params["strategy"] == "exponential":
                    # Exponential decay — smoother tails.
                    prev_frames[0] = gray
                    prev_frames[1] = cv2.addWeighted(
                        prev_frames[1], params["expA"], gray, params["expA2"], 0
                    )
                    prev_frames[2] = cv2.addWeighted(
                        prev_frames[2], params["expB"], gray, params["expB2"], 0
                    )
                elif params["strategy"] == "sequential":
                    # Plain frame-over-frame ring buffer.
                    prev_frames[2] = prev_frames[1]
                    prev_frames[1] = prev_frames[0]
                    prev_frames[0] = gray

                # chromatic_tail_only: emphasise only the leading tail edge.
                if params["chromatic_tail_only"] == "true":
                    tb = cv2.subtract(diffs[0], diffs[1])
                    tr = cv2.subtract(diffs[2], diffs[1])
                    tg = cv2.subtract(diffs[1], diffs[0])

                    blue = cv2.addWeighted(
                        gray,
                        params["lum_weight"],
                        tb,
                        params["rgb_multipliers"][2],
                        params["motion_threshold"],
                    )
                    green = cv2.addWeighted(
                        gray,
                        params["lum_weight"],
                        tg,
                        params["rgb_multipliers"][1],
                        params["motion_threshold"],
                    )
                    red = cv2.addWeighted(
                        gray,
                        params["lum_weight"],
                        tr,
                        params["rgb_multipliers"][0],
                        params["motion_threshold"],
                    )
                else:
                    blue = cv2.addWeighted(
                        gray,
                        params["lum_weight"],
                        diffs[0],
                        params["rgb_multipliers"][2],
                        params["motion_threshold"],
                    )
                    green = cv2.addWeighted(
                        gray,
                        params["lum_weight"],
                        diffs[1],
                        params["rgb_multipliers"][1],
                        params["motion_threshold"],
                    )
                    red = cv2.addWeighted(
                        gray,
                        params["lum_weight"],
                        diffs[2],
                        params["rgb_multipliers"][0],
                        params["motion_threshold"],
                    )

                motion_image = cv2.merge((blue, green, red)).astype(np.uint8)

            # ---- 3c: primary detections -------------------------------
            # GUARD CHANGE: predicate is now "did a model actually load?"
            # rather than "is a stream configured?". A configured stream with
            # no weights falls through to the merge step with zero detections.
            all_detections = []

            # Primary STATIC detection
            if model_static is not None:
                results_static = model_static.predict(
                    frame, conf=params["primary_conf_thresh"], verbose=False
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
            if model_motion is not None:
                results_motion = model_motion.predict(
                    motion_image, conf=params["primary_conf_thresh"], verbose=False
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
            # Merge by proximity (centroid distance) or overlap (IoU). The
            # dominant_source setting decides which stream's class wins; the
            # losing stream's label is kept in the *_combined fields for the
            # CSV.
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
                        # Same-source merge OR confidence-based policy:
                        # keep whichever detection has the higher conf.
                        if (
                            det["source"] == ms_source
                            or params["dominant_source"] == "confidence"
                        ):
                            if det["source"] == "static":
                                if (
                                    "primary_conf" not in md
                                    or det["primary_conf"] > md["primary_conf"]
                                ):
                                    md["primary_class_combined"] = md["primary_class"]
                                    md["primary_conf_combined"] = md["primary_conf"]
                                    md["primary_class"] = det["primary_class"]
                                    md["primary_conf"] = det["primary_conf"]
                                    md["coords"] = det["coords"]
                                    md["centroid"] = (cx, cy)
                                    md["source"] = det["source"]
                            else:  # motion
                                if (
                                    "primary_conf" not in md
                                    or det["primary_conf"] > md["primary_conf"]
                                ):
                                    md["primary_class_combined"] = md["primary_class"]
                                    md["primary_conf_combined"] = md["primary_conf"]
                                    md["primary_class"] = det["primary_class"]
                                    md["primary_conf"] = det["primary_conf"]
                                    md["coords"] = det["coords"]
                                    md["centroid"] = (cx, cy)
                                    md["source"] = det["source"]
                        elif (
                            det["source"] == "static"
                            and params["dominant_source"] == "static"
                        ):
                            md["primary_class_combined"] = md["primary_class"]
                            md["primary_conf_combined"] = md["primary_conf"]
                            md["primary_class"] = det["primary_class"]
                            md["primary_conf"] = det["primary_conf"]
                            md["coords"] = det["coords"]
                            md["centroid"] = (cx, cy)
                            md["source"] = det["source"]
                        elif (
                            det["source"] == "motion"
                            and params["dominant_source"] == "motion"
                        ):
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
                    new_det = {
                        "coords": det["coords"],
                        "centroid": (cx, cy),
                        "source": det["source"],
                        "primary_class_combined": "",
                        "primary_conf_combined": 0.0,
                    }
                    new_det["primary_class"] = det["primary_class"]
                    new_det["primary_conf"] = det["primary_conf"]
                    merged_detections.append(new_det)

            # ================================================================
            # STAGE 4 — Secondary (hierarchical) classification
            # ----------------------------------------------------------------
            # For each merged detection, optionally crop the box out and feed
            # it to a per-class YOLO classifier. Missing secondary models for
            # a primary class are handled by .get(..., None) falling through
            # to the default (secondary_class = primary_class, conf = 1.0).
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

                if params["hierarchical_mode"]:
                    x1, y1, x2, y2 = coords

                    # Pick which secondary model and which image to crop from.
                    # Falls back to the opposite stream if the preferred one
                    # isn't configured.
                    sec_model = None
                    crop_img = None

                    if source == "static":
                        if len(params["secondary_static_classes"]) >= 2:
                            sec_model = secondary_static_models.get(primary_class, None)
                            crop_img = frame
                        elif len(params["secondary_motion_classes"]) >= 2:
                            sec_model = secondary_motion_models.get(primary_class, None)
                            crop_img = (
                                motion_image
                                if params["primary_motion_classes"][0] != "0"
                                else frame
                            )
                    else:  # motion source
                        if len(params["secondary_motion_classes"]) >= 2:
                            sec_model = secondary_motion_models.get(primary_class, None)
                            crop_img = motion_image
                        elif len(params["secondary_static_classes"]) >= 2:
                            sec_model = secondary_static_models.get(primary_class, None)
                            crop_img = frame

                    crop = None
                    if crop_img is not None:
                        crop = crop_img[y1:y2, x1:x2]

                    # Sensible defaults if no secondary model / no valid crop.
                    secondary_class = primary_class
                    secondary_conf = 1.0

                    if sec_model and crop is not None and crop.size > 0:
                        sec_results = sec_model.predict(crop, verbose=False)
                        if sec_results[0].probs is not None:
                            secondary_class_idx = sec_results[0].probs.top1
                            secondary_conf = sec_results[0].probs.top1conf.item()
                            secondary_class = sec_model.names[secondary_class_idx]

                    if source == "static":
                        det["secondary_static_class"] = secondary_class
                        det["secondary_static_conf"] = secondary_conf
                    else:
                        det["secondary_motion_class"] = secondary_class
                        det["secondary_motion_conf"] = secondary_conf

                processed_detections.append(det)

            # ================================================================
            # STAGE 5 — Tracking, rendering, CSV output
            # ================================================================

            # Feed centroids to the Kalman tracker -> {det_idx: track_id}.
            cents = [d["centroid"] for d in processed_detections]
            assignment = tracker.update(cents)

            # Draw each tracked detection on the output frame and log it.
            for idx, det in enumerate(processed_detections):
                tid = assignment.get(idx, None)
                if tid is None or tid not in tracker.tracks:
                    continue

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

                primary_col = params["primary_colors"][
                    params["primary_classes"].index(primary_cls)
                ]
                secondary_col = (255, 255, 255)

                # ---- 5a: draw bounding box + label ---------------------
                if params["hierarchical_mode"]:
                    # Pick secondary colour from whichever secondary fired.
                    if sm_class != "" and sm_class != primary_cls:
                        secondary_cls = sm_class
                        secondary_col = params["secondary_colors"][
                            params["secondary_classes"].index(secondary_cls)
                        ]
                    if ss_class != "" and ss_class != primary_cls:
                        secondary_cls = ss_class
                        secondary_col = params["secondary_colors"][
                            params["secondary_classes"].index(secondary_cls)
                        ]

                    # If no secondary to display, draw a single box + label.
                    if primary_cls in params["primary_classes"]:
                        label = f"{tid} {primary_cls.upper()}"
                        label_size, _ = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            params["font_size"],
                            params["line_thickness"],
                        )
                        label_w, label_h = label_size
                        cv2.rectangle(
                            frame,
                            (
                                x1 - params["line_thickness"],
                                y1 - label_h - params["line_thickness"] * 4,
                            ),
                            (x1 + label_w + params["line_thickness"] * 2, y1),
                            (0, 0, 0),
                            -1,
                        )
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            primary_col,
                            params["line_thickness"],
                        )
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - params["line_thickness"] * 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            params["font_size"],
                            primary_col,
                            params["line_thickness"],
                            cv2.LINE_AA,
                        )
                    else:
                        # Nested boxes: outer = primary, inner = secondary.
                        outer_thickness = params["line_thickness"] + 2
                        cv2.rectangle(
                            frame,
                            (x1 - outer_thickness, y1 - outer_thickness),
                            (x2 + outer_thickness, y2 + outer_thickness),
                            primary_col,
                            outer_thickness,
                        )
                        label = f"{tid} {primary_cls.upper()} {secondary_cls}"
                        label_size, _ = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            params["font_size"],
                            params["line_thickness"],
                        )
                        label_w, label_h = label_size
                        cv2.rectangle(
                            frame,
                            (
                                x1 - params["line_thickness"],
                                y1 - label_h - params["line_thickness"] * 4,
                            ),
                            (x1 + label_w + params["line_thickness"] * 2, y1),
                            (0, 0, 0),
                            -1,
                        )
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            secondary_col,
                            params["line_thickness"],
                        )
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - params["line_thickness"] * 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            params["font_size"],
                            secondary_col,
                            params["line_thickness"],
                            cv2.LINE_AA,
                        )
                else:
                    # Flat mode — single box, primary label only.
                    label = f"{tid} {primary_cls}"
                    label_size, _ = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        params["font_size"],
                        params["line_thickness"],
                    )
                    label_w, label_h = label_size
                    cv2.rectangle(
                        frame,
                        (
                            x1 - params["line_thickness"],
                            y1 - label_h - params["line_thickness"] * 4,
                        ),
                        (x1 + label_w + params["line_thickness"] * 2, y1),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        primary_col,
                        params["line_thickness"],
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - params["line_thickness"] * 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        params["font_size"],
                        primary_col,
                        params["line_thickness"],
                        cv2.LINE_AA,
                    )

                # ---- 5b: draw the KF motion vector ---------------------
                if tid in tracker.tracks:
                    state_post = tracker.tracks[tid]["kf"].statePost
                    x, y = state_post[0, 0], state_post[1, 0]
                    vx, vy = state_post[2, 0], state_post[3, 0]
                    next_x = x + vx
                    next_y = y + vy

                    light_color = tuple(int(0.8 * ch + 0.2 * 255) for ch in primary_col)
                    cv2.line(
                        frame,
                        (int(x), int(y)),
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
                # One row per tracked detection per frame. Empty strings and
                # zero conf indicate the corresponding stream didn't fire or
                # wasn't available.
                csv_writer.writerow(
                    [
                        frame_idx,
                        tid,
                        cx,
                        cy,
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

            writer.write(frame)

            if print_tick > params["progress_update"]:
                elapsed = time.time() - start_time
                current_fps = current_frame / elapsed if elapsed > 0 else 0
                pc_done = (
                    100 * (params["frame_skip"] + 1) * current_frame / total_frames
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

    # save csv in object
    data = pd.read_csv(os.path.join(out_dir, base + "_tracking.csv"))

    # ---- STAGE 5e: close outputs -------------------------------------
    cap.release()
    writer.release()
    csv_file.close()
    print(f"Done processing {base} | {current_fps:.1f} FPS")

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

        temp["video"] = os.path.relpath(vid, input_root)

        data = pd.concat([data, temp], ignore_index=True)

    data.to_csv(os.path.join(params["output_folder"], "tracking_data.csv"), index=False)
