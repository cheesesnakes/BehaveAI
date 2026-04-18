import csv
import glob
import os
import shutil

# ~ import config_watcher
import time
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from load_configs import params

# --- NCNN helper utilities -----------------------


def ncnn_dir_for_weights(weights_path):
    """Return the expected NCNN export directory for a given .pt path."""
    base, ext = os.path.splitext(weights_path)
    # Ultralytics export typically creates a folder named like "<base>_ncnn_model"
    return base + "_ncnn_model"


def ncnn_files_exist(ncnn_dir):
    """Return True if NCNN param+bin appear to exist in the export dir."""
    if not os.path.isdir(ncnn_dir):
        return False
    # Look for .param and .bin files (ncnn export creates *.param and *.bin)
    has_param = any(f.endswith(".param") for f in os.listdir(ncnn_dir))
    has_bin = any(f.endswith(".bin") for f in os.listdir(ncnn_dir))
    return has_param and has_bin


def ensure_ncnn_export(weights_path, task, timeout=300):
    """
    Ensure an NCNN conversion exists for weights_path.
    Returns the ncnn_dir on success, None on failure (falls back to .pt).
    This will skip conversion if the ncnn folder already exists.
    """
    ncnn_dir = ncnn_dir_for_weights(weights_path)
    if ncnn_files_exist(ncnn_dir):
        return ncnn_dir

    try:
        print(f"Exporting {weights_path} -> NCNN (this may take a while)...")
        model = YOLO(weights_path, task=task)
        # Use Ultralytics export API. This creates the folder "<base>_ncnn_model".
        # Some installs can be slow; we try and catch errors below.
        model.export(format="ncnn")
        # Wait a short time for files to appear (export is synchronous in most versions).
        start = time.time()
        while time.time() - start < timeout:
            if ncnn_files_exist(ncnn_dir):
                print(f"NCNN export complete: {ncnn_dir}")
                return ncnn_dir
            time.sleep(0.5)
        # timed out
        print(f"NCNN export timeout for {weights_path}")
        return None
    except Exception as e:
        # Don't crash — export can fail on some systems; print useful debugging info and return None
        print(f"Warning: NCNN export failed for {weights_path}: {e}")
        return None


def load_model_with_ncnn_preference(weights_path, task):
    """
    Attempt to use NCNN if available (or convert it). If conversion or loading fails,
    fall back to the original PyTorch .pt path.
    Returns a YOLO model instance (which may wrap NCNN or .pt).
    """
    # If a .pt was not provided (maybe already a folder), just try loading directly
    if not weights_path.endswith(".pt"):
        try:
            return YOLO(weights_path, task=task)
        except Exception as e:
            print(f"Error loading model {weights_path}: {e}")
            raise

    ncnn_dir = ncnn_dir_for_weights(weights_path)
    # prefer existing NCNN dir if present
    if ncnn_files_exist(ncnn_dir):
        try:
            print(f"Loading NCNN model from {ncnn_dir}")
            return YOLO(ncnn_dir, task=task)
        except Exception as e:
            print(f"Failed to load NCNN model at {ncnn_dir}: {e} (falling back to .pt)")

    # Otherwise attempt conversion (one-time). If it fails, fall back to .pt.
    exported = ensure_ncnn_export(weights_path, task)
    if exported:
        try:
            return YOLO(exported, task=task)
        except Exception as e:
            print(
                f"Failed to load NCNN-exported model {exported}: {e} (falling back to .pt)"
            )

    # Finally, fallback to direct .pt load
    print(f"Using original weights (PyTorch) at {weights_path}")
    return YOLO(weights_path, task=task)


# --------------------------------------------------------------------


def move_to_expected(project_path, run_name="train", runs_root="runs"):
    """
    Locate the most recently written best.pt anywhere Ultralytics might have
    put it, and relocate its parent run directory into project_path/run_name.
    """
    expected_weights = os.path.join(project_path, run_name, "weights", "best.pt")
    if os.path.exists(expected_weights):
        # Already in the right place, nothing to do.
        return os.path.join(project_path, run_name)

    # Search broadly for best.pt files, excluding the project_path itself
    # (in case of a partial previous move) and any *_backup dirs.
    search_roots = [runs_root, "../../runs", "."]
    candidates = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for path in glob.glob(
            os.path.join(root, "**", "weights", "best.pt"), recursive=True
        ):
            abs_path = os.path.abspath(path)
            # Skip anything already inside project_path or a backup
            if os.path.abspath(project_path) in abs_path:
                continue
            if "_backup" in abs_path:
                continue
            candidates.append(abs_path)

    if not candidates:
        # raise exception
        raise FileNotFoundError(
            f"Could not find 'best.pt' after training. Searched in '{search_roots}' and current directory, excluding '{project_path}' and backups."
        )

    # Most recently modified wins
    candidates.sort(key=os.path.getmtime, reverse=True)
    best_pt = candidates[0]

    # The run directory is two levels up from best.pt: <run_dir>/weights/best.pt
    src_run_dir = os.path.dirname(os.path.dirname(best_pt))
    dst_run_dir = os.path.join(project_path, run_name)

    try:
        os.makedirs(project_path, exist_ok=True)
        if os.path.exists(dst_run_dir):
            shutil.rmtree(dst_run_dir)
        shutil.move(src_run_dir, dst_run_dir)
        print(f"Moved YOLO training output: '{src_run_dir}' -> '{dst_run_dir}'")

        # Best-effort cleanup of now-empty ancestor dirs under runs_root
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


# ~ # check whether settings have been changed, and motion annotation library needs rebuilding
# ~ settings_changed = config_watcher.check_settings_changed(current_config_path=config_path, saved_config_path=None, model_dirs=['model_primary_motion'])
# ~ # Globals for prompting/behaviour inside maybe_retrain
# ~ regen_prompt_shown = False
# ~ force_rebuild_motion = False


global_response = 0  # if 'yes' is selected for any model re-training, retraining should be perfoemd for all models


def count_images_in_dataset(path):
    ## Count images in a dataset, handling both YAML-based and directory-based datasets
    # If path is a YAML file (primary models)
    if path.endswith(".yaml"):
        try:
            import yaml

            with open(path, "r") as f:
                data = yaml.safe_load(f)

            # Get the path to the training images
            train_path = data["train"]
            base_dir = os.path.dirname(path)
            abs_train_path = os.path.join(base_dir, train_path)

            # Handle different dataset formats
            if abs_train_path.endswith(".txt"):
                # Text file with image paths
                with open(abs_train_path, "r") as f:
                    return len(f.readlines())
            else:
                # Directory with images
                image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
                return len(
                    [
                        f
                        for f in os.listdir(abs_train_path)
                        if os.path.splitext(f)[1].lower() in image_exts
                    ]
                )
        except Exception as e:
            print(f"Error counting images: {e}")
            return 0

    # If path is a directory (secondary models)
    elif os.path.isdir(path):
        total_count = 0
        image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

        # Walk through all subdirectories
        for root, dirs, files in os.walk(path):
            # Only count files in leaf directories (class directories)
            if not dirs:  # This is a leaf directory (no subdirectories)
                count = sum(
                    1 for f in files if os.path.splitext(f)[1].lower() in image_exts
                )
                total_count += count

        return total_count

    else:
        print(f"Unsupported dataset format: {path}")
        return 0


def maybe_retrain(
    model_type, yaml_path, project_path, model_path, classifier, epochs, imgsz
):
    """
    Decide whether to (re)train a model based on existence and image counts.
    - If model_path exists and the recorded train_count differs from the current dataset,
      prompt the user to retrain (Yes/No).
    - If model_path does not exist, perform first-time training.
    Returns True if a training run was performed, False otherwise.
    """

    # If model exists: compare recorded image count (train_count.txt) with current dataset
    if os.path.exists(model_path):
        if os.path.exists(os.path.join(project_path, "train_count.txt")):
            try:
                with open(os.path.join(project_path, "train_count.txt"), "r") as f:
                    last_count = int(f.read().strip())
            except Exception:
                last_count = -1
        else:
            last_count = -1

        current_count = count_images_in_dataset(yaml_path)

        if current_count != last_count:
            # Ask user whether to retrain
            root = tk.Tk()
            root.withdraw()
            msg = (
                f"New annotations detected for '{model_type}' model.\n"
                f"Image count changed from {last_count} to {current_count}.\n\n"
                "Do you want to re-train this model?"
            )
            response = messagebox.askyesno("Retrain model?", msg)
            root.destroy()

            if response:
                # Backup existing model dir/project and retrain from its weights
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
                # Update saved train count
                with open(os.path.join(project_path, "train_count.txt"), "w") as f:
                    f.write(str(current_count))
                # copy existing settings ini file for reference (so you know which settings were used for each model)
                os.makedirs(project_path, exist_ok=True)
                # ~ dst = os.path.join(project_path, os.path.basename(config_path))
                dst = os.path.join(project_path, "saved_settings.ini")
                try:
                    shutil.copy2(params["config_path"], dst)
                    print(f"Saved settings snapshot to {dst}")
                except Exception as e:
                    print(f"Warning: could not copy settings to model dir: {e}")
                return True

        # else counts match -> nothing to do
        return False

    else:
        # Model missing -> do first-time training
        print(f"{model_type} model not found, building it...")
        current_count = count_images_in_dataset(yaml_path)
        if current_count < 2:
            print(
                f"Error: Not enough images to train {model_type} model (found {current_count}, need at least 2)."
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

        current_count = count_images_in_dataset(yaml_path)
        os.makedirs(project_path, exist_ok=True)
        with open(os.path.join(project_path, "train_count.txt"), "w") as f:
            f.write(str(current_count))

        # copy existing settings ini file for reference (so you know which settings were used for each model)
        os.makedirs(project_path, exist_ok=True)
        # ~ dst = os.path.join(project_path, os.path.basename(config_path))
        dst = os.path.join(project_path, "saved_settings.ini")
        try:
            shutil.copy2(params["config_path"], dst)
            print(f"Saved settings snapshot to {dst}")
        except Exception as e:
            print(f"Warning: could not copy settings to model dir: {e}")

        return True


# Train secondary classifiers for each static class


secondary_static_models = None
secondary_motion_models = None

if params["hierarchical_mode"]:
    secondary_static_models = {}
    static_class_map = [
        [None] * len(params["secondary_classes"])
        for _ in range(len(params["primary_classes"]))
    ]
    if len(params["secondary_static_classes"]) >= 2:
        for primary_class in params["primary_classes"]:
            idx = params["primary_classes"].index(primary_class)
            hotkey = params["primary_hotkeys"][idx]
            if hotkey in params["secondary_hotkeys"]:
                continue

            if primary_class in params["ignore_secondary"]:
                continue

            data_dir = os.path.join(params["secondary_static_data_path"], primary_class)
            # Skip if directory doesn't exist
            if not os.path.isdir(data_dir):
                continue

            # Create model directory for this static class
            model_dir = f"model_static_static_{primary_class}"
            weights_path = os.path.join(model_dir, "train", "weights", "best.pt")

            n_image = count_images_in_dataset(data_dir)

            if n_image < 2:
                print(
                    f"Error: Not enough images to train secondary motion model for primary class '{primary_class}' (found {n_image}, need at least 2). Skipping this secondary model."
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

            # Load the trained model
            if params["use_ncnn"] == "true":
                secondary_static_models[primary_class] = (
                    load_model_with_ncnn_preference(weights_path, "classify")
                )
            else:
                secondary_static_models[primary_class] = YOLO(weights_path)

        # ~ print(f"secondary_static_models {secondary_static_models}")

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

            data_dir = os.path.join(params["secondary_motion_data_path"], primary_class)
            # Skip if directory doesn't exist
            if not os.path.isdir(data_dir):
                continue

            # Create model directory for this static class
            model_dir = f"model_secondary_motion_{primary_class}"
            weights_path = os.path.join(model_dir, "train", "weights", "best.pt")

            n_image = count_images_in_dataset(data_dir)

            if n_image < 2:
                print(
                    f"Error: Not enough images to train secondary motion model for primary class '{primary_class}' (found {n_image}, need at least 2). Skipping this secondary model."
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

            # Load the trained model
            if params["use_ncnn"] == "true":
                secondary_motion_models[primary_class] = (
                    load_model_with_ncnn_preference(weights_path, "classify")
                )
            else:
                secondary_motion_models[primary_class] = YOLO(weights_path)

        # ~ print(f"secondary_motion_models {secondary_motion_models}")

# -------CHECK PRIMARY MODEL EXISTS----------
if params["primary_static_classes"][0] != "0":
    maybe_retrain(
        "primary static",
        params["primary_static_yaml_path"],
        params["primary_static_project_path"],
        params["primary_static_model_path"],
        params["primary_classifier"],
        params["primary_epochs"],
        640,
    )


if params["primary_motion_classes"][0] != "0":
    maybe_retrain(
        "primary motion",
        params["primary_motion_yaml_path"],
        params["primary_motion_project_path"],
        params["primary_motion_model_path"],
        params["primary_classifier"],
        params["primary_epochs"],
        640,
    )

# iou function that returns the larger proportional overlap of box1 and box2, relative to their own areas. This way if one box is entirely inside another, it will return 1.0, rather than a smaller value as with traditional iou.


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
    # return the larger proportional overlap - e.g. if one box is entirely inside another, this will return a 1.0, whereas the previous wouldn't
    if prop1 > prop2:
        return prop1 if prop1 > 0 else 0
    else:
        return prop2 if prop2 > 0 else 0
    # ~ union = area1 + area2 - inter
    # ~ return inter/union if union > 0 else 0


# --- TRACKER CLASS -------------------------------------------------------
class KalmanTracker:
    def __init__(self, dist_thresh, max_missed):
        self.next_id = 1
        self.tracks = {}  # tid -> {'kf': KalmanFilter, 'missed': int}
        self.prev_positions = {}  # Track previous positions
        self.dist_thresh = dist_thresh
        self.max_missed = max_missed

    def _create_kf(self, initial_pt):

        # Create a 4D state (x, y, vx, vy) Kalman Filter measuring (x, y).
        kf = cv2.KalmanFilter(4, 2)
        # State transition: x' = x + vx, y' = y + vy
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        # Measurement: we only observe x, y
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        # Tune these covariances to your scene

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
        # Initialize state
        kf.statePre = np.array(
            [[initial_pt[0]], [initial_pt[1]], [0.0], [0.0]], dtype=np.float32
        )
        kf.statePost = kf.statePre.copy()
        return kf

    def predict_all(self):
        """
        Predict the next position for every track.
        Returns list of (tid, predicted_pt).
        """
        preds = []
        for tid, tr in self.tracks.items():
            pred = tr["kf"].predict()
            preds.append((tid, (float(pred[0, 0]), float(pred[1, 0]))))
        return preds

    def _prune_duplicate_tracks(self):
        """
        Merge any two tracks whose current posteriors are very close.
        Call this at the end of update().
        """
        tids = list(self.tracks.keys())
        posts = {}
        for tid in tids:
            sp = self.tracks[tid]["kf"].statePost
            posts[tid] = (float(sp[0, 0]), float(sp[1, 0]))
        to_drop = set()
        for i, t1 in enumerate(tids):
            x1, y1 = posts[t1]
            for t2 in tids[i + 1 :]:
                x2, y2 = posts[t2]
                if np.hypot(x1 - x2, y1 - y2) < self.dist_thresh * 0.5:
                    # mark the higher ID for deletion
                    to_drop.add(max(t1, t2))
        for tid in to_drop:
            del self.tracks[tid]

    def update(self, detections):

        # detections: list of (x, y) centroids
        # Returns a dict: detection_index -> track_id

        # 1) Predict all tracks forward one step
        preds = self.predict_all()  # list of (tid, (px, py))
        track_ids = [t[0] for t in preds]
        pred_pts = [t[1] for t in preds]

        # 2) Build cost matrix = Euclidean distance
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

        # 3) Associate tracks ↔ detections
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < self.dist_thresh:
                tid = track_ids[r]
                matched_tracks.add(tid)
                matched_dets.add(c)
                assigned_detects[c] = tid

                # Get the measurement point
                dpt = detections[c]
                meas = np.array([[np.float32(dpt[0])], [np.float32(dpt[1])]])

                # Correct KF with the detection measurement
                self.tracks[tid]["kf"].correct(meas)
                self.tracks[tid]["missed"] = 0

                # Update previous position
                self.prev_positions[tid] = (dpt[0], dpt[1])

        # 4) Process unassigned detections
        for i, dpt in enumerate(detections):
            if i in matched_dets:
                continue

            # try to find an existing track under the threshold
            best_tid, best_dist = None, float("inf")
            for tid, (px, py) in preds:
                d = np.hypot(dpt[0] - px, dpt[1] - py)
                if d < best_dist:
                    best_dist, best_tid = d, tid

            if best_dist < self.dist_thresh:
                assigned_detects[i] = best_tid
                self.tracks[best_tid]["missed"] = 0
                meas = np.array([[np.float32(dpt[0])], [np.float32(dpt[1])]])
                self.tracks[best_tid]["kf"].correct(meas)

                # Update previous position
                self.prev_positions[best_tid] = (dpt[0], dpt[1])
                matched_tracks.add(best_tid)  # Add to matched tracks

            else:
                # New track
                tid = self.next_id
                kf = self._create_kf(dpt)
                self.tracks[tid] = {"kf": kf, "missed": 0}
                assigned_detects[i] = tid
                self.prev_positions[tid] = (dpt[0], dpt[1])  # Initialize position
                matched_tracks.add(tid)  # Add to matched tracks
                self.next_id += 1

        # 5) Handle unmatched tracks
        for tid in list(self.tracks.keys()):
            if tid not in matched_tracks:
                self.tracks[tid]["missed"] += 1
                # Increase uncertainty when missing detections
                noise_scale = min(2.0, 1.0 + self.tracks[tid]["missed"] * 0.2)

                # FIXED: Preserve matrix type and structure
                kf = self.tracks[tid]["kf"]
                new_noise = kf.processNoiseCov.copy()
                new_noise *= noise_scale
                kf.processNoiseCov = new_noise

                # Remove track if missed too many times
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]
                    if tid in self.prev_positions:
                        del self.prev_positions[tid]

        return assigned_detects


# --- MAIN PROCESSING -----------------------------------------------------
def process_video(file):
    os.makedirs(params["output_folder"], exist_ok=True)
    base = os.path.splitext(os.path.basename(file))[0]
    cap = cv2.VideoCapture(file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened():
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * params["scale_factor"])
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * params["scale_factor"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(
        os.path.join(params["output_folder"], base + "_detected.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    if params["primary_static_classes"][0] != "0":
        if params["use_ncnn"] == "true":
            model_static = load_model_with_ncnn_preference(
                params["primary_static_model_path"], "detect"
            )
        else:
            model_static = YOLO(params["primary_static_model_path"])

    if params["primary_motion_classes"][0] != "0":
        if params["use_ncnn"] == "true":
            model_motion = load_model_with_ncnn_preference(
                params["primary_motion_model_path"], "detect"
            )
        else:
            model_motion = YOLO(params["primary_motion_model_path"])

    tracker = KalmanTracker(
        params["match_distance_thresh"], params["delete_after_missed"]
    )

    prev_frames, frame_idx = None, 0
    csv_file = open(
        os.path.join(params["output_folder"], base + "_tracking.csv"), "w", newline=""
    )
    csv_writer = csv.writer(csv_file)
    # Updated CSV header with four streams
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

    frame_count = 0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_count == 0:
            if params["scale_factor"] != 1.0:
                raw_frame = cv2.resize(
                    raw_frame,
                    None,
                    fx=params["scale_factor"],
                    fy=params["scale_factor"],
                )
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            frame = raw_frame.copy()
            if prev_frames is None:
                prev_frames = [gray.copy() for _ in range(3)]
                continue

            # only process motion information if necessary
            if params["primary_motion_classes"][0] != "0":
                diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]

                if params["strategy"] == "exponential":
                    prev_frames[0] = gray
                    prev_frames[1] = cv2.addWeighted(
                        prev_frames[1], params["expA"], gray, params["expA2"], 0
                    )
                    prev_frames[2] = cv2.addWeighted(
                        prev_frames[2], params["expB"], gray, params["expB2"], 0
                    )
                elif params["strategy"] == "sequential":
                    prev_frames[2] = prev_frames[1]
                    prev_frames[1] = prev_frames[0]
                    prev_frames[0] = gray

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

            # Collect all primary detections
            all_detections = []

            # Primary static detection
            if params["primary_static_classes"][0] != "0":
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

            # Primary motion detection
            if params["primary_motion_classes"][0] != "0":
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

            # Merge detections based on proximity
            merged_detections = []
            for det in all_detections:
                x1, y1, x2, y2 = det["coords"]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Find matching existing detection
                matched = False
                for md in merged_detections:
                    md_cx, md_cy = md["centroid"]
                    dist = np.hypot(cx - md_cx, cy - md_cy)

                    # Calculate IOU
                    md_x1, md_y1, md_x2, md_y2 = md["coords"]
                    overlap = iou((x1, y1, x2, y2), (md_x1, md_y1, md_x2, md_y2))
                    ms_source = md["source"]

                    if (
                        dist < params["centroid_merge_thresh"]
                        or overlap > params["iou_thresh"]
                    ):
                        # Merge classes - keep highest confidence detection for each source
                        if (
                            det["source"] == ms_source
                            or params["dominant_source"] == "confidence"
                        ):  # mathcing sources so select best, or confidence strategy used
                            if det["source"] == "static":
                                # Keep highest confidence static detection
                                if (
                                    "primary_conf" not in md
                                    or det["primary_conf"] > md["primary_conf"]
                                ):
                                    md["primary_class_combined"] = md[
                                        "primary_class"
                                    ]  # retain the combined primary
                                    md["primary_conf_combined"] = md["primary_conf"]
                                    md["primary_class"] = det["primary_class"]
                                    md["primary_conf"] = det["primary_conf"]
                                    md["coords"] = det[
                                        "coords"
                                    ]  # Update to higher conf box
                                    md["centroid"] = (cx, cy)
                                    md["source"] = det["source"]
                            else:  # motion source
                                # Keep highest confidence motion detection
                                if (
                                    "primary_conf" not in md
                                    or det["primary_conf"] > md["primary_conf"]
                                ):
                                    md["primary_class_combined"] = md[
                                        "primary_class"
                                    ]  # retain the combined primary
                                    md["primary_conf_combined"] = md["primary_conf"]
                                    md["primary_class"] = det["primary_class"]
                                    md["primary_conf"] = det["primary_conf"]
                                    md["coords"] = det[
                                        "coords"
                                    ]  # Update to higher conf box
                                    md["centroid"] = (cx, cy)
                                    md["source"] = det["source"]
                        elif (
                            det["source"] == "static"
                            and params["dominant_source"] == "static"
                        ):
                            # Keep static detection
                            md["primary_class_combined"] = md[
                                "primary_class"
                            ]  # retain the combined primary
                            md["primary_conf_combined"] = md["primary_conf"]
                            md["primary_class"] = det["primary_class"]
                            md["primary_conf"] = det["primary_conf"]
                            md["coords"] = det["coords"]  # Update to higher conf box
                            md["centroid"] = (cx, cy)
                            md["source"] = det["source"]
                        elif (
                            det["source"] == "motion"
                            and params["dominant_source"] == "motion"
                        ):
                            # Keep motion detection
                            md["primary_class_combined"] = md[
                                "primary_class"
                            ]  # retain the combined primary
                            md["primary_conf_combined"] = md["primary_conf"]
                            md["primary_class"] = det["primary_class"]
                            md["primary_conf"] = det["primary_conf"]
                            md["coords"] = det["coords"]  # Update to higher conf box
                            md["centroid"] = (cx, cy)
                            md["source"] = det["source"]

                        matched = True
                        break

                if not matched:
                    # Add as new detection
                    new_det = {
                        "coords": det["coords"],
                        "centroid": (cx, cy),
                        "source": det["source"],
                        "primary_class_combined": "",
                        "primary_conf_combined": 0.0,
                    }
                    if det["source"] == "static":
                        new_det["primary_class"] = det["primary_class"]
                        new_det["primary_conf"] = det["primary_conf"]
                        # ~ if 'secondary_static_class' in det:
                        # ~ new_det['secondary_static_class'] = det['secondary_static_class']
                        # ~ new_det['secondary_static_conf'] = det['secondary_static_conf']
                    else:  # motion source
                        new_det["primary_class"] = det["primary_class"]
                        new_det["primary_conf"] = det["primary_conf"]
                        # ~ if 'secondary_motion_class' in det:
                        # ~ new_det['secondary_motion_class'] = det['secondary_motion_class']
                        # ~ new_det['secondary_motion_conf'] = det['secondary_motion_conf']
                    merged_detections.append(new_det)

            # Run secondary classification on each primary detection
            processed_detections = []
            for det in merged_detections:
                coords = det["coords"]
                primary_class = det["primary_class"]
                primary_conf = det["primary_conf"]
                source = det["source"]
                primary_class_combined = det["primary_class_combined"]
                primary_conf_combined = det["primary_conf_combined"]

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

                    # Determine which secondary model to use based on source and configuration
                    sec_model = None
                    crop_img = None

                    if source == "static":
                        # Use static secondary model if configured
                        if len(params["secondary_static_classes"]) >= 2:
                            sec_model = secondary_static_models.get(primary_class, None)
                            crop_img = frame
                        # Fallback to motion secondary model if static not available
                        elif len(params["secondary_motion_classes"]) >= 2:
                            sec_model = secondary_motion_models.get(primary_class, None)
                            crop_img = (
                                motion_image
                                if params["primary_motion_classes"][0] != "0"
                                else frame
                            )
                    else:  # motion source
                        # Use motion secondary model if configured
                        if len(params["secondary_motion_classes"]) >= 2:
                            sec_model = secondary_motion_models.get(primary_class, None)
                            crop_img = motion_image
                        # Fallback to static secondary model if motion not available
                        elif len(params["secondary_static_classes"]) >= 2:
                            sec_model = secondary_static_models.get(primary_class, None)
                            crop_img = frame

                    # Get the cropped region
                    crop = None
                    if crop_img is not None:
                        crop = crop_img[y1:y2, x1:x2]

                    secondary_class = primary_class
                    secondary_conf = 1.0

                    # Run secondary classification if we have a model and valid crop
                    if sec_model and crop is not None and crop.size > 0:
                        sec_results = sec_model.predict(crop, verbose=False)
                        if sec_results[0].probs is not None:
                            secondary_class_idx = sec_results[0].probs.top1
                            secondary_conf = sec_results[0].probs.top1conf.item()
                            secondary_class = sec_model.names[secondary_class_idx]

                    # Add secondary results to detection
                    if source == "static":
                        det["secondary_static_class"] = secondary_class
                        det["secondary_static_conf"] = secondary_conf
                    else:  # motion source
                        det["secondary_motion_class"] = secondary_class
                        det["secondary_motion_conf"] = secondary_conf

                processed_detections.append(det)

            # Prepare for tracking
            cents = [d["centroid"] for d in processed_detections]
            assignment = tracker.update(cents)

            # ~ frame = motion_image ## enable this line ot save the motion video instead of static

            # Process tracked objects
            for idx, det in enumerate(processed_detections):
                tid = assignment.get(idx, None)
                if tid is None or tid not in tracker.tracks:
                    continue

                x1, y1, x2, y2 = det["coords"]
                cx, cy = det["centroid"]

                # Get all class info with default values
                ps_class = det.get("primary_static_class", "")
                ps_conf = det.get("primary_static_conf", 0)
                pm_class = det.get("primary_motion_class", "")
                pm_conf = det.get("primary_motion_conf", 0)
                ss_class = det.get("secondary_static_class", "")
                ss_conf = det.get("secondary_static_conf", 0)
                sm_class = det.get("secondary_motion_class", "")
                sm_conf = det.get("secondary_motion_conf", 0)
                p_source = det.get("source", "")

                # Create display label
                label_parts = []
                # ~ if ps_class:
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

                if params["hierarchical_mode"]:
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
                        # Draw outer static box (slightly larger)
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
                        frame, (x1, y1), (x2, y2), primary_col, params["line_thickness"]
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

                # Draw motion vector (if tracking available)
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

                # Write to CSV
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

            # ~ # print frame number
            text_color = (255, 255, 255)  # white text
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

        frame_count += 1

        if frame_count > params["frame_skip"]:
            frame_count = 0

    cap.release()
    writer.release()
    csv_file.close()
    print(f"Done processing {base} | {current_fps:.1f} FPS")


if __name__ == "__main__":
    for vid in glob.glob(os.path.join(params["input_folder"], "*.*")):
        process_video(vid)
