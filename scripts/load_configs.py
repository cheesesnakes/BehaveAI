import configparser
import os
import sys
import tkinter as tk
from tkinter import filedialog

# ---------- Tracker configuration ----------------------------------------

# Every knob the BoxMOT adapter accepts, with the default used when the key is
# absent from the INI. Keys live in a [tracker] section; the older [kalman]
# section is still read as a fallback so existing project files keep working.
#
#   name -> (default, type)
TRACKER_DEFAULTS = {
    # Backend. "builtin" uses the bundled KalmanTracker (no torch dependency).
    "tracker_type": ("ocsort", str),
    # Confidence YOLO itself is run at. Must be BELOW det_thresh, otherwise the
    # tracker's low-score recovery pass has nothing to recover: the detector
    # will already have thrown those boxes away.
    "det_conf_floor": (0.05, float),
    # The tracker's high/low confidence split.
    "det_thresh": (0.25, float),
    # Processed frames a track survives unmatched.
    "max_age": (45, int),
    # Detections required before a track is written to the CSV.
    "min_hits": (3, int),
    # Association IoU gate.
    "iou_threshold": (0.20, float),
    # Separate track pools per primary class.
    "per_class": (False, bool),
    "device": ("cpu", str),
    "half": (False, bool),
    # Only loaded by the ReID trackers.
    "reid_weights": ("osnet_x0_25_msmt17.pt", str),
    # Centroid history length used to derive vx/vy.
    "velocity_window": (5, int),
    # Abort rather than continue when the chosen backend silently ignores a
    # critical setting above.
    "strict_kwargs": (True, bool),
    "verbose": (True, bool),
}

# Backends that support a second, low-confidence association pass. For anything
# else, det_conf_floor is meaningless and the detector runs at
# primary_conf_thresh instead.
TWO_STAGE_TRACKERS = {
    "bytetrack",
    "botsort",
    "ocsort",
    "deepocsort",
    "hybridsort",
    "boosttrack",
    "occluboost",
    "strongsort",
    "sfsort",
}


def _cfg_bool(value, fallback=False):
    """INI booleans, tolerant of the true/yes/1/on family."""
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def read_tracker_params(config):
    """
    Read the [tracker] section into a plain dict, falling back to [kalman] for
    keys that used to live there, then to TRACKER_DEFAULTS.
    """
    section = config["tracker"] if config.has_section("tracker") else {}
    legacy = config["kalman"] if config.has_section("kalman") else {}

    out = {}
    for key, (default, kind) in TRACKER_DEFAULTS.items():
        raw = section.get(key, legacy.get(key, None))
        if raw is None or str(raw).strip() == "":
            out[key] = default
            continue
        try:
            if kind is bool:
                out[key] = _cfg_bool(raw, default)
            else:
                out[key] = kind(str(raw).strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"[tracker] {key}: could not read '{raw}' as {kind.__name__}"
            )

    out["tracker_type"] = out["tracker_type"].strip().lower()
    return out


# ---------- Project-aware configuration loading --------------------------


def pick_ini_via_dialog():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select BehaveAI settings INI",
        filetypes=[("INI files", "*.ini"), ("All files", "*.*")],
    )
    root.destroy()
    return path


# Determine config_path (accept project dir or direct INI path)


def get_config_path():

    if len(sys.argv) > 1:
        arg = os.path.abspath(sys.argv[1])
        if os.path.isdir(arg):
            config_path = os.path.join(arg, "BehaveAI_settings.ini")
        else:
            config_path = arg
    else:
        config_path = pick_ini_via_dialog()
        if config_path is None:
            tk.messagebox.showinfo(
                "No settings file", "No settings INI selected — exiting."
            )
            sys.exit(0)

    if config_path is not None:
        config_path = os.path.abspath(config_path)
    else:
        tk.messagebox.showinfo(
            "No settings file", "No settings INI selected — exiting."
        )
        sys.exit(0)

    if not os.path.exists(config_path):
        tk.messagebox.showerror(
            "Missing settings", f"Configuration file not found: {config_path}"
        )
        sys.exit(1)

    return config_path


# Set project directory to the INI parent and make it the working directory


def set_project_directory(config_path):
    project_dir = os.path.dirname(config_path)
    os.chdir(project_dir)
    print(f"Working directory set to project dir: {project_dir}")
    print(f"Using settings file: {config_path}")
    return project_dir


def load_configs(config_path):
    # Load configuration
    config = configparser.ConfigParser()
    config.optionxform = str  # keep case
    config.read(config_path)
    return config


# Helper: resolve a path from INI (absolute or relative to project_dir)
def resolve_project_path(project_dir, value, fallback):
    if value is None or str(value).strip() == "":
        value = fallback
    value = str(value)
    if os.path.isabs(value):
        return os.path.normpath(value)
    return os.path.normpath(os.path.join(project_dir, value))


# Read dataset / directory keys from INI (defaults are relative names inside the project)


def setup_directories():
    global \
        clips_dir, \
        input_folder, \
        output_folder, \
        ANNOTATION_FOLDER, \
        MODEL_FOLDER, \
        config, \
        config_path, \
        project_dir

    config_path = get_config_path()
    project_dir = set_project_directory(config_path)
    config = load_configs(config_path)
    clips_dir_ini = config["DEFAULT"].get("clips_dir", "clips")
    input_dir_ini = config["DEFAULT"].get("input_dir", "input")
    output_dir_ini = config["DEFAULT"].get("output_dir", "output")

    clips_dir = resolve_project_path(project_dir, clips_dir_ini, "clips")
    input_folder = resolve_project_path(project_dir, input_dir_ini, "input")
    output_folder = resolve_project_path(project_dir, output_dir_ini, "output")
    # Define model and annotation folders relative to the project directory
    ANNOTATION_FOLDER = os.path.join(project_dir, "annotations")
    MODEL_FOLDER = os.path.join(project_dir, "models")

    os.makedirs(ANNOTATION_FOLDER, exist_ok=True)
    os.makedirs(MODEL_FOLDER, exist_ok=True)

    return 0


# Read parameters
def read_parameters():
    params = {}
    params["clips_dir"] = clips_dir
    params["input_folder"] = input_folder
    params["output_folder"] = output_folder
    params["annotation_folder"] = ANNOTATION_FOLDER
    params["model_folder"] = MODEL_FOLDER
    params["config_path"] = config_path
    params["project_dir"] = project_dir
    params["progress_update"] = int(config["DEFAULT"].get("progress_update", "10"))

    try:
        params["primary_motion_classes"] = [
            name.strip()
            for name in config["DEFAULT"]["primary_motion_classes"].split(",")
        ]
        cols = [
            c.strip()
            for c in config["DEFAULT"].get("primary_motion_colors", "").split(";")
            if c.strip()
        ]
        params["primary_motion_colors"] = [
            tuple(map(int, c.split(",")))[::-1] for c in cols
        ]
        params["primary_motion_hotkeys"] = [
            key.strip()
            for key in config["DEFAULT"]["primary_motion_hotkeys"].split(",")
        ]

        params["secondary_motion_classes"] = [
            name.strip()
            for name in config["DEFAULT"]["secondary_motion_classes"].split(",")
        ]
        cols = [
            c.strip()
            for c in config["DEFAULT"].get("secondary_motion_colors", "").split(";")
            if c.strip()
        ]
        params["secondary_motion_colors"] = [
            tuple(map(int, c.split(",")))[::-1] for c in cols
        ]
        params["secondary_motion_hotkeys"] = [
            key.strip()
            for key in config["DEFAULT"]["secondary_motion_hotkeys"].split(",")
        ]

        params["primary_static_classes"] = [
            name.strip()
            for name in config["DEFAULT"]["primary_static_classes"].split(",")
        ]
        cols = [
            c.strip()
            for c in config["DEFAULT"].get("primary_static_colors", "").split(";")
            if c.strip()
        ]
        params["primary_static_colors"] = [
            tuple(map(int, c.split(",")))[::-1] for c in cols
        ]
        params["primary_static_hotkeys"] = [
            key.strip()
            for key in config["DEFAULT"]["primary_static_hotkeys"].split(",")
        ]

        params["secondary_static_classes"] = [
            name.strip()
            for name in config["DEFAULT"]["secondary_static_classes"].split(",")
        ]
        cols = [
            c.strip()
            for c in config["DEFAULT"].get("secondary_static_colors", "").split(";")
            if c.strip()
        ]
        params["secondary_static_colors"] = [
            tuple(map(int, c.split(",")))[::-1] for c in cols
        ]
        params["secondary_static_hotkeys"] = [
            key.strip()
            for key in config["DEFAULT"]["secondary_static_hotkeys"].split(",")
        ]

        params["static_train_images_dir"] = (
            f"{ANNOTATION_FOLDER}/annot_static/images/train"
        )
        params["static_val_images_dir"] = f"{ANNOTATION_FOLDER}/annot_static/images/val"
        params["static_train_labels_dir"] = (
            f"{ANNOTATION_FOLDER}/annot_static/labels/train"
        )
        params["static_val_labels_dir"] = f"{ANNOTATION_FOLDER}/annot_static/labels/val"

        params["motion_train_images_dir"] = (
            f"{ANNOTATION_FOLDER}/annot_motion/images/train"
        )
        params["motion_val_images_dir"] = f"{ANNOTATION_FOLDER}/annot_motion/images/val"
        params["motion_train_labels_dir"] = (
            f"{ANNOTATION_FOLDER}/annot_motion/labels/train"
        )
        params["motion_val_labels_dir"] = f"{ANNOTATION_FOLDER}/annot_motion/labels/val"
        params["primary_static_external_model"] = (
            config["DEFAULT"].get("primary_static_external_model", "").strip()
        )
        params["secondary_static_external_model"] = (
            config["DEFAULT"].get("secondary_static_external_model", "").strip()
        )

        # pseudo-labelling parameters
        params["primary_static_pseudo_labeling"] = config["DEFAULT"].get(
            "primary_static_pseudo_labeling", False
        )
        params["secondary_static_pseudo_labeling"] = config["DEFAULT"].get(
            "secondary_static_pseudo_labeling", False
        )
        if (
            len(params["secondary_motion_classes"]) >= 2
            or len(params["secondary_static_classes"]) >= 2
            or params["secondary_static_external_model"] != ""
        ):
            params["hierarchical_mode"] = True
            params["motion_cropped_base_dir"] = f"{ANNOTATION_FOLDER}/annot_motion_crop"
            params["static_cropped_base_dir"] = f"{ANNOTATION_FOLDER}/annot_static_crop"

            # secondary classes need more than one value, so clear if there's only one value
            if len(params["secondary_motion_classes"]) == 1:
                params["secondary_motion_classes"] = []
                params["secondary_motion_colors"] = []
                params["secondary_motion_hotkeys"] = []

            if len(params["secondary_static_classes"]) == 1:
                params["secondary_static_classes"] = []
                params["secondary_static_colors"] = []
                params["secondary_static_hotkeys"] = []

        else:
            params["hierarchical_mode"] = False

        params["primary_classes"] = (
            params["primary_static_classes"] + params["primary_motion_classes"]
        )
        params["primary_colors"] = (
            params["primary_static_colors"] + params["primary_motion_colors"]
        )
        params["primary_hotkeys"] = (
            params["primary_static_hotkeys"] + params["primary_motion_hotkeys"]
        )

        params["secondary_classes"] = (
            params["secondary_static_classes"] + params["secondary_motion_classes"]
        )
        params["secondary_colors"] = (
            params["secondary_static_colors"] + params["secondary_motion_colors"]
        )
        params["secondary_hotkeys"] = (
            params["secondary_static_hotkeys"] + params["secondary_motion_hotkeys"]
        )
        params["primary_static_project_path"] = f"{MODEL_FOLDER}/model_primary_static"
        params["primary_static_model_path"] = os.path.join(
            f"{MODEL_FOLDER}/model_primary_static", "train", "weights", "best.pt"
        )
        params["primary_static_yaml_path"] = (
            f"{ANNOTATION_FOLDER}/static_annotations.yaml"
        )

        params["primary_motion_project_path"] = f"{MODEL_FOLDER}/model_primary_motion"
        params["primary_motion_model_path"] = os.path.join(
            f"{MODEL_FOLDER}/model_primary_motion", "train", "weights", "best.pt"
        )
        params["primary_motion_yaml_path"] = (
            f"{ANNOTATION_FOLDER}/motion_annotations.yaml"
        )

        params["ignore_secondary"] = [
            name.strip() for name in config["DEFAULT"]["ignore_secondary"].split(",")
        ]
        params["dominant_source"] = config["DEFAULT"]["dominant_source"].lower()

        params["primary_classifier"] = config["DEFAULT"].get(
            "primary_classifier", "yolo11s.pt"
        )
        params["primary_epochs"] = int(config["DEFAULT"].get("primary_epochs", "50"))
        params["secondary_classifier"] = config["DEFAULT"].get(
            "secondary_classifier", "yolo11s-cls.pt"
        )
        params["secondary_epochs"] = int(
            config["DEFAULT"].get("secondary_epochs", "50")
        )

        if params["hierarchical_mode"]:
            params["secondary_static_project_path"] = (
                f"{MODEL_FOLDER}/model_secondary_static"
            )
            params["secondary_static_data_path"] = (
                f"{ANNOTATION_FOLDER}/annot_static_crop"
            )
            params["secondary_static_model_path"] = os.path.join(
                f"{MODEL_FOLDER}/model_secondary_static", "train", "weights", "best.pt"
            )

            params["secondary_motion_project_path"] = (
                f"{MODEL_FOLDER}/model_secondary_motion"
            )
            params["secondary_motion_data_path"] = (
                f"{ANNOTATION_FOLDER}/annot_motion_crop"
            )
            params["secondary_motion_model_path"] = os.path.join(
                f"{MODEL_FOLDER}/model_secondary_motion", "train", "weights", "best.pt"
            )

            params["secondary_class_ids"] = list(
                range(len(params["secondary_classes"]))
            )
            paired = list(
                zip(
                    params["secondary_classes"],
                    params["secondary_colors"],
                    params["secondary_class_ids"],
                    params["secondary_hotkeys"],
                )
            )
            paired_sorted = sorted(paired, key=lambda x: x[0].lower())
            (
                secondary_classes,
                secondary_colors,
                secondary_class_ids,
                secondary_hotkeys,
            ) = zip(*paired_sorted)
            # Convert back to lists
            params["secondary_classes"] = list(secondary_classes)
            params["secondary_colors"] = list(secondary_colors)
            params["secondary_class_ids"] = list(secondary_class_ids)
            params["secondary_hotkeys"] = list(secondary_hotkeys)

        # Common parameters
        params["scale_factor"] = float(config["DEFAULT"].get("scale_factor", "1.0"))
        params["expA"] = float(config["DEFAULT"].get("expA", "0.5"))
        params["expB"] = float(config["DEFAULT"].get("expB", "0.8"))
        params["lum_weight"] = float(config["DEFAULT"].get("lum_weight", "0.7"))
        params["strategy"] = config["DEFAULT"].get("strategy", "exponential")
        params["chromatic_tail_only"] = config["DEFAULT"]["chromatic_tail_only"].lower()
        params["rgb_multipliers"] = [
            float(x) for x in config["DEFAULT"]["rgb_multipliers"].split(",")
        ]
        params["use_ncnn"] = config["DEFAULT"]["use_ncnn"].lower()
        params["primary_conf_thresh"] = float(
            config["DEFAULT"].get("primary_conf_thresh", "0.5")
        )
        params["secondary_conf_thresh"] = float(
            config["DEFAULT"].get("secondary_conf_thresh", "0.5")
        )
        params["match_distance_thresh"] = float(
            config["DEFAULT"].get("match_distance_thresh", "200")
        )
        params["delete_after_missed"] = float(
            config["DEFAULT"].get("delete_after_missed", "5")
        )
        params["centroid_merge_thresh"] = float(
            config["DEFAULT"].get("centroid_merge_thresh", "50")
        )
        params["iou_thresh"] = float(config["DEFAULT"].get("iou_thresh", "0.95"))
        params["line_thickness"] = int(config["DEFAULT"].get("line_thickness", "1"))
        params["font_size"] = float(config["DEFAULT"].get("font_size", "0.5"))
        params["frame_skip"] = int(config["DEFAULT"].get("frame_skip", "0"))
        params["motion_blocks_static"] = config["DEFAULT"][
            "motion_blocks_static"
        ].lower()
        params["static_blocks_motion"] = config["DEFAULT"][
            "static_blocks_motion"
        ].lower()
        params["save_empty_frames"] = config["DEFAULT"]["save_empty_frames"].lower()

        params["process_noise_pos"] = float(
            config["kalman"].get("process_noise_pos", "0.01")
        )
        params["process_noise_vel"] = float(
            config["kalman"].get("process_noise_vel", "0.1")
        )
        params["measurement_noise"] = float(
            config["kalman"].get("measurement_noise", "0.1")
        )
        params["motion_threshold"] = -1 * int(
            config["DEFAULT"].get("motion_threshold", "0")
        )

        # ---- tracker block --------------------------------------------
        params["tracker"] = read_tracker_params(config)
        # Kept at the top level for backwards compatibility with any code that
        # still reads params["tracker_type"] directly.
        params["tracker_type"] = params["tracker"]["tracker_type"]

        # Confidence the DETECTOR is actually run at. For a two-stage tracker
        # this is deliberately below the tracker's det_thresh so the low-score
        # association pass has candidates; otherwise there is no consumer for
        # the extra boxes and we stay at primary_conf_thresh.
        if params["tracker_type"] in TWO_STAGE_TRACKERS:
            params["detector_conf_thresh"] = min(
                params["tracker"]["det_conf_floor"],
                params["primary_conf_thresh"],
            )
        else:
            params["detector_conf_thresh"] = params["primary_conf_thresh"]

    except KeyError as e:
        raise KeyError(f"Missing configuration parameter: {e}")

    return params


# Validate configuration


def validate_configuration(params):

    if len(params["primary_motion_classes"]) != len(
        params["primary_motion_colors"]
    ) or len(params["primary_motion_classes"]) != len(params["primary_motion_hotkeys"]):
        raise ValueError(
            "Primary motion classes, colors and hotkeys must match in configuration."
        )
    if len(params["secondary_motion_classes"]) != len(
        params["secondary_motion_colors"]
    ) or len(params["secondary_motion_classes"]) != len(
        params["secondary_motion_hotkeys"]
    ):
        raise ValueError(
            "Secondary motion classes, colors and hotkeys must match in configuration."
        )
    if len(params["primary_static_classes"]) != len(
        params["primary_static_colors"]
    ) or len(params["primary_static_classes"]) != len(params["primary_static_hotkeys"]):
        raise ValueError(
            "Primary static classes, colors and hotkeys must match in configuration."
        )
    if len(params["secondary_static_classes"]) != len(
        params["secondary_static_colors"]
    ) or len(params["secondary_static_classes"]) != len(
        params["secondary_static_hotkeys"]
    ):
        raise ValueError(
            "Secondary static classes, colors and hotkeys must match in configuration."
        )
    if (
        params["dominant_source"] != "motion"
        and params["dominant_source"] != "static"
        and params["dominant_source"] != "confidence"
    ):
        raise ValueError("dominant_source must be motion, static, or confidence")

    if len(params["primary_static_classes"]) > 0:
        if not os.path.exists(params["primary_static_yaml_path"]):
            print(
                "Error: Primary static YAML file not found. Run the Annotation script once to fix this"
            )
            sys.exit(1)

    if len(params["primary_motion_classes"]) > 0:
        if not os.path.exists(params["primary_motion_yaml_path"]):
            print(
                "Error: Primary motion YAML file not found. Run the Annotation script once to fix this"
            )
            sys.exit(1)
    if params["motion_blocks_static"] not in ("true", "false"):
        raise ValueError("motion_blocks_static must be 'true' or 'false'")
    if params["static_blocks_motion"] not in ("true", "false"):
        raise ValueError("static_blocks_motion must be 'true' or 'false'")
    if params["save_empty_frames"] not in ("true", "false"):
        raise ValueError("save_empty_frames must be 'true' or 'false'")

    # ---- tracker block ------------------------------------------------
    tp = params["tracker"]
    known = set(TWO_STAGE_TRACKERS) | {"builtin"}
    if tp["tracker_type"] not in known:
        raise ValueError(
            f"[tracker] tracker_type '{tp['tracker_type']}' is not recognised. "
            f"Choose one of: {', '.join(sorted(known))}"
        )
    if not 0.0 < tp["det_thresh"] < 1.0:
        raise ValueError("[tracker] det_thresh must be between 0 and 1")
    if not 0.0 < tp["det_conf_floor"] < 1.0:
        raise ValueError("[tracker] det_conf_floor must be between 0 and 1")
    if tp["max_age"] < 1:
        raise ValueError("[tracker] max_age must be at least 1")
    if tp["min_hits"] < 1:
        raise ValueError("[tracker] min_hits must be at least 1")
    if not 0.0 <= tp["iou_threshold"] < 1.0:
        raise ValueError("[tracker] iou_threshold must be in [0, 1)")

    # The trap this whole block exists to prevent: if the detector filters at
    # or above the tracker's split, the low-score bin is permanently empty and
    # the second association pass never runs.
    if (
        tp["tracker_type"] in TWO_STAGE_TRACKERS
        and params["detector_conf_thresh"] >= tp["det_thresh"]
    ):
        print(
            f"Warning: detector confidence ({params['detector_conf_thresh']}) is "
            f"not below the tracker's det_thresh ({tp['det_thresh']}), so "
            f"{tp['tracker_type']}'s low-confidence recovery pass will never "
            f"fire. Lower det_conf_floor (and primary_conf_thresh) in your INI."
        )

    return True


# main function to load configs and return params dict


def load_params():
    setup_directories()
    params = read_parameters()
    if not validate_configuration(params):
        print("Configuration validation failed. Please check your settings.")
        sys.exit(1)
    return params
