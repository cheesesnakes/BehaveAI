import configparser
import os
import re
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

# ---------- Secondary-classifier defaults ---------------------------------
# The secondary crop tree is stored FLAT:
#
#     annot_motion_crop/<primary_class>/<subclass>/*.jpg
#
# classify_track.stage_split_dataset() builds a train/val view of it inside
# the model directory at training time. These keys steer that. Everything
# here has a working default, so an INI that predates them still loads.

# Everything in a crop filename before the first _id / _frame / _track marker
# is treated as the source video. Whole videos go to one side of the split, so
# near-duplicate frames from the same track cannot straddle train and val.
DEFAULT_VIDEO_REGEX = r"^(.*?)(?:_id\d+|_frame\d+|_track\d+)"

# Exponent on inverse class frequency for the training sampler.
#   0.0 -> off (natural frequencies)
#   0.5 -> sqrt-inverse (default)
#   1.0 -> full inverse; usually overfits the rarest class
DEFAULT_SAMPLER_POWER = 0.5


def _cfg_bool(value, fallback=False):
    """INI booleans, tolerant of the true/yes/1/on family."""
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _cfg_list(value):
    """Comma-separated INI value -> list of non-empty stripped strings."""
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


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
    #
    # interpolation=None: BasicInterpolation treats '%' as a reference marker,
    # so a regex or format string containing '%' in the INI would raise at
    # read time. secondary_video_regex is a regex, so interpolation is off.
    # Nothing in this project used %(name)s substitution.
    config = configparser.ConfigParser(interpolation=None)
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

        # pseudo-labelling parameters. Parsed as real booleans — these used to
        # be compared as raw strings at the call sites, which is how the
        # `== "False" == "True"` chained-comparison bug got in.
        params["primary_static_pseudo_labeling"] = _cfg_bool(
            config["DEFAULT"].get("primary_static_pseudo_labeling", "false"), False
        )
        params["secondary_static_pseudo_labeling"] = _cfg_bool(
            config["DEFAULT"].get("secondary_static_pseudo_labeling", "false"), False
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

        # NOTE ON DUPLICATES IN primary_classes
        # -------------------------------------
        # These are concatenations, not unions. When the same name appears in
        # both primary_static_classes and primary_motion_classes — which is
        # the normal case for a single-species project, e.g. "fish" detected
        # on both streams — primary_classes legitimately contains it twice,
        # and primary_colors / primary_hotkeys stay aligned with it index for
        # index. That alignment is why the duplicate is NOT removed here.
        #
        # Callers that iterate primary_classes to do per-class WORK (training
        # a secondary model, building a crop directory) must deduplicate
        # first — `dict.fromkeys(...)` preserves order — or they will do the
        # same work twice. classify_track.train_models() does exactly that.
        # Callers that index into the parallel colour/hotkey lists should keep
        # using the raw list.
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

        # ---- local vs external primary static model -------------------
        # ONE decision, consumed by both train_models() and process_video().
        # Previously each computed its own version of this and they disagreed:
        # training was skipped because an external model was configured, while
        # inference still looked for the local weights that were never built,
        # so the static stream was silently dropped from every video.
        #
        #   no external model            -> train and use the local model
        #   external model, no pseudo    -> use the external model directly
        #   external model + pseudo      -> the external model labels the data,
        #                                   so a local model is trained and used
        params["use_local_static_model"] = (
            params["primary_static_external_model"] == ""
            or params["primary_static_pseudo_labeling"]
        )

        # PRIMARY classes for which no secondary classifier is trained or run.
        # Empty entries are dropped: "ignore_secondary = " used to yield [""],
        # a list containing one empty string, which is truthy-looking and
        # matched nothing.
        params["ignore_secondary"] = _cfg_list(
            config["DEFAULT"].get("ignore_secondary", "")
        )
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

        # ---- image sizes ----------------------------------------------
        # Training resolution for the primary detectors and the secondary
        # crop classifiers. These were previously hardcoded to 640 and 224 at
        # the maybe_retrain() call sites, so any value set in the INI was
        # silently ignored.
        params["primary_imgsz"] = int(config["DEFAULT"].get("primary_imgsz", "640"))
        params["secondary_imgsz"] = int(config["DEFAULT"].get("secondary_imgsz", "224"))
        # Resolution used at INFERENCE. Defaults to the training resolution,
        # because a detector evaluated at a different scale than it was
        # trained at loses accuracy on small objects. Override only if you
        # know why you want them to differ.
        params["inference_imgsz"] = int(
            config["DEFAULT"].get("inference_imgsz", str(params["primary_imgsz"]))
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
        # motion_threshold is a NEGATIVE offset applied before the gain, so it
        # acts as a noise floor: a difference below motion_threshold/gain is
        # suppressed. Historically it was written under [kalman] but read from
        # [DEFAULT] — configparser propagates DEFAULT keys into sections, never
        # the reverse, so the [kalman] value was invisible and always resolved
        # to 0. Accept it from either place.
        _mt = config["DEFAULT"].get("motion_threshold", None)
        if _mt is None and config.has_section("kalman"):
            _mt = config["kalman"].get("motion_threshold", None)
        params["motion_threshold"] = -1 * int(_mt if _mt is not None else 0)

        # ---- builtin tracker knobs ------------------------------------
        # Only used when tracker_type = builtin. Read from [kalman], which is
        # where the rest of the builtin tracker's settings live.
        _kal = config["kalman"] if config.has_section("kalman") else {}
        params["tracker_iou_weight"] = float(_kal.get("tracker_iou_weight", "0.4"))
        params["tracker_class_penalty"] = float(
            _kal.get("tracker_class_penalty", "2.0")
        )

        # ---- training hardware ----------------------------------------
        # Previously hardcoded as batch=16, device=0 inside maybe_retrain().
        # batch matters a lot once primary_imgsz goes up: activation memory
        # scales with imgsz^2, so 1280 needs roughly 4x what 640 did.
        params["train_batch"] = int(config["DEFAULT"].get("train_batch", "16"))
        params["train_device"] = config["DEFAULT"].get("train_device", "0").strip()

        params["val_frequency"] = float(config["DEFAULT"].get("val_frequency", "0.2"))

        # ================================================================
        # Secondary classifier: split staging, class balancing, cropping
        # ----------------------------------------------------------------
        # Read unconditionally (not only under hierarchical_mode) so
        # params.get() never has to guess and validation always runs.
        # ================================================================

        # Fraction of each subclass held out for validation when
        # classify_track stages the split. Defaults to val_frequency so a
        # project that already tuned that gets consistent behaviour, but it
        # is a SEPARATE knob: val_frequency governs which annotated FRAMES
        # the annotation tool holds out for the primary detectors, this one
        # governs the secondary crop tree.
        params["secondary_val_fraction"] = float(
            config["DEFAULT"].get(
                "secondary_val_fraction", str(params["val_frequency"])
            )
        )

        # Seed for the split. Fixed by default so re-running training does
        # not silently reshuffle train and val underneath you — a moving
        # split makes two runs incomparable.
        params["secondary_split_seed"] = int(
            config["DEFAULT"].get("secondary_split_seed", "0")
        )

        # SUBCLASS names excluded from secondary training entirely. Distinct
        # from ignore_secondary, which names PRIMARY classes. Use this for
        # behaviours annotated too rarely to learn or evaluate; they stay in
        # secondary_motion_classes so the annotation tool keeps its hotkeys.
        # Subclasses too small to split are dropped automatically regardless.
        params["secondary_ignore_subclasses"] = _cfg_list(
            config["DEFAULT"].get("secondary_ignore_subclasses", "")
        )

        # Regex whose first capture group is the source video of a crop.
        # Whole videos are assigned to one side of the split, so consecutive
        # near-duplicate frames cannot appear in both train and val — which
        # would inflate validation accuracy badly. If this matches nothing,
        # the staging code says so and falls back to a per-image split.
        params["secondary_video_regex"] = config["DEFAULT"].get(
            "secondary_video_regex", DEFAULT_VIDEO_REGEX
        )

        # Exponent on inverse class frequency for the training sampler. See
        # DEFAULT_SAMPLER_POWER above. 0 disables balancing.
        params["secondary_sampler_power"] = float(
            config["DEFAULT"].get("secondary_sampler_power", str(DEFAULT_SAMPLER_POWER))
        )

        # Fraction of its own width/height a detection box is expanded by
        # before the secondary crop is taken. MUST match whatever was used to
        # generate the training crops — train-time and inference-time framing
        # have to agree, or the classifier sees a different composition than
        # it learned on.
        params["secondary_crop_margin"] = float(
            config["DEFAULT"].get("secondary_crop_margin", "0.0")
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

    # ---- image sizes --------------------------------------------------
    for key in ("primary_imgsz", "secondary_imgsz", "inference_imgsz"):
        v = params[key]
        if v < 32 or v % 32 != 0:
            raise ValueError(
                f"{key} must be a positive multiple of 32 (got {v}). YOLO's "
                f"stride is 32; other values are silently rounded."
            )
    if params["inference_imgsz"] != params["primary_imgsz"]:
        print(
            f"Warning: inference_imgsz ({params['inference_imgsz']}) differs from "
            f"primary_imgsz ({params['primary_imgsz']}). Detectors lose accuracy "
            f"on small objects when evaluated at a different scale than they "
            f"were trained at."
        )

    # ---- training + secondary thresholds ------------------------------
    if params["train_batch"] < 1:
        raise ValueError("train_batch must be at least 1")
    if not 0.0 <= params["secondary_conf_thresh"] < 1.0:
        raise ValueError("secondary_conf_thresh must be in [0, 1)")
    if not 0.0 <= params["val_frequency"] < 1.0:
        raise ValueError("val_frequency must be in [0, 1)")

    # ---- secondary split / balancing / cropping -----------------------
    if not 0.0 < params["secondary_val_fraction"] < 1.0:
        raise ValueError(
            "secondary_val_fraction must be in (0, 1) — a secondary "
            "classifier with no validation split cannot be checkpointed."
        )
    if not 0.0 <= params["secondary_sampler_power"] <= 1.0:
        raise ValueError(
            "secondary_sampler_power must be in [0, 1]. 0 disables class "
            "balancing, 0.5 is sqrt-inverse frequency, 1 is full inverse."
        )
    if not 0.0 <= params["secondary_crop_margin"] <= 1.0:
        raise ValueError(
            "secondary_crop_margin must be in [0, 1] — it is a fraction of "
            "the box's own width/height added to EACH side, so 1.0 already "
            "triples the crop."
        )

    try:
        compiled = re.compile(params["secondary_video_regex"])
    except re.error as e:
        raise ValueError(f"secondary_video_regex is not a valid regex: {e}")
    if compiled.groups < 1:
        raise ValueError(
            "secondary_video_regex must contain at least one capture group; "
            "group 1 is used as the source-video identifier."
        )

    # A subclass listed for exclusion that is not a configured secondary class
    # is almost always a typo, and a silent one — it would simply never match.
    unknown = [
        c
        for c in params["secondary_ignore_subclasses"]
        if c not in params["secondary_classes"]
    ]
    if unknown:
        print(
            f"Warning: secondary_ignore_subclasses names {unknown}, which are "
            f"not in secondary_classes ({params['secondary_classes']}). "
            f"Check for typos — unmatched entries do nothing."
        )

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
