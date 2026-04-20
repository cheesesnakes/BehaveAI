<img width="400" height="69" alt="BehaveAI_400" src="https://github.com/user-attachments/assets/6fb5cd16-d266-4e8b-9513-1734a45813bf" />


# A framework for detecting, classifying and tracking moving objects

BehaveAI is a user-friendly tool that identifies and classifies animals and other objects in video footage from their movement as well as their static appearance. The framework converts motion information into false colours that allow both the human annotator and convolutional neural network (CNN) to easily identify patterns of movement. The framework integrates both motion streams and static streams in a similar fashion to the mammalian visual system, separating the tasks of detection and classification.

The framework also supports hierarchical models (e.g. detect something from it's movement, then work out what exactly it is from conventional static appearance, or vice-versa); and semi-supervised annotation, allowing the annotator to rapidly correct errors made by initial models, making for a more efficient and effective training process.

> **This is a fork** of [troscianko/BehaveAI](https://github.com/troscianko/BehaveAI). The fork keeps the full upstream workflow and adds a few extras aimed at users who want to bootstrap a project from an existing pretrained detector rather than annotate from scratch. See [What's new in this fork](#whats-new-in-this-fork) below.

![BehaveAI_flies](https://github.com/user-attachments/assets/99ae83fb-c001-4d5a-8338-0607a914d0c4)


#### Key features:
- Identifies objects and their behaviour based on motion and/or static appearance
- Fast user-friendly annotation - in under an hour you can create a powerful tracking model
- Identifies small (2 px!), fast moving, and motion-blurred targets
- Can track any type(s) of animal/object
- Tracks multiple individuals, classifying the behaviour of each independently
- Semi-supervised annotation allows you to learn where the models are weak and focus on improving performance
- Tools for inspecting and editing the annotation library
- Live (edge) support - record videos, train, then run live video processing using the user interface
- Built around the versatile [YOLO](https://github.com/ultralytics/ultralytics) (You Only Look Once) architecture
- Computationally efficient - runs fine on low-end devices without GPUs
- Intuitive user interface with installers for Windows and Linux (including Raspberry Pi) and full user-interface. Also works on MacOS but currently no installer.
- Free & open source ([GNU Afferro General Public License](https://github.com/troscianko/BehaveAI/blob/main/LICENSE))

## What's new in this fork

This fork re-packages BehaveAI as a standard Python project managed by [`uv`](https://docs.astral.sh/uv/), and adds a pseudo-labelling workflow so you can bootstrap annotations from an existing pretrained detector.

### Pseudo-labeller for the primary static stream

`scripts/pseudo_label.py` is a one-shot script that runs a pretrained YOLO detector over the videos in your `clips_dir` and writes out BehaveAI-compatible annotations (images + YOLO-format `.txt` labels) into `annot_static/images/{train,val}/` and `annot_static/labels/{train,val}/`. These annotations then appear in the standard annotation GUI exactly like hand-drawn ones, ready to be reviewed and corrected.

Typical use case: you have a YOLO26 "fish" detector and a few hours of footage. Rather than annotating from zero, run the pseudo-labeller, open the annotation GUI, and spend your time correcting errors instead of drawing every box.

The pseudo-labeller assumes the external model's class IDs match the order of `primary_static_classes` in your project's `config.ini` — e.g. a single-class fish model maps straight to a project whose first primary static class is `fish`. There is no class remap.

```bash
# after editing config.ini to point primary_static_external_model at your .pt file
uv run python scripts/pseudo_label.py --sample-every 30 --dry-run     # preview
uv run python scripts/pseudo_label.py --sample-every 30               # run it
```

Once you're happy with the corrected annotations, clear `primary_static_external_model` in `config.ini` so the next training run uses your (corrected) pseudo-labels as ground truth.

### Use an external pretrained model instead of training from annotations

Two new `config.ini` keys let you skip BehaveAI's training step entirely for the primary or secondary static stream and use a pretrained model instead:

```ini
primary_static_external_model   = /path/to/primary.pt
secondary_static_external_model = /path/to/secondary.pt

# Used by the pseudo-labeller:
pseudo_label_conf = 0.6
```

When either path is set, `classify_track.py` loads that model for inference instead of retraining from `annot_static/`. Accepted formats: `.pt`, `.torchscript`, `.onnx`, `.engine`, or an NCNN folder. The paths are also exposed as Browse-able fields in the Settings GUI (Tab 3).

### `uv`-based project, with one-command installers

The project now ships with `pyproject.toml` + `uv.lock` and two installer scripts:

- `install.ps1` — Windows (PowerShell)
- `install.sh` — Linux (bash)

Both scripts install `uv` if needed, install the pinned Python 3.13 runtime, run `uv sync` to create `.venv/` and install every dependency, and (on Linux) offer to install `ffmpeg` via the detected package manager. See [Installation](#installation) below.

### Repository layout

The Python modules have been moved out of the repo root into a `scripts/` folder and renamed (e.g. `BehaveAI_annotation.py` → `scripts/annotation.py`). The application entry point is `app.py`. Functionally identical to upstream; the restructure just makes imports cleaner.

## Installation

### Quick start

From the repo root, run the installer for your platform:

**Windows** (PowerShell):
```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

**Linux**:
```bash
bash install.sh
```

Each installer:
1. installs `uv` if not already present;
2. installs Python 3.13 via `uv python install`;
3. runs `uv sync` to install all dependencies into `.venv/`;
4. checks for `ffmpeg` (on Linux, offers to install it).

The dependency install downloads PyTorch, CUDA wheels (on Linux), and Ultralytics — expect **1–4 GB** of downloads depending on platform.

### Manual install

If you'd rather not use the installer scripts:

```bash
# install uv (see https://docs.astral.sh/uv/ for platform-specific instructions)
curl -LsSf https://astral.sh/uv/install.sh | sh        # Linux / macOS
# or on Windows:
#   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

uv python install 3.13
uv sync
```

### Running the app

```bash
uv run python app.py
```

Or activate the venv and run directly:

```bash
# Linux
source .venv/bin/activate && python app.py

# Windows
.\.venv\Scripts\Activate.ps1
python app.py
```

## User guide

For the core workflow (recording clips, annotating, training, classifying, tracking, live inference), see the upstream project wiki: <https://github.com/troscianko/BehaveAI/wiki>. The GUI, hotkeys, and training behaviour are identical to upstream; the only additions in this fork are the ones documented above.

## Paper & Citation:
[PLOS Biology publication](https://doi.org/10.1371/journal.pbio.3003632)

If you use BehaveAI please cite:

* Troscianko, Jolyon, Thomas A. O’Shea-Wheller, James A. M. Galloway, and Kevin J. Gaston. (2026) 'BehaveAI Enables Rapid Detection and Classification of Objects and Behavior from Motion’. _PLOS Biology_ 24, no. 2 (2026): e3003632. https://doi.org/10.1371/journal.pbio.3003632.


## Video Guide (v1.2):
[<img width="350" alt="Screenshot from 2025-11-04 17-38-49" src="https://github.com/user-attachments/assets/5d76855e-d24f-4107-a6b9-c13aa98e6f79" />](https://www.youtube.com/watch?v=atEL14nxz9s)


See the [project Wiki](https://github.com/troscianko/BehaveAI/wiki) for detailed instructions.