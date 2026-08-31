"""
motion.py
=========

The false-colour motion-image encoder, extracted from classify_track.py so it
can be shared without importing that module (classify_track.py calls
load_params() at import time, which either reads sys.argv[1] or pops a Tk file
dialog — not something a second script can safely trigger). This module has no
import-time side effects.

Three call sites use it, in two different shapes:

  classify_track.py / pseudo_label.py
      one motion image per processed frame  ->  create_motion_image()

  annotation.py / regenerate_annotations.py
      walk a window of frames advancing the history, then build ONE image from
      the final frame's diffs  ->  advance_history() in the loop, then
      compose_motion_image() once afterwards

The arithmetic is unchanged from the original inline copies. Splitting the
function in two is what keeps it that way: the colour merge depends only on
(gray, diffs), so composing once at the end gives identical output to the old
code while skipping the merges that would otherwise be computed and discarded
on every frame of the window.

The one deliberate behaviour change is in annotation.py's copy, which used
`else` where the other two used `elif strategy == "sequential"`. This module
keeps the `elif`, so a strategy value that is neither "exponential" nor
"sequential" now freezes the history instead of falling through to the
sequential ring buffer. That makes annotation.py agree with the other two
scripts; nothing changes for either supported strategy value.

To use this from classify_track.py:

    1. delete the create_motion_image() definition there
    2. add:    from motion import create_motion_image
    3. change the single call site in stage 3b to:
                   create_motion_image(prev_frames, gray, params)
"""

import cv2
import numpy as np


def advance_history(prev_frames, gray, params):
    """
    Compute the three temporally-offset differences for `gray` and ADVANCE the
    frame history. Returns the diffs; pass them to compose_motion_image().

    MUTATES prev_frames — call exactly once per frame you want the history to
    advance over, and never once per detection.
    """
    diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]

    if params["strategy"] == "exponential":
        # Exponential decay — smoother tails.
        prev_frames[0] = gray
        prev_frames[1] = cv2.addWeighted(
            prev_frames[1], params["expA"], gray, 1 - params["expA"], 0
        )
        prev_frames[2] = cv2.addWeighted(
            prev_frames[2], params["expB"], gray, 1 - params["expB"], 0
        )
    elif params["strategy"] == "sequential":
        # Plain frame-over-frame ring buffer.
        prev_frames[2] = prev_frames[1]
        prev_frames[1] = prev_frames[0]
        prev_frames[0] = gray

    return diffs


def compose_motion_image(gray, diffs, params):
    """
    Map the luminance plus the three differences into a BGR image. Pure — it
    reads no history and mutates nothing, so it can be called once at the end
    of a frame window or once per frame with identical results.
    """
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

    return cv2.merge((blue, green, red)).astype(np.uint8)


def create_motion_image(prev_frames, gray, params):
    """
    Build the false-colour motion image and ADVANCE the frame history.

    NOTE: this MUTATES prev_frames. It must be called exactly once per
    PROCESSED frame — not once per detection, and not only on the frames you
    happen to be sampling. The three temporal offsets are what the motion
    detector was trained on; advancing the history at a different cadence than
    inference uses produces motion tails the model has never seen.
    """
    diffs = advance_history(prev_frames, gray, params)
    return compose_motion_image(gray, diffs, params)


def new_history(gray):
    """Prime a fresh 3-frame history. Call once per video, never across videos."""
    return [gray.copy() for _ in range(3)]
