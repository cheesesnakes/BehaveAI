#!/usr/bin/env python3
"""
Shared crop geometry for BehaveAI.

annotation.py, the inspector and regenerate_annotations.py all cut
secondary-classifier crops from a processed frame. They must cut the
*same* rectangle for a given box, or the classifier's framing changes
depending on which tool last touched the file.

The crop filename still records the UNEXPANDED box corner
(<video>_<frame>_<x1>_<y1>.jpg) — that corner is what regeneration
matches against the YOLO label file, so it must stay the label box, not
the margined one.
"""


def expand_box(x1, y1, x2, y2, img_w, img_h, margin=0.0):
    """Expand a box by `margin` (a fraction of its own w/h), clipped to the image."""
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * margin)
    my = int(bh * margin)
    return (
        max(0, x1 - mx),
        max(0, y1 - my),
        min(img_w, x2 + mx),
        min(img_h, y2 + my),
    )


def crop_with_margin(img, x1, y1, x2, y2, margin=0.0):
    """Cut a crop expanded by `margin`. Returns None if the region is empty."""
    if img is None:
        return None
    h, w = img.shape[:2]
    ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, w, h, margin)
    if ex2 <= ex1 or ey2 <= ey1:
        return None
    crop = img[ey1:ey2, ex1:ex2]
    return crop if crop.size else None
