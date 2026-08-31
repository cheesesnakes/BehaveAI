# index_annotations.py
# Helper class to list annotated images and load saved labels/masks / find video files.

import os


class AnnotationIndex:
    def __init__(
        self,
        static_train_images_dir,
        static_val_images_dir,
        static_train_labels_dir,
        static_val_labels_dir,
        motion_train_images_dir,
        motion_val_images_dir,
        motion_train_labels_dir,
        motion_val_labels_dir,
        motion_cropped_base_dir,
        static_cropped_base_dir,
        clips_dir,
        primary_static_classes,
        primary_classes,
        secondary_classes,
        hierarchical_mode,
        ignore_secondary=None,
    ):
        # directories
        self.static_train_images_dir = static_train_images_dir
        self.static_val_images_dir = static_val_images_dir
        self.static_train_labels_dir = static_train_labels_dir
        self.static_val_labels_dir = static_val_labels_dir
        self.motion_train_images_dir = motion_train_images_dir
        self.motion_val_images_dir = motion_val_images_dir
        self.motion_train_labels_dir = motion_train_labels_dir
        self.motion_val_labels_dir = motion_val_labels_dir
        self.motion_cropped_base_dir = motion_cropped_base_dir
        self.static_cropped_base_dir = static_cropped_base_dir
        self.clips_dir = clips_dir

        # class lists & mode
        self.primary_static_classes = (
            list(primary_static_classes) if primary_static_classes is not None else []
        )
        self.primary_classes = (
            list(primary_classes) if primary_classes is not None else []
        )
        self.secondary_classes = (
            list(secondary_classes) if secondary_classes is not None else []
        )
        self.hierarchical_mode = bool(hierarchical_mode)
        self.ignore_secondary = set(ignore_secondary or [])

    # ------------------------------------------------------------------
    # Build list of annotated images.
    #
    # FIX: add_dir used to write every directory it scanned into the SAME
    # "static_*" keys, guarded by `if "static_img" not in rec`. For a frame
    # that exists only in the motion dataset that meant static_img,
    # static_lbl and static_origin_lbl_dir all pointed at MOTION files. The
    # inspector then filled in motion_lbl from the same directory, so
    # load_labels_and_masks_for_item read one label file twice — once
    # unshifted (as static) and once shifted by len(primary_static_classes)
    # (as motion). Every box appeared twice, and the phantom copy carried a
    # bogus primary class, so no crop could ever attach to it. That is the
    # "N/2N box(es) had no matching crop" report.
    #
    # Each stream now writes its own namespaced keys, and train wins over val
    # within a stream (same precedence as before).
    # ------------------------------------------------------------------
    _EMPTY_ITEM = {
        "static_img": None,
        "static_lbl": None,
        "static_mask": None,
        "static_origin_img_dir": None,
        "static_origin_lbl_dir": None,
        "motion_img": None,
        "motion_lbl": None,
        "motion_mask": None,
        "motion_origin_img_dir": None,
        "motion_origin_lbl_dir": None,
    }

    def list_images_labels_and_masks(self):
        items = {}

        def add_dir(img_dir, lbl_dir, stream):
            """Index one image directory into the `stream` ('static'/'motion') keys."""
            if not img_dir or not os.path.isdir(img_dir):
                return
            for fname in os.listdir(img_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                base = os.path.splitext(fname)[0]
                img_path = os.path.join(img_dir, fname)
                lbl_path = (
                    os.path.join(lbl_dir, base + ".txt")
                    if lbl_dir and os.path.isdir(lbl_dir)
                    else None
                )
                mask_dir = lbl_dir.replace("labels", "masks") if lbl_dir else None
                mask_path = (
                    os.path.join(mask_dir, base + ".mask.txt")
                    if mask_dir and os.path.isdir(mask_dir)
                    else None
                )

                rec = items.setdefault(base, {})
                if rec.get(stream + "_img"):
                    # already claimed by the train split for this stream
                    continue
                rec[stream + "_img"] = img_path
                rec[stream + "_lbl"] = (
                    lbl_path if lbl_path and os.path.exists(lbl_path) else None
                )
                rec[stream + "_mask"] = (
                    mask_path if mask_path and os.path.exists(mask_path) else None
                )
                rec[stream + "_origin_img_dir"] = img_dir
                rec[stream + "_origin_lbl_dir"] = lbl_dir

        add_dir(self.static_train_images_dir, self.static_train_labels_dir, "static")
        add_dir(self.static_val_images_dir, self.static_val_labels_dir, "static")
        add_dir(self.motion_train_images_dir, self.motion_train_labels_dir, "motion")
        add_dir(self.motion_val_images_dir, self.motion_val_labels_dir, "motion")

        ordered = []
        for base, rec in sorted(items.items()):
            entry = dict(self._EMPTY_ITEM)
            entry["basename"] = base
            entry.update(rec)
            ordered.append(entry)
        return ordered

    # ------------------------------------------------------------------
    # Find a video in clips_dir corresponding to an annotation item basename
    # ------------------------------------------------------------------
    def find_video_for_item(self, item):
        if not os.path.isdir(self.clips_dir):
            return None, None
        base = item.get("basename", "")
        if "_" not in base:
            return None, None
        video_label_guess, tail = base.rsplit("_", 1)
        frame_number_guess = None
        try:
            frame_number_guess = int(tail)
        except Exception:
            frame_number_guess = None

        for fname in os.listdir(self.clips_dir):
            if not fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue
            stem = os.path.splitext(fname)[0]
            if stem.lower() == video_label_guess.lower():
                return os.path.join(self.clips_dir, fname), frame_number_guess

        for fname in os.listdir(self.clips_dir):
            if not fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue
            stem = os.path.splitext(fname)[0]
            if stem.lower().startswith(video_label_guess.lower()):
                return os.path.join(self.clips_dir, fname), frame_number_guess

        return None, None

    # ------------------------------------------------------------------
    # Load the labels & masks for an item into boxes and grey_boxes lists.
    # ------------------------------------------------------------------
    def load_labels_and_masks_for_item(self, item, fr, original_frame):
        boxes = []
        grey_boxes = []

        static_lbl = item.get("static_lbl")
        motion_lbl = item.get("motion_lbl")

        # Belt and braces: if anything upstream ever points both streams at the
        # same file again, read it once rather than duplicating every box.
        if (
            static_lbl
            and motion_lbl
            and os.path.abspath(static_lbl) == os.path.abspath(motion_lbl)
        ):
            print(
                f"WARNING: {item.get('basename')} has static_lbl == motion_lbl "
                f"({static_lbl}); reading it once as motion only."
            )
            static_lbl = None

        # static labels
        if static_lbl and os.path.exists(static_lbl):
            try:
                with open(static_lbl, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls = int(parts[0])
                        xc, yc, bw, bh = parts[1:5]
                        if fr is None:
                            # can't compute pixel coords; skip
                            continue
                        h, w = fr.shape[:2]
                        x1, y1, x2, y2 = self._norm_to_pixels(xc, yc, bw, bh, w, h)
                        if self.hierarchical_mode:
                            # use -1 for "no secondary assigned"
                            boxes.append((x1, y1, x2, y2, cls, -1, -1, -1))
                        else:
                            boxes.append((x1, y1, x2, y2, cls, -1))
            except Exception:
                pass

        # motion labels
        if motion_lbl and os.path.exists(motion_lbl):
            try:
                with open(motion_lbl, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls = int(parts[0])
                        xc, yc, bw, bh = parts[1:5]
                        if original_frame is None:
                            continue
                        h, w = original_frame.shape[:2]
                        x1, y1, x2, y2 = self._norm_to_pixels(xc, yc, bw, bh, w, h)
                        global_primary_cls = cls + len(self.primary_static_classes)
                        if self.hierarchical_mode:
                            boxes.append(
                                (x1, y1, x2, y2, global_primary_cls, -1, -1, -1)
                            )
                        else:
                            boxes.append((x1, y1, x2, y2, global_primary_cls, -1))
            except Exception:
                pass

        # masks (prefer static mask then motion mask)
        mask_path = item.get("static_mask") or item.get("motion_mask")
        if mask_path and os.path.exists(mask_path):
            try:
                with open(mask_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            gx1, gy1, gx2, gy2 = map(int, parts[:4])
                            grey_boxes.append((gx1, gy1, gx2, gy2))
            except Exception:
                pass

        # if hierarchical_mode is enabled, attach secondary crops now
        if self.hierarchical_mode and boxes:
            boxes = self._attach_secondary_crops(item, boxes)

        return boxes, grey_boxes

    # ------------------------------------------------------------------
    # Convenience: load labels by basename (used by the annotation script,
    # which constructs the basename itself).
    # ------------------------------------------------------------------
    def load_labels_for_basename(self, base_fn, fr, original_frame):
        item = dict(self._EMPTY_ITEM)
        item["basename"] = base_fn

        # static label search
        for d in (self.static_train_labels_dir, self.static_val_labels_dir):
            if d and os.path.isdir(d):
                p = os.path.join(d, base_fn + ".txt")
                if os.path.exists(p):
                    item["static_lbl"] = p
                    item["static_origin_lbl_dir"] = d
                    item["static_origin_img_dir"] = d.replace("labels", "images")
                    img_dir = d.replace("labels", "images")
                    if os.path.isdir(img_dir):
                        item["static_img"] = os.path.join(img_dir, base_fn + ".jpg")
                    break
        # motion label search
        for d in (self.motion_train_labels_dir, self.motion_val_labels_dir):
            if d and os.path.isdir(d):
                p = os.path.join(d, base_fn + ".txt")
                if os.path.exists(p):
                    item["motion_lbl"] = p
                    item["motion_origin_lbl_dir"] = d
                    img_dir = d.replace("labels", "images")
                    item["motion_origin_img_dir"] = img_dir
                    if os.path.isdir(img_dir):
                        item["motion_img"] = os.path.join(img_dir, base_fn + ".jpg")
                    break
        # masks — keep static and motion separate so they can't be confused
        for d in (self.static_train_labels_dir, self.static_val_labels_dir):
            if d and os.path.isdir(d):
                mp = os.path.join(d.replace("labels", "masks"), base_fn + ".mask.txt")
                if os.path.exists(mp):
                    item["static_mask"] = mp
                    break
        for d in (self.motion_train_labels_dir, self.motion_val_labels_dir):
            if d and os.path.isdir(d):
                mp = os.path.join(d.replace("labels", "masks"), base_fn + ".mask.txt")
                if os.path.exists(mp):
                    item["motion_mask"] = mp
                    break

        return self.load_labels_and_masks_for_item(item, fr, original_frame)

    # ------------------------------------------------------------------
    # helper: parse crop filename pattern: <video_label>_<frame>_<x1>_<y1>.<ext>
    # ------------------------------------------------------------------
    def _parse_crop_filename(self, fn):
        stem = os.path.splitext(fn)[0]
        parts = stem.split("_")
        if len(parts) < 4:
            return None
        try:
            y1 = int(parts[-1])
            x1 = int(parts[-2])
            frame = int(parts[-3])
            video_label_part = "_".join(parts[:-3])
            return video_label_part, frame, x1, y1
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Attach secondary crop matches to boxes.
    # Exact (x1, y1, primary_name) match first, then a small neighbourhood.
    # ------------------------------------------------------------------
    def _attach_secondary_crops(self, item, boxes):
        MATCH_TOL = 2

        def _with_secondary(b, sec_idx):
            if len(b) >= 8:
                return (b[0], b[1], b[2], b[3], b[4], sec_idx, b[6], b[7])
            primary_cls = b[4] if len(b) > 4 else 0
            conf = b[6] if len(b) > 6 else -1
            return (b[0], b[1], b[2], b[3], primary_cls, sec_idx, conf, -1)

        # build map by (x1, y1, primary_name) -> list of box indices
        box_index = {}
        for bi, b in enumerate(boxes):
            bx1 = int(round(b[0]))
            by1 = int(round(b[1]))
            primary_idx = b[4] if len(b) > 4 else None
            primary_name = (
                self.primary_classes[primary_idx]
                if primary_idx is not None
                and 0 <= primary_idx < len(self.primary_classes)
                else None
            )
            key = (bx1, by1, primary_name)
            box_index.setdefault(key, []).append(bi)

        sec_name_to_idx = {name: idx for idx, name in enumerate(self.secondary_classes)}

        # parse video_label and frame from basename
        if "_" in item.get("basename", ""):
            video_label_guess, tail = item["basename"].rsplit("_", 1)
            try:
                frame_number_guess = int(tail)
            except Exception:
                frame_number_guess = None
        else:
            video_label_guess = item.get("basename", "")
            frame_number_guess = None

        # scan both cropped base dirs (motion then static)
        for base_crop_dir in (
            self.motion_cropped_base_dir,
            self.static_cropped_base_dir,
        ):
            if not base_crop_dir or not os.path.isdir(base_crop_dir):
                continue
            for primary_name in os.listdir(base_crop_dir):
                prim_dir = os.path.join(base_crop_dir, primary_name)
                if not os.path.isdir(prim_dir):
                    continue
                for secondary_name in os.listdir(prim_dir):
                    sec_dir = os.path.join(prim_dir, secondary_name)
                    if not os.path.isdir(sec_dir):
                        continue
                    sec_idx = sec_name_to_idx.get(secondary_name)
                    if sec_idx is None:
                        continue
                    for fn in os.listdir(sec_dir):
                        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                            continue
                        parsed = self._parse_crop_filename(fn)
                        if parsed is None:
                            continue
                        vlabel_part, fn_frame, x1_fn, y1_fn = parsed
                        if (
                            vlabel_part != video_label_guess
                            or fn_frame != frame_number_guess
                        ):
                            continue

                        matched = False
                        # exact key match first
                        key = (x1_fn, y1_fn, primary_name)
                        if key in box_index:
                            for bi in box_index[key]:
                                boxes[bi] = _with_secondary(boxes[bi], sec_idx)
                                matched = True
                        if matched:
                            continue

                        # otherwise small neighbourhood search
                        for dx in range(-MATCH_TOL, MATCH_TOL + 1):
                            if matched:
                                break
                            for dy in range(-MATCH_TOL, MATCH_TOL + 1):
                                cand = (x1_fn + dx, y1_fn + dy, primary_name)
                                if cand in box_index:
                                    for bi in box_index[cand]:
                                        boxes[bi] = _with_secondary(boxes[bi], sec_idx)
                                    matched = True
                                    break
                            if matched:
                                break
        return boxes

    # small helper used above
    def _norm_to_pixels(self, xc, yc, bw, bh, w, h):
        cx = float(xc) * w
        cy = float(yc) * h
        bw_p = float(bw) * w
        bh_p = float(bh) * h
        x1 = int(cx - bw_p / 2)
        y1 = int(cy - bh_p / 2)
        x2 = int(cx + bw_p / 2)
        y2 = int(cy + bh_p / 2)
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        return x1, y1, x2, y2

    # ------------------------------------------------------------------
    # Delete all saved files for a basename (labels, masks, images, original
    # motion images, and cropped secondary images in hierarchical mode).
    # Returns the list of deleted file paths.
    # ------------------------------------------------------------------
    def delete_frame(self, base_filename):
        deleted = []

        label_dirs = [
            self.static_train_labels_dir,
            self.static_val_labels_dir,
            self.motion_train_labels_dir,
            self.motion_val_labels_dir,
        ]
        mask_dirs = [d.replace("labels", "masks") if d else None for d in label_dirs]
        image_dirs = [
            self.static_train_images_dir,
            self.static_val_images_dir,
            self.motion_train_images_dir,
            self.motion_val_images_dir,
        ]
        image_exts = (".jpg", ".jpeg", ".png")

        # --- delete label files (.txt) ---
        for d in label_dirs:
            if not d:
                continue
            p = os.path.join(d, base_filename + ".txt")
            if os.path.exists(p):
                try:
                    os.remove(p)
                    deleted.append(p)
                except Exception:
                    pass

        # --- delete mask files (.mask.txt) ---
        for d in mask_dirs:
            if not d:
                continue
            p = os.path.join(d, base_filename + ".mask.txt")
            if os.path.exists(p):
                try:
                    os.remove(p)
                    deleted.append(p)
                except Exception:
                    pass

        # --- delete image files in expected image dirs ---
        for d in image_dirs:
            if not d:
                continue
            for ext in image_exts:
                p = os.path.join(d, base_filename + ext)
                if os.path.exists(p):
                    try:
                        os.remove(p)
                        deleted.append(p)
                    except Exception:
                        pass

        # --- delete 'original' motion images if they exist ---
        for parent in (self.motion_train_images_dir, self.motion_val_images_dir):
            if not parent:
                continue
            od = os.path.join(parent, "original")
            if not os.path.isdir(od):
                continue
            for ext in image_exts:
                p = os.path.join(od, base_filename + ext)
                if os.path.exists(p):
                    try:
                        os.remove(p)
                        deleted.append(p)
                    except Exception:
                        pass

        # --- delete cropped secondary images when hierarchical_mode is enabled ---
        # Filenames look like <video_label>_<frame>_<x1>_<y1>.jpg. Match on
        # base_filename + "_" and require the remainder to be exactly
        # <x1>_<y1>, so a longer frame number can never be mistaken for a
        # coordinate (deleting frame 105 used to take out 1050-1059's crops).
        if self.hierarchical_mode and base_filename is not None:
            prefix = f"{base_filename}_"
            for base_cropped_dir in (
                self.motion_cropped_base_dir,
                self.static_cropped_base_dir,
            ):
                if not base_cropped_dir or not os.path.isdir(base_cropped_dir):
                    continue
                for root, _, files in os.walk(base_cropped_dir):
                    for fname in files:
                        lf = fname.lower()
                        if not any(lf.endswith(ext) for ext in image_exts):
                            continue
                        if not fname.startswith(prefix):
                            continue
                        rest = os.path.splitext(fname)[0][len(prefix) :]
                        bits = rest.split("_")
                        if len(bits) != 2:
                            continue
                        try:
                            int(bits[0])
                            int(bits[1])
                        except ValueError:
                            continue
                        full = os.path.join(root, fname)
                        if os.path.exists(full):
                            try:
                                os.remove(full)
                                deleted.append(full)
                            except Exception:
                                pass

        return deleted
