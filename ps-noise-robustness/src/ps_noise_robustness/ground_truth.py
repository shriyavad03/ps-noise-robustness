"""Locating and parsing CVAT ground-truth annotations."""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .video_id_utils import detect_side

_gt_file_index = None  # lazy cache: {stem.upper(): [Path, ...]}


def _build_gt_file_index(gt_dir):
    """Scan gt_dir once and cache stem -> [paths], to avoid re-scanning a
    (often network-mounted) directory once per video."""
    gt_dir = Path(gt_dir)
    index = {}
    if not gt_dir.exists():
        return index
    paths = [p for p in gt_dir.glob("*") if p.is_file()]
    if not paths:
        paths = [p for p in gt_dir.rglob("*") if p.is_file()]
    for p in paths:
        index.setdefault(p.stem.upper(), []).append(p)
    return index


def find_gt_for_video(video_id, gt_dir):
    """Locate the CVAT ground-truth file for a video_id.

    Builds the expected stem "P{num}_{L|R}" from the participant number in
    video_id and the hand side, then looks it up case-insensitively
    (matching zero-padded or unpadded participant numbers, any extension).
    """
    global _gt_file_index
    gt_dir = Path(gt_dir)
    if not gt_dir.exists():
        print(f"  [WARN] GT_DIR '{gt_dir}' does not exist.")
        return None

    if _gt_file_index is None:
        _gt_file_index = _build_gt_file_index(gt_dir)
        n_total = sum(len(v) for v in _gt_file_index.values())
        print(f"[INFO] Indexed {n_total} file(s) under GT_DIR='{gt_dir}'.")

    stem = str(video_id).upper()
    m = re.search(r"P0*(\d+)", stem)
    if not m:
        print(f"  [WARN] Could not find a participant number in video_id '{video_id}'.")
        return None
    participant = m.group(1)
    letter = "L" if detect_side(video_id) == "Left" else "R"

    target_stems = {
        f"P{participant}_{letter}",
        f"P{int(participant):02d}_{letter}",
        f"P{int(participant):03d}_{letter}",
    }

    matches = sorted({p for stem_key in target_stems for p in _gt_file_index.get(stem_key, [])},
                      key=str)
    if not matches:
        n_total = sum(len(v) for v in _gt_file_index.values())
        print(f"  [WARN] No ground-truth file found for video_id='{video_id}' "
              f"(looked for stems {sorted(target_stems)}; {n_total} file(s) indexed).")
        return None
    if len(matches) > 1:
        print(f"  [WARN] Multiple ground-truth files matched video_id='{video_id}': "
              f"{[str(p) for p in matches]}. Using the first: {matches[0]}")
    return matches[0]


def parse_cvat_annotations(xml_path):
    """Parse a CVAT 'CVAT for video 1.1' points-annotation export.
    Returns gt_long (tidy: frame, label, x_px, y_px, outside, occluded) and
    the (W, H) CVAT recorded when the job was created."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    width = root.find(".//width")
    height = root.find(".//height")
    if width is None or height is None:
        raise ValueError("Width/Height tags not found.")
    W, H = int(width.text), int(height.text)

    rows = []
    for track in root.findall("track"):
        label = (track.get("label") or "").strip()
        for pt in track.findall("points"):
            outside = pt.get("outside") == "1"
            occluded = pt.get("occluded") == "1"
            x_str, y_str = pt.get("points").split(",")
            rows.append({"frame": int(pt.get("frame")), "label": label,
                         "x_px": float(x_str), "y_px": float(y_str),
                         "outside": outside, "occluded": occluded})
    gt_long = pd.DataFrame(rows).sort_values(["label", "frame"]).reset_index(drop=True)
    return gt_long, W, H


# Canonical keypoint names, matched case-/whitespace-insensitively against
# whatever label strings actually appear in a given CVAT export.
_CANONICAL_GT_LABELS = {"index_mcp": "Index_MCP", "pinky_mcp": "Pinky_MCP"}


def _resolve_gt_label_columns(columns):
    """Map label strings present in a CVAT file to the canonical
    'Index_MCP' / 'Pinky_MCP' names. Returns {canonical: actual_column}."""
    resolved = {}
    for col in columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in _CANONICAL_GT_LABELS:
            resolved[_CANONICAL_GT_LABELS[key]] = col
    return resolved


def build_gt_arrays(xml_path, n_frames, video_w, video_h):
    """Returns gt_index_xy, gt_pinky_xy: (n_frames, 2) arrays in pixel
    space. NaN wherever CVAT has no annotation for that frame, marked it
    outside-of-frame, or occluded.

    Ground truth is annotated on the clean video and reused as-is for
    every noise severity of the same video_id, since noise degradation
    doesn't move the true keypoint locations -- only what MediaPipe/RTMPose
    manage to detect.
    """
    gt_long, GT_W, GT_H = parse_cvat_annotations(xml_path)
    if (GT_W, GT_H) != (video_w, video_h):
        print(f"[WARN] CVAT recorded {GT_W}x{GT_H} but video is {video_w}x{video_h} -- "
              f"ground-truth pixel coords are assumed to already be in the video's own "
              f"pixel space.")

    gt_long = gt_long.copy()
    gt_long.loc[gt_long["outside"] | gt_long["occluded"], ["x_px", "y_px"]] = np.nan

    # CVAT exports can contain duplicate <points> entries for the same
    # (frame, label). Dedupe, preferring a valid (non-NaN) entry.
    gt_long["_valid"] = gt_long[["x_px", "y_px"]].notna().all(axis=1)
    gt_long = (gt_long.sort_values(["frame", "label", "_valid"], ascending=[True, True, False])
                        .drop_duplicates(subset=["frame", "label"], keep="first")
                        .drop(columns="_valid"))

    x_wide = gt_long.pivot(index="frame", columns="label", values="x_px")
    y_wide = gt_long.pivot(index="frame", columns="label", values="y_px")

    label_cols = _resolve_gt_label_columns(x_wide.columns)
    missing = [name for name in ("Index_MCP", "Pinky_MCP") if name not in label_cols]
    if missing:
        print(f"[WARN] {xml_path}: could not find CVAT label(s) {missing} "
              f"(labels present in file: {list(x_wide.columns)}).")

    gt_index_xy = np.full((n_frames, 2), np.nan)
    gt_pinky_xy = np.full((n_frames, 2), np.nan)
    idx_col = label_cols.get("Index_MCP")
    pinky_col = label_cols.get("Pinky_MCP")
    for frame in x_wide.index:
        if frame >= n_frames or frame < 0:
            continue
        if idx_col is not None:
            gt_index_xy[frame] = [x_wide.loc[frame, idx_col], y_wide.loc[frame, idx_col]]
        if pinky_col is not None:
            gt_pinky_xy[frame] = [x_wide.loc[frame, pinky_col], y_wide.loc[frame, pinky_col]]

    n_valid = int((~np.isnan(gt_index_xy[:, 0]) & ~np.isnan(gt_pinky_xy[:, 0])).sum())
    print(f"  Ground truth: {n_valid}/{n_frames} frames annotated for both Index_MCP and Pinky_MCP")
    return gt_index_xy, gt_pinky_xy
