"""Helpers for interpreting video_id / folder-path conventions."""
import re
from pathlib import Path


def detect_side(video_id):
    """Left/Right hand from video_id (e.g. 'P01PSFLM1_output'), since the
    noisy video files themselves may not carry the L/R or FLM/FRM token."""
    stem = str(video_id).upper()
    for tok in re.split(r"[_\-]", stem):
        if tok == "L":
            return "Left"
        if tok == "R":
            return "Right"
    if "FLM" in stem:
        return "Left"
    if "FRM" in stem:
        return "Right"
    print(f"  [WARN] Side not found in '{video_id}'; defaulting to Left.")
    return "Left"


def parse_noise_path(video_path, root_dir):
    """video_id / noise_type / severity from ROOT_DIR/video_id/noise_type/severity/*.mp4."""
    rel = Path(video_path).relative_to(root_dir)
    parts = rel.parts
    video_id = parts[0]
    noise_type = parts[1]
    severity = parts[1]  # no separate severity subfolder in this layout
    return video_id, noise_type, severity
