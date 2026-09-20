"""Splits side-by-side dual-panel recordings into separate frontal
(left half) and lateral (right half) video files via ffmpeg."""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config

VIDEO_EXTENSIONS = (".mp4", ".mov")
MAX_WORKERS = 4


def find_videos(root):
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]


def split_video(input_path, frontal_dir, lateral_dir):
    filename = input_path.stem
    frontal_output = Path(frontal_dir) / f"{filename}_panel1.mp4"
    lateral_output = Path(lateral_dir) / f"{filename}_panel2.mp4"

    command = [
        "ffmpeg", "-i", str(input_path),
        "-filter_complex",
        "[0:v]crop=iw/2:ih:0:0[left];[0:v]crop=iw/2:ih:iw/2:0[right]",
        "-map", "[left]", str(frontal_output),   # left half = frontal view
        "-map", "[right]", str(lateral_output),  # right half = lateral view
        "-y",
    ]
    print(f"Processing: {input_path}")
    result = subprocess.run(command)
    print(f"  {'Failed' if result.returncode != 0 else 'Done'}: {input_path.name}")


def run_batch(root_dir=None, frontal_dir=None, lateral_dir=None):
    root_dir = root_dir or config.SPLIT_ROOT_DIR
    frontal_dir = frontal_dir or config.SPLIT_FRONTAL_DIR
    lateral_dir = lateral_dir or config.SPLIT_LATERAL_DIR
    os.makedirs(frontal_dir, exist_ok=True)
    os.makedirs(lateral_dir, exist_ok=True)

    videos = find_videos(root_dir)
    print(f"Found {len(videos)} video(s)")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(lambda v: split_video(v, frontal_dir, lateral_dir), videos)

    print("All videos processed.")
