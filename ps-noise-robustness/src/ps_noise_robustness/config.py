"""Loads configs/config.yaml (fallback: config.example.yaml) into module
constants. Point PS_CONFIG at a different file to override.

Where to change paths: see configs/config.example.yaml -- almost always
just the `experiment_root` line there, not anything in this file.
"""
import os
from pathlib import Path

import onnxruntime as ort
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = Path(os.environ.get(
    "PS_CONFIG", _REPO_ROOT / "configs" / "config.yaml"))
if not _CONFIG_PATH.exists():
    _CONFIG_PATH = _REPO_ROOT / "configs" / "config.example.yaml"

with open(_CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

# Every path below is resolved relative to this one folder (see the
# "CHANGE ME" line in configs/config.example.yaml). A path value that is
# already absolute (starts with "/") overrides it instead of nesting under
# it -- that's plain pathlib behaviour, not special-cased here.
EXPERIMENT_ROOT = Path(_cfg["experiment_root"]).expanduser()


def _resolve(section, key):
    return (EXPERIMENT_ROOT / _cfg[section][key]).expanduser()


# ── Paths (configs/config.yaml > paths:) ────────────────────────────────
TRIMMED_VIDEOS_DIR = _resolve("paths", "trimmed_videos_dir")
ROOT_DIR = _resolve("paths", "root_dir")
GT_DIR = _resolve("paths", "gt_dir")
OUT_DIR = _resolve("paths", "out_dir")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE = _resolve("paths", "model_cache")

FEATURES_CSV = OUT_DIR / "ps_features_ALL_NOISE_VIDEOS.csv"
ERROR_CSV = OUT_DIR / "ps_error_vs_gt_ALL_NOISE_VIDEOS.csv"
SPATIAL_CSV = OUT_DIR / "ps_spatial_accuracy_ALL_NOISE_VIDEOS.csv"

# Fixed column order, so pandas never infers it from a row dict (which
# varies depending on whether any cycles were detected).
SUMMARY_COLUMNS = [
    "video_id", "source", "n_cycles",
    "mean_duration_s", "mean_amplitude_p2p_deg", "mean_cumulative_amplitude_deg",
    "mean_speed_deg_s", "mean_peak_speed_deg_s", "mean_frequency_hz",
    "mean_confidence", "detection_rate",
]

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

# ── add_noise.py: input == TRIMMED_VIDEOS_DIR above; output feeds straight
# back into ROOT_DIR above, so the two stay in sync without a separate key.
NOISE_INPUT_DIR = TRIMMED_VIDEOS_DIR
NOISE_OUTPUT_DIR = ROOT_DIR

# ── Video splitting (configs/config.yaml > split:), split_video.py only ─
SPLIT_ROOT_DIR = _resolve("split", "root_dir")
SPLIT_FRONTAL_DIR = _resolve("split", "frontal_dir")
SPLIT_LATERAL_DIR = _resolve("split", "lateral_dir")

# ── Run options (configs/config.yaml > run:) ────────────────────────────
FORCE_HAND_SIDE = _cfg["run"]["force_hand_side"]   # None = auto-detect, or "Left"/"Right"
FORCED_FPS = _cfg["run"]["forced_fps"]              # 0 = read fps from each video

# ── MediaPipe ────────────────────────────────────────────────────────────
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
INDEX_MCP, PINKY_MCP = 5, 17

# ── RTMPose (configs/config.yaml > rtmpose:) ────────────────────────────
RTMPOSE_MODE = _cfg["rtmpose"]["mode"]
RTMPOSE_BACKEND = _cfg["rtmpose"]["backend"]
RTMPOSE_DEVICE = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"

# RTMPose Wholebody 133-kp indices
WB_L_INDEX_MCP, WB_L_PINKY_MCP = 96, 108
WB_R_INDEX_MCP, WB_R_PINKY_MCP = 117, 129

# ── Peak/trough detection tuning (configs/config.yaml > cycle_detection:) ─
PEAK_PROMINENCE_DEG = _cfg["cycle_detection"]["peak_prominence_deg"]
PEAK_MIN_DISTANCE_S = _cfg["cycle_detection"]["peak_min_distance_s"]
