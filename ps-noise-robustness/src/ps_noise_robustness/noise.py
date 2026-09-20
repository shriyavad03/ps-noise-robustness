"""Degrades trimmed videos with sensor, motion-blur, and illumination noise
at several severities, for the robustness batch (see batch_runner.py)."""
import os

import cv2
import numpy as np

from . import config

SENSOR_CONFIGS = [
    {"name": "sensor_clean",    "poisson_strength": 0.00},
    {"name": "sensor_mild",     "poisson_strength": 0.04},
    {"name": "sensor_moderate", "poisson_strength": 0.16},
    {"name": "sensor_strong",   "poisson_strength": 0.24},
]

MOTION_CONFIGS = [
    {"name": "motion_clean",    "ghost_alpha": 1.00},
    {"name": "motion_mild",     "ghost_alpha": 0.90},
    {"name": "motion_moderate", "ghost_alpha": 0.80},
    {"name": "motion_strong",   "ghost_alpha": 0.70},
]

ILLUMINATION_CONFIGS = [
    {"name": "illum_clean",    "brightness_jitter": 0},
    {"name": "illum_mild",     "brightness_jitter": 5},
    {"name": "illum_moderate", "brightness_jitter": 10},
    {"name": "illum_strong",   "brightness_jitter": 15},
]

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv",
                     ".MP4", ".AVI", ".MOV", ".MKV")


def add_poisson_noise(frame, strength=0.0, photon_budget=60.0):
    """Simulates camera-sensor shot noise at a given blend strength."""
    if strength <= 0:
        return frame
    frame_float = frame.astype(np.float32) / 255.0
    noisy = np.random.poisson(frame_float * photon_budget) / photon_budget
    output = frame_float * (1 - strength) + noisy * strength
    return np.clip(output * 255, 0, 255).astype(np.uint8)


def apply_temporal_ghosting(frame, previous_frame, alpha=0.7):
    """Blends in the previous frame to simulate motion-blur ghosting."""
    if previous_frame is None:
        return frame
    return cv2.addWeighted(frame, alpha, previous_frame, 1 - alpha, 0)


def apply_brightness_variation(frame, amount=10):
    """Random per-frame brightness jitter in [-amount, amount]."""
    if amount <= 0:
        return frame
    delta = np.random.randint(-amount, amount + 1)
    frame = frame.astype(np.int16) + delta
    return np.clip(frame, 0, 255).astype(np.uint8)


def process_video(input_video, cfg, output_path, noise_type):
    print(f"  [{noise_type}] {cfg['name']} -> {output_path}")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("  [ERROR] Cannot open input video -- skipping.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    previous_frame = None
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if noise_type == "sensor":
            frame = add_poisson_noise(frame, cfg["poisson_strength"])
        elif noise_type == "illumination":
            frame = apply_brightness_variation(frame, cfg["brightness_jitter"])

        # Ghosting is applied on the undegraded frame so consecutive
        # ghosted frames don't compound sensor/illumination noise.
        original_frame = frame.copy()
        if noise_type == "motion":
            frame = apply_temporal_ghosting(original_frame, previous_frame, cfg["ghost_alpha"])

        out.write(frame)
        previous_frame = original_frame

    cap.release()
    out.release()


def run_batch(input_folder=None, output_dir=None):
    input_folder = str(input_folder or config.NOISE_INPUT_DIR)
    output_dir = str(output_dir or config.NOISE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(input_folder) if f.endswith(VIDEO_EXTENSIONS)]
    print(f"Found {len(video_files)} video(s) in {input_folder}")

    for video_file in video_files:
        input_video = os.path.join(input_folder, video_file)
        base_name = os.path.splitext(video_file)[0]
        print(f"\nProcessing {base_name}")

        video_output_dir = os.path.join(output_dir, base_name)
        dirs = {
            "sensor": os.path.join(video_output_dir, "sensor_noise"),
            "motion": os.path.join(video_output_dir, "motion_blur"),
            "illumination": os.path.join(video_output_dir, "illumination"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        for noise_type, cfgs in [("sensor", SENSOR_CONFIGS),
                                  ("motion", MOTION_CONFIGS),
                                  ("illumination", ILLUMINATION_CONFIGS)]:
            for cfg in cfgs:
                output_path = os.path.join(dirs[noise_type], f"{base_name}_{cfg['name']}.mp4")
                process_video(input_video, cfg, output_path, noise_type)

    print("\nAll videos processed.")
