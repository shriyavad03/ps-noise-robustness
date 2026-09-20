"""Recursive, checkpointed batch run over config.ROOT_DIR.

Discovers every video recursively under ROOT_DIR. For each file, video_id /
noise_type / severity are read off the three path components directly
above it (ROOT_DIR/video_id/noise_type/severity/file.*), so it doesn't
matter what the file itself is named. Hand side and ground truth are
resolved from video_id, not from the noisy filename. Each video's row is
appended to the output CSV as soon as it finishes, and videos already
present in that CSV are skipped on re-run -- so an interrupted run can just
be restarted.
"""
import traceback

import pandas as pd

from . import config
from .ground_truth import find_gt_for_video
from .pipeline import run_pipeline_for_noise_video
from .video_id_utils import detect_side, parse_noise_path


def _load_last_done_key(csv_path):
    """(video_id, noise_type, severity, filename) of the last row written
    to csv_path, used to resume the batch from where it left off.

    dtype=str matters here: without it, a purely-numeric folder name (e.g.
    severity "5") could be read back as an int and fail to match the
    string keys built from Path.parts.
    """
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, dtype=str)
        if len(df) == 0:
            return None
        last = df.iloc[-1]
        return (last["video_id"], last["noise_type"], last["severity"], last["filename"])
    except Exception as e:
        print(f"  [WARN] Could not read {csv_path} to resume ({e}); starting from scratch.")
        return None


def discover_videos(root_dir):
    return sorted(p for p in root_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in config.VIDEO_EXTS)


def resume_start_index(video_files, root_dir, features_csv):
    """Index into video_files to resume from, based on the last checkpointed
    video found in features_csv."""
    last_key = _load_last_done_key(features_csv)
    if last_key is None:
        print(f"[RESUME] No existing progress found at {features_csv} -- starting from the beginning.")
        return 0

    print(f"[RESUME] Reading progress from {features_csv} ...")
    print(f"[RESUME] Last checkpointed video: {last_key[0]}/{last_key[1]}/{last_key[2]}/{last_key[3]}")
    file_keys = [(*parse_noise_path(vp, root_dir), vp.name) for vp in video_files]
    if last_key in file_keys:
        return file_keys.index(last_key) + 1
    print(f"  [WARN] That video is no longer among the {len(video_files)} videos discovered "
          f"under ROOT_DIR (renamed/moved/deleted?) -- starting from the beginning instead.")
    return 0


def run_batch():
    print(f"[CHECK] ROOT_DIR = {config.ROOT_DIR.resolve()}  (exists: {config.ROOT_DIR.exists()})")
    video_files = discover_videos(config.ROOT_DIR)
    print(f"Found {len(video_files)} video(s) under {config.ROOT_DIR}")

    start_idx = resume_start_index(video_files, config.ROOT_DIR, config.FEATURES_CSV)
    remaining = video_files[start_idx:]
    if start_idx > 0:
        if remaining:
            print(f"[RESUME] Continuing from: {remaining[0]}  ({len(remaining)} video(s) left)")
        else:
            print("[RESUME] All discovered videos are already checkpointed -- nothing left to do.")

    shared_wholebody = None
    n_written_this_run = 0

    for vp in remaining:
        video_id, noise_type, severity = parse_noise_path(vp, config.ROOT_DIR)
        hand_side = config.FORCE_HAND_SIDE or detect_side(video_id)
        print(f"\n{'=' * 70}\n{video_id} | {noise_type} | {severity} | {vp.name} "
              f"| Hand side: {hand_side}\n{'=' * 70}")

        try:
            xml_path_v = find_gt_for_video(video_id, config.GT_DIR)
            if xml_path_v is not None:
                print(f"  [GT] Using ground truth: {xml_path_v}")
            summary_df_v, err_df_v, spatial_df_v, shared_wholebody = run_pipeline_for_noise_video(
                vp, video_id, hand_side, xml_path=xml_path_v,
                forced_fps=config.FORCED_FPS, wholebody_model=shared_wholebody)

            summary_df_v.insert(1, "noise_type", noise_type)
            summary_df_v.insert(2, "severity", severity)
            summary_df_v.insert(3, "filename", vp.name)
            summary_df_v.insert(4, "hand_side", hand_side)

            if len(err_df_v):
                err_df_v = err_df_v.rename(columns={"video": "video_id"})
                err_df_v.insert(1, "noise_type", noise_type)
                err_df_v.insert(2, "severity", severity)
                err_df_v.insert(3, "filename", vp.name)

            if len(spatial_df_v):
                spatial_df_v.insert(1, "noise_type", noise_type)
                spatial_df_v.insert(2, "severity", severity)
                spatial_df_v.insert(3, "filename", vp.name)
                spatial_df_v.insert(4, "hand_side", hand_side)

            header = not config.FEATURES_CSV.exists()
            summary_df_v.to_csv(config.FEATURES_CSV, mode="a", header=header, index=False)
            if len(err_df_v):
                header_e = not config.ERROR_CSV.exists()
                err_df_v.to_csv(config.ERROR_CSV, mode="a", header=header_e, index=False)
            if len(spatial_df_v):
                header_s = not config.SPATIAL_CSV.exists()
                spatial_df_v.to_csv(config.SPATIAL_CSV, mode="a", header=header_s, index=False)
            n_written_this_run += 1
            print(f"  -> checkpointed {video_id}/{noise_type}/{severity}/{vp.name}")
        except Exception:
            # Full traceback: a bare exception message is often useless for
            # figuring out where something failed.
            print(f"[ERROR] Failed on {video_id}/{noise_type}/{severity}/{vp.name}:")
            traceback.print_exc()

    print(f"\n[SUMMARY] {n_written_this_run}/{len(remaining)} video(s) checkpointed this run.")
    if n_written_this_run == 0 and len(remaining) > 0:
        print("[SUMMARY] WARNING: 0 videos were written this run despite videos being "
              "available to process -- scroll up for [ERROR] tracebacks to see why.")

    for csv_path in (config.FEATURES_CSV, config.ERROR_CSV, config.SPATIAL_CSV):
        if csv_path.exists():
            size_kb = csv_path.stat().st_size / 1024
            n_rows = sum(1 for _ in open(csv_path)) - 1  # minus header
            print(f"[SUMMARY] {csv_path} exists -- {size_kb:.1f} KB, {n_rows} data row(s).")
        else:
            print(f"[SUMMARY] {csv_path} does NOT exist on disk.")

    print(f"\nDone. Per-video, per-noise-type, per-severity, per-algorithm "
          f"(GT / RTMPose / MediaPipe) features -> {config.FEATURES_CSV}")
    if config.ERROR_CSV.exists():
        print(f"Error-vs-GT cycle metrics -> {config.ERROR_CSV}")
    if config.SPATIAL_CSV.exists():
        print(f"Spatial accuracy (keypoint pixel error) vs GT -> {config.SPATIAL_CSV}")


if __name__ == "__main__":
    run_batch()
