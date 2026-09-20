"""Runs the full extraction -> cycle-metrics -> GT-comparison pipeline for
one video."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rtmlib import Wholebody

from . import config
from .accuracy_metrics import (detection_rate, error_report, mean_confidence,
                               spatial_accuracy_vs_gt)
from .cycle_metrics import analyze_algorithm_no_plot
from .ground_truth import build_gt_arrays
from .pose_extraction import process_video_mediapipe, process_video_rtmpose


def run_pipeline_for_noise_video(video_path, video_id, hand_side, xml_path=None,
                                  forced_fps=0, wholebody_model=None):
    video_path = Path(video_path)
    cap_probe = cv2.VideoCapture(str(video_path))
    fps = float(forced_fps) if forced_fps > 0 else (cap_probe.get(cv2.CAP_PROP_FPS) or 30.0)
    n_frames_total = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()

    print("  [MediaPipe] running ...")
    idx_mp, pk_mp, conf_mp = process_video_mediapipe(video_path, hand_side, fps, n_frames_total)

    wb = wholebody_model or Wholebody(mode=config.RTMPOSE_MODE, backend=config.RTMPOSE_BACKEND,
                                       device=config.RTMPOSE_DEVICE)
    print("  [RTMPose] running (onnxruntime backend) ...")
    idx_rtm, pk_rtm, conf_rtm = process_video_rtmpose(video_path, hand_side, wb, n_frames_total, w, h)

    n_frames = min(len(idx_mp), len(idx_rtm))
    times_v = np.arange(n_frames) / fps
    idx_mp, pk_mp, conf_mp = idx_mp[:n_frames], pk_mp[:n_frames], conf_mp[:n_frames]
    idx_rtm, pk_rtm, conf_rtm = idx_rtm[:n_frames], pk_rtm[:n_frames], conf_rtm[:n_frames]

    have_gt = xml_path is not None and Path(xml_path).exists()
    res = {}
    res["MediaPipe"] = analyze_algorithm_no_plot("MediaPipe", idx_mp, pk_mp, times_v, fps)
    res["RTMPose"] = analyze_algorithm_no_plot("RTMPose", idx_rtm, pk_rtm, times_v, fps)

    gt_idx = gt_pk = None
    if have_gt:
        # Isolated try/except: a bad annotation file should only cost the
        # GT comparison for this video, not the MediaPipe/RTMPose results
        # already computed above.
        try:
            gt_idx, gt_pk = build_gt_arrays(xml_path, n_frames, w, h)
            res["Ground truth"] = analyze_algorithm_no_plot("Ground truth", gt_idx, gt_pk, times_v, fps)
        except Exception as e:
            print(f"  [WARN] Failed to build ground truth from {xml_path} ({e}) -- "
                  f"continuing with MediaPipe vs RTMPose only for this video.")
            have_gt = False
            gt_idx = gt_pk = None
    if not have_gt:
        print(f"  [INFO] No ground truth available for video_id={video_id} -- "
              f"MediaPipe vs RTMPose only.")

    conf_by_source = {"MediaPipe": conf_mp, "RTMPose": conf_rtm}

    summary_rows = []
    for label, (_, pp_df) in res.items():
        row = {"video_id": video_id, "source": label, "n_cycles": len(pp_df)}
        if len(pp_df):
            row.update({
                "mean_duration_s": pp_df["duration_s"].mean(),
                "mean_amplitude_p2p_deg": pp_df["amplitude_p2p_deg"].mean(),
                "mean_cumulative_amplitude_deg": pp_df["cumulative_path_deg"].mean(),
                "mean_speed_deg_s": pp_df["mean_speed_deg_s"].mean(),
                "mean_peak_speed_deg_s": pp_df["peak_speed_deg_s"].mean(),
                "mean_frequency_hz": (1.0 / pp_df["duration_s"]).mean(),
            })
        # Confidence is only meaningful for the two trackers, not GT.
        if label in conf_by_source:
            row["mean_confidence"] = mean_confidence(conf_by_source[label])
            row["detection_rate"] = detection_rate(conf_by_source[label])
        summary_rows.append(row)

    # Explicit fixed column order (see config.SUMMARY_COLUMNS) so a video
    # with 0 cycles for some source doesn't shift columns for its CSV rows.
    summary_df_v = pd.DataFrame(summary_rows, columns=config.SUMMARY_COLUMNS).round(3)

    err_df_v = error_report(res, video_id) if have_gt else pd.DataFrame()

    spatial_rows = []
    if have_gt:
        for label, pred_idx, pred_pk in [("MediaPipe", idx_mp, pk_mp), ("RTMPose", idx_rtm, pk_rtm)]:
            stats = spatial_accuracy_vs_gt(pred_idx, pred_pk, gt_idx, gt_pk, w, h)
            spatial_rows.append({"video_id": video_id, "source": label, **stats})
    spatial_df_v = pd.DataFrame(spatial_rows).round(2)

    return summary_df_v, err_df_v, spatial_df_v, wb
