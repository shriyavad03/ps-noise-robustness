"""Per-video accuracy reporting: tracker confidence, spatial (pixel) error
vs ground truth, and per-cycle-feature error vs ground truth."""
import numpy as np
import pandas as pd


def mean_confidence(conf_array):
    """Average confidence over frames with a detection (NaN = missed
    detection, tracked separately by detection_rate())."""
    conf_array = np.asarray(conf_array, dtype=float)
    valid = conf_array[~np.isnan(conf_array)]
    return float(np.mean(valid)) if len(valid) else np.nan


def detection_rate(conf_array):
    """Fraction of frames on which the algorithm returned a detection at all."""
    conf_array = np.asarray(conf_array, dtype=float)
    return float(np.mean(~np.isnan(conf_array))) if len(conf_array) else np.nan


def spatial_accuracy_vs_gt(pred_index_xy, pred_pinky_xy, gt_index_xy, gt_pinky_xy,
                            video_w, video_h, pred_is_normalized=True):
    """Per-frame Euclidean pixel distance between predicted and ground-truth
    keypoints, for index_mcp and pinky_mcp, over frames where both are
    valid. pred_*_xy are normalised [0,1]; gt_*_xy are pixel space.

    Returns MAE / RMSE (pixels) per keypoint and pooled, plus frame counts.
    """
    scale = np.array([video_w, video_h]) if pred_is_normalized else np.array([1.0, 1.0])
    pred_index_px = pred_index_xy * scale
    pred_pinky_px = pred_pinky_xy * scale

    def _errs(pred_px, gt_px):
        d = pred_px - gt_px
        valid = ~np.isnan(d).any(axis=1)
        if valid.sum() == 0:
            return np.array([]), 0
        return np.linalg.norm(d[valid], axis=1), int(valid.sum())

    err_index, n_index = _errs(pred_index_px, gt_index_xy)
    err_pinky, n_pinky = _errs(pred_pinky_px, gt_pinky_xy)
    err_pooled = np.concatenate([err_index, err_pinky]) if (n_index + n_pinky) else np.array([])

    def _stats(err):
        if len(err) == 0:
            return np.nan, np.nan
        return float(np.mean(err)), float(np.sqrt(np.mean(err ** 2)))

    idx_mae, idx_rmse = _stats(err_index)
    pk_mae, pk_rmse = _stats(err_pinky)
    pooled_mae, pooled_rmse = _stats(err_pooled)

    return dict(
        index_mcp_mae_px=idx_mae, index_mcp_rmse_px=idx_rmse, index_mcp_n=n_index,
        pinky_mcp_mae_px=pk_mae, pinky_mcp_rmse_px=pk_rmse, pinky_mcp_n=n_pinky,
        overall_mae_px=pooled_mae, overall_rmse_px=pooled_rmse,
        n_frames_compared=n_index + n_pinky,
    )


def match_cycles_to_gt(pred_df, gt_df, max_gap_s=None):
    """Nearest-neighbour match each predicted cycle to the closest
    ground-truth cycle by peak_start_s. A match whose time gap exceeds
    max_gap_s (default: median GT cycle duration) is flagged unmatched."""
    if len(pred_df) == 0 or len(gt_df) == 0:
        return pd.DataFrame()

    def _prep(df):
        d = df.copy().sort_values("peak_start_s").reset_index(drop=True)
        d["frequency_hz"] = 1.0 / d["duration_s"].replace(0, np.nan)
        return d

    p, g = _prep(pred_df), _prep(gt_df)
    g = g.rename(columns={"peak_start_s": "gt_peak_start_s"})
    if max_gap_s is None:
        max_gap_s = float(g["duration_s"].median())

    merged = pd.merge_asof(p, g, left_on="peak_start_s", right_on="gt_peak_start_s",
                            direction="nearest", suffixes=("", "_gt"))
    merged["peak_start_s_gt"] = merged["gt_peak_start_s"]
    merged["match_gap_s"] = (merged["peak_start_s"] - merged["gt_peak_start_s"]).abs()
    merged["matched"] = merged["match_gap_s"] <= max_gap_s
    return merged


def compute_error_metrics(merged, feature, gt_suffix="_gt"):
    """MAE / RMSE / mean signed bias / mean abs %% error for one feature,
    over matched cycles only."""
    m = merged[merged["matched"]]
    pred = m[feature].to_numpy(dtype=float)
    gt = m[f"{feature}{gt_suffix}"].to_numpy(dtype=float)
    valid = ~(np.isnan(pred) | np.isnan(gt))
    pred, gt = pred[valid], gt[valid]
    if len(pred) == 0:
        return dict(n=0, mae=np.nan, rmse=np.nan, bias=np.nan, pct_error=np.nan)
    err = pred - gt
    pct = np.where(gt != 0, 100 * err / gt, np.nan)
    return dict(n=len(pred), mae=float(np.mean(np.abs(err))),
                rmse=float(np.sqrt(np.mean(err ** 2))),
                bias=float(np.mean(err)), pct_error=float(np.nanmean(np.abs(pct))))


FEATURES_OF_INTEREST = {
    "cumulative_path_deg": ("Mean cumulative amplitude (deg)", True),
    "frequency_hz":        ("Mean frequency / speed (Hz, cycles-per-s)", True),
    "amplitude_p2p_deg":   ("Mean amplitude p2p (deg)", False),
    "mean_speed_deg_s":    ("Mean angular speed (deg/s)", False),
    "peak_speed_deg_s":    ("Mean peak angular speed (deg/s)", False),
    "duration_s":          ("Mean cycle duration (s)", False),
}


def error_report(results, video_name=""):
    """For every non-GT source, match its cycles to Ground truth and
    compute error metrics for each feature in FEATURES_OF_INTEREST. Tidy
    long-format DataFrame (one row per source x feature); empty if no GT."""
    if "Ground truth" not in results:
        return pd.DataFrame()

    _, gt_df = results["Ground truth"]
    rows = []
    for label, (_, pred_df) in results.items():
        if label == "Ground truth" or len(pred_df) == 0:
            continue
        merged = match_cycles_to_gt(pred_df, gt_df)
        if merged.empty:
            continue
        for feat, (nice_name, highlight) in FEATURES_OF_INTEREST.items():
            stats = compute_error_metrics(merged, feat)
            rows.append({"video": video_name, "source": label, "feature": feat,
                         "feature_name": nice_name, "highlight": highlight, **stats})

    return pd.DataFrame(rows)
