"""Angle trace: raw angle -> phasor interpolation -> smoothing + peak/trough detection."""
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from . import config


def raw_angle_deg(index_xy, pinky_xy):
    """atan2 angle (deg, wrapped [-180,180]) index_mcp -> pinky_mcp, per frame.
    NaN wherever either point wasn't detected."""
    vx = pinky_xy[:, 0] - index_xy[:, 0]
    vy = pinky_xy[:, 1] - index_xy[:, 1]
    ang = np.degrees(np.arctan2(vy, vx))
    ang[np.isnan(vx) | np.isnan(vy)] = np.nan
    return ang


def phasor_interpolate(angle_deg):
    """Fill NaN gaps by interpolating the angle's phasor (unit complex
    exponential) rather than the angle itself -- wrap-safe across the
    +/-180 seam."""
    n = len(angle_deg)
    idx = np.arange(n)
    valid = ~np.isnan(angle_deg)
    if valid.sum() < 2:
        return angle_deg.copy()

    re = np.cos(np.radians(angle_deg))
    im = np.sin(np.radians(angle_deg))
    re_filled = np.interp(idx, idx[valid], re[valid])
    im_filled = np.interp(idx, idx[valid], im[valid])
    return np.degrees(np.arctan2(im_filled, re_filled))


def despike_isolated_detections(angle_deg, min_run=2):
    """Any run of consecutive valid detections shorter than min_run frames
    is re-marked NaN so phasor_interpolate() bridges over it instead of
    trusting it (more likely on noisy videos, where a detector can briefly
    latch onto the wrong point)."""
    valid = ~np.isnan(angle_deg)
    out = angle_deg.copy()
    run_start = None
    for i in range(len(valid) + 1):
        is_valid = valid[i] if i < len(valid) else False
        if is_valid and run_start is None:
            run_start = i
        elif not is_valid and run_start is not None:
            if i - run_start < min_run:
                out[run_start:i] = np.nan
            run_start = None
    return out


def condition_and_find_cycles(angle_raw, times, fps, label):
    angle_check = despike_isolated_detections(angle_raw, min_run=2)
    angle_phasor = phasor_interpolate(angle_check)
    angle_unwrapped = np.degrees(np.unwrap(np.radians(angle_phasor)))

    win = min(11, len(angle_unwrapped) - (1 - len(angle_unwrapped) % 2))
    win = max(win, 5)
    if win % 2 == 0:
        win += 1
    angle_conditioned = (savgol_filter(angle_unwrapped, win, 3)
                         if len(angle_unwrapped) > win else angle_unwrapped.copy())

    min_distance = max(int(config.PEAK_MIN_DISTANCE_S * fps), 1)
    peak_idx, _ = find_peaks(angle_conditioned, prominence=config.PEAK_PROMINENCE_DEG,
                              distance=min_distance)
    trough_idx, _ = find_peaks(-angle_conditioned, prominence=config.PEAK_PROMINENCE_DEG,
                                distance=min_distance)

    peak_val_map = {i: angle_conditioned[i] for i in peak_idx}
    trough_val_map = {i: angle_conditioned[i] for i in trough_idx}

    print(f"  [{label}] {len(peak_idx)} peaks, {len(trough_idx)} troughs detected "
          f"(prominence >= {config.PEAK_PROMINENCE_DEG} deg, min spacing {config.PEAK_MIN_DISTANCE_S}s)")

    return dict(angle_raw=angle_raw, angle_phasor=angle_phasor,
                angle_conditioned=angle_conditioned,
                peak_idx=peak_idx, trough_idx=trough_idx,
                peak_val_map=peak_val_map, trough_val_map=trough_val_map)
