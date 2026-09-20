"""Peak-to-peak cycle metrics for a single algorithm's angle trace."""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .angle_signal import condition_and_find_cycles, raw_angle_deg


def peak_to_peak_cycles(times, angle_conditioned, peak_idx, trough_idx, fps,
                         peak_val_map, trough_val_map):
    """Cycle-by-cycle metrics anchored from the first detected peak."""
    dt = 1.0 / fps
    peak_idx = np.sort(np.asarray(peak_idx, dtype=int))
    trough_set = set(trough_idx.tolist())

    n_cycles = len(peak_idx) - 1
    if n_cycles < 1:
        empty = np.array([])
        return dict(cycle_num=empty, peak_start_t=empty, peak_end_t=empty,
                    trough_t=empty, duration_s=empty, amplitude_p2p_deg=empty,
                    mean_speed_deg_s=empty, peak_speed_deg_s=empty,
                    cumulative_path_deg=empty)

    (cycle_num, pk_start_t, pk_end_t,
     tr_t, durations, p2p_amp, mean_spd, peak_spd, cum_amp) = ([], [], [], [], [], [], [], [], [])

    for k in range(n_cycles):
        i0, i1 = peak_idx[k], peak_idx[k + 1]
        seg = angle_conditioned[i0:i1 + 1]
        if len(seg) < 3 or np.all(np.isnan(seg)):
            continue

        local_troughs = [idx for idx in trough_set if i0 < idx < i1]
        if local_troughs:
            tr_idx = min(local_troughs, key=lambda idx: angle_conditioned[idx])
            tr_val = trough_val_map.get(tr_idx, angle_conditioned[tr_idx])
        else:
            valid = ~np.isnan(seg)
            if valid.sum() < 2:
                continue
            seg_uw = seg.copy()
            seg_uw[valid] = np.degrees(np.unwrap(np.radians(seg[valid])))
            local_min = int(np.nanargmin(seg_uw))
            tr_idx = i0 + local_min
            tr_val = angle_conditioned[tr_idx]

        pk_start_val = peak_val_map.get(i0, angle_conditioned[i0])
        amp = float(abs(pk_start_val - tr_val))
        dur = float(times[i1] - times[i0])

        valid_seg = ~np.isnan(seg)
        seg_for_vel = seg.copy()
        if valid_seg.sum() >= 2:
            seg_for_vel[valid_seg] = np.degrees(np.unwrap(np.radians(seg[valid_seg])))
        inst_spd = gaussian_filter1d(np.abs(np.gradient(seg_for_vel, dt)), sigma=2)
        ps = float(np.nanmax(inst_spd))
        cum_path = float(np.nansum(np.abs(np.diff(seg_for_vel))))
        ms = cum_path / dur if dur > 0 else 0.0

        cycle_num.append(k + 1)
        pk_start_t.append(times[i0])
        pk_end_t.append(times[i1])
        tr_t.append(times[tr_idx])
        durations.append(dur)
        p2p_amp.append(amp)
        mean_spd.append(ms)
        peak_spd.append(ps)
        cum_amp.append(cum_path)

    return dict(
        cycle_num=np.array(cycle_num, dtype=int), peak_start_t=np.array(pk_start_t),
        peak_end_t=np.array(pk_end_t), trough_t=np.array(tr_t),
        duration_s=np.array(durations), amplitude_p2p_deg=np.array(p2p_amp),
        mean_speed_deg_s=np.array(mean_spd), peak_speed_deg_s=np.array(peak_spd),
        cumulative_path_deg=np.array(cum_amp))


def analyze_algorithm_no_plot(label, index_xy, pinky_xy, times, fps):
    """angle trace -> phasor fill -> conditioning -> peak/trough detection ->
    peak-to-peak cycle metrics. No figures, no per-source CSV: built for a
    large recursive batch where per-video inline figures would just slow
    things down."""
    angle_raw = raw_angle_deg(index_xy, pinky_xy)
    cond = condition_and_find_cycles(angle_raw, times, fps, label)
    pp = peak_to_peak_cycles(times, cond["angle_conditioned"], cond["peak_idx"],
                              cond["trough_idx"], fps, cond["peak_val_map"],
                              cond["trough_val_map"])
    pp_df = pd.DataFrame({
        "cycle": pp["cycle_num"],
        "peak_start_s": pp["peak_start_t"].round(3),
        "trough_s": pp["trough_t"].round(3),
        "peak_end_s": pp["peak_end_t"].round(3),
        "duration_s": pp["duration_s"].round(3),
        "amplitude_p2p_deg": pp["amplitude_p2p_deg"].round(1),
        "mean_speed_deg_s": pp["mean_speed_deg_s"].round(1),
        "peak_speed_deg_s": pp["peak_speed_deg_s"].round(1),
        "cumulative_path_deg": pp["cumulative_path_deg"].round(1),
    })
    return cond, pp_df
