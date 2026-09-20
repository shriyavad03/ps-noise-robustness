# ps-noise-robustness

Noise-robustness evaluation of **MediaPipe** vs **RTMPose** hand-keypoint tracking on the
**pronation-supination (PS)** motor task, benchmarked against manually-annotated (CVAT) ground truth.

Videos are degraded along three independent noise axes -- illumination, motion blur, sensor noise --
at four severities each (clean / mild / moderate / strong), then run through both detectors to see
how the extracted kinematic features (amplitude, speed, frequency, cycle count) and keypoint-level
spatial accuracy hold up as video quality drops.

---

## Repository layout

```
ps-noise-robustness/
├── src/ps_noise_robustness/       # Main package (src-layout)
│   ├── config.py                  #   Loads configs/config.yaml
│   ├── video_split.py             #   Splits dual-panel recordings into frontal/lateral
│   ├── noise.py                   #   Illumination / motion-blur / sensor-noise augmentation
│   ├── setup_environment.py       #   GPU env bootstrap (installs, CUDA env)
│   ├── pose_extraction.py         #   MediaPipe + RTMPose per-frame keypoint extraction
│   ├── ground_truth.py            #   CVAT XML parsing + ground-truth file lookup
│   ├── video_id_utils.py          #   Hand-side detection, noise-path parsing
│   ├── angle_signal.py            #   Angle trace, phasor gap-filling, peak/trough detection
│   ├── cycle_metrics.py           #   Peak-to-peak cycle metric computation
│   ├── accuracy_metrics.py        #   Confidence, spatial error, cycle error vs. ground truth
│   ├── pipeline.py                #   Runs the full pipeline for one video
│   └── batch_runner.py            #   Recursive, checkpointed batch run + resume logic
├── configs/
│   └── config.example.yaml        #   Annotated template -- copy to config.yaml
├── scripts/
│   ├── split_video.py             #   Entry point: video splitting
│   ├── add_noise.py               #   Entry point: noise augmentation
│   ├── setup_environment.py       #   Entry point: GPU env bootstrap
│   └── run_pipeline.py            #   Entry point: main batch pipeline
├── environment.yml
├── pyproject.toml
├── LICENSE
└── CITATION.cff
```

---

## Pipeline overview

```
Trimmed_Videos/
    │
    ▼
scripts/split_video.py         (only needed for dual-panel source recordings)
    └── crops each video into frontal / lateral halves
    │
    ▼
scripts/add_noise.py
    └── writes ModifiedNoise/{video_id}/{noise_type}/{severity}/*.mp4
    │
    ▼
scripts/run_pipeline.py
    └── batch_runner.py          -- recursive, checkpointed batch loop
            ├── pose_extraction.py   MediaPipe + RTMPose keypoint tracking
            ├── ground_truth.py      CVAT ground-truth lookup + parsing
            ├── angle_signal.py      angle trace + peak/trough detection
            ├── cycle_metrics.py     per-cycle kinematic features
            └── accuracy_metrics.py  confidence / spatial / cycle-error vs. GT
    │
    ▼
results_noise/
    ├── ps_features_ALL_NOISE_VIDEOS.csv
    ├── ps_error_vs_gt_ALL_NOISE_VIDEOS.csv
    └── ps_spatial_accuracy_ALL_NOISE_VIDEOS.csv
```

---

## Quick start

```bash
# 1. Create the environment (video preprocessing)
conda env create -f environment.yml
conda activate pd_features

# 2. Configure
cp configs/config.example.yaml configs/config.yaml
# edit configs/config.yaml with your paths

# 3. (optional) split dual-panel recordings
python scripts/split_video.py

# 4. Generate noise-degraded videos
python scripts/add_noise.py

# 5. Run the extraction pipeline (GPU environment -- see Installation)
python scripts/setup_environment.py   # once per environment/session
python scripts/run_pipeline.py
```

---

## Installation

```bash
# Video preprocessing only (CPU, local machine):
conda env create -f environment.yml

# Extraction pipeline (adds MediaPipe, RTMPose, onnxruntime-gpu -- GPU machine):
pip install -e ".[pipeline]"
```

`scripts/run_pipeline.py` needs a GPU for RTMPose to run at a reasonable speed, and
`split_video.py` needs `ffmpeg` on PATH. `scripts/setup_environment.py` installs the GPU pipeline
packages and points onnxruntime at their CUDA libraries; skip it if your environment already has
the right packages and CUDA libraries set up.

---

## Configuration

All paths and tunables live in `configs/config.yaml` (copy from `configs/config.example.yaml`).
Everything is nested under a single `experiment_root` -- **change that one line** to your own
project folder and every other path below it moves with it:

```yaml
experiment_root: /Users/yourname/Experiment/YourProject   # <-- change this

paths:
  trimmed_videos_dir: data/Trimmed_Videos       # add_noise.py input
  root_dir: data/ModifiedNoise                  # add_noise.py output == run_pipeline.py input
  gt_dir: data/Annotations_GroundTruth          # CVAT ground-truth exports
  out_dir: results/results_noise                # run_pipeline.py CSV output
  model_cache: models/hand_landmarker.task      # auto-downloaded MediaPipe model

split:                                          # split_video.py only
  root_dir: data/PS_Data
  frontal_dir: data/PS_Data/Frontal
  lateral_dir: data/PS_Data/Noise_Videos/Lateral

run:
  force_hand_side: null   # null = auto-detect, or "Left" / "Right"
  forced_fps: 0

rtmpose:
  mode: performance
  backend: onnxruntime

cycle_detection:
  peak_prominence_deg: 15.0
  peak_min_distance_s: 0.3
```

Any single path can be swapped for a full `/...` path of its own to move just that one folder
outside `experiment_root`. Point at a different config file at runtime with the `PS_CONFIG`
environment variable. See the comments at the top of `configs/config.example.yaml` for the full
folder tree this expects.

---

## Input file formats

### Video layout expected by `run_pipeline.py`

```
ModifiedNoise/
  {video_id}/
    illumination/{clean,mild,moderate,strong}/*.mp4
    sensor_noise/{clean,mild,moderate,strong}/*.mp4
    motion_blur/{clean,mild,moderate,strong}/*.mp4
```

`video_id`, `noise_type`, and `severity` are read off the three path components directly above each
file, not off the filename, so noisy files can be named anything.

### Ground truth (CVAT)

Any file under `paths.gt_dir` whose stem matches `P{num}_{L|R}` (case-insensitive, zero-padding
optional -- `P5_L`, `P05_L`, `p05_l.xml`) is matched to the corresponding `video_id` automatically.

---

## Output files

| File | Grain | Content |
|---|---|---|
| `ps_features_ALL_NOISE_VIDEOS.csv` | video x hand x source | Kinematic features per video (amplitude, speed, frequency, cycles, detection quality) |
| `ps_error_vs_gt_ALL_NOISE_VIDEOS.csv` | video x source x feature | Feature-level error vs. ground truth (MAE, RMSE, bias, % error) |
| `ps_spatial_accuracy_ALL_NOISE_VIDEOS.csv` | video x hand x source | Frame-level spatial (pixel) localisation error of `index_mcp` / `pinky_mcp` vs. ground truth |

`run_pipeline.py` appends each video's rows to these CSVs as soon as it finishes and skips
already-checkpointed videos on re-run, so an interrupted batch can just be restarted.

---

## Design decisions

- **Phasor interpolation**: angle gaps are filled on the unit-complex-exponential representation of
  the angle, not the angle itself, so interpolation is safe across the +/-180 deg wrap.
- **Despiking**: detection runs shorter than 2 frames are discarded before interpolation, since a
  detector can briefly latch onto the wrong point on noisy video.
- **Fixed CSV column order**: the features-summary schema is pinned explicitly so a video with zero
  detected cycles for one source can't silently shift columns for its row.
- **Checkpointed, resumable batch**: each video's rows are written immediately and the batch resumes
  from the last checkpoint on re-run, rather than reprocessing everything after an interruption.

---

## Citation

```
@software{vadavalli_ps_noise_robustness,
  author  = {Vadavalli, Shriya},
  title   = {ps-noise-robustness},
  url     = {https://github.com/<your-username>/ps-noise-robustness},
  license = {MIT}
}
```

See `CITATION.cff` for the full citation metadata.

---

## License

MIT -- see [LICENSE](LICENSE).
