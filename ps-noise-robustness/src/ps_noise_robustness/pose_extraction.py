"""Per-frame index_mcp / pinky_mcp keypoint extraction, MediaPipe + RTMPose."""
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from . import config


def process_video_mediapipe(video_path, hand_side, video_fps, video_frames):
    """Returns index_xy, pinky_xy: (n_frames, 2) arrays, normalised [0,1],
    NaN on any frame the target hand wasn't detected; and conf: (n_frames,)
    array holding the handedness confidence score for the chosen hand
    (MediaPipe's HandLandmarker has no per-landmark score, so handedness
    score is the best available per-frame confidence proxy), NaN elsewhere.
    """
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(config.MODEL_CACHE)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    detector = mp_vision.HandLandmarker.create_from_options(hand_opts)
    cap = cv2.VideoCapture(str(video_path))
    NaN2 = np.array([np.nan, np.nan])
    index_xy, pinky_xy, conf = [], [], []
    try:
        for fi in range(video_frames):
            ret, frame = cap.read()
            if not ret:
                break
            t = fi / video_fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect_for_video(mp_img, int(t * 1000))

            chosen, chosen_score = None, np.nan
            if res.hand_landmarks:
                for handedness, lms in zip(res.handedness, res.hand_landmarks):
                    if handedness[0].category_name == hand_side:
                        chosen = lms
                        chosen_score = float(handedness[0].score)
                        break
            if chosen:
                index_xy.append(np.array([chosen[config.INDEX_MCP].x, chosen[config.INDEX_MCP].y]))
                pinky_xy.append(np.array([chosen[config.PINKY_MCP].x, chosen[config.PINKY_MCP].y]))
                conf.append(chosen_score)
            else:
                index_xy.append(NaN2.copy())
                pinky_xy.append(NaN2.copy())
                conf.append(np.nan)
    finally:
        cap.release()
        detector.close()
    return np.array(index_xy), np.array(pinky_xy), np.array(conf)


def process_video_rtmpose(video_path, hand_side, wholebody_model, video_frames,
                           video_w, video_h):
    """Returns index_xy, pinky_xy: (n_frames, 2) arrays, normalised [0,1];
    and conf: (n_frames,) array holding the mean of RTMPose's per-keypoint
    confidence for index_mcp & pinky_mcp on the selected person, NaN on
    frames with no detection."""
    idx_i, idx_p = (config.WB_L_INDEX_MCP, config.WB_L_PINKY_MCP) if hand_side == "Left" \
        else (config.WB_R_INDEX_MCP, config.WB_R_PINKY_MCP)
    cap = cv2.VideoCapture(str(video_path))
    NaN2 = np.array([np.nan, np.nan])
    index_xy, pinky_xy, conf = [], [], []
    try:
        for fi in range(video_frames):
            ret, frame = cap.read()
            if not ret:
                break
            keypoints, sc_all = wholebody_model(frame)
            if keypoints is not None and len(keypoints) > 0:
                bp = int(np.argmax(sc_all[:, idx_i]))
                kp = keypoints[bp]
                index_xy.append(np.array([kp[idx_i, 0] / video_w, kp[idx_i, 1] / video_h]))
                pinky_xy.append(np.array([kp[idx_p, 0] / video_w, kp[idx_p, 1] / video_h]))
                conf.append(float(np.mean([sc_all[bp, idx_i], sc_all[bp, idx_p]])))
            else:
                index_xy.append(NaN2.copy())
                pinky_xy.append(NaN2.copy())
                conf.append(np.nan)
    finally:
        cap.release()
    return np.array(index_xy), np.array(pinky_xy), np.array(conf)
