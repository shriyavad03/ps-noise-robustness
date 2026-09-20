#!/usr/bin/env python
"""Split dual-panel recordings into frontal/lateral halves.

Usage: python scripts/split_video.py
"""
from ps_noise_robustness import video_split

if __name__ == "__main__":
    video_split.run_batch()
