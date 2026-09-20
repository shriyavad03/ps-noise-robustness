#!/usr/bin/env python
"""Degrade trimmed videos with sensor/motion/illumination noise.

Usage: python scripts/add_noise.py
"""
from ps_noise_robustness import noise

if __name__ == "__main__":
    noise.run_batch()
