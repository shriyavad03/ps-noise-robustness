#!/usr/bin/env python
"""Run once per GPU environment/session, before run_pipeline.py.

Usage: python scripts/setup_environment.py
"""
from ps_noise_robustness import setup_environment

if __name__ == "__main__":
    setup_environment.run_full_setup()
