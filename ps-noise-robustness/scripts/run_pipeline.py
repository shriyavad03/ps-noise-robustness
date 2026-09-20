#!/usr/bin/env python
"""Main pipeline entry point: checkpointed recursive batch run over
configs paths.root_dir.

Usage: python scripts/run_pipeline.py
"""
from ps_noise_robustness.batch_runner import run_batch

if __name__ == "__main__":
    run_batch()
