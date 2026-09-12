#!/usr/bin/env python3
import sys
import os

# Add root directory to pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import SleepDetectionPipeline


def main():
    print("Initializing Sleep Detection Pipeline...")
    pipeline = SleepDetectionPipeline("configs/config.yaml")
    results = pipeline.run_baseline()
    print("Baseline 0 completed successfully.")


if __name__ == "__main__":
    main()

