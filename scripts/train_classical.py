#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import SleepDetectionPipeline


def main():
    print("Initializing Classical ML Training Pipeline...")
    pipeline = SleepDetectionPipeline("configs/config.yaml")
    results = pipeline.run_classical_models()
    print("Classical ML training and evaluation completed successfully.")


if __name__ == "__main__":
    main()

