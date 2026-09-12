#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import SleepDetectionPipeline


def main():
    print("Initializing Robustness & Perturbation Pipeline...")
    pipeline = SleepDetectionPipeline("configs/config.yaml")
    df_rob = pipeline.run_robustness_experiments()
    
    os.makedirs("results", exist_ok=True)
    df_rob.to_csv("results/robustness_results.csv", index=False)
    print("Saved results to results/robustness_results.csv")


if __name__ == "__main__":
    main()

