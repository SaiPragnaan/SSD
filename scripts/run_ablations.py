#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import SleepDetectionPipeline


def main():
    print("Initializing Ablation Studies Pipeline...")
    pipeline = SleepDetectionPipeline("configs/config.yaml")
    df_abl = pipeline.run_ablations()
    
    os.makedirs("results", exist_ok=True)
    df_abl.to_csv("results/ablation_results.csv", index=False)
    print("Saved results to results/ablation_results.csv")


if __name__ == "__main__":
    main()

