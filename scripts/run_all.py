import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import SleepDetectionPipeline


def main():
    print("="*70)
    print("RUNNING FULL END-TO-END SLEEP/WAKE EVENT DETECTION BENCHMARK")
    print("="*70)
    
    os.makedirs("results", exist_ok=True)
    pipeline = SleepDetectionPipeline("configs/config.yaml")
    
    res_base = pipeline.run_baseline()
    
    res_class = pipeline.run_classical_models()
    
    res_cnn = pipeline.run_cnn()
    
    all_models = [
        ("Baseline 0 (Threshold)", res_base),
        ("Logistic Regression", res_class["logistic_regression"]),
        ("Random Forest", res_class["random_forest"]),
        ("LightGBM", res_class["lightgbm"]),
        ("1D CNN", res_cnn)
    ]
    
    comparison_rows = []
    for name, res in all_models:
        sm = res["state_metrics"]
        em = res["event_metrics"]
        
        f1_30 = em[(em["event_type"] == "overall") & (em["tolerance_sec"] == 30)]["f1"].values[0]
        f1_120 = em[(em["event_type"] == "overall") & (em["tolerance_sec"] == 120)]["f1"].values[0]
        f1_300 = em[(em["event_type"] == "overall") & (em["tolerance_sec"] == 300)]["f1"].values[0]
        
        comparison_rows.append({
            "Model": name,
            "State F1": round(sm["f1"], 4),
            "State PR-AUC": round(sm["pr_auc"], 4),
            "Event F1 (±30s)": round(f1_30, 4),
            "Event F1 (±2m)": round(f1_120, 4),
            "Event F1 (±5m)": round(f1_300, 4)
        })
        
    df_comp = pd.DataFrame(comparison_rows)
    print("\n" + "="*70)
    print("FINAL MODEL BENCHMARK COMPARISON")
    print("="*70)
    try:
        print(df_comp.to_markdown(index=False))
    except Exception:
        print(df_comp.to_string(index=False))
    df_comp.to_csv("results/model_comparison.csv", index=False)
    print("\nSaved summary comparison to results/model_comparison.csv")
    
    df_abl = pipeline.run_ablations()
    df_abl.to_csv("results/ablation_results.csv", index=False)
    
    df_rob = pipeline.run_robustness_experiments()
    df_rob.to_csv("results/robustness_results.csv", index=False)
    
    print("\n" + "="*70)
    print("FULL BENCHMARK SUITE FINISHED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    main()
