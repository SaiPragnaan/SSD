import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from src.data import (
    load_series_and_events,
    split_series_ids,
    create_step_labels,
    summarize_dataset
)
from src.features import extract_features_dataset
from src.datasets import extract_raw_windows_dataset, build_dataloaders
from src.preprocessing import (
    inject_gaussian_noise,
    inject_missing_chunks,
    inject_sensor_spikes,
    inject_constant_offset
)
from src.models.baseline import ActivityThresholdBaseline
from src.models.classical import ClassicalSleepClassifier
from src.models.cnn import Sleep1DCNN, train_torch_model, predict_torch_model
from src.postprocess import extract_events_from_probabilities
from src.evaluation import (
    evaluate_state_classification,
    evaluate_event_detection,
    print_event_metrics_table
)


class SleepDetectionPipeline:
    """
    End-to-end orchestration pipeline for Sleep/Wake Event Detection.
    """
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.sample_rate = self.config["data"].get("sample_rate_sec", 5)
        self.tolerances = self.config["evaluation"].get("tolerances_seconds", [30, 60, 120, 300, 900])
        
        self.df_series, self.df_events = load_series_and_events(
            series_path=self.config["data"].get("series_path", "data/train_series.parquet"),
            events_path=self.config["data"].get("events_path", "data/train_events.csv"),
            fallback_synthetic=True,
            synthetic_kwargs=self.config["data"].get("synthetic", {})
        )
        
        self.labels = create_step_labels(self.df_series, self.df_events)
        
        series_ids = self.df_series["series_id"].unique().tolist()
        self.splits = split_series_ids(
            series_ids,
            train_ratio=self.config["splits"].get("train_ratio", 0.70),
            val_ratio=self.config["splits"].get("val_ratio", 0.15),
            test_ratio=self.config["splits"].get("test_ratio", 0.15),
            random_seed=self.config["splits"].get("random_seed", 42)
        )
        print(f"Dataset split by series_id: {len(self.splits['train'])} train, {len(self.splits['val'])} val, {len(self.splits['test'])} test series.")
        
    def get_split_dfs(self, split_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        target_series = self.splits[split_name]
        mask_series = self.df_series["series_id"].isin(target_series)
        df_sub = self.df_series[mask_series].copy()
        labels_sub = self.labels[mask_series].copy()
        events_sub = self.df_events[self.df_events["series_id"].isin(target_series)].copy()
        return df_sub, events_sub, labels_sub
        
    def run_baseline(self) -> Dict[str, Any]:
        """
        Runs Phase 2: Baseline 0 - Rule-based Activity Threshold.
        """
        print("\n" + "="*60 + "\n--- Running Baseline 0: Activity Threshold ---\n" + "="*60)
        train_df, _, train_y = self.get_split_dfs("train")
        val_df, _, val_y = self.get_split_dfs("val")
        test_df, test_events, test_y = self.get_split_dfs("test")
        
        win_size = self.config["windowing"].get("window_size_steps", 60)
        stride = self.config["windowing"].get("stride_steps", 12)
        
        X_train, y_train, _ = extract_features_dataset(train_df, train_y, win_size, stride, use_anglez_stats=False, use_derivatives=False, use_spectral=False, use_temporal_context=False)
        X_val, y_val, _ = extract_features_dataset(val_df, val_y, win_size, stride, use_anglez_stats=False, use_derivatives=False, use_spectral=False, use_temporal_context=False)
        X_test, y_test, meta_test = extract_features_dataset(test_df, test_y, win_size, stride, use_anglez_stats=False, use_derivatives=False, use_spectral=False, use_temporal_context=False)
        
        baseline = ActivityThresholdBaseline()
        baseline.fit(X_train, y_train)
        
        test_probs = baseline.predict_proba(X_test)
        test_preds = (test_probs >= 0.5).astype(int)
        
        state_metrics = evaluate_state_classification(y_test, test_preds, test_probs)

        pred_events = extract_events_from_probabilities(
            meta_test, test_probs,
            sleep_onset_threshold=0.6,
            wake_threshold=0.4,
            smooth_window=5
        )
        
        event_metrics = evaluate_event_detection(test_events, pred_events, tolerances_seconds=self.tolerances, sample_rate_sec=self.sample_rate)
        print_event_metrics_table(event_metrics, "Baseline 0: Activity Threshold Event Metrics")
        
        return {
            "model_name": "Activity Threshold (Baseline 0)",
            "state_metrics": state_metrics,
            "event_metrics": event_metrics,
            "pred_events": pred_events
        }
        
    def run_classical_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Runs Phase 5: Classical ML Models (Logistic Regression, Random Forest, LightGBM).
        """
        print("\n" + "="*60 + "\n--- Running Classical ML Models ---\n" + "="*60)
        train_df, _, train_y = self.get_split_dfs("train")
        val_df, _, val_y = self.get_split_dfs("val")
        test_df, test_events, test_y = self.get_split_dfs("test")
        
        win_size = self.config["windowing"].get("window_size_steps", 60)
        stride = self.config["windowing"].get("stride_steps", 12)
        
        print("Extracting engineered features (statistical, dynamic, spectral, temporal)...")
        X_train, y_train, _ = extract_features_dataset(train_df, train_y, win_size, stride)
        X_test, y_test, meta_test = extract_features_dataset(test_df, test_y, win_size, stride)
        
        results = {}
        for m_type in ["logistic_regression", "random_forest", "lightgbm"]:
            print(f"\nFitting {m_type.upper()} on {len(X_train)} windows ({X_train.shape[1]} features)...")
            clf = ClassicalSleepClassifier(model_type=m_type, params=self.config["models"].get(m_type, {}))
            clf.fit(X_train, y_train)
            
            test_probs = clf.predict_proba(X_test)
            test_preds = (test_probs >= 0.5).astype(int)
            
            state_metrics = evaluate_state_classification(y_test, test_preds, test_probs)
            
            pp_cfg = self.config["postprocessing"]
            pred_events = extract_events_from_probabilities(
                meta_test, test_probs,
                sleep_onset_threshold=pp_cfg.get("sleep_onset_threshold", 0.75),
                wake_threshold=pp_cfg.get("wake_threshold", 0.25),
                min_onset_persistence_steps=pp_cfg.get("min_onset_persistence_steps", 12),
                min_wake_persistence_steps=pp_cfg.get("min_wake_persistence_steps", 6),
                min_event_separation_steps=pp_cfg.get("min_event_separation_steps", 36),
                smooth_window=pp_cfg.get("smooth_window", 7)
            )
            
            event_metrics = evaluate_event_detection(test_events, pred_events, tolerances_seconds=self.tolerances, sample_rate_sec=self.sample_rate)
            print_event_metrics_table(event_metrics, f"{m_type.upper()} Event Detection Performance")
            
            results[m_type] = {
                "model_name": m_type,
                "model": clf,
                "state_metrics": state_metrics,
                "event_metrics": event_metrics,
                "pred_events": pred_events
            }
            
        return results

    def run_cnn(self) -> Dict[str, Any]:
        """
        Runs Phase 8: 1D CNN over raw continuous sensor windows.
        """
        print("\n" + "="*60 + "\n--- Running 1D CNN on Raw Accelerometer Signals ---\n" + "="*60)
        train_df, _, train_y = self.get_split_dfs("train")
        val_df, _, val_y = self.get_split_dfs("val")
        test_df, test_events, test_y = self.get_split_dfs("test")
        
        win_size = self.config["windowing"].get("window_size_steps", 60)
        stride = self.config["windowing"].get("stride_steps", 12)
        
        train_raw = extract_raw_windows_dataset(train_df, train_y, win_size, stride)
        val_raw = extract_raw_windows_dataset(val_df, val_y, win_size, stride)
        test_raw = extract_raw_windows_dataset(test_df, test_y, win_size, stride)
        
        cnn_cfg = self.config["models"]["cnn"]
        loaders = build_dataloaders(train_raw, val_raw, test_raw, batch_size=cnn_cfg.get("batch_size", 64))
        
        model = Sleep1DCNN(
            in_channels=cnn_cfg.get("in_channels", 2),
            conv_filters=tuple(cnn_cfg.get("conv_filters", [32, 64, 128])),
            kernel_sizes=tuple(cnn_cfg.get("kernel_sizes", [7, 5, 3])),
            dense_dim=cnn_cfg.get("dense_dim", 64),
            dropout=cnn_cfg.get("dropout", 0.3)
        )
        
        trained_model, history = train_torch_model(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            epochs=cnn_cfg.get("epochs", 15),
            lr=cnn_cfg.get("learning_rate", 0.001),
            weight_decay=cnn_cfg.get("weight_decay", 1e-4)
        )
        
        test_probs = predict_torch_model(trained_model, loaders["test"])
        y_test = test_raw[1]
        meta_test = test_raw[2]
        test_preds = (test_probs >= 0.5).astype(int)
        
        state_metrics = evaluate_state_classification(y_test, test_preds, test_probs)
        
        pp_cfg = self.config["postprocessing"]
        pred_events = extract_events_from_probabilities(
            meta_test, test_probs,
            sleep_onset_threshold=pp_cfg.get("sleep_onset_threshold", 0.75),
            wake_threshold=pp_cfg.get("wake_threshold", 0.25),
            min_onset_persistence_steps=pp_cfg.get("min_onset_persistence_steps", 12),
            min_wake_persistence_steps=pp_cfg.get("min_wake_persistence_steps", 6),
            min_event_separation_steps=pp_cfg.get("min_event_separation_steps", 36),
            smooth_window=pp_cfg.get("smooth_window", 7)
        )
        
        event_metrics = evaluate_event_detection(test_events, pred_events, tolerances_seconds=self.tolerances, sample_rate_sec=self.sample_rate)
        print_event_metrics_table(event_metrics, "1D CNN Event Detection Performance")
        
        return {
            "model_name": "1D CNN",
            "model": trained_model,
            "state_metrics": state_metrics,
            "event_metrics": event_metrics,
            "pred_events": pred_events,
            "history": history
        }

    def run_ablations(self) -> pd.DataFrame:
        """
        Runs Phase 13 Ablation Studies:
          1. ENMO Only vs ENMO + ANGLEZ
          2. Raw Statistics vs + Temporal Difference/Derivative Features
          3. Classical ML vs 1D CNN
          4. Raw Unfiltered Predictions vs Hysteresis Post-Processing
        """
        print("\n" + "="*60 + "\n--- Running Phase 13: Ablation Studies ---\n" + "="*60)
        train_df, _, train_y = self.get_split_dfs("train")
        test_df, test_events, test_y = self.get_split_dfs("test")
        win_size = self.config["windowing"].get("window_size_steps", 60)
        stride = self.config["windowing"].get("stride_steps", 12)
        
        ablation_results = []
        
        # Ablation 1: ENMO only vs ENMO + ANGLEZ
        print("[Ablation 1] Evaluating Feature Signal Source: ENMO only vs ENMO + ANGLEZ...")
        for name, use_angle in [("ENMO Only", False), ("ENMO + ANGLEZ", True)]:
            X_tr, y_tr, _ = extract_features_dataset(train_df, train_y, win_size, stride, use_anglez_stats=use_angle)
            X_te, y_te, meta_te = extract_features_dataset(test_df, test_y, win_size, stride, use_anglez_stats=use_angle)
            clf = ClassicalSleepClassifier(model_type="lightgbm").fit(X_tr, y_tr)
            probs = clf.predict_proba(X_te)
            ev = extract_events_from_probabilities(meta_te, probs)
            m = evaluate_event_detection(test_events, ev, tolerances_seconds=[120], sample_rate_sec=self.sample_rate)
            f1_120 = m[m["event_type"] == "overall"]["f1"].values[0]
            ablation_results.append({"Ablation": "1. Signal Source", "Variant": name, "Event F1 (±2m)": round(f1_120, 4)})
            
        # Ablation 2: Raw Stats vs + Derivatives
        print("[Ablation 2] Evaluating Feature Families: Raw Stats vs + Derivatives...")
        for name, use_deriv in [("Static Stats Only", False), ("Stats + Derivatives", True)]:
            X_tr, y_tr, _ = extract_features_dataset(train_df, train_y, win_size, stride, use_derivatives=use_deriv)
            X_te, y_te, meta_te = extract_features_dataset(test_df, test_y, win_size, stride, use_derivatives=use_deriv)
            clf = ClassicalSleepClassifier(model_type="lightgbm").fit(X_tr, y_tr)
            probs = clf.predict_proba(X_te)
            ev = extract_events_from_probabilities(meta_te, probs)
            m = evaluate_event_detection(test_events, ev, tolerances_seconds=[120], sample_rate_sec=self.sample_rate)
            f1_120 = m[m["event_type"] == "overall"]["f1"].values[0]
            ablation_results.append({"Ablation": "2. Dynamics", "Variant": name, "Event F1 (±2m)": round(f1_120, 4)})
            
        # Ablation 4: Raw Predictions vs Post-processed
        print("[Ablation 4] Evaluating Post-processing: Raw Argmax vs Hysteresis Filtering...")
        X_tr, y_tr, _ = extract_features_dataset(train_df, train_y, win_size, stride)
        X_te, y_te, meta_te = extract_features_dataset(test_df, test_y, win_size, stride)
        clf = ClassicalSleepClassifier(model_type="lightgbm").fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)
        
        # Raw naive  (no hysteresis, threshold=0.5)
        raw_events = extract_events_from_probabilities(meta_te, probs, sleep_onset_threshold=0.5, wake_threshold=0.5, min_onset_persistence_steps=1, min_wake_persistence_steps=1, min_event_separation_steps=1, smooth_window=1)
        m_raw = evaluate_event_detection(test_events, raw_events, tolerances_seconds=[120], sample_rate_sec=self.sample_rate)
        f1_raw = m_raw[m_raw["event_type"] == "overall"]["f1"].values[0]
        ablation_results.append({"Ablation": "4. Post-processing", "Variant": "Naive Threshold (No Hysteresis)", "Event F1 (±2m)": round(f1_raw, 4)})
        
        pp_events = extract_events_from_probabilities(meta_te, probs, sleep_onset_threshold=0.75, wake_threshold=0.25, min_onset_persistence_steps=12, min_wake_persistence_steps=6, min_event_separation_steps=36, smooth_window=7)
        m_pp = evaluate_event_detection(test_events, pp_events, tolerances_seconds=[120], sample_rate_sec=self.sample_rate)
        f1_pp = m_pp[m_pp["event_type"] == "overall"]["f1"].values[0]
        ablation_results.append({"Ablation": "4. Post-processing", "Variant": "Hysteresis + Persistence Filtering", "Event F1 (±2m)": round(f1_pp, 4)})
        
        df_abl = pd.DataFrame(ablation_results)
        print("\n### Ablation Study Summary")
        try:
            print(df_abl.to_markdown(index=False))
        except Exception:
            print(df_abl.to_string(index=False))
        return df_abl

    def run_robustness_experiments(self) -> pd.DataFrame:
        """
        Runs Phase 14 Sensor Perturbation & Robustness Experiments:
          - Clean Baseline
          - Gaussian Noise
          - Missing Sensor Drops (10%)
          - Mechanical Impact Spikes
          - Calibration Drift / Offset
        """
        print("\n" + "="*60 + "\n--- Running Phase 14: Sensor Robustness Perturbations ---\n" + "="*60)
        train_df, _, train_y = self.get_split_dfs("train")
        test_df, test_events, test_y = self.get_split_dfs("test")
        win_size = self.config["windowing"].get("window_size_steps", 60)
        stride = self.config["windowing"].get("stride_steps", 12)
        
        X_train, y_train, _ = extract_features_dataset(train_df, train_y, win_size, stride)
        clf = ClassicalSleepClassifier(model_type="lightgbm").fit(X_train, y_train)
        
        perturbations = {
            "Clean Baseline": test_df,
            "Gaussian Noise (σ=0.05)": inject_gaussian_noise(test_df, enmo_noise_std=0.05),
            "10% Missing Sensor Chunks": inject_missing_chunks(test_df, missing_fraction=0.10),
            "Impact Spikes (1% rate)": inject_sensor_spikes(test_df, spike_rate=0.01),
            "Sensor Angle Drift (+15°)": inject_constant_offset(test_df, enmo_offset=0.01, anglez_offset=15.0)
        }
        
        robustness_rows = []
        for name, df_pert in perturbations.items():
            X_test_p, y_test_p, meta_test_p = extract_features_dataset(df_pert, test_y, win_size, stride)
            probs_p = clf.predict_proba(X_test_p)
            events_p = extract_events_from_probabilities(meta_test_p, probs_p)
            
            m_event = evaluate_event_detection(test_events, events_p, tolerances_seconds=[120], sample_rate_sec=self.sample_rate)
            m_state = evaluate_state_classification(y_test_p, (probs_p >= 0.5).astype(int), probs_p)
            
            f1_120 = m_event[m_event["event_type"] == "overall"]["f1"].values[0]
            robustness_rows.append({
                "Perturbation Condition": name,
                "State PR-AUC": round(m_state["pr_auc"], 4),
                "State F1": round(m_state["f1"], 4),
                "Event F1 (±2m)": round(f1_120, 4)
            })
            
        df_rob = pd.DataFrame(robustness_rows)
        print("\n### Sensor Robustness Evaluation Summary")
        try:
            print(df_rob.to_markdown(index=False))
        except Exception:
            print(df_rob.to_string(index=False))
        return df_rob
