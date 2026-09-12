import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class ClassicalSleepClassifier:
    """
    Unified interface for classical tabular machine learning models:
      1. Logistic Regression
      2. Random Forest
      3. LightGBM / Gradient Boosting
    """
    def __init__(self, model_type: str = "lightgbm", params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type.lower()
        self.params = params or {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model = self._init_model()
        
    def _init_model(self):
        if self.model_type == "logistic_regression":
            lr_params = {
                "C": 1.0,
                "max_iter": 500,
                "class_weight": "balanced",
                "random_state": 42
            }
            lr_params.update(self.params)
            return LogisticRegression(**lr_params)
            
        elif self.model_type == "random_forest":
            rf_params = {
                "n_estimators": 100,
                "max_depth": 12,
                "n_jobs": -1,
                "random_state": 42
            }
            rf_params.update(self.params)
            return RandomForestClassifier(**rf_params)
            
        elif self.model_type in ["lightgbm", "lgbm"]:
            try:
                import lightgbm as lgb
                lgb_params = {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "random_state": 42,
                    "verbose": -1
                }
                lgb_params.update(self.params)
                return lgb.LGBMClassifier(**lgb_params)
            except ImportError:
                print("LightGBM not installed. Falling back to sklearn GradientBoostingClassifier.")
                gb_params = {
                    "n_estimators": 150,
                    "learning_rate": 0.05,
                    "max_depth": 5,
                    "random_state": 42
                }
                return GradientBoostingClassifier(**gb_params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
            
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ClassicalSleepClassifier":
        self.feature_names = list(X.columns)
        X_mat = X.values
        
        if self.model_type == "logistic_regression":
            X_mat = self.scaler.fit_transform(X_mat)
            
        self.model.fit(X_mat, y)
        return self
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns 1D array of predicted sleep probabilities P(sleep = 1).
        """
        X_mat = X.values
        if self.model_type == "logistic_regression":
            X_mat = self.scaler.transform(X_mat)
        probs = self.model.predict_proba(X_mat)
        # Class 1 is sleep
        return probs[:, 1]
        
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
        
    def get_feature_importances(self) -> pd.DataFrame:
        """
        Returns feature importance table for tree models or absolute coefficients for linear models.
        """
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            imp = np.abs(self.model.coef_[0])
        else:
            imp = np.zeros(len(self.feature_names))
            
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": imp
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df_imp

