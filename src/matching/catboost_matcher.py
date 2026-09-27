try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

import pandas as pd
import numpy as np

class CatBoostMatcher:
    def __init__(self, params=None):
        """
        Initializes the CatBoost matching model.
        """
        if CatBoostClassifier is None:
            raise ImportError("catboost must be installed to use CatBoostMatcher.")
            
        if params is None:
            self.params = {
                'iterations': 300,
                'learning_rate': 0.05,
                'depth': 6,
                'loss_function': 'Logloss',
                'eval_metric': 'Logloss',
                'random_seed': 42,
                'verbose': 0
            }
        else:
            self.params = params
            
        self.model = CatBoostClassifier(**self.params)
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val=None, y_val=None):
        """Trains the CatBoost model."""
        print("Training CatBoost Matcher...")
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
            
        self.model.fit(
            X_train, 
            y_train,
            eval_set=eval_set,
            use_best_model=True if eval_set else False
        )
        self.is_fitted = True
        print("CatBoost Training complete.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importances(self) -> pd.DataFrame:
        if not self.is_fitted:
            return pd.DataFrame()
        return pd.DataFrame({
            'feature': self.model.feature_names_,
            'importance': self.model.get_feature_importance()
        }).sort_values('importance', ascending=False)
