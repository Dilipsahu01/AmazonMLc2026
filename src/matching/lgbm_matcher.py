import lightgbm as lgb
import pandas as pd
import numpy as np

class LGBMMatcher:
    def __init__(self, params=None):
        """
        Initializes the LightGBM matching model.
        Default parameters are tuned for binary classification with unbalanced data.
        """
        if params is None:
            self.params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': -1,
                'feature_fraction': 0.8,
                'n_estimators': 300,
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            self.params = params
            
        self.model = lgb.LGBMClassifier(**self.params)
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val=None, y_val=None):
        """
        Trains the LightGBM model. 
        If validation data is provided, it uses early stopping.
        """
        print("Training LightGBM Matcher...")
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        self.model.fit(
            X_train, 
            y_train,
            eval_set=eval_set,
            eval_metric='binary_logloss'
        )
        self.is_fitted = True
        print("Training complete.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns the probability that the pair is a MATCH."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        
        # predict_proba returns [prob_class_0, prob_class_1]
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importances(self) -> pd.DataFrame:
        """Returns a dataframe of feature importances."""
        if not self.is_fitted:
            return pd.DataFrame()
            
        return pd.DataFrame({
            'feature': self.model.feature_name_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
