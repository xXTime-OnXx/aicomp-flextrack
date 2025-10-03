"""
Baseline Random Forest model for demand response classification
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any
import joblib


class BaselineRandomForest:
    """
    Baseline Random Forest classifier for demand response flag prediction
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 20,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: str = 'sqrt',
        class_weight: str = 'balanced',
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        """
        Initialize Random Forest classifier
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required in leaf
            max_features: Number of features to consider for best split
            class_weight: Weights for class imbalance
            random_state: Random seed
            n_jobs: Number of parallel jobs
            **kwargs: Additional arguments for RandomForestClassifier
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
        
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None):
        """
        Fit the model
        
        Args:
            X: Training features
            y: Training labels
            feature_names: Names of features
        """
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        
        self.feature_names = feature_names
        self.is_fitted = True
        
        print(f"Model trained on {X.shape[0]} samples with {X.shape[1]} features")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict demand response flags
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importances = self.model.feature_importances_
        
        if self.feature_names is not None:
            feature_importance = dict(zip(self.feature_names, importances))
        else:
            feature_importance = {f"feature_{i}": imp for i, imp in enumerate(importances)}
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features[:top_n])
    
    def save(self, filepath: str):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaselineRandomForest':
        """Load model from disk"""
        model_data = joblib.load(filepath)
        
        instance = cls()
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {filepath}")
        return instance