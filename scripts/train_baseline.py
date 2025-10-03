"""
Training script for baseline Random Forest model
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.feature_engineering import TimeSeriesFeatureEngineer
from src.models.baseline_rf import BaselineRandomForest
from src.evaluation.metrics import print_evaluation_report


def load_data(data_path: str) -> pd.DataFrame:
    """Load training data"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {df['Site'].nunique()} sites")
    return df


def split_by_site(df: pd.DataFrame, test_sites: list = None, test_size: float = 0.2):
    """
    Split data into train/validation sets by site
    
    Args:
        df: Full dataframe
        test_sites: Specific sites to use for testing
        test_size: Fraction of sites to use for testing
    """
    sites = df['Site'].unique()
    
    if test_sites is None:
        # Randomly split sites
        np.random.seed(42)
        n_test_sites = max(1, int(len(sites) * test_size))
        test_sites = np.random.choice(sites, size=n_test_sites, replace=False)
    
    train_df = df[~df['Site'].isin(test_sites)].copy()
    val_df = df[df['Site'].isin(test_sites)].copy()
    
    print(f"\nTrain sites: {train_df['Site'].unique().tolist()}")
    print(f"Val sites: {val_df['Site'].unique().tolist()}")
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    return train_df, val_df


def main(args):
    """Main training pipeline"""
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    df = load_data(args.data_path)
    
    # Split data
    train_df, val_df = split_by_site(df, test_size=args.val_split)
    
    # Initialize feature engineer
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    feature_engineer = TimeSeriesFeatureEngineer(
        lag_features=[1, 2, 4, 8, 12, 24, 48, 96],
        rolling_windows=[4, 8, 12, 24, 48, 96]
    )
    
    # Transform training data
    train_features, train_target = feature_engineer.fit_transform(train_df)
    X_train = train_features[feature_engineer.feature_names].values
    y_train = train_target.values
    
    # Transform validation data
    val_features, val_target = feature_engineer.fit_transform(val_df)
    X_val = val_features[feature_engineer.feature_names].values
    y_val = val_target.values
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"\nClass distribution in training:")
    print(pd.Series(y_train).value_counts().sort_index())
    
    # Train model
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    model = BaselineRandomForest(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, feature_names=feature_engineer.feature_names)
    
    # Evaluate on training set
    print("\n" + "="*60)
    print("TRAINING SET EVALUATION")
    print("="*60)
    train_pred = model.predict(X_train)
    train_metrics = print_evaluation_report(y_train, train_pred, verbose=args.verbose)
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("VALIDATION SET EVALUATION")
    print("="*60)
    val_pred = model.predict(X_val)
    val_metrics = print_evaluation_report(y_val, val_pred, verbose=args.verbose)
    
    # Feature importance
    print("\n" + "="*60)
    print("TOP FEATURE IMPORTANCES")
    print("="*60)
    feature_importance = model.get_feature_importance(top_n=20)
    for i, (feature, importance) in enumerate(feature_importance.items(), 1):
        print(f"{i:2d}. {feature:50s}: {importance:.4f}")
    
    # Save model
    model_path = os.path.join(args.output_dir, 'baseline_model.pkl')
    model.save(model_path)
    
    # Save feature engineer
    import joblib
    feature_engineer_path = os.path.join(args.output_dir, 'feature_engineer.pkl')
    joblib.dump(feature_engineer, feature_engineer_path)
    print(f"\nFeature engineer saved to {feature_engineer_path}")
    
    # Save results
    results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'feature_importance': feature_importance,
        'feature_names': feature_engineer.feature_names
    }
    
    results_path = os.path.join(args.output_dir, 'training_results.pkl')
    joblib.dump(results, results_path)
    print(f"Results saved to {results_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nValidation Geometric Mean Score: {val_metrics['geometric_mean_score']:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline Random Forest model")
    
    # Data arguments
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to training CSV file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Directory to save model and results'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help='Fraction of sites to use for validation'
    )
    
    # Model arguments
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=200,
        help='Number of trees in Random Forest'
    )
    parser.add_argument(
        '--max_depth',
        type=int,
        default=20,
        help='Maximum depth of trees'
    )
    parser.add_argument(
        '--min_samples_split',
        type=int,
        default=10,
        help='Minimum samples required to split'
    )
    parser.add_argument(
        '--min_samples_leaf',
        type=int,
        default=5,
        help='Minimum samples required in leaf'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed'
    )
    
    # Other arguments
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed classification report'
    )
    
    args = parser.parse_args()
    main(args)