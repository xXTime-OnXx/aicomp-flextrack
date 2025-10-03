"""
Prediction script for generating submission files
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baseline_rf import BaselineRandomForest
from src.utils.feature_engineering import TimeSeriesFeatureEngineer


def load_test_data(test_path: str) -> pd.DataFrame:
    """Load test data"""
    print(f"Loading test data from {test_path}...")
    df = pd.read_csv(test_path)
    print(f"Loaded {len(df)} rows from {df['Site'].nunique()} sites")
    return df


def generate_predictions(
    model: BaselineRandomForest,
    feature_engineer: TimeSeriesFeatureEngineer,
    test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate predictions for test data
    
    Args:
        model: Trained model
        feature_engineer: Fitted feature engineer
        test_df: Test dataframe
        
    Returns:
        Dataframe with predictions
    """
    print("\nGenerating features for test data...")
    
    # Transform test data
    test_features, _ = feature_engineer.fit_transform(test_df.copy())
    X_test = test_features[feature_engineer.feature_names].values
    
    print(f"Test data shape: {X_test.shape}")
    
    # Generate predictions
    print("Making predictions...")
    predictions = model.predict(X_test)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'Site': test_df['Site'],
        'Timestamp_Local': test_df['Timestamp_Local'],
        'Demand_Response_Flag': predictions
    })
    
    print("\nPrediction distribution:")
    print(submission_df['Demand_Response_Flag'].value_counts().sort_index())
    
    return submission_df


def main(args):
    """Main prediction pipeline"""
    
    # Load model
    print("="*60)
    print("LOADING MODEL")
    print("="*60)
    model = BaselineRandomForest.load(args.model_path)
    
    # Load feature engineer
    feature_engineer_path = os.path.join(
        os.path.dirname(args.model_path),
        'feature_engineer.pkl'
    )
    
    if os.path.exists(feature_engineer_path):
        print(f"Loading feature engineer from {feature_engineer_path}")
        feature_engineer = joblib.load(feature_engineer_path)
    else:
        print("Warning: Feature engineer not found, creating new one")
        feature_engineer = TimeSeriesFeatureEngineer()
    
    # Load test data
    print("\n" + "="*60)
    print("LOADING TEST DATA")
    print("="*60)
    test_df = load_test_data(args.test_path)
    
    # Generate predictions
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    submission_df = generate_predictions(model, feature_engineer, test_df)
    
    # Save submission
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)
    
    submission_df.to_csv(output_path, index=False)
    print(f"\n✅ Submission saved to: {output_path}")
    
    # Display sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    print(submission_df.head(20))
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions for test data")
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model (.pkl file)'
    )
    parser.add_argument(
        '--test_path',
        type=str,
        required=True,
        help='Path to test CSV file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Directory to save predictions'
    )
    parser.add_argument(
        '--output_filename',
        type=str,
        default='submission.csv',
        help='Name of output submission file'
    )
    
    args = parser.parse_args()
    main(args)