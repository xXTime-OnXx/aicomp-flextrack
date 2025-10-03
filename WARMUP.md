# FlexTrack Challenge 2025 - Baseline Implementation (WARMUP PHASE) Summary

## 🎯 What's Included

This baseline implementation provides everything you need to get started with the FlexTrack Challenge warm-up phase:

### Core Components

1. **Random Forest Classifier** (`src/models/baseline_rf.py`)
   - Handles class imbalance with balanced weights
   - Includes feature scaling
   - Provides feature importance analysis
   - Save/load functionality

2. **Feature Engineering** (`src/utils/feature_engineering.py`)
   - Temporal features with cyclic encoding
   - Lag features (up to 96 timesteps = 24 hours)
   - Rolling statistics (multiple windows)
   - Difference and interaction features

3. **Evaluation Metrics** (`src/evaluation/metrics.py`)
   - Geometric Mean Score (primary metric)
   - F1-Score (secondary metric)
   - Comprehensive per-class metrics
   - Detailed classification reports

4. **Training Pipeline** (`scripts/train_baseline.py`)
   - Site-aware train/validation splitting
   - Progress logging
   - Model and pipeline serialization

5. **Prediction Pipeline** (`scripts/predict.py`)
   - Easy submission file generation
   - Consistent preprocessing

6. **EDA Notebook** (`notebooks/01_exploratory_data_analysis.ipynb`)
   - Comprehensive data exploration
   - Visualization of patterns
   - Feature correlation analysis

## 🚀 Quick Start

### Option 1: With Sample Data (No Real Data Required)
```bash
# Generate sample data and run full pipeline
./quick_start.sh
```

### Option 2: With Competition Data
```bash
# 1. Place your data in data/raw/
#    - train.csv
#    - test.csv

# 2. Train the model
python scripts/train_baseline.py --data_path data/classification/classification-train.csv

# 3. Generate predictions
python scripts/predict.py --model_path outputs/baseline_model.pkl --test_path data/classification/raw/classification-test.csv
```

## 📊 Expected Performance

This baseline should achieve:
- **Geometric Mean Score**: ~0.70-0.80 on validation set
- **F1-Score**: ~0.65-0.75 on validation set

Performance will vary based on:
- Data quality and size
- Site characteristics
- Hyperparameter tuning

## 🔧 Customization Points

### Easy Improvements (Low Effort, Good Return)

1. **Hyperparameter Tuning**
   ```python
   # In train_baseline.py, modify:
   --n_estimators 300        # More trees
   --max_depth 25           # Deeper trees
   --min_samples_leaf 2     # More granular splits
   ```

2. **Feature Engineering**
   ```python
   # In TimeSeriesFeatureEngineer, adjust:
   lag_features = [1, 2, 4, 8, 12, 24, 48, 96, 192]  # Add longer lags
   rolling_windows = [4, 8, 12, 24, 48, 96, 192]     # Add longer windows
   ```

3. **Add More Features**
   - Previous day same time power consumption
   - Hour-of-day specific rolling averages
   - Site-specific normalization

### Advanced Improvements (Higher Effort)

1. **Better Models**
   - XGBoost / LightGBM for gradient boosting
   - LSTM / GRU for sequence modeling
   - Transformer architectures for time-series

2. **Ensemble Methods**
   - Combine multiple model predictions
   - Stacking with meta-learner
   - Weighted voting

3. **Advanced Feature Engineering**
   - Fourier features for periodicity
   - Wavelet transforms
   - Autoregressive features

4. **Data Strategies**
   - SMOTE for class balancing
   - Time-series cross-validation
   - Site-specific models

## 📁 Project Structure

```
flextrack-baseline/
├── README.md                 # Project overview
├── USAGE_GUIDE.md           # Detailed usage instructions
├── requirements.txt         # Python dependencies
├── quick_start.sh          # Demo script
│
├── notebooks/              # Jupyter notebooks
│   └── 01_exploratory_data_analysis.ipynb
│
├── scripts/               # Executable scripts
│   ├── train_baseline.py
│   ├── predict.py
│   └── generate_sample_data.py
│
├── src/                   # Source code
│   ├── models/
│   │   └── baseline_rf.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── utils/
│       └── feature_engineering.py
│
├── data/                  # Data directory
│   ├── raw/              # Place competition data here
│   └── processed/        # Processed features (auto-generated)
│
└── outputs/              # Model outputs
    ├── baseline_model.pkl
    ├── feature_engineer.pkl
    └── submission.csv
```

## 🎓 Key Concepts

### Geometric Mean Score
The competition uses GMS to balance sensitivity and specificity:
```
GMS = sqrt(TPR × TNR)
```
This is better than accuracy for imbalanced datasets because:
- Penalizes poor performance on minority classes
- Rewards balanced predictions across all classes
- Range: 0 to 1 (higher is better)

### Feature Engineering Philosophy
The baseline focuses on:
1. **Temporal patterns**: Buildings follow daily/weekly schedules
2. **Historical context**: Recent behavior predicts future behavior
3. **Weather correlation**: Temperature and radiation affect power consumption
4. **Rate of change**: Sudden changes may indicate DR events

### Site-Aware Splitting
The baseline uses site-based train/validation splits because:
- Tests model generalization to new buildings
- More realistic evaluation
- Prevents data leakage from same site

## 🐛 Troubleshooting

### ModuleNotFoundError
```bash
# Ensure you're in the project root and install dependencies
pip install -r requirements.txt
```

### Memory Issues
```bash
# Reduce feature complexity or process sites separately
# Modify TimeSeriesFeatureEngineer parameters
```

### Poor Performance
- Check class distribution (should see -1, 0, 1 flags)
- Verify temporal features are correct
- Ensure no data leakage in validation split
- Review feature importance for sanity check

## 📚 Next Steps

1. **Understand Your Data**
   - Run the EDA notebook
   - Visualize DR events
   - Check for site-specific patterns

2. **Establish Baseline**
   - Train with default parameters
   - Evaluate on validation set
   - Submit to leaderboard

3. **Iterate and Improve**
   - Analyze errors
   - Try feature engineering ideas
   - Experiment with models
   - Ensemble predictions

4. **Prepare for Competition Phase**
   - Save your best approach
   - Document your methodology
   - Prepare for regression task (Phase 2)

## 🏆 Competition Tips

1. **Start Simple**: This baseline is a solid foundation
2. **Understand Metrics**: GMS rewards balanced predictions
3. **Cross-Validate**: Use multiple validation strategies
4. **Feature Importance**: Let the model tell you what matters
5. **Domain Knowledge**: Think about building energy behavior
6. **Time Series**: Respect temporal dependencies
7. **Document Everything**: Required for final submission

## 📞 Support

- Competition Forum: Check AIcrowd discussion board
- Documentation: See USAGE_GUIDE.md for details
- Issues: Examine error messages and logs

Good luck! 🚀
