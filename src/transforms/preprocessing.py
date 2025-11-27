"""Data preprocessing utilities for LSTM regression."""

import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import StandardScaler

from utils.config import (
    ENTRIES_PER_DAY,
    CONTINUOUS_FEATURE_COLUMNS,
    CATEGORICAL_FEATURE_COLUMNS,
    CYCLIC_FEATURE_COLUMNS,
    TARGET_COLUMN
)


def add_time_features(df):
    """Add time-based features to the dataframe."""
    australian_holidays = holidays.AU()
    
    df['Timestamp_Local'] = pd.to_datetime(df['Timestamp_Local'])
    df['hour'] = df['Timestamp_Local'].dt.hour
    df['minute'] = df['Timestamp_Local'].dt.minute
    df['month'] = df['Timestamp_Local'].dt.month
    df['day_of_week'] = df['Timestamp_Local'].dt.dayofweek
    df['is_weekend'] = df['Timestamp_Local'].dt.dayofweek >= 5
    df['is_holiday'] = df['Timestamp_Local'].dt.date.apply(lambda x: x in australian_holidays)
    
    return df


def filter_business_hours(df):
    """Filter data to business hours (6:00 AM - 7:00 PM)."""
    df = df[
        ((df['hour'] > 6) & (df['hour'] < 19)) |
        ((df['hour'] == 6) & (df['minute'] >= 0)) |
        ((df['hour'] == 19) & (df['minute'] == 0))
    ].copy()
    
    return df


def add_cyclic_features(df):
    """Add cyclic encoding for month."""
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] / 12))
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] / 12))
    df.drop(columns=['month'], inplace=True)
    
    return df


def add_difference_features(df):
    """Add difference features for power and weather."""
    df['Building_Power_kW_diff_15min'] = df.groupby('Site')['Building_Power_kW'].diff(1)
    df['Building_Power_kW_diff_1h'] = df.groupby('Site')['Building_Power_kW'].diff(4)
    df['Building_Power_kW_diff_1d'] = df.groupby('Site')['Building_Power_kW'].diff(ENTRIES_PER_DAY)
    
    df['Dry_Bulb_Temperature_C_diff_15min'] = df.groupby('Site')['Dry_Bulb_Temperature_C'].diff(1)
    df['Global_Horizontal_Radiation_W/m2_diff_15min'] = df.groupby('Site')['Global_Horizontal_Radiation_W/m2'].diff(1)
    
    return df


def add_rolling_features(df):
    """Add rolling window statistics for building power."""
    group = df.groupby('Site')['Building_Power_kW']
    
    # Rolling means
    df['Building_Power_kW_rolling_mean_1h'] = group.rolling(4).mean().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_mean_2h'] = group.rolling(8).mean().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_mean_1d'] = group.rolling(ENTRIES_PER_DAY).mean().reset_index(level=0, drop=True)
    
    # Rolling standard deviations
    df['Building_Power_kW_rolling_std_1h'] = group.rolling(4).std().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_std_2h'] = group.rolling(8).std().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_std_1d'] = group.rolling(ENTRIES_PER_DAY).std().reset_index(level=0, drop=True)
    
    # Rolling min/max
    df['Building_Power_kW_rolling_min_1h'] = group.rolling(4).min().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_min_2h'] = group.rolling(8).min().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_max_1h'] = group.rolling(4).max().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_max_2h'] = group.rolling(8).max().reset_index(level=0, drop=True)
    
    return df


def encode_categorical_features(df):
    """One-hot encode categorical features."""
    df = pd.get_dummies(df, columns=['minute', 'day_of_week'])
    df = pd.get_dummies(df, columns=['Demand_Response_Flag'])
    
    df['is_holiday'] = df['is_holiday'].astype(int)
    df['is_weekend'] = df['is_weekend'].astype(int)
    
    return df


def preprocess_data(df):
    """Complete preprocessing pipeline."""
    df = add_time_features(df)
    df = filter_business_hours(df)
    df = add_cyclic_features(df)
    df = add_difference_features(df)
    df = add_rolling_features(df)
    df = encode_categorical_features(df)
    
    return df


def prepare_data(df_train):
    """Prepare and return all data splits with scalers."""
    # Split by site
    df_train_site_a = df_train[0:19345]
    df_train_site_b = df_train[19345:38690]
    df_train_site_c = df_train[38690:58035]

    # Extract features by type
    X_continuous_site_a = df_train_site_a[CONTINUOUS_FEATURE_COLUMNS].values
    X_continuous_site_b = df_train_site_b[CONTINUOUS_FEATURE_COLUMNS].values
    X_continuous_site_c = df_train_site_c[CONTINUOUS_FEATURE_COLUMNS].values

    X_categorical_site_a = df_train_site_a[CATEGORICAL_FEATURE_COLUMNS].values
    X_categorical_site_b = df_train_site_b[CATEGORICAL_FEATURE_COLUMNS].values
    X_categorical_site_c = df_train_site_c[CATEGORICAL_FEATURE_COLUMNS].values

    X_cyclic_site_a = df_train_site_a[CYCLIC_FEATURE_COLUMNS].values
    X_cyclic_site_b = df_train_site_b[CYCLIC_FEATURE_COLUMNS].values
    X_cyclic_site_c = df_train_site_c[CYCLIC_FEATURE_COLUMNS].values

    # Extract targets
    y_site_a = df_train_site_a[TARGET_COLUMN].values.reshape(-1, 1)
    y_site_b = df_train_site_b[TARGET_COLUMN].values.reshape(-1, 1)
    y_site_c = df_train_site_c[TARGET_COLUMN].values.reshape(-1, 1)

    y_site_a_unscaled = y_site_a.copy()

    # Initialize scalers
    scaler_X_site_a = StandardScaler()
    scaler_X_site_b = StandardScaler()
    scaler_X_site_c = StandardScaler()

    scaler_y_site_a = StandardScaler()
    scaler_y_site_b = StandardScaler()
    scaler_y_site_c = StandardScaler()

    # Scale continuous features
    X_scaled_site_a = scaler_X_site_a.fit_transform(X_continuous_site_a)
    X_scaled_site_b = scaler_X_site_b.fit_transform(X_continuous_site_b)
    X_scaled_site_c = scaler_X_site_c.fit_transform(X_continuous_site_c)

    # Scale targets
    y_site_a = scaler_y_site_a.fit_transform(y_site_a)
    y_site_b = scaler_y_site_b.fit_transform(y_site_b)
    y_site_c = scaler_y_site_c.fit_transform(y_site_c)

    # Concatenate all features
    X_site_a = np.concatenate([X_scaled_site_a, X_categorical_site_a, X_cyclic_site_a], axis=1).astype(np.float32)
    X_site_b = np.concatenate([X_scaled_site_b, X_categorical_site_b, X_cyclic_site_b], axis=1).astype(np.float32)
    X_site_c = np.concatenate([X_scaled_site_c, X_categorical_site_c, X_cyclic_site_c], axis=1).astype(np.float32)

    y_site_a = y_site_a.astype(np.float32)
    y_site_b = y_site_b.astype(np.float32)
    y_site_c = y_site_c.astype(np.float32)

    # Remove NaN entries (first day due to rolling features)
    mask_site_a = np.ones(len(X_site_a), dtype=bool)
    mask_site_b = np.ones(len(X_site_b), dtype=bool)
    mask_site_c = np.ones(len(X_site_c), dtype=bool)

    mask_site_a[0:ENTRIES_PER_DAY] = False
    mask_site_b[0:ENTRIES_PER_DAY] = False
    mask_site_c[0:ENTRIES_PER_DAY] = False

    X_site_a = X_site_a[mask_site_a]
    X_site_b = X_site_b[mask_site_b]
    X_site_c = X_site_c[mask_site_c]

    y_site_a = y_site_a[mask_site_a]
    y_site_b = y_site_b[mask_site_b]
    y_site_c = y_site_c[mask_site_c]

    return {
        'X_site_a': X_site_a, 'X_site_b': X_site_b, 'X_site_c': X_site_c,
        'y_site_a': y_site_a, 'y_site_b': y_site_b, 'y_site_c': y_site_c,
        'scaler_X_site_a': scaler_X_site_a, 'scaler_X_site_b': scaler_X_site_b, 'scaler_X_site_c': scaler_X_site_c,
        'scaler_y_site_a': scaler_y_site_a, 'scaler_y_site_b': scaler_y_site_b, 'scaler_y_site_c': scaler_y_site_c,
        'y_site_a_unscaled': y_site_a_unscaled
    }
