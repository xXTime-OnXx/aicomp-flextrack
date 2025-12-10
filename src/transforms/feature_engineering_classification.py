import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import StandardScaler

from transforms.feature_engineering import (
    add_time_features, 
    add_cyclic_features, 
    add_difference_features, 
    add_rolling_features,
)


ENTRIES_PER_DAY = 96

# Feature column definitions
CONTINUOUS_FEATURE_COLUMNS = [
    'Dry_Bulb_Temperature_C', 'Global_Horizontal_Radiation_W/m2', 'Building_Power_kW',
    'Building_Power_kW_diff_15min', 'Building_Power_kW_diff_1h', 'Building_Power_kW_diff_1d',
    'Dry_Bulb_Temperature_C_diff_15min', 'Global_Horizontal_Radiation_W/m2_diff_15min',
    'Building_Power_kW_rolling_mean_1h', 'Building_Power_kW_rolling_mean_2h', 'Building_Power_kW_rolling_mean_1d',
    'Building_Power_kW_rolling_std_1h', 'Building_Power_kW_rolling_std_2h', 'Building_Power_kW_rolling_std_1d',
    'Building_Power_kW_rolling_min_1h', 'Building_Power_kW_rolling_min_2h', 'Building_Power_kW_rolling_max_1h',
    'Building_Power_kW_rolling_max_2h', 'hour'
]
# ATTENTION: DR Flags are remove here and used as target column
CATEGORICAL_FEATURE_COLUMNS = [
    'minute_0', 'minute_15', 'minute_30', 'minute_45', 'day_of_week_0',
    'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4',
    'day_of_week_5', 'day_of_week_6', 'is_weekend', 'is_holiday' 
]
CYCLIC_FEATURE_COLUMNS = ['month_sin', 'month_cos']

# ATTENTION: DR Capacity is replaced with DR Flag for classificaiton
TARGET_COLUMN = 'Demand_Response_Flag'

CLASSIFICATION_MAPPING = {-1: 0, 0: 1, 1: 2}

def add_all_features(df):
    df = add_time_features(df)
    df = add_cyclic_features(df)
    df = add_difference_features(df)
    df = add_rolling_features(df)
    df = encode_categorical_features(df)

    return df

def encode_categorical_features(df):
    """One-hot encode categorical features but keep original columns."""
    
    minute_dummies = pd.get_dummies(df['minute'], prefix="minute")
    dow_dummies = pd.get_dummies(df['day_of_week'], prefix="day_of_week")
    
    # Concatenate all dummies
    df = pd.concat([df, minute_dummies, dow_dummies], axis=1)

    df['is_holiday'] = df['is_holiday'].astype(int)
    df['is_weekend'] = df['is_weekend'].astype(int)
    
    df["Demand_Response_Flag"] = df["Demand_Response_Flag"].map(CLASSIFICATION_MAPPING)

    return df


def filter_business_hours(df):
    """Filter data to business hours (6:00 AM - 7:00 PM)."""
    df = df[
        ((df['hour'] > 6) & (df['hour'] < 19)) |
        ((df['hour'] == 6) & (df['minute'] >= 0)) |
        ((df['hour'] == 19) & (df['minute'] == 0))
    ].copy()
    
    return df


def entries_per_day_per_site(df):
    # Ensure date is a date (not datetime)
    df_temp = df.copy(deep=True)
    df_temp['date'] = df_temp['Timestamp_Local'].dt.date

    # Count entries per day per site
    counts = df_temp.groupby(['date', 'Site']).size()

    # Count entries per site per day (to check consistency across days)
    daily_totals = counts.groupby(level='Site').unique()

    # Assert: each site must have exactly one unique daily count
    for site, uniques in daily_totals.items():
        assert len(uniques) == 1, f"Site {site} has inconsistent daily counts: {uniques}"

    # If assertion passes, store counts in a variable:
    site_counts = {site: uniques[0] for site, uniques in daily_totals.items()}

    values = list(site_counts.values())
    assert len(set(values)) == 1, f"Not all sites have the same count: {values}"
    
    return values[0]