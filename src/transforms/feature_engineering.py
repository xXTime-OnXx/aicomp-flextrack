import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import StandardScaler

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
CATEGORICAL_FEATURE_COLUMNS = [
    'minute_0', 'minute_15', 'minute_30', 'minute_45', 'day_of_week_0',
    'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4',
    'day_of_week_5', 'day_of_week_6', 'Demand_Response_Flag_-1',
    'Demand_Response_Flag_0', 'Demand_Response_Flag_1',
    'is_weekend', 'is_holiday'
]
CYCLIC_FEATURE_COLUMNS = ['month_sin', 'month_cos']
TARGET_COLUMN = 'Demand_Response_Capacity_kW'


def add_all_features(df):
    df = add_time_features(df)
    df = add_cyclic_features(df)
    df = add_difference_features(df)
    df = add_rolling_features(df)
    df = encode_categorical_features(df)

    return df


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
    """One-hot encode categorical features but keep original columns."""
    
    minute_dummies = pd.get_dummies(df['minute'], prefix="minute")
    dow_dummies = pd.get_dummies(df['day_of_week'], prefix="day_of_week")
    dr_dummies = pd.get_dummies(df['Demand_Response_Flag'], prefix="Demand_Response_Flag")
    
    # Concatenate all dummies
    df = pd.concat([df, minute_dummies, dow_dummies, dr_dummies], axis=1)

    df['is_holiday'] = df['is_holiday'].astype(int)
    df['is_weekend'] = df['is_weekend'].astype(int)

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