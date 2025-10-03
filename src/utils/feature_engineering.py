"""
Feature engineering utilities for FlexTrack Challenge
"""
import numpy as np
import pandas as pd
from typing import List, Tuple


class TimeSeriesFeatureEngineer:
    """Engineer time-series features for demand response prediction"""
    
    def __init__(
        self,
        lag_features: List[int] = [1, 2, 4, 8, 12, 24],
        rolling_windows: List[int] = [4, 8, 12, 24, 96],
    ):
        """
        Args:
            lag_features: List of lag periods (in 15-min intervals)
            rolling_windows: List of window sizes for rolling statistics
        """
        self.lag_features = lag_features
        self.rolling_windows = rolling_windows
        self.feature_names = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from timestamp"""
        df = df.copy()
        df['Timestamp_Local'] = pd.to_datetime(df['Timestamp_Local'])
        
        # Basic temporal features
        df['hour'] = df['Timestamp_Local'].dt.hour
        df['day_of_week'] = df['Timestamp_Local'].dt.dayofweek
        df['day_of_year'] = df['Timestamp_Local'].dt.dayofyear
        df['month'] = df['Timestamp_Local'].dt.month
        df['quarter'] = df['Timestamp_Local'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclic encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_lag_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str]
    ) -> pd.DataFrame:
        """Create lag features for specified columns"""
        df = df.copy()
        
        for col in columns:
            for lag in self.lag_features:
                df[f'{col}_lag_{lag}'] = df.groupby('Site')[col].shift(lag)
                
        return df
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Create rolling statistics features"""
        df = df.copy()
        
        for col in columns:
            for window in self.rolling_windows:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}'] = (
                    df.groupby('Site')[col]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                
                # Rolling std
                df[f'{col}_rolling_std_{window}'] = (
                    df.groupby('Site')[col]
                    .rolling(window=window, min_periods=1)
                    .std()
                    .reset_index(level=0, drop=True)
                )
                
                # Rolling min
                df[f'{col}_rolling_min_{window}'] = (
                    df.groupby('Site')[col]
                    .rolling(window=window, min_periods=1)
                    .min()
                    .reset_index(level=0, drop=True)
                )
                
                # Rolling max
                df[f'{col}_rolling_max_{window}'] = (
                    df.groupby('Site')[col]
                    .rolling(window=window, min_periods=1)
                    .max()
                    .reset_index(level=0, drop=True)
                )
        
        return df
    
    def create_difference_features(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Create difference features (changes over time)"""
        df = df.copy()
        
        for col in columns:
            # First difference
            df[f'{col}_diff_1'] = df.groupby('Site')[col].diff(1)
            
            # Second difference
            df[f'{col}_diff_2'] = df.groupby('Site')[col].diff(2)
            
            # Percentage change
            df[f'{col}_pct_change'] = df.groupby('Site')[col].pct_change()
            
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables"""
        df = df.copy()
        
        # Temperature * Radiation (proxy for cooling/heating load)
        df['temp_radiation_interaction'] = (
            df['Dry_Bulb_Temperature_C'] * df['Global_Horizontal_Radiation_W/m2']
        )
        
        # Power normalized by temperature
        df['power_per_temp'] = df['Building_Power_kW'] / (df['Dry_Bulb_Temperature_C'] + 1e-6)
        
        return df
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = 'Demand_Response_Flag'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply all feature engineering steps
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            
        Returns:
            Tuple of (features_df, target_series)
        """
        print("Creating temporal features...")
        df = self.create_temporal_features(df)
        
        print("Creating lag features...")
        power_cols = ['Building_Power_kW', 'Dry_Bulb_Temperature_C', 'Global_Horizontal_Radiation_W/m2']
        df = self.create_lag_features(df, power_cols)
        
        print("Creating rolling features...")
        df = self.create_rolling_features(df, ['Building_Power_kW'])
        
        print("Creating difference features...")
        df = self.create_difference_features(df, ['Building_Power_kW'])
        
        print("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        # Store original columns for later
        if target_col in df.columns:
            target = df[target_col].copy()
        else:
            target = None
            
        # Get feature columns (exclude metadata and target)
        exclude_cols = ['Site', 'Timestamp_Local', target_col, 'Demand_Response_Capacity_kW']
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Total features created: {len(self.feature_names)}")
        
        return df, target
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted feature engineering"""
        df, _ = self.fit_transform(df, target_col=None)
        return df