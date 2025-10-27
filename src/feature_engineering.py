# src/feature_engineering.py
import pandas as pd
import numpy as np
from src.utils import setup_logging, save_dataframe, load_config

class FeatureEngineer:
    """Feature Engineering - Create new features from raw data"""
    
    def __init__(self):
        self.config = load_config()
        self.logger = setup_logging('feature_engineering', 'logs/feature_engineering.log')
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform comprehensive feature engineering"""
        self.logger.info("Starting feature engineering...")
        
        df_eng = df.copy()
        
        # Convert date to datetime
        df_eng['date'] = pd.to_datetime(df_eng['date'])
        
        # Time-based features
        if self.config['feature_engineering']['time_features']:
            df_eng = self._create_time_features(df_eng)
        
        # Encoding categorical variables
        if self.config['feature_engineering']['encoding']:
            df_eng = self._encode_categorical_features(df_eng)
        
        # Lag and rolling features
        if self.config['feature_engineering']['lag_features']:
            df_eng = self._create_lag_features(df_eng)
        
        # Holiday features
        df_eng = self._create_holiday_features(df_eng)
        
        # Store features
        df_eng = self._create_store_features(df_eng)
        
        self.logger.info(f"Feature engineering completed. Final shape: {df_eng.shape}")
        return df_eng
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        self.logger.info("Creating time-based features...")
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        self.logger.info("Encoding categorical features...")
        
        # Store type encoding
        store_type_dummies = pd.get_dummies(df['type'], prefix='store_type')
        df = pd.concat([df, store_type_dummies], axis=1)
        
        # Family encoding (top categories)
        top_families = df['family'].value_counts().head(8).index
        df['family_top'] = df['family'].apply(lambda x: x if x in top_families else 'Other')
        family_dummies = pd.get_dummies(df['family_top'], prefix='family')
        df = pd.concat([df, family_dummies], axis=1)
        
        # City encoding (top categories)
        top_cities = df['city'].value_counts().head(5).index
        df['city_top'] = df['city'].apply(lambda x: x if x in top_cities else 'Other')
        city_dummies = pd.get_dummies(df['city_top'], prefix='city')
        df = pd.concat([df, city_dummies], axis=1)
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag and rolling window features"""
        self.logger.info("Creating lag features...")
        
        # Sort by store, family, and date
        df = df.sort_values(['store_nbr', 'family', 'date'])
        
        # Lag features for sales
        df['sales_lag_1'] = df.groupby(['store_nbr', 'family'])['sales'].shift(1)
        df['sales_lag_7'] = df.groupby(['store_nbr', 'family'])['sales'].shift(7)
        
        # Rolling window features
        windows = self.config['feature_engineering']['rolling_windows']
        for window in windows:
            df[f'sales_rolling_mean_{window}'] = df.groupby(['store_nbr', 'family'])['sales'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'sales_rolling_std_{window}'] = df.groupby(['store_nbr', 'family'])['sales'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        return df
    
    def _create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create holiday-related features"""
        self.logger.info("Creating holiday features...")
        
        df['is_holiday'] = (~df['type_y'].isna()).astype(int)
        df['is_national_holiday'] = (df['locale'] == 'National').fillna(0).astype(int)
        df['is_local_holiday'] = (df['locale'] == 'Local').fillna(0).astype(int)
        df['is_regional_holiday'] = (df['locale'] == 'Regional').fillna(0).astype(int)
        
        # Days until next holiday and days since last holiday
        holiday_dates = df[df['is_holiday'] == 1]['date'].unique()
        holiday_dates = pd.to_datetime(holiday_dates)
        
        def days_to_next_holiday(date):
            date = pd.to_datetime(date)
            future_holidays = holiday_dates[holiday_dates > date]
            return (future_holidays.min() - date).days if len(future_holidays) > 0 else np.nan
        
        def days_since_last_holiday(date):
            date = pd.to_datetime(date)
            past_holidays = holiday_dates[holiday_dates < date]
            return (date - past_holidays.max()).days if len(past_holidays) > 0 else np.nan
        
        df['days_to_next_holiday'] = df['date'].apply(days_to_next_holiday)
        df['days_since_last_holiday'] = df['date'].apply(days_since_last_holiday)
        
        return df
    
    def _create_store_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create store-specific features"""
        self.logger.info("Creating store features...")
        
        # Store performance metrics
        store_stats = df.groupby('store_nbr').agg({
            'sales': ['mean', 'std', 'max'],
            'transactions': 'mean'
        }).round(2)
        store_stats.columns = ['store_sales_mean', 'store_sales_std', 'store_sales_max', 'store_transactions_mean']
        store_stats = store_stats.reset_index()
        
        df = df.merge(store_stats, on='store_nbr', how='left')
        
        return df

if __name__ == "__main__":
    engineer = FeatureEngineer()
    # Example usage after preprocessing
    # df = pd.read_csv('data/processed/cleaned_data.csv')
    # df_engineered = engineer.engineer_features(df)
    # save_dataframe(df_engineered, 'data/processed/engineered_data.csv', 'Engineered data')