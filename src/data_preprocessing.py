# src/data_preprocessing.py
import pandas as pd
import numpy as np
from src.utils import setup_logging, save_dataframe

class DataPreprocessor:
    """Data Preprocessing - Handle missing values and basic cleaning"""
    
    def __init__(self):
        self.logger = setup_logging('data_preprocessing', 'logs/data_preprocessing.log')
        self.imputers = {}
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        self.logger.info("Handling missing values...")
        
        df_clean = df.copy()
        
        # Fill numeric columns with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                self.logger.info(f"Filled missing values in {col} with median")
        
        # Fill categorical columns with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
                    self.logger.info(f"Filled missing values in {col} with mode: {mode_val[0]}")
                else:
                    df_clean[col] = df_clean[col].fillna('Unknown')
        
        self.logger.info("Missing values handling completed")
        return df_clean
    
    def remove_outliers(self, df: pd.DataFrame, columns: list, method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers from specified columns"""
        self.logger.info(f"Removing outliers using {method} method...")
        
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns and df_clean[col].dtype in [np.int64, np.float64]:
                if method == 'iqr':
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    before = len(df_clean)
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                    after = len(df_clean)
                    
                    self.logger.info(f"Removed {before - after} outliers from {col}")
        
        return df_clean

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    # Example usage after loading merged data
    # df = pd.read_csv('data/processed/merged_data.csv')
    # df_clean = preprocessor.handle_missing_values(df)
    # save_dataframe(df_clean, 'data/processed/cleaned_data.csv', 'Cleaned data')