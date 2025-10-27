# src/data_validation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils import setup_logging, save_dataframe

class DataValidator:
    """Data Validation - Comprehensive data quality checks"""
    
    def __init__(self):
        self.logger = setup_logging('data_validation', 'logs/data_validation.log')
        self.validation_results = {}
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """Comprehensive data validation"""
        self.logger.info(f"Validating dataset: {dataset_name}")
        
        validation_report = {
            'basic_info': self._get_basic_info(df),
            'data_quality': self._check_data_quality(df),
            'temporal_analysis': self._check_temporal_consistency(df) if 'date' in df.columns else {},
            'business_rules': self._check_business_rules(df, dataset_name)
        }
        
        self.validation_results[dataset_name] = validation_report
        return validation_report
    
    def _get_basic_info(self, df: pd.DataFrame) -> dict:
        """Get basic dataset information"""
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            'data_types': df.dtypes.astype(str).to_dict()
        }
    
    def _check_data_quality(self, df: pd.DataFrame) -> dict:
        """Check data quality issues"""
        quality_report = {
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            'duplicates': df.duplicated().sum(),
            'numeric_stats': {},
            'categorical_stats': {}
        }
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            quality_report['numeric_stats'][col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            }
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            quality_report['categorical_stats'][col] = {
                'unique_count': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None
            }
        
        return quality_report
    
    def _check_temporal_consistency(self, df: pd.DataFrame) -> dict:
        """Check temporal consistency"""
        df_temp = df.copy()
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        
        return {
            'date_range': {
                'start': df_temp['date'].min().strftime('%Y-%m-%d'),
                'end': df_temp['date'].max().strftime('%Y-%m-%d'),
                'days': (df_temp['date'].max() - df_temp['date'].min()).days
            },
            'missing_dates': self._find_missing_dates(df_temp['date'])
        }
    
    def _find_missing_dates(self, date_series: pd.Series) -> list:
        """Find missing dates in the series"""
        full_range = pd.date_range(start=date_series.min(), end=date_series.max(), freq='D')
        missing = full_range.difference(date_series)
        return missing.tolist()
    
    def _check_business_rules(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """Check dataset-specific business rules"""
        rules = {}
        
        if dataset_name == 'train':
            if 'sales' in df.columns:
                rules['negative_sales'] = (df['sales'] < 0).sum()
        
        return rules
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        self.logger.info("Generating validation report...")
        
        # Create visualizations
        self._create_validation_plots()
        
        # Save validation results
        import json
        with open('reports/validation_results.json', 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        self.logger.info("Validation report generated")
    
    def _create_validation_plots(self):
        """Create validation visualizations"""
        # Missing values heatmap
        for dataset_name, results in self.validation_results.items():
            missing_data = results['data_quality']['missing_percentage']
            missing_series = pd.Series(missing_data)
            missing_series = missing_series[missing_series > 0]
            
            if len(missing_series) > 0:
                plt.figure(figsize=(10, 6))
                missing_series.sort_values().plot(kind='barh')
                plt.title(f'Missing Values - {dataset_name}')
                plt.tight_layout()
                plt.savefig(f'reports/figures/missing_values_{dataset_name}.png')
                plt.close()

if __name__ == "__main__":
    validator = DataValidator()
    # Example usage after loading data
    # df = pd.read_csv('data/processed/merged_data.csv')
    # validation_results = validator.validate_dataset(df, 'merged_data')
    # validator.generate_validation_report()