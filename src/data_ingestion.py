import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from src.utils import load_config, setup_logging, save_dataframe

class DataIngestion:
    """Data Ingestion - Load and merge all datasets"""
    
    def __init__(self):
        self.config = load_config()
        self.logger = setup_logging('data_ingestion', 'logs/data_ingestion.log')
        self.data_path = Path(self.config['data']['raw_path'])
            
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all datasets from raw directory"""
        self.logger.info("Loading all datasets...")
        
        datasets = {}
        for file_key, file_name in self.config['data']['files'].items():
            file_path = self.data_path / file_name
            if file_path.exists():
                datasets[file_key] = pd.read_csv(file_path)
                self.logger.info(f"Loaded {file_name}: {datasets[file_key].shape}")
            else:
                self.logger.warning(f"File not found: {file_name}")
        
        return datasets
    
    def merge_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all datasets using specified join logic"""
        self.logger.info("Merging datasets...")
        
        # Standardize date formats
        for name, df in datasets.items():
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y').dt.strftime('%Y-%m-%d')
        
        # Perform merges
        merged_data = (
            datasets['train']
            .merge(datasets['stores'], on='store_nbr', how='left')
            .merge(datasets['holidays'], on='date', how='left')
            .merge(datasets['oil'], on='date', how='left')
            .merge(datasets['transactions'], on=['date', 'store_nbr'], how='left')
        )
        
        self.logger.info(f"Merged dataset shape: {merged_data.shape}")
        save_dataframe(merged_data, 'data/processed/merged_data.csv', 'Merged dataset')
        
        return merged_data

if __name__ == "__main__":
    ingestion = DataIngestion()
    loaded_datasets = ingestion.load_all_data()
    merged_data = ingestion.merge_datasets(loaded_datasets)