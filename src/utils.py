# src/utils.py
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(log_name: str, log_file: str = None) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def create_directory_structure():
    """Create the complete directory structure"""
    directories = [
        'data/raw', 'data/processed', 'data/interim',
        'notebooks', 'models/trained_models', 'models/model_artifacts',
        'logs', 'reports/figures', 'config'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")

def save_dataframe(df: pd.DataFrame, file_path: str, description: str = ""):
    """Save DataFrame with logging"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"✅ {description} saved to: {file_path}")