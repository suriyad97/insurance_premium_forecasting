# src/model_training.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import pickle
from pathlib import Path
from src.utils import setup_logging, load_config, save_dataframe

class ModelTrainer:
    """Model Training - Train multiple ML models with hyperparameter tuning"""
    
    def __init__(self):
        self.config = load_config()
        self.logger = setup_logging('model_training', 'logs/model_training.log')
        self.models = {}
        self.best_models = {}
        self.results = {}
        self.scaler = StandardScaler()
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target for modeling"""
        self.logger.info("Preparing features for modeling...")
        
        target = self.config['model']['target_variable']
        
        # Select feature columns (exclude identifiers, dates, and target)
        exclude_cols = ['id', 'date', target, 'family', 'city', 'state', 
                       'type_x', 'type_y', 'locale', 'locale_name', 
                       'description', 'transferred', 'family_top', 'city_top']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target]
        
        self.logger.info(f"Features: {len(feature_cols)} columns")
        self.logger.info(f"Target: {target}")
        self.logger.info(f"Feature columns: {feature_cols}")
        
        return X, y, feature_cols
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Split data into train and test sets"""
        self.logger.info("Splitting data into train and test sets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state'],
            shuffle=False  # Important for time series
        )
        
        self.logger.info(f"Train set: {X_train.shape}")
        self.logger.info(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                    y_train: pd.Series, y_test: pd.Series) -> dict:
        """Train multiple models with hyperparameter tuning"""
        self.logger.info("Training multiple ML models...")
        
        # Define models and their hyperparameter grids
        models_config = {
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'XGBoost': {
                'model': xgb.XGBRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'Ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'Lasso': {
                'model': Lasso(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'max_iter': [1000, 2000]
                }
            }
        }
        
        # Train each model
        for model_name in self.config['model']['models']:
            if model_name not in models_config:
                self.logger.warning(f"Model {model_name} not found in configuration")
                continue
                
            self.logger.info(f"Training {model_name}...")
            
            config = models_config[model_name]
            
            # Randomized Search with Time Series Cross Validation
            search = RandomizedSearchCV(
                config['model'],
                config['params'],
                n_iter=10,
                cv=TimeSeriesSplit(n_splits=self.config['model']['cv_folds']),
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=42
            )
            
            # For linear models, use scaled features
            if model_name in ['Ridge', 'Lasso']:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                search.fit(X_train_scaled, y_train)
                best_model = search.best_estimator_
                y_pred = best_model.predict(X_test_scaled)
                X_train_used = X_train_scaled
                X_test_used = X_test_scaled
            else:
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                y_pred = best_model.predict(X_test)
                X_train_used = X_train
                X_test_used = X_test
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
            
            # Store results
            self.best_models[model_name] = best_model
            self.results[model_name] = {
                'best_params': search.best_params_,
                'model': best_model,
                'predictions': y_pred,
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape
                },
                'X_train': X_train_used,
                'X_test': X_test_used,
                'y_train': y_train,
                'y_test': y_test
            }
            
            self.logger.info(f"✅ {model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
        
        return self.results
    
    def save_models(self):
        """Save trained models to disk"""
        self.logger.info("Saving trained models...")
        
        models_dir = Path('models/trained_models')
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model_info in self.results.items():
            model_path = models_dir / f'{model_name.lower().replace(" ", "_")}.pkl'
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_info['model'], f)
            
            self.logger.info(f"Saved {model_name} to {model_path}")
        
        # Save scaler for linear models
        scaler_path = models_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save training results
        results_df = pd.DataFrame({
            model: info['metrics'] for model, info in self.results.items()
        }).T
        
        save_dataframe(results_df, 'reports/model_performance.csv', 'Model performance metrics')

if __name__ == "__main__":
    trainer = ModelTrainer()
    # Example usage after feature engineering
    # df = pd.read_csv('data/processed/engineered_data.csv')
    # X, y, feature_cols = trainer.prepare_features(df)
    # X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    # results = trainer.train_models(X_train, X_test, y_train, y_test)
    # trainer.save_models()