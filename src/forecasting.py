# src/forecasting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
from pathlib import Path
from src.utils import setup_logging, save_dataframe, load_config

class ForecastingEngine:
    """Forecasting Engine - Generate future predictions using trained models"""
    
    def __init__(self):
        self.config = load_config()
        self.logger = setup_logging('forecasting', 'logs/forecasting.log')
        self.models = {}
        self.scaler = None
    
    def load_trained_models(self):
        """Load trained ML models from disk"""
        self.logger.info("Loading trained models...")
        
        models_dir = Path('models/trained_models')
        
        for model_file in models_dir.glob('*.pkl'):
            if model_file.name != 'scaler.pkl':
                model_name = model_file.stem.replace('_', ' ').title()
                
                with open(model_file, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                
                self.logger.info(f"Loaded model: {model_name}")
        
        # Load scaler
        scaler_path = models_dir / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.logger.info("Loaded scaler")
    
    def create_future_features(self, df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
        """Create future feature set for forecasting"""
        self.logger.info(f"Creating future features for {horizon} days...")
        
        # Get the last date from historical data
        last_date = pd.to_datetime(df['date']).max()
        
        # Create future dates
        future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
        
        # Get unique store-family combinations
        store_families = df[['store_nbr', 'family']].drop_duplicates()
        
        future_data = []
        
        for date in future_dates:
            for _, row in store_families.iterrows():
                future_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'store_nbr': row['store_nbr'],
                    'family': row['family'],
                    # Set default values for other features
                    'onpromotion': 0,  # Assume no promotion by default
                    'year': date.year,
                    'month': date.month,
                    'quarter': date.quarter,
                    'day_of_week': date.weekday(),
                    'day_of_year': date.timetuple().tm_yday,
                    'week_of_year': date.isocalendar()[1],
                    'is_weekend': 1 if date.weekday() >= 5 else 0,
                    'is_month_start': 1 if date.day == 1 else 0,
                    'is_month_end': 1 if date.day == date.days_in_month else 0,
                    # Add other engineered features with default values
                    'sales_lag_1': df['sales'].mean(),  # Use historical average
                    'sales_lag_7': df['sales'].mean(),
                    'sales_rolling_mean_7': df['sales'].mean(),
                    'sales_rolling_mean_30': df['sales'].mean(),
                    'is_holiday': 0,  # Assume no holidays by default
                    'is_national_holiday': 0,
                    'is_local_holiday': 0,
                    'is_regional_holiday': 0
                })
        
        future_df = pd.DataFrame(future_data)
        
        # Add store information
        stores_df = pd.read_csv('data/raw/stores.csv')
        future_df = future_df.merge(stores_df, on='store_nbr', how='left')
        
        # Add encoded features
        future_df = self._add_encoded_features(future_df)
        
        self.logger.info(f"Created future dataset with {len(future_df)} records")
        return future_df
    
    def _add_encoded_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add encoded features to future data"""
        # Store type encoding
        store_type_dummies = pd.get_dummies(df['type'], prefix='store_type')
        df = pd.concat([df, store_type_dummies], axis=1)
        
        # Family encoding (using top categories from historical data)
        top_families = ['GROCERY', 'BEVERAGES', 'PRODUCE', 'DAIRY', 'CLEANING']  # Example top families
        for family in top_families:
            df[f'family_{family}'] = (df['family'] == family).astype(int)
        
        # City encoding
        top_cities = ['Quito', 'Guayaquil', 'Cuenca', 'Ambato', 'Santo Domingo']  # Example top cities
        for city in top_cities:
            df[f'city_{city}'] = (df['city'] == city).astype(int)
        
        return df
    
    def generate_ml_forecasts(self, future_df: pd.DataFrame, best_model_name: str) -> pd.DataFrame:
        """Generate forecasts using the best ML model"""
        self.logger.info(f"Generating ML forecasts using {best_model_name}...")
        
        if best_model_name not in self.models:
            self.logger.error(f"Model {best_model_name} not found in loaded models")
            return None
        
        model = self.models[best_model_name]
        
        # Prepare features (same as training)
        exclude_cols = ['id', 'date', 'sales', 'family', 'city', 'state', 
                       'type', 'locale', 'locale_name', 'description', 'transferred']
        
        feature_cols = [col for col in future_df.columns if col not in exclude_cols]
        X_future = future_df[feature_cols]
        
        # Handle missing values in future data
        X_future = X_future.fillna(0)
        
        # Scale features if it's a linear model
        if best_model_name in ['Ridge', 'Lasso'] and self.scaler is not None:
            X_future = self.scaler.transform(X_future)
        
        # Generate predictions
        predictions = model.predict(X_future)
        
        # Add predictions to future dataframe
        future_df['predicted_sales'] = predictions
        future_df['prediction_model'] = best_model_name
        
        # Ensure non-negative predictions
        future_df['predicted_sales'] = future_df['predicted_sales'].clip(lower=0)
        
        self.logger.info(f"Generated {len(predictions)} ML forecasts")
        return future_df
    
    def generate_ensemble_forecast(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """Generate ensemble forecast using multiple models"""
        self.logger.info("Generating ensemble forecast...")
        
        exclude_cols = ['id', 'date', 'sales', 'family', 'city', 'state', 
                       'type', 'locale', 'locale_name', 'description', 'transferred']
        
        feature_cols = [col for col in future_df.columns if col not in exclude_cols]
        X_future = future_df[feature_cols].fillna(0)
        
        # Collect predictions from all models
        all_predictions = {}
        
        for model_name, model in self.models.items():
            if model_name in ['Ridge', 'Lasso'] and self.scaler is not None:
                X_scaled = self.scaler.transform(X_future)
                predictions = model.predict(X_scaled)
            else:
                predictions = model.predict(X_future)
            
            all_predictions[model_name] = predictions
        
        # Create ensemble prediction (mean of all models)
        predictions_df = pd.DataFrame(all_predictions)
        future_df['ensemble_prediction'] = predictions_df.mean(axis=1)
        future_df['ensemble_std'] = predictions_df.std(axis=1)
        future_df['prediction_model'] = 'Ensemble'
        
        # Calculate confidence intervals
        future_df['lower_bound'] = future_df['ensemble_prediction'] - 1.96 * future_df['ensemble_std']
        future_df['upper_bound'] = future_df['ensemble_prediction'] + 1.96 * future_df['ensemble_std']
        
        # Ensure non-negative
        future_df['lower_bound'] = future_df['lower_bound'].clip(lower=0)
        future_df['ensemble_prediction'] = future_df['ensemble_prediction'].clip(lower=0)
        
        self.logger.info("Generated ensemble forecasts")
        return future_df
    
    def plot_forecasts(self, historical_df: pd.DataFrame, forecast_df: pd.DataFrame, 
                      model_type: str = "ML"):
        """Plot forecast results"""
        self.logger.info("Creating forecast visualizations...")
        
        # Aggregate historical sales by date
        historical_agg = historical_df.groupby('date')['sales'].sum().reset_index()
        historical_agg['date'] = pd.to_datetime(historical_agg['date'])
        
        # Aggregate forecast sales by date
        forecast_agg = forecast_df.groupby('date')['predicted_sales'].sum().reset_index()
        forecast_agg['date'] = pd.to_datetime(forecast_agg['date'])
        
        # Plot 1: Overall forecast
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        # Show last 90 days of historical data for context
        recent_historical = historical_agg[historical_agg['date'] > historical_agg['date'].max() - timedelta(days=90)]
        
        plt.plot(recent_historical['date'], recent_historical['sales'], 
                label='Historical Sales', color='blue', linewidth=2)
        plt.plot(forecast_agg['date'], forecast_agg['predicted_sales'], 
                label='Forecast', color='red', linewidth=2, marker='o')
        
        # Add confidence intervals if available
        if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
            forecast_ci = forecast_df.groupby('date').agg({
                'lower_bound': 'sum',
                'upper_bound': 'sum'
            }).reset_index()
            
            plt.fill_between(forecast_ci['date'], 
                           forecast_ci['lower_bound'], 
                           forecast_ci['upper_bound'], 
                           color='pink', alpha=0.3, label='95% Confidence Interval')
        
        plt.title(f'Sales Forecast - {model_type} Model', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Total Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Forecast by store type
        plt.subplot(2, 1, 2)
        store_forecasts = forecast_df.groupby(['date', 'type'])['predicted_sales'].sum().reset_index()
        
        for store_type in store_forecasts['type'].unique():
            store_data = store_forecasts[store_forecasts['type'] == store_type]
            plt.plot(store_data['date'], store_data['predicted_sales'], 
                    label=f'Store Type {store_type}', linewidth=2)
        
        plt.title('Forecast by Store Type', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'reports/figures/{model_type.lower()}_forecast_comprehensive.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 3: Top product families forecast
        plt.figure(figsize=(15, 8))
        family_forecasts = forecast_df.groupby(['date', 'family'])['predicted_sales'].sum().reset_index()
        top_families = family_forecasts.groupby('family')['predicted_sales'].sum().nlargest(5).index
        
        for family in top_families:
            family_data = family_forecasts[family_forecasts['family'] == family]
            plt.plot(family_data['date'], family_data['predicted_sales'], 
                    label=family, linewidth=2)
        
        plt.title('Forecast for Top 5 Product Families', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'reports/figures/{model_type.lower()}_forecast_families.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_forecasts(self, forecast_df: pd.DataFrame, filename: str):
        """Save forecast results to file"""
        save_dataframe(forecast_df, f'reports/{filename}', 'Forecast results')
        
        # Also save aggregated forecasts
        daily_forecast = forecast_df.groupby('date').agg({
            'predicted_sales': 'sum',
            'store_nbr': 'count'
        }).reset_index()
        daily_forecast = daily_forecast.rename(columns={'store_nbr': 'store_count'})
        
        save_dataframe(daily_forecast, f'reports/aggregated_{filename}', 
                      'Aggregated forecast results')

if __name__ == "__main__":
    forecaster = ForecastingEngine()
    # Example usage:
    # forecaster.load_trained_models()
    # df = pd.read_csv('data/processed/engineered_data.csv')
    # future_df = forecaster.create_future_features(df, horizon=30)
    # ml_forecasts = forecaster.generate_ml_forecasts(future_df, 'Random Forest')
    # ensemble_forecasts = forecaster.generate_ensemble_forecast(future_df)
    # forecaster.plot_forecasts(df, ensemble_forecasts, "Ensemble")
    # forecaster.save_forecasts(ensemble_forecasts, 'ensemble_forecasts.csv')