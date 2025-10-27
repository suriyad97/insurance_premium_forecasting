# src/statistical_modeling.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils import setup_logging, save_dataframe

class StatisticalModeler:
    """Statistical Modeling - Traditional time series forecasting methods"""
    
    def __init__(self):
        self.logger = setup_logging('statistical_modeling', 'logs/statistical_modeling.log')
        self.models = {}
        self.results = {}
    
    def prepare_time_series_data(self, df: pd.DataFrame, frequency: str = 'D') -> pd.DataFrame:
        """Prepare data for time series analysis"""
        self.logger.info("Preparing time series data...")
        
        # Convert to time series
        df_ts = df.copy()
        df_ts['date'] = pd.to_datetime(df_ts['date'])
        df_ts = df_ts.set_index('date')
        
        # Aggregate sales by date
        ts_data = df_ts.groupby('date')['sales'].sum().resample(frequency).sum()
        
        # Handle any missing dates
        ts_data = ts_data.asfreq(frequency, fill_value=0)
        
        self.logger.info(f"Time series data prepared: {len(ts_data)} points")
        return ts_data
    
    def decompose_time_series(self, ts_data: pd.Series, model: str = 'additive'):
        """Decompose time series into trend, seasonal, and residual components"""
        self.logger.info("Decomposing time series...")
        
        # Ensure sufficient data for decomposition
        if len(ts_data) < 2 * 365:  # Less than 2 years
            period = 30  # Monthly seasonality
        else:
            period = 365  # Yearly seasonality
        
        try:
            decomposition = seasonal_decompose(ts_data, model=model, period=period)
            
            # Plot decomposition
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            decomposition.observed.plot(ax=axes[0], title='Original')
            decomposition.trend.plot(ax=axes[1], title='Trend')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
            decomposition.resid.plot(ax=axes[3], title='Residual')
            
            plt.tight_layout()
            plt.savefig('reports/figures/time_series_decomposition.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return decomposition
            
        except Exception as e:
            self.logger.warning(f"Decomposition failed: {str(e)}")
            return None
    
    def check_stationarity(self, ts_data: pd.Series):
        """Check stationarity using Augmented Dickey-Fuller test"""
        self.logger.info("Checking stationarity...")
        
        result = adfuller(ts_data.dropna())
        
        stationarity_report = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] <= 0.05
        }
        
        self.logger.info(f"ADF p-value: {result[1]:.4f}")
        self.logger.info(f"Stationary: {stationarity_report['is_stationary']}")
        
        # Plot ACF and PACF
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(ts_data, ax=axes[0], lags=40)
        plot_pacf(ts_data, ax=axes[1], lags=40)
        plt.tight_layout()
        plt.savefig('reports/figures/acf_pacf_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return stationarity_report
    
    def fit_sarima(self, ts_data: pd.Series, order: tuple = (1, 1, 1), 
                   seasonal_order: tuple = (1, 1, 1, 7)) -> dict:
        """Fit SARIMA model to time series data"""
        self.logger.info("Fitting SARIMA model...")
        
        try:
            # Split data
            train_size = int(len(ts_data) * 0.8)
            train, test = ts_data[:train_size], ts_data[train_size:]
            
            # Fit SARIMA model
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit(disp=False)
            
            # Forecast
            forecast = fitted_model.get_forecast(steps=len(test))
            forecast_values = forecast.predicted_mean
            confidence_intervals = forecast.conf_int()
            
            # Calculate metrics
            mae = mean_absolute_error(test, forecast_values)
            rmse = np.sqrt(mean_squared_error(test, forecast_values))
            r2 = r2_score(test, forecast_values)
            
            results = {
                'model': fitted_model,
                'forecast': forecast_values,
                'confidence_intervals': confidence_intervals,
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                },
                'order': order,
                'seasonal_order': seasonal_order
            }
            
            self.models['SARIMA'] = results
            self.logger.info(f"SARIMA - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
            
            # Plot results
            self._plot_sarima_results(ts_data, train, test, forecast_values, confidence_intervals)
            
            return results
            
        except Exception as e:
            self.logger.error(f"SARIMA fitting failed: {str(e)}")
            return None
    
    def _plot_sarima_results(self, full_data, train, test, forecast, conf_int):
        """Plot SARIMA model results"""
        plt.figure(figsize=(15, 8))
        
        # Plot historical data
        plt.plot(train.index, train.values, label='Training Data', color='blue', alpha=0.7)
        plt.plot(test.index, test.values, label='Test Data', color='green', alpha=0.7)
        
        # Plot forecast
        plt.plot(forecast.index, forecast.values, label='SARIMA Forecast', color='red', linewidth=2)
        
        # Plot confidence intervals
        plt.fill_between(conf_int.index, 
                        conf_int.iloc[:, 0], 
                        conf_int.iloc[:, 1], 
                        color='pink', alpha=0.3, label='Confidence Interval')
        
        plt.title('SARIMA Model - Forecast vs Actual', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reports/figures/sarima_forecast.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def fit_exponential_smoothing(self, ts_data: pd.Series, trend: str = 'add', 
                                 seasonal: str = 'add', seasonal_periods: int = 7) -> dict:
        """Fit Exponential Smoothing model (Holt-Winters)"""
        self.logger.info("Fitting Exponential Smoothing model...")
        
        try:
            # Split data
            train_size = int(len(ts_data) * 0.8)
            train, test = ts_data[:train_size], ts_data[train_size:]
            
            # Fit model
            model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, 
                                       seasonal_periods=seasonal_periods)
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(len(test))
            
            # Calculate metrics
            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mean_squared_error(test, forecast))
            r2 = r2_score(test, forecast)
            
            results = {
                'model': fitted_model,
                'forecast': forecast,
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                },
                'params': {
                    'trend': trend,
                    'seasonal': seasonal,
                    'seasonal_periods': seasonal_periods
                }
            }
            
            self.models['ExponentialSmoothing'] = results
            self.logger.info(f"Exponential Smoothing - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
            
            # Plot results
            self._plot_es_results(train, test, forecast)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Exponential Smoothing fitting failed: {str(e)}")
            return None
    
    def _plot_es_results(self, train, test, forecast):
        """Plot Exponential Smoothing results"""
        plt.figure(figsize=(15, 8))
        
        plt.plot(train.index, train.values, label='Training Data', color='blue', alpha=0.7)
        plt.plot(test.index, test.values, label='Test Data', color='green', alpha=0.7)
        plt.plot(forecast.index, forecast.values, label='Exponential Smoothing Forecast', 
                color='red', linewidth=2)
        
        plt.title('Exponential Smoothing - Forecast vs Actual', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reports/figures/exponential_smoothing_forecast.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_statistical_models(self) -> pd.DataFrame:
        """Compare performance of all statistical models"""
        self.logger.info("Comparing statistical models...")
        
        comparison_data = []
        
        for model_name, model_info in self.models.items():
            metrics = model_info['metrics']
            
            comparison_data.append({
                'Model': model_name,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'R2': metrics['r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Plot comparison
        if len(comparison_df) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            comparison_df.set_index('Model')['MAE'].plot(kind='bar', ax=axes[0], color='skyblue')
            axes[0].set_title('MAE Comparison')
            axes[0].tick_params(axis='x', rotation=45)
            
            comparison_df.set_index('Model')['RMSE'].plot(kind='bar', ax=axes[1], color='lightcoral')
            axes[1].set_title('RMSE Comparison')
            axes[1].tick_params(axis='x', rotation=45)
            
            comparison_df.set_index('Model')['R2'].plot(kind='bar', ax=axes[2], color='lightgreen')
            axes[2].set_title('R² Comparison')
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('reports/figures/statistical_models_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        save_dataframe(comparison_df, 'reports/statistical_models_performance.csv', 
                      'Statistical models performance')
        
        return comparison_df
    
    def generate_forecasts(self, ts_data: pd.Series, horizon: int = 30) -> dict:
        """Generate future forecasts using the best statistical model"""
        self.logger.info(f"Generating {horizon}-day forecasts...")
        
        if not self.models:
            self.logger.warning("No models trained. Training default models...")
            self.fit_sarima(ts_data)
            self.fit_exponential_smoothing(ts_data)
        
        # Find best model based on RMSE
        best_model_name = min(
            self.models.items(), 
            key=lambda x: x[1]['metrics']['rmse']
        )[0]
        
        best_model_info = self.models[best_model_name]
        model = best_model_info['model']
        
        # Generate forecasts
        if best_model_name == 'SARIMA':
            forecast = model.get_forecast(steps=horizon)
            forecast_values = forecast.predicted_mean
            conf_int = forecast.conf_int()
        else:  # Exponential Smoothing
            forecast_values = model.forecast(horizon)
            conf_int = None  # Exponential Smoothing doesn't provide confidence intervals by default
        
        # Create forecast dataframe
        last_date = ts_data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_sales': forecast_values,
            'model': best_model_name
        })
        
        if conf_int is not None:
            forecast_df['lower_bound'] = conf_int.iloc[:, 0].values
            forecast_df['upper_bound'] = conf_int.iloc[:, 1].values
        
        self.logger.info(f"Best statistical model: {best_model_name}")
        
        # Plot forecasts
        self._plot_future_forecasts(ts_data, forecast_df, best_model_name)
        
        return forecast_df
    
    def _plot_future_forecasts(self, historical_data, forecast_df, model_name):
        """Plot future forecasts with historical context"""
        plt.figure(figsize=(15, 8))
        
        # Plot historical data (last 90 days for clarity)
        recent_data = historical_data[-90:]
        plt.plot(recent_data.index, recent_data.values, 
                label='Historical Data', color='blue', linewidth=2)
        
        # Plot forecasts
        plt.plot(forecast_df['date'], forecast_df['predicted_sales'], 
                label=f'{model_name} Forecast', color='red', linewidth=2, marker='o')
        
        # Plot confidence intervals if available
        if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
            plt.fill_between(forecast_df['date'], 
                           forecast_df['lower_bound'], 
                           forecast_df['upper_bound'], 
                           color='pink', alpha=0.3, label='Confidence Interval')
        
        plt.title(f'Future Sales Forecast - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reports/figures/future_forecasts_statistical.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    modeler = StatisticalModeler()
    # Example usage:
    # df = pd.read_csv('data/processed/engineered_data.csv')
    # ts_data = modeler.prepare_time_series_data(df)
    # decomposition = modeler.decompose_time_series(ts_data)
    # stationarity = modeler.check_stationarity(ts_data)
    # sarima_results = modeler.fit_sarima(ts_data)
    # es_results = modeler.fit_exponential_smoothing(ts_data)
    # comparison = modeler.compare_statistical_models()
    # forecasts = modeler.generate_forecasts(ts_data, horizon=30)