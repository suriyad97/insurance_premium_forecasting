# src/model_evaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils import setup_logging, save_dataframe

class ModelEvaluator:
    """Model Evaluation - Comprehensive model performance assessment"""
    
    def __init__(self):
        self.logger = setup_logging('model_evaluation', 'logs/model_evaluation.log')
        self.metrics_df = None
    
    def evaluate_models(self, results: dict) -> pd.DataFrame:
        """Evaluate all models and generate performance metrics"""
        self.logger.info("Evaluating model performance...")
        
        metrics_data = []
        
        for model_name, model_info in results.items():
            metrics = model_info['metrics']
            
            metrics_data.append({
                'Model': model_name,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'R2': metrics['r2'],
                'MAPE': metrics['mape'],
                'Best_Params': str(model_info['best_params'])
            })
            
            self.logger.info(f"{model_name}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
        
        self.metrics_df = pd.DataFrame(metrics_data)
        save_dataframe(self.metrics_df, 'reports/model_performance_detailed.csv', 'Detailed model performance')
        
        return self.metrics_df
    
    def create_model_comparison_plots(self, results: dict, metrics_df: pd.DataFrame):
        """Create comprehensive model comparison visualizations"""
        self.logger.info("Creating model comparison plots...")
        
        # Model performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # MAE comparison
        metrics_df.set_index('Model')['MAE'].sort_values().plot(
            kind='bar', ax=axes[0,0], color='skyblue', alpha=0.7
        )
        axes[0,0].set_title('MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('MAE')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        metrics_df.set_index('Model')['RMSE'].sort_values().plot(
            kind='bar', ax=axes[0,1], color='lightcoral', alpha=0.7
        )
        axes[0,1].set_title('RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('RMSE')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        metrics_df.set_index('Model')['R2'].sort_values(ascending=False).plot(
            kind='bar', ax=axes[1,0], color='lightgreen', alpha=0.7
        )
        axes[1,0].set_title('R² Comparison (Higher is Better)', fontsize=14, fontweight='bold')
        axes[1,0].set_ylabel('R² Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        metrics_df.set_index('Model')['MAPE'].sort_values().plot(
            kind='bar', ax=axes[1,1], color='gold', alpha=0.7
        )
        axes[1,1].set_title('MAPE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        axes[1,1].set_ylabel('MAPE (%)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('reports/figures/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Actual vs Predicted plots
        self._create_actual_vs_predicted_plots(results)
        
        # Residual plots
        self._create_residual_plots(results)
    
    def _create_actual_vs_predicted_plots(self, results: dict):
        """Create actual vs predicted plots for all models"""
        n_models = len(results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for idx, (model_name, model_info) in enumerate(results.items()):
            if idx < len(axes):
                y_true = model_info['y_test']
                y_pred = model_info['predictions']
                
                axes[idx].scatter(y_true, y_pred, alpha=0.6, s=50)
                axes[idx].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
                axes[idx].set_xlabel('Actual Values')
                axes[idx].set_ylabel('Predicted Values')
                axes[idx].set_title(f'{model_name}\nR² = {model_info["metrics"]["r2"]:.3f}')
                axes[idx].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('reports/figures/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_residual_plots(self, results: dict):
        """Create residual plots for all models"""
        n_models = len(results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for idx, (model_name, model_info) in enumerate(results.items()):
            if idx < len(axes):
                y_true = model_info['y_test']
                y_pred = model_info['predictions']
                residuals = y_true - y_pred
                
                axes[idx].scatter(y_pred, residuals, alpha=0.6, s=50)
                axes[idx].axhline(y=0, color='r', linestyle='--')
                axes[idx].set_xlabel('Predicted Values')
                axes[idx].set_ylabel('Residuals')
                axes[idx].set_title(f'{model_name}\nResiduals')
                axes[idx].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('reports/figures/residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def identify_best_model(self, metrics_df: pd.DataFrame) -> dict:
        """Identify the best performing model based on multiple metrics"""
        self.logger.info("Identifying best model...")
        
        # Normalize metrics (lower is better for MAE, RMSE, MAPE; higher for R²)
        normalized_df = metrics_df.copy()
        
        for metric in ['MAE', 'RMSE', 'MAPE']:
            normalized_df[f'{metric}_norm'] = (
                metrics_df[metric] - metrics_df[metric].min()
            ) / (metrics_df[metric].max() - metrics_df[metric].min())
        
        # For R², higher is better so we invert
        normalized_df['R2_norm'] = 1 - (
            (metrics_df['R2'] - metrics_df['R2'].min()) / 
            (metrics_df['R2'].max() - metrics_df['R2'].min())
        )
        
        # Calculate combined score (lower is better)
        normalized_df['combined_score'] = (
            normalized_df['MAE_norm'] + 
            normalized_df['RMSE_norm'] + 
            normalized_df['MAPE_norm'] + 
            normalized_df['R2_norm']
        )
        
        best_model_row = normalized_df.loc[normalized_df['combined_score'].idxmin()]
        best_model = best_model_row['Model']
        
        self.logger.info(f"🏆 Best model: {best_model}")
        self.logger.info(f"   MAE: {best_model_row['MAE']:.2f}")
        self.logger.info(f"   RMSE: {best_model_row['RMSE']:.2f}")
        self.logger.info(f"   R²: {best_model_row['R2']:.3f}")
        self.logger.info(f"   MAPE: {best_model_row['MAPE']:.2f}%")
        
        return {
            'model_name': best_model,
            'metrics': {
                'MAE': best_model_row['MAE'],
                'RMSE': best_model_row['RMSE'],
                'R2': best_model_row['R2'],
                'MAPE': best_model_row['MAPE']
            }
        }

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    # Example usage after model training
    # metrics_df = evaluator.evaluate_models(results)
    # evaluator.create_model_comparison_plots(results, metrics_df)
    # best_model = evaluator.identify_best_model(metrics_df)