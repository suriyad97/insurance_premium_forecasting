# src/model_interpretation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from src.utils import setup_logging, save_dataframe

class ModelInterpreter:
    """Model Interpretation - SHAP analysis and feature importance"""
    
    def __init__(self):
        self.logger = setup_logging('model_interpretation', 'logs/model_interpretation.log')
        self.shap_results = {}
    
    def perform_shap_analysis(self, results: dict, feature_names: list, top_n: int = 15):
        """Perform SHAP analysis for model interpretability"""
        self.logger.info("Performing SHAP analysis...")
        
        # Find the best model based on RMSE
        best_model_name = min(
            results.items(), 
            key=lambda x: x[1]['metrics']['rmse']
        )[0]
        
        best_model_info = results[best_model_name]
        model = best_model_info['model']
        X_test = best_model_info['X_test']
        
        self.logger.info(f"Analyzing best model: {best_model_name}")
        
        # Create SHAP explainer based on model type
        try:
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X_test)
            
            # Calculate SHAP values
            shap_values = explainer(X_test)
            
            # Store results
            self.shap_results[best_model_name] = {
                'shap_values': shap_values,
                'explainer': explainer
            }
            
            # Create SHAP plots
            self._create_shap_plots(shap_values, X_test, feature_names, best_model_name, top_n)
            
            return self.shap_results
            
        except Exception as e:
            self.logger.warning(f"SHAP analysis failed: {str(e)}")
            return {}
    
    def _create_shap_plots(self, shap_values, X_test, feature_names: list, 
                          model_name: str, top_n: int):
        """Create comprehensive SHAP plots"""
        self.logger.info("Creating SHAP plots...")
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary Plot - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('reports/figures/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance plot
        feature_importance = self._calculate_feature_importance(shap_values, feature_names)
        
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(top_n)
        
        plt.subplot(1, 2, 1)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance\n{model_name}', fontweight='bold')
        plt.xlabel('Mean |SHAP Value|')
        
        plt.subplot(1, 2, 2)
        # Create a beeswarm plot for top features
        top_feature_indices = [feature_names.index(f) for f in top_features['feature']]
        shap.summary_plot(
            shap_values[:, top_feature_indices], 
            X_test[:, top_feature_indices],
            feature_names=[feature_names[i] for i in top_feature_indices],
            show=False,
            plot_size=(8, 6)
        )
        plt.title(f'SHAP Beeswarm Plot - Top {top_n} Features', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('reports/figures/feature_importance_shap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save feature importance
        save_dataframe(feature_importance, 'reports/feature_importance.csv', 'Feature importance')
        
        # Create dependency plots for top 3 features
        self._create_dependency_plots(shap_values, X_test, feature_names, feature_importance, model_name)
    
    def _calculate_feature_importance(self, shap_values, feature_names: list) -> pd.DataFrame:
        """Calculate feature importance from SHAP values"""
        if hasattr(shap_values, 'values'):
            shap_importance = np.abs(shap_values.values).mean(0)
        else:
            shap_importance = np.abs(shap_values).mean(0)
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': shap_importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def _create_dependency_plots(self, shap_values, X_test, feature_names: list, 
                               feature_importance: pd.DataFrame, model_name: str):
        """Create SHAP dependency plots for top features"""
        top_features = feature_importance.head(3)['feature'].tolist()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, feature in enumerate(top_features):
            if idx < 3:
                feature_idx = feature_names.index(feature)
                shap.dependence_plot(
                    feature_idx, shap_values.values if hasattr(shap_values, 'values') else shap_values,
                    X_test, feature_names=feature_names, ax=axes[idx], show=False
                )
                axes[idx].set_title(f'SHAP Dependency: {feature}')
        
        plt.tight_layout()
        plt.savefig('reports/figures/shap_dependency_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_interpretation_report(self, results: dict, best_model_info: dict):
        """Generate comprehensive model interpretation report"""
        self.logger.info("Generating model interpretation report...")
        
        report_content = f"""
# MODEL INTERPRETATION REPORT

## Best Model: {best_model_info['model_name']}

### Performance Metrics:
- **MAE**: {best_model_info['metrics']['MAE']:.2f}
- **RMSE**: {best_model_info['metrics']['RMSE']:.2f}
- **R²**: {best_model_info['metrics']['R2']:.3f}
- **MAPE**: {best_model_info['metrics']['MAPE']:.2f}%

### All Models Performance:
"""
        
        # Add performance comparison
        for model_name, model_info in results.items():
            metrics = model_info['metrics']
            report_content += f"""
#### {model_name}
- MAE: {metrics['mae']:.2f}
- RMSE: {metrics['rmse']:.2f}
- R²: {metrics['r2']:.3f}
- MAPE: {metrics['mape']:.2f}%
"""
        
        # Save report
        report_path = Path('reports/model_interpretation_report.md')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Interpretation report saved to: {report_path}")

if __name__ == "__main__":
    interpreter = ModelInterpreter()
    # Example usage after model evaluation
    # shap_results = interpreter.perform_shap_analysis(results, feature_cols)
    # interpreter.generate_interpretation_report(results, best_model_info)