# main_enhanced.py
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.utils import create_directory_structure, load_config
from src.data_ingestion import DataIngestion
from src.data_validation import DataValidator
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.model_interpretation import ModelInterpreter
from src.statistical_modeling import StatisticalModeler
from src.forecasting import ForecastingEngine

def main():
    """Enhanced main function with statistical modeling and forecasting"""
    print("🚀 STARTING ENHANCED SALES FORECASTING PIPELINE")
    print("=" * 60)
    
    # Create directory structure
    create_directory_structure()
    
    # Load configuration
    config = load_config()
    
    try:
        # Step 1: Data Ingestion
        print("\n📥 STEP 1: Data Ingestion")
        ingestion = DataIngestion()
        datasets = ingestion.load_all_data()
        merged_data = ingestion.merge_datasets(datasets)
        
        # Step 2: Data Validation
        print("\n🔍 STEP 2: Data Validation")
        validator = DataValidator()
        validation_results = validator.validate_dataset(merged_data, 'merged_data')
        validator.generate_validation_report()
        
        # Step 3: Data Preprocessing
        print("\n🔄 STEP 3: Data Preprocessing")
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor.handle_missing_values(merged_data)
        
        # Step 4: Feature Engineering
        print("\n🔧 STEP 4: Feature Engineering")
        engineer = FeatureEngineer()
        engineered_data = engineer.engineer_features(cleaned_data)
        
        # Step 5: Machine Learning Modeling
        print("\n🤖 STEP 5: Machine Learning Modeling")
        trainer = ModelTrainer()
        X, y, feature_cols = trainer.prepare_features(engineered_data)
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        ml_results = trainer.train_models(X_train, X_test, y_train, y_test)
        trainer.save_models()
        
        # Step 6: ML Model Evaluation
        print("\n📊 STEP 6: ML Model Evaluation")
        ml_evaluator = ModelEvaluator()
        ml_metrics_df = ml_evaluator.evaluate_models(ml_results)
        ml_evaluator.create_model_comparison_plots(ml_results, ml_metrics_df)
        best_ml_model = ml_evaluator.identify_best_model(ml_metrics_df)
        
        # Step 7: ML Model Interpretation
        print("\n🔍 STEP 7: ML Model Interpretation")
        ml_interpreter = ModelInterpreter()
        shap_results = ml_interpreter.perform_shap_analysis(ml_results, feature_cols)
        ml_interpreter.generate_interpretation_report(ml_results, best_ml_model)
        
        # Step 8: Statistical Modeling
        print("\n📈 STEP 8: Statistical Modeling")
        stats_modeler = StatisticalModeler()
        ts_data = stats_modeler.prepare_time_series_data(engineered_data)
        
        # Time series analysis
        decomposition = stats_modeler.decompose_time_series(ts_data)
        stationarity = stats_modeler.check_stationarity(ts_data)
        
        # Fit statistical models
        sarima_results = stats_modeler.fit_sarima(ts_data)
        es_results = stats_modeler.fit_exponential_smoothing(ts_data)
        stats_comparison = stats_modeler.compare_statistical_models()
        
        # Step 9: Forecasting
        print("\n🔮 STEP 9: Forecasting")
        forecaster = ForecastingEngine()
        forecaster.load_trained_models()
        
        # Generate ML forecasts
        future_df = forecaster.create_future_features(engineered_data, 
                                                     horizon=config['forecasting']['horizon'])
        ml_forecasts = forecaster.generate_ml_forecasts(future_df, best_ml_model['model_name'])
        
        # Generate ensemble forecasts
        ensemble_forecasts = forecaster.generate_ensemble_forecast(future_df)
        
        # Plot ML forecasts
        if ml_forecasts is not None:
            forecaster.plot_forecasts(engineered_data, ml_forecasts, f"ML ({best_ml_model['model_name']})")
            forecaster.save_forecasts(ml_forecasts, 'ml_forecasts.csv')
        
        # Plot ensemble forecasts
        forecaster.plot_forecasts(engineered_data, ensemble_forecasts, "Ensemble")
        forecaster.save_forecasts(ensemble_forecasts, 'ensemble_forecasts.csv')
        
        # Generate statistical forecasts
        stats_forecasts = stats_modeler.generate_forecasts(ts_data, 
                                                          horizon=config['forecasting']['horizon'])
        
        # Final Summary
        print("\n🎉 ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
        print("\n📁 OUTPUT FILES:")
        print("   data/processed/ - Processed datasets")
        print("   models/trained_models/ - Saved models")
        print("   reports/ - Performance metrics and interpretation")
        print("   reports/figures/ - Visualizations and plots")
        print("   logs/ - Processing logs")
        print("\n📊 FORECASTS GENERATED:")
        print("   - ML forecasts (best model)")
        print("   - Ensemble forecasts (all models)")
        print("   - Statistical forecasts (SARIMA/Exponential Smoothing)")
        
        # Performance Summary
        print(f"\n🏆 BEST ML MODEL: {best_ml_model['model_name']}")
        print(f"   RMSE: {best_ml_model['metrics']['RMSE']:.2f}")
        print(f"   R²: {best_ml_model['metrics']['R2']:.3f}")
        
        if stats_comparison is not None and len(stats_comparison) > 0:
            best_stats_model = stats_comparison.loc[stats_comparison['RMSE'].idxmin()]
            print(f"\n📈 BEST STATISTICAL MODEL: {best_stats_model['Model']}")
            print(f"   RMSE: {best_stats_model['RMSE']:.2f}")
            print(f"   R²: {best_stats_model['R2']:.3f}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()