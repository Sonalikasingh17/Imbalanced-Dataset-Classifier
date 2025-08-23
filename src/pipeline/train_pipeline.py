import os
import sys
from datetime import datetime

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        pass

    def start_training_pipeline(self):
        try:
            logging.info("=" * 60)
            logging.info("🚀 STARTING APS FAILURE PREDICTION TRAINING PIPELINE")
            logging.info("=" * 60)
            
            print(f"\n🚀 APS FAILURE PREDICTION - TRAINING PIPELINE")
            print(f"=" * 60)
            print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"=" * 60)

            # Step 1: Data Ingestion
            print(f"\n1️⃣ DATA INGESTION PHASE")
            print(f"-" * 30)
            logging.info("Starting data ingestion phase")
            
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            
            logging.info(f"Data ingestion completed - Train: {train_data_path}, Test: {test_data_path}")

            # Step 2: Data Transformation
            print(f"\n2️⃣ DATA TRANSFORMATION PHASE")
            print(f"-" * 35)
            logging.info("Starting data transformation phase")
            
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            
            logging.info(f"Data transformation completed - Preprocessor: {preprocessor_path}")
            print(f"✅ Preprocessing completed! Data ready for training.")

            # Step 3: Model Training
            print(f"\n3️⃣ MODEL TRAINING PHASE")
            print(f"-" * 30)
            logging.info("Starting model training phase")
            
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            
            logging.info(f"Model training completed with F1 Score: {model_score:.4f}")

            # Pipeline Summary
            print(f"\n" + "=" * 60)
            print(f"🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"=" * 60)
            print(f"📊 Final Model Performance:")
            print(f"  • F1-Score (Macro): {model_score:.4f}")
            print(f"📁 Generated Artifacts:")
            print(f"  • Raw Data: artifacts/data.csv")
            print(f"  • Processed Data: artifacts/data_transformation/")
            print(f"  • Preprocessors: artifacts/preprocessors/")
            print(f"  • Best Model: artifacts/models/best_model.pkl")
            print(f"  • Training Report: artifacts/model_trainer/training_report.txt")
            print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"=" * 60)
            
            logging.info("🎉 Training pipeline completed successfully!")
            
            return model_score

        except Exception as e:
            error_msg = f"Error in training pipeline: {str(e)}"
            logging.error(error_msg)
            print(f"\n❌ TRAINING PIPELINE FAILED!")
            print(f"Error: {error_msg}")
            raise CustomException(e, sys)

def run_full_pipeline():
    """
    Convenience function to run the complete training pipeline
    """
    try:
        # Create necessary directories
        directories = [
            'artifacts',
            'artifacts/data_transformation', 
            'artifacts/preprocessors',
            'artifacts/models',
            'artifacts/model_trainer',
            'artifacts/evaluations'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize and run pipeline
        training_pipeline = TrainingPipeline()
        score = training_pipeline.start_training_pipeline()
        
        return score
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        print("🔄 Initializing APS Failure Prediction Training Pipeline...")
        
        # Run the complete pipeline
        final_score = run_full_pipeline()
        
        print(f"\n🎯 PIPELINE EXECUTION SUMMARY:")
        print(f"✅ Status: SUCCESS")
        print(f"📊 Final F1-Score: {final_score:.4f}")
        print(f"🚀 Model ready for deployment!")
        
    except Exception as e:
        print(f"\n💥 PIPELINE EXECUTION FAILED:")
        print(f"❌ Error: {str(e)}")
        sys.exit(1)