import os
import sys
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
            logging.info("Starting training pipeline")
            
            # Data Ingestion
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            
            # Data Transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            
            # Model Training
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            
            logging.f"Training pipeline completed. Model F1 Score: {model_score}")
            print(f"Training pipeline completed. Model F1 Score: {model_score}")
            
            return model_score
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline.start_training_pipeline()
