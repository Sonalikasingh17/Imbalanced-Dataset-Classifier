import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        
        try:
            # Read the dataset - update with your actual data path
            logging.info("Reading APS failure training dataset...")
            df = pd.read_csv('aps_failure_training_set.csv', na_values=['na'])
            
            logging.info('Read the dataset as dataframe')
            print(f"📊 Dataset loaded successfully!")
            print(f"📏 Original Dataset Shape: {df.shape}")
            print(f"💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Check class distribution
            class_counts = df['class'].value_counts()
            total_samples = len(df)
            
            print(f"\n🎯 Class Distribution Analysis:")
            print(f"  • Total samples: {total_samples:,}")
            for class_name, count in class_counts.items():
                percentage = (count / total_samples) * 100
                print(f"  • {class_name}: {count:,} ({percentage:.2f}%)")
            
            # Calculate imbalance ratio
            if len(class_counts) >= 2:
                majority_class = class_counts.iloc[0]
                minority_class = class_counts.iloc[1]
                imbalance_ratio = majority_class / minority_class
                print(f"  • Imbalance ratio: {imbalance_ratio:.1f}:1")
                
                if imbalance_ratio > 10:
                    print(f"  ⚠️ Severe class imbalance detected!")
                    logging.warning(f"Severe class imbalance detected: {imbalance_ratio:.1f}:1")
            
            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved successfully")
            
            # Perform stratified train-test split to maintain class distribution
            logging.info("Performing stratified train-test split...")
            print(f"\n🔄 Performing stratified train-test split (80/20)...")
            
            train_set, test_set = train_test_split(
                df, 
                test_size=0.2, 
                random_state=42, 
                stratify=df['class'],
                shuffle=True
            )
            
            # Verify stratification worked correctly
            train_class_dist = train_set['class'].value_counts(normalize=True)
            test_class_dist = test_set['class'].value_counts(normalize=True)
            
            print(f"✅ Stratified split completed successfully!")
            print(f"📊 Training set: {len(train_set):,} samples")
            for class_name in train_class_dist.index:
                print(f"  • {class_name}: {train_set['class'].value_counts()[class_name]:,} ({train_class_dist[class_name]*100:.2f}%)")
            
            print(f"📊 Test set: {len(test_set):,} samples")
            for class_name in test_class_dist.index:
                print(f"  • {class_name}: {test_set['class'].value_counts()[class_name]:,} ({test_class_dist[class_name]*100:.2f}%)")
            
            # Verify class distributions are similar
            distribution_diff = abs(train_class_dist - test_class_dist).max()
            if distribution_diff < 0.01:  # Less than 1% difference
                print(f"✅ Class distributions maintained across train/test splits (max diff: {distribution_diff*100:.2f}%)")
            else:
                print(f"⚠️ Class distribution difference: {distribution_diff*100:.2f}%")
                logging.warning(f"Class distribution difference between train/test: {distribution_diff*100:.2f}%")
            
            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Train and test data saved successfully")
            logging.info("Ingestion of the data is completed")
            
            # Print summary
            print(f"\n📁 Data Ingestion Summary:")
            print(f"  • Raw data: {self.ingestion_config.raw_data_path}")
            print(f"  • Train data: {self.ingestion_config.train_data_path}")  
            print(f"  • Test data: {self.ingestion_config.test_data_path}")
            print(f"✅ Data ingestion completed successfully!")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    print(f"\n🎉 Data ingestion pipeline completed!")
    print(f"📊 Training data: {train_data}")
    print(f"📊 Test data: {test_data}")