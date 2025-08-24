import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
import joblib
import json

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # Main preprocessor path (for compatibility)
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    
    # Individual preprocessor paths (matching notebook structure)
    imputer_path = os.path.join('artifacts', 'preprocessors', 'imputer.pkl')
    scaler_path = os.path.join('artifacts', 'preprocessors', 'scaler.pkl')
    variance_selector_path = os.path.join('artifacts', 'preprocessors', 'variance_selector.pkl')
    label_encoder_path = os.path.join('artifacts', 'preprocessors', 'label_encoder.pkl')
    
    # Processed data paths (matching notebook structure)
    X_train_processed_path = os.path.join('artifacts', 'data_transformation', 'X_train_processed.csv')
    X_test_processed_path = os.path.join('artifacts', 'data_transformation', 'X_test_processed.csv')
    y_train_processed_path = os.path.join('artifacts', 'data_transformation', 'y_train_processed.csv')
    y_test_processed_path = os.path.join('artifacts', 'data_transformation', 'y_test_processed.csv')
    feature_info_path = os.path.join('artifacts', 'data_transformation', 'feature_info.json')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        # Create necessary directories
        os.makedirs('artifacts/preprocessors', exist_ok=True)
        os.makedirs('artifacts/data_transformation', exist_ok=True)

    def get_data_transformer_object(self):
        """
        This function creates individual preprocessing components
        """
        try:
            logging.info("Creating individual preprocessing components")
            
            # Create individual preprocessors
            imputer = SimpleImputer(strategy="median")
            variance_selector = VarianceThreshold(threshold=0.0)  # Remove zero-variance features
            scaler = StandardScaler()
            
            logging.info("Individual preprocessing components created successfully")
            
            return imputer, variance_selector, scaler
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            
            # Separate features and target
            target_column_name = "class"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Getting preprocessing components")
            
            # Get individual preprocessing components
            imputer, variance_selector, scaler = self.get_data_transformer_object()
            
            # Apply transformations step by step
            logging.info("Applying imputation...")
            
            # Step 1: Imputation
            X_train_imputed = imputer.fit_transform(input_feature_train_df)
            X_test_imputed = imputer.transform(input_feature_test_df)
            
            # Convert back to DataFrame to maintain feature names
            X_train_imputed_df = pd.DataFrame(X_train_imputed, columns=input_feature_train_df.columns)
            X_test_imputed_df = pd.DataFrame(X_test_imputed, columns=input_feature_test_df.columns)
            
            logging.info("Applying variance threshold...")
            
            # Step 2: Remove low-variance features
            X_train_selected = variance_selector.fit_transform(X_train_imputed_df)
            X_test_selected = variance_selector.transform(X_test_imputed_df)
            
            # Get selected feature names
            selected_features = X_train_imputed_df.columns[variance_selector.get_support()].tolist()
            removed_features = X_train_imputed_df.columns[~variance_selector.get_support()].tolist()
            
            # Convert back to DataFrame
            X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
            X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)
            
            logging.info("Applying scaling...")
            
            # Step 3: Scaling
            X_train_scaled = scaler.fit_transform(X_train_selected_df)
            X_test_scaled = scaler.transform(X_test_selected_df)
            
            # Convert back to DataFrame
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=selected_features)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=selected_features)
            
            logging.info("Encoding target labels...")
            
            # Step 4: Encode target labels
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(target_feature_train_df)
            y_test_encoded = label_encoder.transform(target_feature_test_df)
            
            # Create target DataFrames
            y_train_df = pd.DataFrame({'target': y_train_encoded})
            y_test_df = pd.DataFrame({'target': y_test_encoded})
            
            # Save processed data (matching notebook structure)
            logging.info("Saving processed data...")
            
            X_train_scaled_df.to_csv(self.data_transformation_config.X_train_processed_path, index=False)
            X_test_scaled_df.to_csv(self.data_transformation_config.X_test_processed_path, index=False)
            y_train_df.to_csv(self.data_transformation_config.y_train_processed_path, index=False)
            y_test_df.to_csv(self.data_transformation_config.y_test_processed_path, index=False)
            
            # Save feature information
            feature_info = {
                'original_features': input_feature_train_df.columns.tolist(),
                'selected_features': selected_features,
                'removed_features': removed_features,
                'n_original_features': len(input_feature_train_df.columns),
                'n_selected_features': len(selected_features),
                'n_removed_features': len(removed_features)
            }
            
            with open(self.data_transformation_config.feature_info_path, 'w') as f:
                json.dump(feature_info, f, indent=2)
            
            # Save individual preprocessors (matching notebook structure)
            logging.info("Saving individual preprocessors...")
            
            joblib.dump(imputer, self.data_transformation_config.imputer_path)
            joblib.dump(variance_selector, self.data_transformation_config.variance_selector_path)
            joblib.dump(scaler, self.data_transformation_config.scaler_path)
            joblib.dump(label_encoder, self.data_transformation_config.label_encoder_path)
            
            # Create combined preprocessor for compatibility
            logging.info("Creating combined preprocessor for compatibility...")
            
            # Create a pipeline that combines all preprocessing steps
            combined_preprocessor = Pipeline([
                ('imputer', imputer),
                ('variance_selector', variance_selector),
                ('scaler', scaler)
            ])
            
            # Save combined preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=combined_preprocessor
            )
            
            # Prepare arrays for model training (backward compatibility)
            train_arr = np.c_[X_train_scaled, y_train_encoded]
            test_arr = np.c_[X_test_scaled, y_test_encoded]
            
            logging.info("Data transformation completed successfully")
            
            print(f"\n📊 FINAL PREPROCESSED DATA SUMMARY")
            print(f"="*50)
            print(f"🎯 Final Dataset Characteristics:")
            print(f"  • Total samples: {len(X_train_scaled_df) + len(X_test_scaled_df):,}")
            print(f"  • Total features: {X_train_scaled_df.shape[1]}")
            print(f"  • Training samples: {len(X_train_scaled_df):,} ({len(X_train_scaled_df)/(len(X_train_scaled_df) + len(X_test_scaled_df))*100:.1f}%)")
            print(f"  • Testing samples: {len(X_test_scaled_df):,} ({len(X_test_scaled_df)/(len(X_train_scaled_df) + len(X_test_scaled_df))*100:.1f}%)")
            print(f"\n🔧 Preprocessing Steps Applied:")
            print(f"  • ✅ Missing values imputed (median strategy)")
            print(f"  • ✅ Low-variance features removed ({len(removed_features)} features)")
            print(f"  • ✅ Features scaled (StandardScaler)")
            print(f"  • ✅ Target variable encoded (LabelEncoder)")
            print(f"  • ✅ Stratified train-test split (80/20)")
            print(f"\n📁 Saved Preprocessing Objects:")
            print(f"  • artifacts/preprocessors/imputer.pkl")
            print(f"  • artifacts/preprocessors/variance_selector.pkl") 
            print(f"  • artifacts/preprocessors/scaler.pkl")
            print(f"  • artifacts/preprocessors/label_encoder.pkl")
            print(f"  • artifacts/preprocessor.pkl (combined)")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)
