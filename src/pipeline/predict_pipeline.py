import sys
import pandas as pd
import numpy as np
import joblib
import os

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Updated paths to match the new structure
            model_path = os.path.join("artifacts", "models", "best_model.pkl")
            
            # Try to load individual preprocessors first (matching notebook structure)
            try:
                imputer_path = os.path.join('artifacts', 'preprocessors', 'imputer.pkl')
                variance_selector_path = os.path.join('artifacts', 'preprocessors', 'variance_selector.pkl')
                scaler_path = os.path.join('artifacts', 'preprocessors', 'scaler.pkl')
                label_encoder_path = os.path.join('artifacts', 'preprocessors', 'label_encoder.pkl')
                
                print("Loading individual preprocessors...")
                imputer = joblib.load(imputer_path)
                variance_selector = joblib.load(variance_selector_path)
                scaler = joblib.load(scaler_path)
                label_encoder = joblib.load(label_encoder_path)
                
                print("Loading best model...")
                model = load_object(file_path=model_path)
                
                # Apply preprocessing steps in sequence
                print("Applying preprocessing steps...")
                
                # Step 1: Imputation
                data_imputed = imputer.transform(features)
                data_imputed_df = pd.DataFrame(data_imputed, columns= features.columns)
                
                # Step 2: Variance threshold (feature selection)
                data_selected = variance_selector.transform(data_imputed_df)
                
                # Step 3: Scaling
                data_scaled = scaler.transform(data_selected)
                
                print("Making predictions...")
                preds = model.predict(data_scaled)
                
                # Decode predictions back to original labels
                preds_decoded = label_encoder.inverse_transform(preds)
                
                return preds_decoded
                
            except FileNotFoundError:
                # Fallback to combined preprocessor (backward compatibility)
                print("Individual preprocessors not found, using combined preprocessor...")
                preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
                
                print("Loading combined preprocessor and model...")
                model = load_object(file_path=model_path)
                preprocessor = load_object(file_path=preprocessor_path)
                
                print("Applying preprocessing...")
                data_scaled = preprocessor.transform(features)
                
                print("Making predictions...")
                preds = model.predict(data_scaled)
                
                # Try to decode if label encoder exists
                try:
                    label_encoder_path = os.path.join('artifacts', 'preprocessors', 'label_encoder.pkl')
                    label_encoder = joblib.load(label_encoder_path)
                    preds_decoded = label_encoder.inverse_transform(preds)
                    return preds_decoded
                except:
                    # Return raw predictions if no label encoder found
                    return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, **kwargs):
        """
        Initialize with feature values
        For APS failure dataset, we have 170 features (after preprocessing it will be reduced)
        """
        self.features = kwargs

    def get_data_as_data_frame(self):
        try:
            # Create a dataframe with the input features
            # This should match the feature structure of your training data
            custom_data_input_dict = self.features
            df = pd.DataFrame([custom_data_input_dict])
            
            # Load feature info to ensure correct feature order
            try:
                feature_info_path = os.path.join('artifacts', 'data_transformation', 'feature_info.json')
                import json
                with open(feature_info_path, 'r') as f:
                    feature_info = json.load(f)
                
                # Ensure all original features are present, fill missing with 0
                original_features = feature_info['original_features']
                for feature in original_features:
                    if feature not in df.columns:
                        df[feature] = 0.0
                
                # Reorder columns to match training data
                df = df[original_features]
                
            except FileNotFoundError:
                print("Feature info not found, using provided features as-is")
                pass
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def create_sample_input():
        """
        Create a sample input for testing (based on APS dataset structure)
        """
        try:
            # Load feature info to get the correct feature names
            feature_info_path = os.path.join('artifacts', 'data_transformation', 'feature_info.json')
            try:
                import json
                with open(feature_info_path, 'r') as f:
                    feature_info = json.load(f)
                original_features = feature_info['original_features']
            except FileNotFoundError:
                # Fallback to common APS dataset features (first few)
                original_features = [
                    'aa_000', 'ab_000', 'ac_000', 'ad_000', 'ae_000', 'af_000', 'ag_000',
                    'ag_001', 'ag_002', 'ag_003', 'ag_004', 'ag_005', 'ag_006', 'ag_007',
                    'ag_008', 'ag_009', 'ah_000', 'ai_000', 'aj_000', 'ak_000'
                ] + [f"feature_{i}" for i in range(20, 170)]  # Placeholder for remaining features
            
            # Create sample data (all zeros as default)
            sample_data = {feature: 0.0 for feature in original_features}
            
            # Set some non-zero values for more realistic sample
            if 'aa_000' in sample_data:
                sample_data['aa_000'] = 76698.0
            if 'ac_000' in sample_data:
                sample_data['ac_000'] = 2130706000.0
            if 'ad_000' in sample_data:
                sample_data['ad_000'] = 280.0
            
            return sample_data
            
        except Exception as e:
            raise CustomException(e, sys)
