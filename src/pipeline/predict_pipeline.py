import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, **kwargs):
        """
        Initialize with feature values
        For APS failure dataset, we have 170 features
        """
        self.features = kwargs
        
    def get_data_as_data_frame(self):
        try:
            # Create a dataframe with the input features
            # This should match the feature structure of your training data
            custom_data_input_dict = self.features
            
            return pd.DataFrame([custom_data_input_dict])

        except Exception as e:
            raise CustomException(e, sys)
