import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from src.exception import CustomException



def save_object(file_path, obj):
    """
    Save object to file using dill
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load object from file using dill
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models_bayesian(self, X_train, y_train, X_test, y_test, models, search_spaces):
        """
        Evaluate models using Bayesian hyperparameter optimization
        """
        try:
            model_report = {}
            trained_models = {}
            
            for model_name, model in models.items():
                logging.info(f"Training {model_name} with Bayesian optimization...")
                print(f"\n🔄 Bayesian tuning {model_name}...")
                
                try:
                    search_space = search_spaces[model_name]
                    bayes_cv = BayesSearchCV(
                        model,
                        search_space,
                        n_iter=10,  # Reduce n_iter for fast tuning
                        scoring='f1_macro',
                        cv=3,
                        n_jobs=-1,
                        verbose=0,
                        random_state=42
                    )
                    
                    # Fit the model
                    bayes_cv.fit(X_train, y_train)
                    
                    # Get best model
                    best_model = bayes_cv.best_estimator_
                    
                    # Make predictions
                    y_pred_test = best_model.predict(X_test)
                    
                    # Calculate F1 score
                    test_f1 = f1_score(y_test, y_pred_test, average='macro')
                    
                    # Store results
                    model_report[model_name] = test_f1
                    trained_models[model_name] = best_model
                    
                    print(f"  ✅ Best CV F1-Score: {bayes_cv.best_score_:.4f}")
                    print(f"  ✅ Test F1-Score: {test_f1:.4f}")
                    print(f"  🔧 Best params: {bayes_cv.best_params_}")
                    
                except Exception as e:
                    print(f"  ❌ Error training {model_name}: {str(e)}")
                    logging.error(f"Error training {model_name}: {str(e)}")
                    continue
            
            return model_report, trained_models
        
        except Exception as e:
            raise CustomException(e, sys)

    

