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
from src.logger import logging

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

def evaluate_models_bayesian(X_train, y_train, X_test, y_test, models, search_spaces):
    """
    Evaluate models using Bayesian hyperparameter optimization
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features  
        y_test: Test targets
        models: Dictionary of model objects
        search_spaces: Dictionary of search spaces for each model
        
    Returns:
        model_report: Dictionary of model names and their F1 scores
        trained_models: Dictionary of model names and their trained model objects
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
                
                logging.info(f"{model_name} - CV Score: {bayes_cv.best_score_:.4f}, Test F1: {test_f1:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error training {model_name}: {str(e)}")
                logging.error(f"Error training {model_name}: {str(e)}")
                continue
        
        return model_report, trained_models
        
    except Exception as e:
        logging.error(f"Error in evaluate_models_bayesian: {str(e)}")
        raise CustomException(e, sys)

def get_model_performance_summary(y_true, y_pred, model_name="Model"):
    """
    Generate comprehensive performance summary for a model
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        
    Returns:
        Dictionary containing various performance metrics
    """
    try:
        performance = {
            'model_name': model_name,
            'f1_score': f1_score(y_true, y_pred, average='macro'),
            'f1_score_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        }
        
        # Add class-specific metrics if binary classification
        unique_labels = np.unique(y_true)
        if len(unique_labels) == 2:
            f1_per_class = f1_score(y_true, y_pred, average=None)
            precision_per_class = precision_score(y_true, y_pred, average=None)
            recall_per_class = recall_score(y_true, y_pred, average=None)
            
            performance.update({
                'f1_class_0': f1_per_class[0],
                'f1_class_1': f1_per_class[1],
                'precision_class_0': precision_per_class[0],
                'precision_class_1': precision_per_class[1],
                'recall_class_0': recall_per_class[0],
                'recall_class_1': recall_per_class[1],
            })
        
        return performance
        
    except Exception as e:
        raise CustomException(e, sys)

def print_model_comparison(model_reports):
    """
    Print a formatted comparison of model performances
    
    Args:
        model_reports: Dictionary of model names and their performance scores
    """
    try:
        if not model_reports:
            print("⚠️ No model results to display")
            return
            
        print(f"\n🏆 MODEL PERFORMANCE COMPARISON")
        print("=" * 60)
        print(f"{'Model Name':<30} {'F1-Score':<10} {'Rank':<5}")
        print("-" * 60)
        
        # Sort models by F1 score (descending)
        sorted_models = sorted(model_reports.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (model_name, score) in enumerate(sorted_models, 1):
            rank_symbol = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
            print(f"{model_name:<30} {score:<10.4f} {rank_symbol}")
        
        print("-" * 60)
        best_model = sorted_models[0]
        print(f"🏆 Best Model: {best_model[0]} (F1: {best_model[1]:.4f})")
        
    except Exception as e:
        raise CustomException(e, sys)

def validate_data_quality(X_train, X_test, y_train, y_test):
    """
    Validate data quality and print summary statistics
    
    Args:
        X_train, X_test: Feature arrays
        y_train, y_test: Target arrays
    """
    try:
        print(f"\n🔍 DATA QUALITY VALIDATION")
        print("=" * 50)
        
        # Check for missing values
        if hasattr(X_train, 'isnull'):
            train_missing = X_train.isnull().sum().sum()
            test_missing = X_test.isnull().sum().sum()
        else:
            train_missing = np.isnan(X_train).sum()
            test_missing = np.isnan(X_test).sum()
        
        print(f"📊 Dataset Shapes:")
        print(f"  • Training: X{X_train.shape}, y{y_train.shape}")
        print(f"  • Testing: X{X_test.shape}, y{y_test.shape}")
        
        print(f"📊 Missing Values:")
        print(f"  • Training features: {train_missing}")
        print(f"  • Test features: {test_missing}")
        
        # Check class distribution
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        
        print(f"📊 Class Distribution:")
        print(f"  • Training: {dict(zip(unique_train, counts_train))}")
        print(f"  • Testing: {dict(zip(unique_test, counts_test))}")
        
        # Check for data leakage (same samples in train and test)
        if hasattr(X_train, 'values'):
            train_hash = pd.util.hash_pandas_object(X_train).values
            test_hash = pd.util.hash_pandas_object(X_test).values
        else:
            train_hash = [hash(tuple(row)) for row in X_train]
            test_hash = [hash(tuple(row)) for row in X_test]
        
        overlapping_samples = len(set(train_hash) & set(test_hash))
        
        if overlapping_samples > 0:
            print(f"⚠️ WARNING: {overlapping_samples} overlapping samples found between train and test sets!")
        else:
            print(f"✅ No data leakage detected")
        
        print("=" * 50)
        
    except Exception as e:
        logging.error(f"Error in data quality validation: {str(e)}")
        raise CustomException(e, sys)