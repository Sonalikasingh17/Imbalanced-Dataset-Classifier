import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    model_report_file_path = os.path.join("artifacts", "model_report.txt")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define models with class imbalance handling
            models = {
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Logistic Regression (Balanced)": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Decision Tree (Balanced)": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
                "SVM": SVC(random_state=42),
                "SVM (Balanced)": SVC(random_state=42, class_weight='balanced'),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Random Forest (Balanced)": RandomForestClassifier(random_state=42, class_weight='balanced')
            }

            params = {
                "Logistic Regression": {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                "Logistic Regression (Balanced)": {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                "Decision Tree": {
                    'max_depth': [10, 15, 20, 25],
                    'max_leaf_nodes': [10, 20, 30]
                },
                "Decision Tree (Balanced)": {
                    'max_depth': [10, 15, 20, 25],
                    'max_leaf_nodes': [10, 20, 30]
                },
                "SVM": {
                    'C': [1, 10],
                    'kernel': ['rbf', 'poly']
                },
                "SVM (Balanced)": {
                    'C': [1, 10],
                    'kernel': ['rbf', 'poly']
                },
                "Random Forest": {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 15, 20]
                },
                "Random Forest (Balanced)": {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 15, 20]
                }
            }

            # Evaluate baseline models
            logging.info("Evaluating baseline models")
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, 
                                               X_test=X_test, y_test=y_test,
                                               models=models, param=params)

            # Apply SMOTE oversampling
            logging.info("Applying SMOTE oversampling")
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

            # Evaluate models with SMOTE
            smote_models = {
                "Logistic Regression (SMOTE)": LogisticRegression(random_state=42, max_iter=1000),
                "Decision Tree (SMOTE)": DecisionTreeClassifier(random_state=42),
                "SVM (SMOTE)": SVC(random_state=42),
                "Random Forest (SMOTE)": RandomForestClassifier(random_state=42)
            }

            smote_params = {
                "Logistic Regression (SMOTE)": {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                "Decision Tree (SMOTE)": {
                    'max_depth': [10, 15, 20, 25],
                    'max_leaf_nodes': [10, 20, 30]
                },
                "SVM (SMOTE)": {
                    'C': [1, 10],
                    'kernel': ['rbf', 'poly']
                },
                "Random Forest (SMOTE)": {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 15, 20]
                }
            }

            smote_report: dict = evaluate_models(X_train=X_train_smote, y_train=y_train_smote, 
                                               X_test=X_test, y_test=y_test,
                                               models=smote_models, param=smote_params)

            # Combine reports
            all_models_report = {**model_report, **smote_report}

            # Get best model score
            best_model_score = max(sorted(all_models_report.values()))

            # Get best model name
            best_model_name = list(all_models_report.keys())[
                list(all_models_report.values()).index(best_model_score)
            ]

            if best_model_name in models:
                best_model = models[best_model_name]
                # Train on original data
                best_model.fit(X_train, y_train)
            else:
                best_model = smote_models[best_model_name]
                # Train on SMOTE data
                best_model.fit(X_train_smote, y_train_smote)

            if best_model_score < 0.4:
                raise CustomException("No best model found with F1 score > 0.4")

            logging.info(f"Best found model: {best_model_name} with F1 score: {best_model_score}")

            # Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Generate detailed report
            y_pred = best_model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='macro')
            
            # Save detailed report
            with open(self.model_trainer_config.model_report_file_path, 'w') as f:
                f.write("=== MODEL TRAINING REPORT ===\n\n")
                f.write(f"Best Model: {best_model_name}\n")
                f.write(f"Best Model F1 Score: {best_model_score:.4f}\n\n")
                f.write("All Models Performance:\n")
                for model_name, score in all_models_report.items():
                    f.write(f"{model_name}: {score:.4f}\n")
                f.write("\nDetailed Classification Report:\n")
                f.write(classification_report(y_test, y_pred))

            return f1
            
        except Exception as e:
            raise CustomException(e, sys)
