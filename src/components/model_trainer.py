import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models_bayesian

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "models", "best_model.pkl")
    model_report_file_path = os.path.join("artifacts", "model_trainer", "training_report.txt")
    model_comparison_file_path = os.path.join("artifacts", "evaluations", "model_comparison.csv")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        # Create necessary directories
        os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.model_trainer_config.model_report_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.model_trainer_config.model_comparison_file_path), exist_ok=True)


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            print("🎯 BASELINE MODEL CONFIGURATION")
            print("="*50)

            # Define baseline models
            baseline_models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42, probability=True),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1)
            }

            # Define class-weighted models
            weighted_models = {
                'Logistic Regression (Balanced)': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
                'SVM (Balanced)': SVC(random_state=42, probability=True, class_weight='balanced'),
                'Decision Tree (Balanced)': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
                'Random Forest (Balanced)': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
            }

            # Define hyperparameters for Bayesian search
            bayes_search_spaces = {
                'Logistic Regression': {
                    'C': Real(1e-3, 1e+3, prior='log-uniform'),
                    'penalty': Categorical(['l1', 'l2']),
                    'solver': Categorical(['liblinear'])
                },
                'Logistic Regression (Balanced)': {
                    'C': Real(1e-3, 1e+3, prior='log-uniform'),
                    'penalty': Categorical(['l1', 'l2']),
                    'solver': Categorical(['liblinear'])
                },
                'SVM': {
                    'C': Real(1, 100, prior='log-uniform'),
                    'kernel': Categorical(['rbf', 'poly']),
                    'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
                    'degree': Integer(2, 5)
                },
                'SVM (Balanced)': {
                    'C': Real(1, 100, prior='log-uniform'),
                    'kernel': Categorical(['rbf', 'poly']),
                    'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
                    'degree': Integer(2, 5)
                },
                'Decision Tree': {
                    'max_depth': Integer(5, 25),
                    'min_samples_split': Integer(2, 10),
                    'min_samples_leaf': Integer(1, 4)
                },
                'Decision Tree (Balanced)': {
                    'max_depth': Integer(5, 25),
                    'min_samples_split': Integer(2, 10),
                    'min_samples_leaf': Integer(1, 4)
                },
                'Random Forest': {
                    'n_estimators': Integer(50, 100),
                    'max_depth': Integer(10, 20),
                    'min_samples_split': Integer(2, 5)
                },
                'Random Forest (Balanced)': {
                    'n_estimators': Integer(50, 100),
                    'max_depth': Integer(10, 20),
                    'min_samples_split': Integer(2, 5)
                }
            }

            print(f"📊 Baseline models configured: {list(baseline_models.keys())}")
            print(f"🔧 Hyperparameter tuning enabled for all models")
            print(f"📈 Evaluation metric: F1-score (macro average)")

            # Evaluate baseline models
            print("\n🚀 TRAINING BASELINE MODELS")
            print("="*50)
            logging.info("Evaluating baseline models with Bayesian optimization")
            baseline_report, baseline_trained_models = evaluate_models_bayesian(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=baseline_models, search_spaces=bayes_search_spaces
            )

            # Evaluate class-weighted models
            print("\n🚀 TRAINING CLASS-WEIGHTED MODELS")
            print("="*50)
            logging.info("Evaluating class-weighted models with Bayesian optimization")
            weighted_report, weighted_trained_models = self.evaluate_models_bayesian(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=weighted_models, search_spaces=bayes_search_spaces
            )

            # Apply SMOTE oversampling
            print("\n🚀 TRAINING MODELS WITH SMOTE")
            print("="*50)
            logging.info("Applying SMOTE oversampling")
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

            print(f"📊 Original training set: {len(X_train)} samples")
            print(f"📊 SMOTE training set: {len(X_train_smote)} samples")

            # Evaluate models with SMOTE
            smote_models = {
                "Logistic Regression (SMOTE)": LogisticRegression(random_state=42, max_iter=1000),
                "Decision Tree (SMOTE)": DecisionTreeClassifier(random_state=42),
                "SVM (SMOTE)": SVC(random_state=42, probability=True),
                "Random Forest (SMOTE)": RandomForestClassifier(random_state=42, n_jobs=-1)
            }

            smote_search_spaces = {
                "Logistic Regression (SMOTE)": bayes_search_spaces['Logistic Regression'],
                "Decision Tree (SMOTE)": bayes_search_spaces['Decision Tree'],
                "SVM (SMOTE)": bayes_search_spaces['SVM'],
                "Random Forest (SMOTE)": bayes_search_spaces['Random Forest']
            }

            smote_report, smote_trained_models = evaluate_models_bayesian(
                X_train=X_train_smote, y_train=y_train_smote, X_test=X_test, y_test=y_test,
                models=smote_models, search_spaces=smote_search_spaces
            )

            # Combine all results
            all_models_report = {**baseline_report, **weighted_report, **smote_report}
            all_trained_models = {**baseline_trained_models, **weighted_trained_models, **smote_trained_models}

            if not all_models_report:
                raise CustomException("No models were successfully trained")

            # Get best model score
            best_model_score = max(all_models_report.values())

            # Get best model name
            best_model_name = max(all_models_report, key=all_models_report.get)

            if best_model_score < 0.4:
                raise CustomException("No best model found with F1 score > 0.4")

            logging.info(f"Best found model: {best_model_name} with F1 score: {best_model_score}")

            # Get the best model object
            best_model = all_trained_models[best_model_name]

            # If it's a SMOTE model, retrain on SMOTE data
            if "SMOTE" in best_model_name:
                best_model.fit(X_train_smote, y_train_smote)
            else:
                best_model.fit(X_train, y_train)

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            print(f"\n🏆 BEST MODEL SELECTION")
            print("="*50)
            print(f"🥇 Selected Best Model: {best_model_name}")
            print(f"🎯 F1-Score: {best_model_score:.4f}")

            # Generate detailed report
            y_pred = best_model.predict(X_test)
            final_f1 = f1_score(y_test, y_pred, average='macro')
            final_precision = precision_score(y_test, y_pred, average='macro')
            final_recall = recall_score(y_test, y_pred, average='macro')
            
            # Save detailed report
            with open(self.model_trainer_config.model_report_file_path, 'w') as f:
                f.write("==========================================\n")
                f.write("🎯 FINAL MODEL TRAINING REPORT\n")
                f.write("==========================================\n\n")
                f.write(f"📊 Dataset Information:\n")
                f.write(f"• Training samples: {len(X_train):,}\n")
                f.write(f"• Testing samples: {len(X_test):,}\n")
                f.write(f"• Features: {X_train.shape[1]}\n\n")
                f.write(f"🏆 Best Model Performance:\n")
                f.write(f"• Model: {best_model_name}\n")
                f.write(f"• F1-Score (Macro): {final_f1:.4f}\n")
                f.write(f"• Precision (Macro): {final_precision:.4f}\n")
                f.write(f"• Recall (Macro): {final_recall:.4f}\n\n")
                f.write(f"📈 All Models Performance:\n")
                for model_name, score in sorted(all_models_report.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"• {model_name}: {score:.4f}\n")
                f.write(f"\n📋 Detailed Classification Report:\n")
                f.write(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

            # Save model comparison CSV
            import pandas as pd
            comparison_data = []
            for model_name, score in all_models_report.items():
                y_pred_temp = all_trained_models[model_name].predict(X_test)
                comparison_data.append({
                    'Model': model_name,
                    'Test_F1': score,
                    'Test_Precision': precision_score(y_test, y_pred_temp, average='macro'),
                    'Test_Recall': recall_score(y_test, y_pred_temp, average='macro')
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Test_F1', ascending=False)
            comparison_df.to_csv(self.model_trainer_config.model_comparison_file_path, index=False)

            print(f"💾 Files saved:")
            print(f"  • Best model: {self.model_trainer_config.trained_model_file_path}")
            print(f"  • Training report: {self.model_trainer_config.model_report_file_path}")
            print(f"  • Model comparison: {self.model_trainer_config.model_comparison_file_path}")

            return final_f1
            
        except Exception as e:
            raise CustomException(e, sys)
