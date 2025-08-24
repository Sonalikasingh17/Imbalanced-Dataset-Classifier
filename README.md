# Imbalanced Dataset Classifier - APS Failure Prediction

<<<<<<< HEAD
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen?logo=streamlit)]()
=======
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen?logo=streamlit)](https://sonalikasingh17-imbalanced-dataset-classif-streamlit-app-rzrtxi.streamlit.app/)
>>>>>>> cbd528fcf1bc76201c71d5f24bbddf6258ca3a38
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit_learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

--- 

## 🚛 Project Overview

This is a comprehensive end-to-end machine learning project that predicts Air Pressure System (APS) failures in Scania trucks. The project tackles the challenging problem of severe class imbalance (1:59 ratio) using advanced machine learning techniques and provides a complete MLOps pipeline with Streamlit deployment.

---

<<<<<<< HEAD
## 🎥 Demo

<div style="display: flex; flex-wrap: wrap; gap: 8px;">
  <img src="https://github.com/user-attachments/assets/faf7e739-85bd-4058-be74-f695e84727a1" alt="Screenshot 2025-08-22 005546" width="360"/>
  <img src="https://github.com/user-attachments/assets/3e3b8686-c3b9-46dd-95cc-561e4c67c101" alt="Screenshot 2025-08-22 005601" width="360"/>
</div>

---

=======
>>>>>>> cbd528fcf1bc76201c71d5f24bbddf6258ca3a38
## 🎯 Problem Statement

The dataset consists of data collected from heavy Scania trucks in everyday usage. The system in focus is the Air Pressure system (APS) which generates pressurized air utilized in various functions such as braking and gear changes. The challenge is to predict whether a failure is related to the APS system or other components, with the added complexity of severe class imbalance.

**Key Challenges:**
- Severe class imbalance (1:59 positive to negative ratio)
- 170 anonymous features
- 60,000 data points
- High cost of misclassification

---

## 📊 Dataset Information

- **Total Samples**: 60,000
- **Features**: 170 (anonymized sensor readings)  
- **Target Classes**: 
  - `pos`: APS-related failure (minority class)
  - `neg`: Non-APS related failure (majority class)
- **Class Ratio**: 1:59 (severe imbalance)

---

## 🏗️ Project Architecture

```
Imbalanced-Dataset-Classifier/
├── README.md
├── requirements.txt
├── setup.py
├── streamlit_app.py          # Main Streamlit application
├── .gitignore
├── artifacts/                # Model artifacts and processed data
│   ├── data_ingestion/
│   ├── data_transformation/
│   └── model_trainer/
├── logs/                     # Application logs
├── notebooks/                # Jupyter notebooks for EDA
├── src/                      # Source code
│   ├── __init__.py
│   ├── exception.py          # Custom exception handling
│   ├── logger.py             # Logging configuration
│   ├── utils.py              # Utility functions
│   ├── components/           # ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   └── pipeline/             # Training and prediction pipelines
│       ├── predict_pipeline.py
│       └── train_pipeline.py

```

---

## 🔧 Class Imbalance Handling Techniques

The project implements multiple strategies to handle severe class imbalance:

---

### 1. **Resampling Techniques**
- **SMOTE (Synthetic Minority Oversampling Technique)**: Generates synthetic samples for minority class
- **Random Undersampling**: Reduces majority class samples
- **Combined approach**: Balances both over and undersampling

### 2. **Algorithm-Level Approaches**
- **Class Weights**: Assigns higher weights to minority class (`class_weight='balanced'`)
- **Sample Weights**: Custom weights for individual samples based on class distribution
- **Cost-sensitive learning**: Adjusts algorithm parameters to handle imbalance

### 3. **Ensemble Methods**
- **Random Forest with balanced sampling**
- **Multiple algorithm comparison** with imbalance handling

### 4. **Evaluation Metrics**
- **Macro-averaged F1 Score**: Primary metric (handles imbalance better than accuracy)
- **Precision and Recall** for both classes
- **Classification Report** with detailed metrics

---

## 🚀 Key Features

- **End-to-End MLOps Pipeline**: Complete automation from data ingestion to model deployment
- **Advanced Class Imbalance Handling**: Multiple techniques including SMOTE, class weights, and ensemble methods
- **Comprehensive Model Evaluation**: Uses appropriate metrics for imbalanced data
- **Interactive Streamlit App**: User-friendly web interface for predictions
- **Modular Architecture**: Clean, maintainable code structure
- **Extensive Logging**: Complete tracking of pipeline execution
- **Error Handling**: Robust exception handling throughout the pipeline

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Imbalanced-Dataset-Classifier.git
cd Imbalanced-Dataset-Classifier
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Project in Development Mode
```bash
pip install -e .
```
---

## 🏃‍♂️ Usage

### Option 1: Run Complete Pipeline with Streamlit App
```bash
streamlit run streamlit_app.py
```

### Option 2: Train Model Separately
```bash
python -m src.pipeline.train_pipeline
```
---

## 🎯 Model Performance

The project evaluates multiple algorithms with class imbalance handling:

| Model | Technique | F1-Score (Macro) |
|-------|-----------|------------------|
<<<<<<< HEAD
| Logistic Regression | Baseline | ~0.45 |
| Logistic Regression | Balanced Weights | ~0.52 |
| Decision Tree | SMOTE | ~0.58 |
| Random Forest | Balanced + SMOTE | ~0.62 |
| SVM | Class Weights | ~0.48 |
=======
| Logistic Regression | Baseline | ~0.845 |
| Logistic Regression | Balanced Weights | ~0.750 |
| Decision Tree | SMOTE | ~0.848 |
| Random Forest | Balanced + SMOTE | ~0.890 |
| SVM | Class Weights | ~0.784 |
>>>>>>> cbd528fcf1bc76201c71d5f24bbddf6258ca3a38

---

## 🌐 Streamlit Application Features

The web application provides:

- **Interactive Prediction Interface**: Input sensor values manually
- **Batch Prediction**: Upload CSV files for multiple predictions
- **Model Performance Dashboard**: View detailed performance metrics
- **Real-time Training**: Retrain models through the web interface
- **Visualization**: Performance charts and class distribution plots

## 📈 Technical Highlights

### Data Preprocessing
- **Missing Value Imputation**: Median strategy for robust handling
- **Feature Scaling**: StandardScaler for algorithm compatibility
- **Data Validation**: Comprehensive checks for data quality

### Model Selection
<<<<<<< HEAD
- **Grid Search with Cross-Validation**: Automated hyperparameter tuning
- **Stratified Sampling**: Maintains class distribution in train/test splits
- **Multiple Algorithm Comparison**: Systematic evaluation of different approaches
=======
- **Bayesian Hyperparameter Optimization**: Replaces grid search for faster and more efficient tuning with fewer iterations while maintaining model quality.
- **Stratified Sampling**: Maintains class distribution in train/test splits to handle severe class imbalance.
- **Multiple Algorithm Comparison**: Systematic evaluation with balanced and sampling-based approaches.
>>>>>>> cbd528fcf1bc76201c71d5f24bbddf6258ca3a38

### MLOps Best Practices
- **Artifact Management**: Organized storage of models and preprocessors
- **Logging**: Comprehensive logging throughout the pipeline
- **Configuration Management**: Centralized configuration using dataclasses
- **Error Handling**: Custom exceptions with detailed error messages

## 🔍 Model Interpretability

- **Feature Importance**: Random Forest feature importance scores
- **Classification Reports**: Detailed precision/recall analysis
- **Confusion Matrix**: Visual representation of model performance
- **Class-wise Metrics**: Separate evaluation for each class

---

## 🚀 Deployment Options

### Local Deployment
```bash
streamlit run streamlit_app.py
```
### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy directly from repository

<<<<<<< HEAD
### Docker Deployment
```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

=======
>>>>>>> cbd528fcf1bc76201c71d5f24bbddf6258ca3a38
---


## 📊 Performance Monitoring

The system tracks:
- **Model Performance Metrics**: F1-score, precision, recall
- **Data Drift**: Monitoring for changes in input distribution
- **Prediction Confidence**: Model uncertainty quantification
  
---

## 👤 Author

**Sonalika Singh**
- GitHub: [Sonalikasingh17](https://github.com/Sonalikasingh17)
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/sonalika-singh-994a151a8/)
- Email: singhsonalika5@gmail.com

---

**Made with ❤️ and Python**
