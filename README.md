# Class-Imbalance-Handling-(IDA2016-Dataset)

This project implements a binary classification model using logistic regression to predict outcomes based on a structured dataset. The notebook and script include steps for data preprocessing, model training, evaluation, and visualization. 
  
---   

## Dataset     
  
- The dataset is loaded from a `.csv` file using `pandas`.
- It contains numeric features and a binary target variable (0 or 1).
- Common tasks include: 
  - Checking for null values
  - Exploratory data analysis
  - Feature scaling or normalization

---

## Workflow Summary

1. **Data Loading**: Read the dataset using `pandas`.
2. **Preprocessing**:
   - Handle missing values
   - Scale the features using `StandardScaler` or `MinMaxScaler`
   - Encode categorical variables (if any)
3. **Splitting**:
   - Split the dataset using `train_test_split` into training and testing sets.
4. **Model Training**:
   - Train a logistic regression model using `sklearn`.
5. **Evaluation**:
   - Assess the model using accuracy, confusion matrix, and F1-score.
6. **Visualization**:
   - Plot the confusion matrix using `seaborn` heatmaps.

---

##  Model Architecture

| Layer         | Description                           |
|---------------|---------------------------------------|
| Input Layer   | Number of features in the dataset     |
| Output Layer  | Single neuron with sigmoid activation |
| Model Type    | Logistic Regression                   |

---

##  Evaluation Metrics

- **Accuracy Score** – Measures overall correctness
- **Confusion Matrix** – Breaks down predictions into TP/FP/TN/FN
- **Classification Report** – Includes:
  - **Precision**: Correctness of positive predictions
  - **Recall**: Coverage of actual positives
  - **F1-Score**: Harmonic mean of precision and recall

---

##  Key Learnings

- Gained hands-on experience in data preprocessing, feature scaling, and train-test splitting.
- Understood how logistic regression behaves on binary classification tasks.
- Learned how to evaluate model performance using precision, recall, F1-score, and confusion matrix.
- Learned to visualize results effectively with `matplotlib` and `seaborn`.
- Reinforced the importance of model interpretability in classification problems.

---

##  Requirements

Install the following Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

# Imbalanced Dataset Classifier - APS Failure Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit_learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## 🚛 Project Overview

This is a comprehensive end-to-end machine learning project that predicts Air Pressure System (APS) failures in Scania trucks. The project tackles the challenging problem of severe class imbalance (1:59 ratio) using advanced machine learning techniques and provides a complete MLOps pipeline with Streamlit deployment.

## 🎯 Problem Statement

The dataset consists of data collected from heavy Scania trucks in everyday usage. The system in focus is the Air Pressure system (APS) which generates pressurized air utilized in various functions such as braking and gear changes. The challenge is to predict whether a failure is related to the APS system or other components, with the added complexity of severe class imbalance.

**Key Challenges:**
- Severe class imbalance (1:59 positive to negative ratio)
- 170 anonymous features
- 60,000 data points
- High cost of misclassification

## 📊 Dataset Information

- **Total Samples**: 60,000
- **Features**: 170 (anonymized sensor readings)  
- **Target Classes**: 
  - `pos`: APS-related failure (minority class)
  - `neg`: Non-APS related failure (majority class)
- **Class Ratio**: 1:59 (severe imbalance)

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
└── tests/                    # Unit and integration tests
```

## 🔧 Class Imbalance Handling Techniques

The project implements multiple strategies to handle severe class imbalance:

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

## 🚀 Key Features

- **End-to-End MLOps Pipeline**: Complete automation from data ingestion to model deployment
- **Advanced Class Imbalance Handling**: Multiple techniques including SMOTE, class weights, and ensemble methods
- **Comprehensive Model Evaluation**: Uses appropriate metrics for imbalanced data
- **Interactive Streamlit App**: User-friendly web interface for predictions
- **Modular Architecture**: Clean, maintainable code structure
- **Extensive Logging**: Complete tracking of pipeline execution
- **Error Handling**: Robust exception handling throughout the pipeline

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Imbalanced-Dataset-Classifier.git
cd Imbalanced-Dataset-Classifier
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Project in Development Mode
```bash
pip install -e .
```

## 🏃‍♂️ Usage

### Option 1: Run Complete Pipeline with Streamlit App
```bash
streamlit run streamlit_app.py
```

### Option 2: Train Model Separately
```bash
python -m src.pipeline.train_pipeline
```

### Option 3: Make Predictions
```python
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Create prediction pipeline
predict_pipeline = PredictPipeline()

# Create custom data (replace with actual sensor values)
data = CustomData(
    aa_000=100.0,
    ac_000=50.0,
    ad_000=25.0,
    # ... add all 170 features
)

# Make prediction
df = data.get_data_as_data_frame()
prediction = predict_pipeline.predict(df)
print(f"Prediction: {'APS Failure' if prediction[0] == 1 else 'Non-APS Failure'}")
```

## 🎯 Model Performance

The project evaluates multiple algorithms with class imbalance handling:

| Model | Technique | F1-Score (Macro) |
|-------|-----------|------------------|
| Logistic Regression | Baseline | ~0.45 |
| Logistic Regression | Balanced Weights | ~0.52 |
| Decision Tree | SMOTE | ~0.58 |
| Random Forest | Balanced + SMOTE | ~0.62 |
| SVM | Class Weights | ~0.48 |

*Note: Actual scores may vary based on data and hyperparameters*

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
- **Grid Search with Cross-Validation**: Automated hyperparameter tuning
- **Stratified Sampling**: Maintains class distribution in train/test splits
- **Multiple Algorithm Comparison**: Systematic evaluation of different approaches

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

## 🚀 Deployment Options

### Local Deployment
```bash
streamlit run streamlit_app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy directly from repository

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

## 🧪 Testing

Run tests using:
```bash
python -m pytest tests/
```

## 📊 Performance Monitoring

The system tracks:
- **Model Performance Metrics**: F1-score, precision, recall
- **Data Drift**: Monitoring for changes in input distribution
- **Prediction Confidence**: Model uncertainty quantification

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Scania for providing the APS failure dataset
- IDA 2016 Challenge for the problem formulation
- Open source community for the amazing tools and libraries

---

## 📞 Support

If you have any questions or issues, please:
1. Check the [Issues](https://github.com/yourusername/Imbalanced-Dataset-Classifier/issues) page
2. Create a new issue if your problem isn't addressed
3. Contact the maintainer directly

**Made with ❤️ and Python**
