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

## Author
Sonalika Singh

IIT Madras
