# 💳 Credit Card Fraud Detection

A Machine Learning based web application that detects potentially fraudulent credit card transactions using a Random Forest classification model.

## 🎯 Project Objective

The main objective of this project is to identify fraudulent credit card transactions using Machine Learning.

Since fraudulent transactions are much fewer than normal transactions, the dataset is highly imbalanced. SMOTE is used during model training to handle this class imbalance.

## 🤖 Machine Learning Model

### Random Forest

Random Forest is used as the primary classification algorithm.

The model combines multiple decision trees to improve prediction performance and reduce overfitting.

### SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) is used to generate synthetic samples for the minority fraud class during training.

## 📊 Dataset

The application analyzes credit card transaction data containing:

- Time
- V1 to V28
- Amount
- Class

Where:

- `0` = Normal Transaction
- `1` = Fraudulent Transaction

## 📈 Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 99.94% |
| Precision | 82.00% |
| Recall | 83.67% |
| F1 Score | 82.83% |
| ROC-AUC | 0.9638 |

## 🚀 Application Features

- 🔐 Admin Login
- 🏠 Home Page
- 📊 Dataset Dashboard
- 📈 Transaction Distribution
- 🎯 Model Performance Analysis
- 📈 ROC-AUC Curve
- 🔲 Confusion Matrix
- 🤖 Model Comparison
- 🔍 Feature Importance
- 💳 Dataset Transaction Prediction
- 🧾 Manual Transaction Prediction
- 📂 CSV Batch Prediction
- ⬇️ Download Prediction Results

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- SMOTE
- Streamlit
- Matplotlib
- Joblib

## 📁 Project Structure

```text
credit-card-fraud-detection/
│
├── app.py
├── data.py
├── model.py
├── fraud_model.pkl
├── auc_score.pkl
├── roc_data.pkl
├── model_comparison.pkl
├── requirements.txt
└── README.md
