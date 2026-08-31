import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE


# =========================================================
# 1. LOAD DATASET
# =========================================================

data = pd.read_csv("creditcard.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# =========================================================
# 2. SEPARATE FEATURES AND TARGET
# =========================================================

X = data.drop("Class", axis=1)
y = data["Class"]


# =========================================================
# 3. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# =========================================================
# 4. BEFORE SMOTE
# =========================================================

print("\nBefore SMOTE:")
print(y_train.value_counts())


# =========================================================
# 5. APPLY SMOTE
# =========================================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())


# =========================================================
# 6. RANDOM FOREST MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(
    X_train_smote,
    y_train_smote
)

print("Random Forest training completed!")


# =========================================================
# 7. RANDOM FOREST PREDICTION
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# 8. RANDOM FOREST EVALUATION
# =========================================================

print("\n================================")
print("RANDOM FOREST CLASSIFICATION")
print("================================")

print(
    classification_report(
        y_test,
        y_pred
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion Matrix:")
print(cm)


# =========================================================
# 9. SAVE RANDOM FOREST MODEL
# =========================================================

joblib.dump(
    model,
    "fraud_model.pkl"
)

print("\nRandom Forest model saved successfully!")


# =========================================================
# 10. ROC-AUC
# =========================================================

y_probability = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

auc_score = roc_auc_score(
    y_test,
    y_probability
)

roc_data = pd.DataFrame({
    "FPR": fpr,
    "TPR": tpr
})

joblib.dump(
    roc_data,
    "roc_data.pkl"
)

joblib.dump(
    auc_score,
    "auc_score.pkl"
)

print(
    "ROC-AUC Score:",
    round(auc_score, 4)
)

print("ROC data saved successfully!")


# =========================================================
# 11. LOGISTIC REGRESSION
# =========================================================

print("\n================================")
print("LOGISTIC REGRESSION")
print("================================")

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

print("Training Logistic Regression...")

lr_model.fit(
    X_train,
    y_train
)

lr_pred = lr_model.predict(
    X_test
)


# =========================================================
# 12. LOGISTIC REGRESSION METRICS
# =========================================================

lr_accuracy = accuracy_score(
    y_test,
    lr_pred
)

lr_precision = precision_score(
    y_test,
    lr_pred
)

lr_recall = recall_score(
    y_test,
    lr_pred
)

lr_f1 = f1_score(
    y_test,
    lr_pred
)


# =========================================================
# 13. RANDOM FOREST METRICS
# =========================================================

rf_accuracy = accuracy_score(
    y_test,
    y_pred
)

rf_precision = precision_score(
    y_test,
    y_pred
)

rf_recall = recall_score(
    y_test,
    y_pred
)

rf_f1 = f1_score(
    y_test,
    y_pred
)


# =========================================================
# 14. MODEL COMPARISON
# =========================================================

print("\n================================")
print("MODEL COMPARISON")
print("================================")

print("\nLogistic Regression")

print(
    "Accuracy :",
    round(lr_accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(lr_precision * 100, 2),
    "%"
)

print(
    "Recall   :",
    round(lr_recall * 100, 2),
    "%"
)

print(
    "F1 Score :",
    round(lr_f1 * 100, 2),
    "%"
)


print("\nRandom Forest")

print(
    "Accuracy :",
    round(rf_accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(rf_precision * 100, 2),
    "%"
)

print(
    "Recall   :",
    round(rf_recall * 100, 2),
    "%"
)

print(
    "F1 Score :",
    round(rf_f1 * 100, 2),
    "%"
)


# =========================================================
# 15. SAVE MODEL COMPARISON
# =========================================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "Accuracy": [
        lr_accuracy * 100,
        rf_accuracy * 100
    ],

    "Precision": [
        lr_precision * 100,
        rf_precision * 100
    ],

    "Recall": [
        lr_recall * 100,
        rf_recall * 100
    ],

    "F1 Score": [
        lr_f1 * 100,
        rf_f1 * 100
    ]
})

joblib.dump(
    comparison,
    "model_comparison.pkl"
)

print("\nModel comparison saved successfully!")

print("\n================================")
print("ALL PROCESS COMPLETED!")
print("================================")