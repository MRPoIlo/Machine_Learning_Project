"""train_logistic_regression.py - Train a Logistic Regression model on the
rural drinking-water quality dataset.

Run this script ONCE after preprocessing to generate the supervised model.

Usage (from the project root):
    python services/train_logistic_regression.py

Output:
    models/logistic_model.pkl              - trained LogisticRegression model
    models/logistic_scaler.pkl             - StandardScaler used during training
    reports/logistic_confusion_matrix.png  - confusion matrix heatmap
    reports/logistic_roc_curve.png         - ROC curve plot
    reports/logistic_metrics.txt           - classification metrics
"""

# 1. Import required libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_curve,
    roc_auc_score,
)


ROOT        = Path(__file__).resolve().parent.parent
CSV_PATH    = ROOT / "data"    / "water_quality_for_human_consumption.csv"
MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

MODEL_PATH        = MODELS_DIR  / "logistic_model.pkl"
SCALER_PATH       = MODELS_DIR  / "logistic_scaler.pkl"
CONF_MATRIX_PLOT  = REPORTS_DIR / "logistic_confusion_matrix.png"
ROC_CURVE_PLOT    = REPORTS_DIR / "logistic_roc_curve.png"
METRICS_FILE      = REPORTS_DIR / "logistic_metrics.txt"

# Risk levels considered "unsafe" for the binary target.
# The rural risk level is computed from IRCArural by fixed thresholds, so
# IRCArural itself is excluded from the features to avoid target leakage.
HIGH_RISK_LEVELS = ["Alto riesgo", "Inviable sanitariamente"]

TARGET_NAMES = ["No high risk", "High risk"]


# 2. Load the dataset
data = pd.read_csv(CSV_PATH, sep=";", thousands=",", encoding="utf-8")

# 3. Explore the dataset
print(data.head())
print(data.info())
print(data.describe())

# 4. Build the binary target: 1 if water is unsafe, 0 otherwise
data["target"] = data["Nivel de riesgo rural"].isin(HIGH_RISK_LEVELS).astype(int)

# Build the feature matrix:
#   - DepartamentoCodigo as one-hot dummies. The codes are categorical
#     identifiers, not continuous magnitudes, so they must be encoded.
#     We use the code (not the name) because the dataset contains two
#     spellings for the same department, but the code is stable.
#   - MunicipioCodigo is dropped: 508 unique values would explode the feature
#     space without adding generalisable signal.
#   - Año stays numeric (it captures the temporal trend across 2021-2024).
department_dummies = pd.get_dummies(
    data["DepartamentoCodigo"].astype(str), prefix="dept", dtype=int
)
X = pd.concat([department_dummies, data[["Año"]]], axis=1)
y = data["target"]

FEATURE_COLS = X.columns.tolist()

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the model
# class_weight="balanced" reweights the loss so the minority class (high risk)
# is not ignored - it is the class we actually want to detect.
logistic_model = LogisticRegression(
    max_iter=1000, random_state=42, class_weight="balanced"
)
logistic_model.fit(X_train_scaled, y_train)

# 6. Predict on the test set
y_pred = logistic_model.predict(X_test_scaled)
y_proba = logistic_model.predict_proba(X_test_scaled)[:, 1]

# 7. Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)

# Confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
    xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig(CONF_MATRIX_PLOT, dpi=150)
plt.close()
print(f"Confusion matrix saved to {CONF_MATRIX_PLOT}")

# Classification report
report = classification_report(y_test, y_pred, target_names=TARGET_NAMES)
print(report)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(ROC_CURVE_PLOT, dpi=150)
plt.close()
print(f"ROC curve saved to {ROC_CURVE_PLOT}")

# Persist artifacts
joblib.dump(logistic_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

METRICS_FILE.write_text(
    f"accuracy: {accuracy:.4f}\n"
    f"roc_auc: {roc_auc:.4f}\n"
    f"positive_class: high_risk (Alto riesgo + Inviable sanitariamente)\n"
    f"features: {', '.join(FEATURE_COLS)}\n"
    f"train_size: {len(X_train)}\n"
    f"test_size: {len(X_test)}\n"
    f"\n"
    f"classification_report:\n{report}\n",
    encoding="utf-8",
)

print(f"\nModel       -> {MODEL_PATH}")
print(f"Scaler      -> {SCALER_PATH}")
print(f"Metrics     -> {METRICS_FILE}")
