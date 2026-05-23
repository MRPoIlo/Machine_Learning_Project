"""train_lda.py - Train a Linear Discriminant Analysis model on the
rural drinking-water quality dataset.

Run this script ONCE after preprocessing to generate the LDA model.

Usage (from the project root):
    python services/train_lda.py

Output:
    models/lda_model.pkl                   - trained LinearDiscriminantAnalysis model
    models/lda_scaler.pkl                  - StandardScaler used during training
    reports/lda_confusion_matrix.png       - confusion matrix heatmap
    reports/lda_roc_curve.png              - ROC curve plot
    reports/lda_discriminant_plot.png      - LDA projection plot (LD1 distribution)
    reports/lda_metrics.txt                - classification metrics
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

MODEL_PATH        = MODELS_DIR  / "lda_model.pkl"
SCALER_PATH       = MODELS_DIR  / "lda_scaler.pkl"
CONF_MATRIX_PLOT  = REPORTS_DIR / "lda_confusion_matrix.png"
ROC_CURVE_PLOT    = REPORTS_DIR / "lda_roc_curve.png"
DISCRIMINANT_PLOT = REPORTS_DIR / "lda_discriminant_plot.png"
METRICS_FILE      = REPORTS_DIR / "lda_metrics.txt"

# Risk levels considered "unsafe" for the binary target.
HIGH_RISK_LEVELS = ["High Risk", "Sanitaryly not feasible"]
TARGET_NAMES = ["No high risk", "High risk"]


# 2. Load the dataset
data = pd.read_csv(CSV_PATH, sep=";", thousands=",", encoding="utf-8")

# 3. Explore the dataset
print(data.head())
print(data.info())
print(data.describe())

# 4. Build the binary target: 1 if water is unsafe, 0 otherwise
data["target"] = data["Nivel de riesgo rural"].isin(HIGH_RISK_LEVELS).astype(int)

# Build the feature matrix (same strategy as Logistic Regression for fair comparison):
#   - DepartamentoCodigo as one-hot dummies
#   - Año kept numeric (temporal trend 2021-2024)
#   - IRCArural EXCLUDED (target leakage)
#   - MunicipioCodigo DROPPED (508 unique values)
department_dummies = pd.get_dummies(
    data["DepartamentoCodigo"].astype(str), prefix="dept", dtype=int
)
X = pd.concat([department_dummies, data[["Año"]]], axis=1)
y = data["target"]

FEATURE_COLS = X.columns.tolist()

# 5. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 7. Train the LDA model
# solver='svd' is the default and does not compute the covariance matrix
# explicitly — numerically stable for high-dimensional feature matrices.
# priors=None lets sklearn estimate class priors from the training data.
lda_model = LinearDiscriminantAnalysis(solver="svd")
lda_model.fit(X_train_scaled, y_train)

# 8. Predict on the test set
y_pred  = lda_model.predict(X_test_scaled)
y_proba = lda_model.predict_proba(X_test_scaled)[:, 1]

# 9. Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)

# Confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
    xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Linear Discriminant Analysis")
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
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"LDA  AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Linear Discriminant Analysis")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(ROC_CURVE_PLOT, dpi=150)
plt.close()
print(f"ROC curve saved to {ROC_CURVE_PLOT}")

# LD1 projection plot — shows how well the single discriminant axis
# separates the two classes
X_train_ld = lda_model.transform(X_train_scaled)
X_test_ld  = lda_model.transform(X_test_scaled)

plt.figure(figsize=(10, 5))
for label, name, color in [(0, "No high risk", "steelblue"), (1, "High risk", "tomato")]:
    mask = y_test.values == label
    plt.hist(
        X_test_ld[mask, 0], bins=30, alpha=0.6,
        label=name, color=color, edgecolor="white",
    )
plt.axvline(0, color="black", linestyle="--", linewidth=1, label="Decision boundary")
plt.xlabel("Linear Discriminant 1 (LD1)")
plt.ylabel("Count")
plt.title("LDA Projection — Class Separation on LD1 (test set)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(DISCRIMINANT_PLOT, dpi=150)
plt.close()
print(f"Discriminant plot saved to {DISCRIMINANT_PLOT}")

# 10. Persist artifacts
joblib.dump(lda_model, MODEL_PATH)
joblib.dump(scaler,    SCALER_PATH)

METRICS_FILE.write_text(
    f"accuracy: {accuracy:.4f}\n"
    f"roc_auc: {roc_auc:.4f}\n"
    f"positive_class: high_risk (High Risk + Sanitaryly not feasible)\n"
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