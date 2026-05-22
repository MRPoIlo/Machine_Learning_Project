"""Logistic Regression service.

Loads the trained Logistic Regression model and exposes helpers to query
its metadata, metrics, and to run predictions. Keep all model-specific
logic here so routes stay thin.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"

CSV_PATH     = DATA_DIR    / "water_quality_rural.csv"
MODEL_PATH   = MODELS_DIR  / "logistic_model.pkl"
SCALER_PATH  = MODELS_DIR  / "logistic_scaler.pkl"
METRICS_PATH = REPORTS_DIR / "logistic_metrics.txt"

HIGH_RISK_LEVELS = ["Alto riesgo", "Inviable sanitariamente"]
TARGET_NAMES     = ["No high risk", "High risk"]


def _load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, sep=";", thousands=",", encoding="utf-8")
    df["Año"] = df["Año"].astype(int)
    df["target"] = df["Nivel de riesgo rural"].isin(HIGH_RISK_LEVELS).astype(int)
    return df


def _build_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Recreate the same feature matrix used during training.

    The model was trained on one-hot encoded department codes plus the year.
    We rebuild it the same way and align the columns with the trained
    feature order so that unseen department codes get a column of zeros.
    """
    department_dummies = pd.get_dummies(
        df["DepartamentoCodigo"].astype(str), prefix="dept", dtype=int
    )
    X = pd.concat([department_dummies, df[["Año"]]], axis=1)
    return X.reindex(columns=feature_cols, fill_value=0)


_DATAFRAME    = _load_dataframe()
_MODEL        = joblib.load(MODEL_PATH)
_SCALER       = joblib.load(SCALER_PATH)
_FEATURE_COLS = list(_SCALER.feature_names_in_)


def dataframe() -> pd.DataFrame:
    return _DATAFRAME


def records() -> list[dict]:
    df = _DATAFRAME.copy()
    df["target_label"] = df["target"].map({0: "No high risk", 1: "High risk"})
    return df.to_dict(orient="records")


def _parse_metrics_file() -> dict:
    if not METRICS_PATH.exists():
        return {}
    parsed: dict = {"report_lines": []}
    in_report = False
    for line in METRICS_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("classification_report:"):
            in_report = True
            continue
        if in_report:
            if line.strip():
                parsed["report_lines"].append(line)
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
    return parsed


def metrics() -> dict:
    """Return the metrics persisted by the training script."""
    parsed = _parse_metrics_file()
    return {
        "accuracy":     float(parsed.get("accuracy", 0.0)),
        "roc_auc":      float(parsed.get("roc_auc", 0.0)),
        "train_size":   int(parsed.get("train_size", 0)),
        "test_size":    int(parsed.get("test_size", 0)),
        "features":     parsed.get("features", ""),
        "report_lines": parsed.get("report_lines", []),
    }


def departments() -> list[dict]:
    """Distinct departments with their codes, for prediction dropdowns."""
    df = _DATAFRAME
    grouped = (
        df.groupby("DepartamentoCodigo")["Departamento"]
        .agg(lambda values: values.mode().iat[0])
        .reset_index()
        .sort_values("Departamento")
    )
    return [
        {"code": int(row["DepartamentoCodigo"]), "name": row["Departamento"]}
        for _, row in grouped.iterrows()
    ]


def years() -> list[int]:
    return sorted(int(y) for y in _DATAFRAME["Año"].unique())


def stats() -> dict:
    df = _DATAFRAME
    positives = int(df["target"].sum())
    total = int(len(df))

    sample_predictions = predict_samples(n=8)

    return {
        "total_rows":        total,
        "positive_count":    positives,
        "negative_count":    total - positives,
        "positive_pct":      round(positives / total * 100, 1) if total else 0.0,
        "departments_count": int(df["Departamento"].nunique()),
        "years":             years(),
        "metrics":           metrics(),
        "sample_predictions": sample_predictions,
    }


def predict(department_code: int, year: int) -> dict:
    """Predict the high-risk probability for a given department and year."""
    row = pd.DataFrame(
        [{"DepartamentoCodigo": int(department_code), "Año": int(year)}]
    )
    X = _build_features(row, _FEATURE_COLS)
    X_scaled = _SCALER.transform(X)
    proba = float(_MODEL.predict_proba(X_scaled)[0, 1])
    label = int(_MODEL.predict(X_scaled)[0])
    return {
        "department_code": int(department_code),
        "year":            int(year),
        "probability":     round(proba, 4),
        "prediction":      TARGET_NAMES[label],
        "is_high_risk":    bool(label),
    }


def detailed_metrics() -> dict:
    """Rebuild the train/test split and compute the full set of metrics.

    The split is deterministic (random_state=42, stratify=y) so this returns
    exactly the same data the model was evaluated on after training.
    """
    df = _DATAFRAME
    X = _build_features(df, _FEATURE_COLS)
    y = df["target"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test_scaled = _SCALER.transform(X_test)
    y_pred  = _MODEL.predict(X_test_scaled)
    y_proba = _MODEL.predict_proba(X_test_scaled)[:, 1]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    auc = roc_auc_score(y_test, y_proba)
    accuracy = float((y_pred == y_test).mean())

    return {
        "accuracy": round(accuracy, 4),
        "roc_auc":  round(float(auc), 4),
        "per_class": [
            {
                "name":      TARGET_NAMES[i],
                "precision": round(float(precision[i]), 4),
                "recall":    round(float(recall[i]), 4),
                "f1":        round(float(f1[i]), 4),
                "support":   int(support[i]),
            }
            for i in range(len(TARGET_NAMES))
        ],
        "confusion_matrix": {
            "labels": TARGET_NAMES,
            "matrix": cm.tolist(),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "test_size":  int(len(y_test)),
        "train_size": int(len(df) - len(y_test)),
        "positive_test_count": int((y_test == 1).sum()),
    }


def predict_samples(n: int = 8) -> list[dict]:
    """Run predictions on a small, deterministic sample of the dataset.

    Used to show console-style prediction examples in the Flask view.
    """
    df = _DATAFRAME.sample(n=min(n, len(_DATAFRAME)), random_state=42)
    X = _build_features(df, _FEATURE_COLS)
    X_scaled = _SCALER.transform(X)
    probabilities = _MODEL.predict_proba(X_scaled)[:, 1]
    predictions = _MODEL.predict(X_scaled)

    out = []
    for (_, row), proba, pred in zip(df.iterrows(), probabilities, predictions):
        out.append({
            "department":  row["Departamento"],
            "municipality": row["Municipio"],
            "year":        int(row["Año"]),
            "actual":      TARGET_NAMES[int(row["target"])],
            "predicted":   TARGET_NAMES[int(pred)],
            "probability": round(float(proba), 4),
            "correct":     bool(int(row["target"]) == int(pred)),
        })
    return out
