"""K-Means (IRCA dataset) service.

Loads the trained K-Means model fit on the rural drinking-water dataset and
exposes helpers to query its metadata, metrics and to run predictions on
single rows. Keep all model-specific logic here so routes stay thin.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"

CSV_PATH      = DATA_DIR    / "water_quality_for_human_consumption.csv"
CLUSTERS_CSV  = DATA_DIR    / "irca_clusters.csv"
MODEL_PATH    = MODELS_DIR  / "kmeans_irca_model.pkl"
SCALER_PATH   = MODELS_DIR  / "kmeans_irca_scaler.pkl"
METRICS_PATH  = REPORTS_DIR / "kmeans_irca_metrics.txt"

HIGH_RISK_LEVELS = ["Alto riesgo", "Inviable sanitariamente"]


def _load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, sep=";", thousands=",", encoding="utf-8")
    df["Año"] = df["Año"].astype(int)
    df["IRCArural"] = pd.to_numeric(df["IRCArural"], errors="coerce")
    df = df.dropna(subset=["IRCArural"]).reset_index(drop=True)
    df["target"] = df["Nivel de riesgo rural"].isin(HIGH_RISK_LEVELS).astype(int)
    return df


_DATAFRAME    = _load_dataframe()
_MODEL        = joblib.load(MODEL_PATH)  if MODEL_PATH.exists()  else None
_SCALER       = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
_FEATURE_COLS = list(_SCALER.feature_names_in_) if _SCALER is not None else []


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate the same feature matrix the trained pipeline expects."""
    department_dummies = pd.get_dummies(
        df["DepartamentoCodigo"].astype(str), prefix="dept", dtype=int
    )
    X = pd.concat([department_dummies, df[["Año", "IRCArural"]]], axis=1)
    return X.reindex(columns=_FEATURE_COLS, fill_value=0)


def _assign_clusters(df: pd.DataFrame) -> np.ndarray:
    X = _build_features(df)
    X_scaled = _SCALER.transform(X)
    return _MODEL.predict(X_scaled)


def dataframe() -> pd.DataFrame:
    return _DATAFRAME


def _parse_metrics_file() -> dict:
    parsed: dict = {}
    if not METRICS_PATH.exists():
        return parsed
    for line in METRICS_PATH.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
    return parsed


def model_metrics() -> dict:
    parsed = _parse_metrics_file()
    pca_variance = []
    if "pca_explained_variance" in parsed:
        pca_variance = [float(v) for v in parsed["pca_explained_variance"].split(",")]
    return {
        "silhouette":         float(parsed.get("silhouette", 0.0)),
        "davies_bouldin":     float(parsed.get("davies_bouldin", 0.0)),
        "calinski_harabasz":  float(parsed.get("calinski_harabasz", 0.0)),
        "best_k":             int(parsed.get("best_k", 0)),
        "train_size":         int(parsed.get("train_size", 0)),
        "n_features":         len(_FEATURE_COLS),
        "features":           parsed.get("features", ""),
        "pca_variance":       [round(v, 4) for v in pca_variance],
        "pca_variance_total": round(sum(pca_variance), 4) if pca_variance else 0.0,
    }


def _cluster_label(mean_irca: float) -> str:
    """Map a cluster centroid IRCA to a human-readable risk profile."""
    if mean_irca < 5:
        return "Sin riesgo · safe water"
    if mean_irca < 14:
        return "Bajo riesgo · monitor"
    if mean_irca < 35:
        return "Riesgo medio · improve"
    if mean_irca < 80:
        return "Alto riesgo · act"
    return "Inviable · critical"


def cluster_breakdown() -> list[dict]:
    """Per-cluster summary on the IRCA dataset."""
    if not CLUSTERS_CSV.exists():
        return []
    df = pd.read_csv(CLUSTERS_CSV, sep=";", thousands=",", encoding="utf-8")
    df["IRCArural"] = pd.to_numeric(df["IRCArural"], errors="coerce")
    df["target"] = df["Nivel de riesgo rural"].isin(HIGH_RISK_LEVELS).astype(int)

    rows: list[dict] = []
    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id]
        mean_irca = float(sub["IRCArural"].mean())
        top_departments = sub["Departamento"].value_counts().head(3).index.tolist()
        dominant_risk = sub["Nivel de riesgo rural"].mode().iat[0] if not sub.empty else "—"
        rows.append({
            "cluster":             int(cluster_id),
            "label":               _cluster_label(mean_irca),
            "size":                int(len(sub)),
            "mean_irca":           round(mean_irca, 2),
            "std_irca":            round(float(sub["IRCArural"].std() or 0.0), 2),
            "min_irca":            round(float(sub["IRCArural"].min()), 2),
            "max_irca":            round(float(sub["IRCArural"].max()), 2),
            "high_risk_pct":       round(float(sub["target"].mean() * 100), 1),
            "n_departments":       int(sub["Departamento"].nunique()),
            "n_municipalities":    int(sub["Municipio"].nunique()),
            "top_departments":     top_departments,
            "dominant_risk_level": dominant_risk,
        })
    return rows


def predict_samples(n: int = 8) -> list[dict]:
    """Run K-Means on a deterministic IRCA sample to mirror Logistic's table."""
    if _MODEL is None or _SCALER is None:
        return []
    sample = _DATAFRAME.sample(n=min(n, len(_DATAFRAME)), random_state=42)
    clusters = _assign_clusters(sample)
    breakdown = {c["cluster"]: c for c in cluster_breakdown()}

    out: list[dict] = []
    for (_, row), cluster_id in zip(sample.iterrows(), clusters):
        meta = breakdown.get(int(cluster_id), {})
        out.append({
            "department":   row["Departamento"],
            "municipality": row["Municipio"],
            "year":         int(row["Año"]),
            "irca":         round(float(row["IRCArural"]), 2),
            "risk_level":   row["Nivel de riesgo rural"],
            "cluster":      int(cluster_id),
            "cluster_label": meta.get("label", f"Cluster {cluster_id}"),
        })
    return out


def detailed_metrics() -> dict:
    """Bundle every K-Means artefact the templates need (parallel to logistic.detailed_metrics)."""
    metrics = model_metrics()
    breakdown = cluster_breakdown()
    return {
        "metrics":            metrics,
        "cluster_breakdown":  breakdown,
        "sample_predictions": predict_samples(n=8),
        "total_rows":         sum(c["size"] for c in breakdown),
        "n_clusters":         len(breakdown),
    }
