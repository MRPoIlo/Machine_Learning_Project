"""train_kmeans.py — Train a K-Means clustering model on the precipitation data.

Run this script ONCE after preprocessing.py to generate the clustering model.

Usage (from the project root):
    python services/train_kmeans.py

Output:
    models/kmeans_model.pkl        — trained KMeans model
    models/scaler.pkl              — StandardScaler used during training
    data/station_clusters.csv      — features + assigned cluster label
    reports/elbow_curve.png        — elbow plot to help pick the best K
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV  = ROOT / "data"    / "features.csv"
CLUSTERS_CSV  = ROOT / "data"    / "station_clusters.csv"
MODELS_DIR    = ROOT / "models"
REPORTS_DIR   = ROOT / "reports"
MODEL_PATH    = MODELS_DIR / "kmeans_model.pkl"
SCALER_PATH   = MODELS_DIR / "scaler.pkl"
ELBOW_PLOT    = REPORTS_DIR / "elbow_curve.png"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


FEATURE_COLS = [
    "mean_precipitation",
    "std_precipitation",
    "max_precipitation",
    "latitude",
    "longitude",
    "mean_hour",
]


BEST_K = 2


def elbow_analysis(X_scaled: np.ndarray):
    inertias    = []
    silhouettes = []
    k_range     = range(2, 11)

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        print(f"  K={k}  inertia={km.inertia_:.1f}  silhouette={silhouettes[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(list(k_range), inertias, marker="o", color="steelblue")
    axes[0].set_title("Elbow Method — Inertia")
    axes[0].set_xlabel("Number of clusters K")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(list(k_range), silhouettes, marker="s", color="seagreen")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of clusters K")
    axes[1].set_ylabel("Silhouette score")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(ELBOW_PLOT, dpi=150)
    plt.close()
    print(f"\nElbow plot saved to {ELBOW_PLOT}")

    best = list(k_range)[silhouettes.index(max(silhouettes))]
    print(f"Best K by silhouette: {best}")


def main():
    df = pd.read_csv(FEATURES_CSV)
    print(f"Feature matrix loaded: {df.shape}")

    X        = df[FEATURE_COLS].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nRunning elbow analysis...")
    elbow_analysis(X_scaled)

    print(f"\nTraining final model with K={BEST_K}...")
    km     = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    print(f"Silhouette score: {sil:.4f}")
    print("Cluster sizes:")

    df["cluster"] = labels
    print(df["cluster"].value_counts().sort_index())

    joblib.dump(km,     MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    df.to_csv(CLUSTERS_CSV, index=False)

    print(f"\nModel  → {MODEL_PATH}")
    print(f"Scaler → {SCALER_PATH}")
    print(f"Result → {CLUSTERS_CSV}")


if __name__ == "__main__":
    main()
