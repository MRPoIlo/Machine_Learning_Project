"""train_kmeans.py — Train a K-Means clustering model on the precipitation data.

Run this script ONCE after preprocessing.py to generate the clustering model.

Usage (from the project root):
    python services/train_kmeans.py

Output:
    models/kmeans_model.pkl        — trained KMeans model
    models/robust_scaler.pkl       — RobustScaler used during training
    models/pca.pkl                 — PCA model
    data/station_clusters.csv      — features + assigned cluster label
    reports/elbow_curve.png        — elbow plot to help pick the best K
    reports/clusters_pca.png       — PCA scatter plot
    reports/metrics.txt            — clustering validation metrics
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans                    
from sklearn.decomposition import PCA                           
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,                                     
    calinski_harabasz_score,                                
)
from sklearn.preprocessing import RobustScaler                


ROOT         = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data"    / "features_advanced.csv"     
CLUSTERS_CSV = ROOT / "data"    / "station_clusters.csv"
MODELS_DIR   = ROOT / "models"
REPORTS_DIR  = ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

MODEL_PATH   = MODELS_DIR / "kmeans_model.pkl"
SCALER_PATH  = MODELS_DIR / "robust_scaler.pkl"               
PCA_PATH     = MODELS_DIR / "pca.pkl"                         
ELBOW_PLOT   = REPORTS_DIR / "elbow_curve.png"
PCA_PLOT     = REPORTS_DIR / "clusters_pca.png"                
METRICS_FILE = REPORTS_DIR / "metrics.txt"                     


FEATURE_COLS = [                                               
    "mean_precipitation",
    "median_precipitation",
    "std_precipitation",
    "variance_precipitation",
    "min_precipitation",
    "max_precipitation",
    "p25_precipitation",
    "p75_precipitation",
    "observation_count",
    "rainy_days_ratio",
    "latitude",
    "longitude",
    "mean_hour",
    "hour_sin_mean",
    "hour_cos_mean",
    "month_sin_mean",
    "month_cos_mean",
    "unique_months",
    "unique_years",
    "precipitation_skewness",
    "coefficient_variation",
    "precipitation_range",
    "iqr_precipitation",
]

BEST_K = 4                                                   


def elbow_analysis(X_scaled: np.ndarray):
    inertias    = []
    silhouettes = []
    k_range     = range(2, 11)

    for k in k_range:
        km     = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)  
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


def plot_pca(X_scaled: np.ndarray, labels: np.ndarray, pca: PCA):  
    X_2d = pca.transform(X_scaled)
    plt.figure(figsize=(8, 6))
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f"Cluster {cluster_id}", s=80)
    plt.title("Clusters visualizados con PCA (2 componentes)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PCA_PLOT, dpi=150)
    plt.close()
    print(f"PCA plot saved to {PCA_PLOT}")


def main():
    df = pd.read_csv(FEATURES_CSV)
    print(f"Feature matrix loaded: {df.shape}")

    X      = df[FEATURE_COLS].values
    scaler = RobustScaler()                                    
    X_scaled = scaler.fit_transform(X)

    X_scaled = np.nan_to_num(
    X_scaled,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)
  
    pca      = PCA(n_components=2, random_state=42)
    pca.fit(X_scaled)
    print(f"PCA variance explained: {pca.explained_variance_ratio_}")

    print("\nRunning elbow analysis...")
    elbow_analysis(X_scaled)

    print(f"\nTraining final model with K={BEST_K}...")
    km     = MiniBatchKMeans(n_clusters=BEST_K, random_state=42, n_init=10)  
    labels = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db  = davies_bouldin_score(X_scaled, labels)
    ch  = calinski_harabasz_score(X_scaled, labels)
    print(f"Silhouette score:          {sil:.4f}")
    print(f"Davies-Bouldin score:      {db:.4f}")
    print(f"Calinski-Harabasz score:   {ch:.4f}")

    METRICS_FILE.write_text(                                    
        f"silhouette: {sil:.4f}\n"
        f"davies_bouldin: {db:.4f}\n"
        f"calinski_harabasz: {ch:.4f}\n"
        f"best_k: {BEST_K}\n"
    )

    print("Cluster sizes:")
    df["cluster"] = labels
    print(df["cluster"].value_counts().sort_index())

                                       
    plot_pca(X_scaled, labels, pca)

    joblib.dump(km,     MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(pca,    PCA_PATH)                              
    df.to_csv(CLUSTERS_CSV, index=False)

    print(f"\nModel  → {MODEL_PATH}")
    print(f"Scaler → {SCALER_PATH}")
    print(f"PCA    → {PCA_PATH}")                             
    print(f"Result → {CLUSTERS_CSV}")


if __name__ == "__main__":
    main()