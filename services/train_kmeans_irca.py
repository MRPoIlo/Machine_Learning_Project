"""train_kmeans_irca.py - Train a K-Means clustering model on the rural
drinking-water IRCA dataset (the same dataset Logistic Regression uses).

Run this script ONCE after preprocessing to generate the unsupervised model.

Usage (from the project root):
    python services/train_kmeans_irca.py

Output:
    models/kmeans_irca_model.pkl           - trained KMeans model
    models/kmeans_irca_scaler.pkl          - StandardScaler used during training
    data/irca_clusters.csv                 - dataset + assigned cluster label
    reports/kmeans_irca_elbow.png          - elbow + silhouette curves
    reports/kmeans_irca_pca.png            - cluster scatter on first two PCs
    reports/kmeans_irca_metrics.txt        - clustering validation metrics
"""

# 1. Import required libraries
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


ROOT        = Path(__file__).resolve().parent.parent
CSV_PATH    = ROOT / "data"    / "water_quality_for_human_consumption.csv"
MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DATA_DIR    = ROOT / "data"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

MODEL_PATH    = MODELS_DIR  / "kmeans_irca_model.pkl"
SCALER_PATH   = MODELS_DIR  / "kmeans_irca_scaler.pkl"
CLUSTERS_CSV  = DATA_DIR    / "irca_clusters.csv"
ELBOW_PLOT    = REPORTS_DIR / "kmeans_irca_elbow.png"
PCA_PLOT      = REPORTS_DIR / "kmeans_irca_pca.png"
METRICS_FILE  = REPORTS_DIR / "kmeans_irca_metrics.txt"

# Risk levels used to map each cluster to a dominant risk profile after training.
HIGH_RISK_LEVELS = ["Alto riesgo", "Inviable sanitariamente"]


# 2. Load the dataset
data = pd.read_csv(CSV_PATH, sep=";", thousands=",", encoding="utf-8")
data["Año"] = data["Año"].astype(int)
data["IRCArural"] = pd.to_numeric(data["IRCArural"], errors="coerce")
data = data.dropna(subset=["IRCArural"]).reset_index(drop=True)

# 3. Explore the dataset
print(data.head())
print(data.info())
print(data["Nivel de riesgo rural"].value_counts())

# Build the feature matrix:
#   - DepartamentoCodigo as one-hot dummies (same encoding as Logistic).
#   - Año numeric (captures the 2021-2024 temporal trend).
#   - IRCArural numeric. Unlike Logistic Regression, K-Means has no target,
#     so the variable is allowed in the feature space; it is the strongest
#     signal of water-quality profiles.
department_dummies = pd.get_dummies(
    data["DepartamentoCodigo"].astype(str), prefix="dept", dtype=int
)
X = pd.concat(
    [department_dummies, data[["Año", "IRCArural"]]], axis=1
)

FEATURE_COLS = X.columns.tolist()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 4. Elbow + silhouette analysis to choose K
inertias    = []
silhouettes = []
k_range     = range(2, 9)

print("\nRunning elbow analysis...")
for k in k_range:
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    print(f"  K={k}  inertia={km.inertia_:.1f}  silhouette={silhouettes[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(list(k_range), inertias, marker="o", color="steelblue")
axes[0].set_title("Elbow Method - Inertia")
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
print(f"Elbow plot saved to {ELBOW_PLOT}")

silhouette_best = list(k_range)[int(np.argmax(silhouettes))]
print(f"Best K by silhouette: {silhouette_best}")

# Fixed K = 4 to keep clusters interpretable as risk profiles
# (sin / bajo / medio / alto). Silhouette is logged above for the report.
best_k = 4
print(f"K used for the final model: {best_k}")


# 5. Train the final model
print(f"\nTraining final model with K={best_k}...")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# 6. Evaluate the model with intrinsic clustering metrics
sil = silhouette_score(X_scaled, labels)
db  = davies_bouldin_score(X_scaled, labels)
ch  = calinski_harabasz_score(X_scaled, labels)
print(f"Silhouette score:        {sil:.4f}")
print(f"Davies-Bouldin score:    {db:.4f}")
print(f"Calinski-Harabasz score: {ch:.4f}")
print("Cluster sizes:")
print(pd.Series(labels).value_counts().sort_index())

# 7. PCA scatter for visualisation
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
for cluster_id in np.unique(labels):
    mask = labels == cluster_id
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f"Cluster {cluster_id}", s=30, alpha=0.6)
plt.title("IRCA clusters projected on the first two PCA components")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(PCA_PLOT, dpi=150)
plt.close()
print(f"PCA plot saved to {PCA_PLOT}")

# 8. Persist artifacts
data["cluster"] = labels
data.to_csv(CLUSTERS_CSV, index=False, sep=";")

joblib.dump(kmeans, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

METRICS_FILE.write_text(
    f"silhouette: {sil:.4f}\n"
    f"davies_bouldin: {db:.4f}\n"
    f"calinski_harabasz: {ch:.4f}\n"
    f"best_k: {best_k}\n"
    f"train_size: {len(data)}\n"
    f"features: {', '.join(FEATURE_COLS)}\n"
    f"pca_explained_variance: {pca.explained_variance_ratio_[0]:.4f}, {pca.explained_variance_ratio_[1]:.4f}\n",
    encoding="utf-8",
)

print(f"\nModel       -> {MODEL_PATH}")
print(f"Scaler      -> {SCALER_PATH}")
print(f"Clusters    -> {CLUSTERS_CSV}")
print(f"Metrics     -> {METRICS_FILE}")
