"""preprocessing.py — Clean and save the precipitation dataset.

Run this script ONCE before training any model. It reads the raw CSV,
applies all cleaning steps, and writes a ready-to-use file to data/.

Usage (from the project root):
    python services/preprocessing.py

Output:
    data/precipitation_clean.csv   — cleaned, full dataset
    data/features.csv              — engineered feature matrix for clustering
"""

import io
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "Precipitación_20260520.csv"
CLEAN_CSV = ROOT / "data" / "precipitation_clean.csv"
FEATURES_CSV = ROOT / "data" / "features.csv"


def load_raw(path: Path) -> pd.DataFrame:
    """Read the double-quote-wrapped CSV and return a raw DataFrame."""
    lines = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            line = line.replace('""', '"')
            lines.append(line)
    return pd.read_csv(io.StringIO("\n".join(lines)))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules and return a cleaned DataFrame."""

    for col in ["ValorObservado", "Latitud", "Longitud"]:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)


    df["FechaObservacion"] = pd.to_datetime(
        df["FechaObservacion"], format="%Y %b %d %I:%M:%S %p"
    )


    df["hour"] = df["FechaObservacion"].dt.hour
    df["month"] = df["FechaObservacion"].dt.month
    df["day_of_week"] = df["FechaObservacion"].dt.dayofweek


    for col in ["NombreEstacion", "Departamento", "Municipio", "ZonaHidrografica"]:
        df[col] = df[col].str.strip().str.upper()

    
    before = len(df)
    df = df.dropna(subset=["ValorObservado"])
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} rows with missing ValorObservado.")

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix used for unsupervised learning.

    One row per station with aggregated precipitation statistics.
    """
    agg = (
        df.groupby("CodigoEstacion")
        .agg(
            mean_precipitation=("ValorObservado", "mean"),
            std_precipitation=("ValorObservado", "std"),
            max_precipitation=("ValorObservado", "max"),
            observation_count=("ValorObservado", "count"),
            latitude=("Latitud", "first"),
            longitude=("Longitud", "first"),
            mean_hour=("hour", "mean"),
        )
        .reset_index()
    )
    agg["std_precipitation"] = agg["std_precipitation"].fillna(0.0)
    return agg


def main():
    print("Loading raw data...")
    df_raw = load_raw(RAW_CSV)
    print(f"  Raw shape: {df_raw.shape}")

    print("Cleaning...")
    df_clean = clean(df_raw)
    print(f"  Clean shape: {df_clean.shape}")

    print("Engineering features...")
    df_features = engineer_features(df_clean)
    print(f"  Feature matrix shape: {df_features.shape}")

    print(f"Saving to {CLEAN_CSV}...")
    df_clean.to_csv(CLEAN_CSV, index=False)

    print(f"Saving to {FEATURES_CSV}...")
    df_features.to_csv(FEATURES_CSV, index=False)

    print("Done.")
    print(df_features.describe())


if __name__ == "__main__":
    main()
