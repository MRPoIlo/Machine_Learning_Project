"""preprocessing.py — Clean and save the precipitation dataset.

Run this script ONCE before training any model. It reads the raw CSV,
applies all cleaning steps, and writes a ready-to-use file to data/.

Usage (from the project root):
    python services/preprocessing.py

Output:
    data/precipitation_clean.csv   — cleaned, full dataset
    data/features_advanced.csv     — engineered feature matrix for clustering
"""

import io
from pathlib import Path

import numpy as np                          
import pandas as pd
from scipy.stats import skew               


ROOT         = Path(__file__).resolve().parent.parent
RAW_CSV      = ROOT / "data" / "Precipitación_20260520.csv"
CLEAN_CSV    = ROOT / "data" / "precipitation_clean.csv"
FEATURES_CSV = ROOT / "data" / "features_advanced.csv"   


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

   
    df["hour"]        = df["FechaObservacion"].dt.hour
    df["month"]       = df["FechaObservacion"].dt.month
    df["day"]         = df["FechaObservacion"].dt.day         
    df["day_of_week"] = df["FechaObservacion"].dt.dayofweek
    df["year"]        = df["FechaObservacion"].dt.year         

                               
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

   
    for col in ["NombreEstacion", "Departamento", "Municipio", "ZonaHidrografica"]:
        df[col] = df[col].str.strip().str.upper()

    before = len(df)
    df = df.dropna(subset=["ValorObservado"])
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} rows with missing ValorObservado.")

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the advanced feature matrix used for unsupervised learning."""

    agg = (
        df.groupby("CodigoEstacion")
        .agg(
            mean_precipitation   =("ValorObservado", "mean"),
            median_precipitation =("ValorObservado", "median"), 
            std_precipitation    =("ValorObservado", "std"),
            variance_precipitation=("ValorObservado", "var"),    
            min_precipitation    =("ValorObservado", "min"),      
            max_precipitation    =("ValorObservado", "max"),
            p25_precipitation    =("ValorObservado", lambda x: x.quantile(0.25)), 
            p75_precipitation    =("ValorObservado", lambda x: x.quantile(0.75)),  
            observation_count    =("ValorObservado", "count"),
            latitude             =("Latitud",  "first"),
            longitude            =("Longitud", "first"),
            mean_hour            =("hour",  "mean"),
            hour_sin_mean        =("hour_sin",  "mean"),         
            hour_cos_mean        =("hour_cos",  "mean"),        
            month_sin_mean       =("month_sin", "mean"),        
            month_cos_mean       =("month_cos", "mean"),       
            unique_months        =("month", "nunique"),        
            unique_years         =("year",  "nunique"),         
        )
        .reset_index()
    )

    
    agg["std_precipitation"]      = agg["std_precipitation"].fillna(0.0)
    agg["variance_precipitation"] = agg["variance_precipitation"].fillna(0.0)

    agg["precipitation_skewness"] = (
        df.groupby("CodigoEstacion")["ValorObservado"]
        .apply(lambda x: skew(x.dropna()))
        .values
    )

    agg["coefficient_variation"] = (
        agg["std_precipitation"] / agg["mean_precipitation"].replace(0, np.nan)
    ).fillna(0.0)

    agg["precipitation_range"] = agg["max_precipitation"] - agg["min_precipitation"]
    agg["iqr_precipitation"]   = agg["p75_precipitation"] - agg["p25_precipitation"]

    agg["rainy_days_ratio"] = (
        df.groupby("CodigoEstacion")["ValorObservado"]
        .apply(lambda x: (x > 0).sum() / len(x))
        .values
    )

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