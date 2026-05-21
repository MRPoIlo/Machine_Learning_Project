"""Precipitation dataset service."""

import io
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"

CSV_PATH     = DATA_DIR   / "Precipitación_20260520.csv"

FEATURE_COLS = [
    "mean_precipitation",
    "std_precipitation",
    "max_precipitation",
    "latitude",
    "longitude",
    "mean_hour",
]


def _load_dataframe() -> pd.DataFrame:
    lines = []
    with open(CSV_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            line = line.replace('""', '"')
            lines.append(line)

    df = pd.read_csv(io.StringIO("\n".join(lines)))

    for col in ["ValorObservado", "Latitud", "Longitud"]:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    df["FechaObservacion"] = pd.to_datetime(
        df["FechaObservacion"], format="%Y %b %d %I:%M:%S %p"
    )

    df["hour"] = df["FechaObservacion"].dt.hour

    for col in ["NombreEstacion", "Departamento", "Municipio", "ZonaHidrografica"]:
        df[col] = df[col].str.strip().str.upper()

    return df


_DATAFRAME = _load_dataframe()


def dataframe() -> pd.DataFrame:
    return _DATAFRAME


def records() -> list[dict]:
    df = _DATAFRAME.copy()
    df["FechaObservacion"] = df["FechaObservacion"].dt.strftime("%Y-%m-%d %H:%M")
    return df.to_dict(orient="records")


def stats() -> dict:
    df = _DATAFRAME

    # Station info: name, municipality, department
    station_info = (
        df.groupby("CodigoEstacion")
        .agg(
            NombreEstacion=("NombreEstacion", "first"),
            Municipio=("Municipio", "first"),
            Departamento=("Departamento", "first"),
            mean_precipitation=("ValorObservado", "mean"),
            latitude=("Latitud", "first"),
            longitude=("Longitud", "first"),
        )
        .reset_index()
    )

    # Load cluster assignments
    clusters_path = DATA_DIR / "station_clusters.csv"
    cluster_info = []
    if clusters_path.exists():
        df_clusters = pd.read_csv(clusters_path)
        df_merged = station_info.merge(
            df_clusters[["CodigoEstacion", "cluster"]],
            on="CodigoEstacion"
        )
        for _, row in df_merged.iterrows():
            cluster_info.append({
                "station":            int(row["CodigoEstacion"]),
                "name":               row["NombreEstacion"].title(),
                "municipality":       row["Municipio"].title(),
                "department":         row["Departamento"].title(),
                "cluster":            int(row["cluster"]),
                "mean_precipitation": round(float(row["mean_precipitation"]), 3),
                "latitude":           round(float(row["latitude"]), 5),
                "longitude":          round(float(row["longitude"]), 5),
            })

    return {
        "total_rows":          int(len(df)),
        "stations":            int(df["CodigoEstacion"].nunique()),
        "municipalities":      int(df["Municipio"].nunique()),
        "departments":         int(df["Departamento"].nunique()),
        "hydrographic_zones":  int(df["ZonaHidrografica"].nunique()),
        "date_min":            df["FechaObservacion"].min().strftime("%Y-%m-%d"),
        "date_max":            df["FechaObservacion"].max().strftime("%Y-%m-%d"),
        "precipitation_mean":  round(float(df["ValorObservado"].mean()), 3),
        "precipitation_max":   round(float(df["ValorObservado"].max()), 3),
        "precipitation_min":   round(float(df["ValorObservado"].min()), 3),
        "clusters":            cluster_info,
        "n_clusters":          len(set(c["cluster"] for c in cluster_info)),
    }
