"""Water quality (IRCArural) dataset service."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
CSV_PATH = DATA_DIR / 'water_quality_for_human_consumption.csv'

RISK_LEVEL_ORDER = [
    'No Risk',
    'Low Risk',
    'Medium Risk',
    'High Risk',
    'Sanitaryly not feasible',
]


def _load_dataframe():
    df = pd.read_csv(CSV_PATH, sep=';', thousands=',', encoding='utf-8')
    df['Año'] = df['Año'].astype(int)
    df['IRCArural'] = pd.to_numeric(df['IRCArural'], errors='coerce')
    df['Nivel de riesgo rural'] = df['Nivel de riesgo rural'].str.strip()
    return df


_DATAFRAME = _load_dataframe()


def dataframe():
    return _DATAFRAME


def records():
    return _DATAFRAME.to_dict(orient='records')


def stats():
    df = _DATAFRAME
    years = sorted(df['Año'].unique().tolist())
    risk_counts = df['Nivel de riesgo rural'].value_counts().to_dict()
    total = int(len(df))

    risk_distribution = [
        {
            'label': level,
            'count': int(risk_counts.get(level, 0)),
            'pct': round((risk_counts.get(level, 0) / total) * 100, 1) if total else 0,
        }
        for level in RISK_LEVEL_ORDER
    ]

    return {
        'total_rows': total,
        'years': years,
        'year_min': years[0],
        'year_max': years[-1],
        'municipalities': int(df['Municipio'].nunique()),
        'departments': int(df['Departamento'].nunique()),
        'risk_distribution': risk_distribution,
        'irca_mean': round(float(df['IRCArural'].mean()), 2),
        'irca_max': round(float(df['IRCArural'].max()), 2),
    }