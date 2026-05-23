import os
from datetime import datetime
from pathlib import Path

import pandas as pd

# Fecha de recolección de datos — congelada, no reemplazar con datetime.today()
DATA_COLLECTED_AT = datetime(2026, 5, 14).date()

CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE.parent / "../../../../datasets/customer_purchases/"


def read_csv(filename: str) -> pd.DataFrame:
    """Lee un CSV del directorio de datos."""
    fullpath = os.path.abspath(os.path.join(DATA_DIR, f"{filename}.csv"))
    return pd.read_csv(fullpath)


def df_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte todas las columnas a numéricas (coerce = NaN si no aplica)."""
    data = df.copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    return data
