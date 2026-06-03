"""
Extracción de features de imágenes de ítems.

Cada ítem tiene una imagen PNG de 128x128 en:
    datasets/customer_purchases/images/{item_img_filename}

La imagen codifica señales visuales latentes que NO aparecen como
columnas en el CSV (color de la prenda, patrón, silueta, etc.).

Implementa extract_image_features para extraer las señales que consideres
utiles. El DataFrame resultante se une al resto de features antes del
ColumnTransformer en preprocessing.py.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from ml26.proyectos.P02_customer_purchases.pipeline.io import (
    DATA_DIR,
)

IMG_DIR = Path(os.path.abspath(DATA_DIR)) / "images"


def _load_image(filename: str) -> np.ndarray:
    """Carga una imagen del directorio de imagenes como array (128, 128, 3)."""
    return np.array(Image.open(IMG_DIR / filename).convert("RGB"))


def extract_mean_color(df: pd.DataFrame) -> pd.DataFrame:
    """Color promedio por canal RGB ignorando el fondo blanco.

    Parameters
    ----------
    df : DataFrame con columna item_img_filename.

    Returns
    -------
    pd.DataFrame con columnas img_mean_r, img_mean_g, img_mean_b.
    """
    records = []
    for filename in df["item_img_filename"]:
        arr = _load_image(filename)
        # Excluir pixeles de fondo blanco (todos los canales > 250)
        is_background = (
            (arr[:, :, 0] > 250) & (arr[:, :, 1] > 250) & (arr[:, :, 2] > 250)
        )
        foreground = arr[~is_background]
        mean = foreground.mean(axis=0) if len(foreground) > 0 else arr.mean(axis=(0, 1))
        records.append(
            {
                "img_mean_r": float(mean[0]),
                "img_mean_g": float(mean[1]),
                "img_mean_b": float(mean[2]),
            }
        )
    return pd.DataFrame(records, index=df.index)


def extract_image_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae features visuales por ítem único y devuelve un DataFrame listo para merge.

    Las imágenes se cargan UNA sola vez por ítem (deduplicando por item_id).
    El resultado incluye item_id como clave para hacer merge en orchestration.py.

    Cómo agregar un nuevo extractor
    --------------------------------
    1. Implementa extract_X(df) -> pd.DataFrame que reciba un DataFrame con
       item_img_filename y devuelva columnas nuevas alineadas por índice.
    2. Añade la función a la lista `extractors` abajo.

    Parameters
    ----------
    df : DataFrame con columnas item_id e item_img_filename.

    Returns
    -------
    pd.DataFrame con item_id + columnas de imagen (una fila por ítem único).
    """
    extractors = [
        extract_mean_color,  # img_mean_r, img_mean_g, img_mean_b
        # descomenta y agrega aquí
    ]

    unique_items = (
        df[["item_id", "item_img_filename"]]
        .drop_duplicates("item_id")
        .reset_index(drop=True)
    )

    parts = [fn(unique_items) for fn in extractors]
    img_features = pd.concat(parts, axis=1)
    img_features.insert(0, "item_id", unique_items["item_id"].values)
    return img_features
