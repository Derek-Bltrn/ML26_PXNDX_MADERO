"""
Capa 2 — Preprocesamiento.

Modifica preprocess() para:
  - Crear features derivadas (e.g. dias desde lanzamiento, match de categoria).
  - Decidir que columnas escalar, codificar o vectorizar.
  - Descartar columnas no disponibles en test (purchase_timestamp, etc.).

No modifiques build_processor — solo define las listas de columnas y las
features derivadas en preprocess().
"""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml26.proyectos.P02_customer_purchases.pipeline.io import (
    DATA_COLLECTED_AT,
    DATA_DIR,
)


def build_processor(
    df: pd.DataFrame,
    numeric_features: list,
    categorical_features: list,
    count_features: list,
    passthrough_features: list = [],
    training: bool = True,
) -> pd.DataFrame:
    """Ajusta o carga un ColumnTransformer y retorna el DataFrame transformado.

    Cuando training=True ajusta el preprocessor y lo guarda en disco.
    Cuando training=False lo carga y aplica — nunca ajustar sobre test.

    Parameters
    ----------
    df                    : DataFrame de entrada (sin label ni IDs).
    numeric_features      : columnas numericas -> StandardScaler.
    categorical_features  : columnas categoricas -> OneHotEncoder.
    count_features         : columnas a aplicar CountVectorizer.
    passthrough_features  : columnas que pasan sin transformar.
    training              : True = fit + save; False = load + transform.
    """
    savepath = Path(os.path.abspath(DATA_DIR)) / "preprocessor.pkl"

    if training:
        text_transformers = [
            (
                col,
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 3),
                    min_df=2,
                    max_features=1000,
                ),
                col,
            )
            for col in count_features
        ]
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                *text_transformers,
                ("passthrough", "passthrough", passthrough_features),
            ],
            remainder="drop",
        )
        processed_array = preprocessor.fit_transform(df)
        joblib.dump(preprocessor, savepath)
    else:
        preprocessor = joblib.load(savepath)
        processed_array = preprocessor.transform(df)

    # CountVectorizer devuelve sparse — convertir a denso para DataFrame uniforme
    if hasattr(processed_array, "toarray"):
        processed_array = processed_array.toarray()

    num_cols = numeric_features
    cat_cols = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(
            categorical_features
        )
    )
    bow_cols = []
    for col in count_features:
        vocab = preprocessor.named_transformers_[col].get_feature_names_out()
        bow_cols.extend([f"{col}_bow_{t}" for t in vocab])

    return pd.DataFrame(
        processed_array,
        columns=list(num_cols) + cat_cols + bow_cols + passthrough_features,
    )


def preprocess(df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
    """Define que features usar y como codificarlas.

    Aqui decides:
      - Que columnas numericas escalar (numeric_features).
      - Que columnas categoricas codificar (categorical_features).
      - Que columnas de texto libre vectorizar (text_features).
      - Que features derivadas crear antes del ColumnTransformer.
      - Que columnas descartar (no disponibles en test o con leakage).

    El flag training se pasa directamente a build_processor — no lo cambies.

    Parameters
    ----------
    df       : DataFrame con todas las features post-merge.
    training : True cuando se llama desde read_train_data (ajusta el
               preprocessor). False cuando se llama desde read_test_data
               (aplica el preprocessor guardado).
    """
    # Columnas a descartar: no disponibles en test o ya integradas en otras features
    drop_raw = [
        "purchase_id",
        "item_release_date",  # conviértela a feature numérica si la necesitas
        "item_img_filename",  # reemplázala por features extraídas de la imagen
        "item_avg_rating",
        "purchase_timestamp",
        "customer_item_views",
        "purchase_item_rating",
        "purchase_device",
        "item_num_ratings"
    ]

    # ── Features derivadas ─────────────────────────────────────────────────
    if training:
        df["item_release_date"] = pd.to_datetime(
            df["item_release_date"],
            errors="coerce",
            dayfirst=True,
        )
    else:
        df["item_release_date"] = pd.to_datetime(
            df["item_release_date"],
            errors="coerce",
            dayfirst=False,
        )
    # item_days_since_release_cutoff: NO borrar — lo usa split_by_days en training.py
    # para separar train/val sin data leakage. Pasa sin escalar via passthrough.
    df["item_days_since_release_cutoff"] = (
        pd.to_datetime(DATA_COLLECTED_AT) - df["item_release_date"]
    ).dt.days

    # ── TODO: crea aquí tus features derivadas ─────────────────────────────
    # Ejemplos:
    
    category_pct_cols = {
        "t-shirt": "customer_pct_t_shirt",
        "blouse": "customer_pct_blouse",
        "dress": "customer_pct_dress",
        "shoes": "customer_pct_shoes",
        "skirt": "customer_pct_skirt",
        "jeans": "customer_pct_jeans",
        "shirt": "customer_pct_shirt",
        "suit": "customer_pct_suit",
        "slacks": "customer_pct_slacks",
        "jacket": "customer_pct_jacket",
    }

    df["customer_item_category_pct"] = 0.0

    for category, pct_col in category_pct_cols.items():
        mask = df["item_category"] == category
        df.loc[mask, "customer_item_category_pct"] = df.loc[mask, pct_col].fillna(0)

    df["customer_preferred_category_match"] = (df["customer_item_category_pct"] > 0.40).astype(int)

    df["customer_recency_score"] = (
        1 - (df["customer_days_since_last_purchase"].fillna(90) / 90)
    ).clip(lower=0, upper=1)

    price_ratio = (
        df["item_price"] / df["customer_avg_price"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1)

    df["item_price_similarity_customer_avg"] = 1 / (
        1 + (price_ratio - 1).abs()
    )

    df["item_price_in_customer_range"] = (
        (df["item_price"] >= df["customer_min_price"])
        & (df["item_price"] <= df["customer_max_price"])
    ).astype(int)

    df["customer_purchase_frequency_90d_score"] = (
        1 - (df["customer_purchase_frequency_90d"].fillna(90) / 90)
    ).clip(lower=0, upper=1)

    df["item_title"] = df["item_title"].fillna("")

    # ── Definicion de grupos de features ───────────────────────────────────
    # Agrega aquí las columnas que quieras escalar con StandardScaler
    numeric_features = [
        "customer_age_years",  # ejemplo: edad del cliente
        "customer_avg_price",
        "customer_item_category_pct",
        "customer_purchase_count",
        "customer_purchase_count_30d",
        "customer_purchase_count_30_90d",
        "customer_purchase_count_90_180d",
        "customer_tenure_months",
        "item_price",
        "img_mean_r",
        "img_mean_g",
        "img_mean_b",
    ]

    # Agrega aquí columnas categóricas para OneHotEncoder
    categorical_features = [
        "customer_gender",
        "item_category",
    ]

    # Agrega aquí columnas para CountVectorizer
    count_features = [
        "item_title",
    ]

    # Columnas que pasan sin transformar — item_days_since_release_cutoff es
    # necesario para split_by_days; se dropea antes de model.fit() en training.py.
    passthrough_features = [
        "item_days_since_release_cutoff",
        "customer_preferred_category_match",
        "customer_recency_score",
        "item_price_similarity_customer_avg",
        "item_price_in_customer_range",
        "customer_purchase_frequency_90d_score",
    ]

    # Tirar columnas de id: NO SIRVEN
    id_cols = ["customer_id", "item_id", "label"]
    cols_to_drop = set(drop_raw) | set(id_cols)
    df_input = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # ── Aplicar ColumnTransformer ───────────────────────────────────────────
    return build_processor(
        df_input,
        numeric_features,
        categorical_features,
        count_features,
        passthrough_features,
        training,
    )
