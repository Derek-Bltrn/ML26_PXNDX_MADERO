"""
Ingeniería de features por cliente.

Modifica extract_customer_features para agregar las estadísticas que
quieras calcular por cliente. El resultado se persiste en
customer_features.csv y se reutiliza al momento de predecir sobre ítems nuevos
(donde no hay historial de compra).
"""

import os

import numpy as np
import pandas as pd

from ml26.proyectos.P02_customer_purchases.pipeline.io import (
    DATA_COLLECTED_AT,
    DATA_DIR,
)


def extract_customer_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features agregadas por cliente a partir del historial de compras.

    Esta función se llama UNA SOLA VEZ sobre los datos de entrenamiento.
    El resultado se guarda en customer_features.csv y se reutiliza en test
    porque el conjunto de test no tiene historial de compra para agregar.

    Parameters
    ----------
    train_df : DataFrame completo de compras de entrenamiento (solo positivos).

    Returns
    -------
    pd.DataFrame con una fila por customer_id.
    """
    df = train_df.copy()
    df["item_release_date"] = pd.to_datetime(df["item_release_date"])
    df["purchase_timestamp"] = pd.to_datetime(df["purchase_timestamp"])
    df["customer_date_of_birth"] = pd.to_datetime(df["customer_date_of_birth"])
    df["customer_signup_date"] = pd.to_datetime(df["customer_signup_date"])

    group = df.groupby("customer_id")

    today_ts = pd.to_datetime(DATA_COLLECTED_AT)

    # ── Ejemplo: edad del cliente en años ──────────────────────────────────
    customer_age_years = (
        today_ts - group["customer_date_of_birth"].first()
    ).dt.days // 365

    # ── Ejemplo: antigüedad en la plataforma en meses ──────────────────────
    customer_tenure_months = (
        today_ts - group["customer_signup_date"].first()
    ).dt.days // 30

    # ── TODO: agrega aquí tus propias features ─────────────────────────────

    # Se agregó el precio promedio de los ítems comprados
    customer_avg_price = group["item_price"].mean()

    # Se agregó el porcentaje de compras por categoría de ítem
    item_categories = [
        "t-shirt",
        "blouse",
        "dress",
        "shoes",
        "skirt",
        "jeans",
        "shirt",
        "suit",
        "slacks",
        "jacket",
    ]

    customer_category_pct = (
        group["item_category"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=item_categories, fill_value=0)
        .rename(columns=lambda cat: f"customer_pct_{cat.replace('-', '_')}")
    )

    # Número de compras totales
    customer_purchase_count_total = group["purchase_id"].count()

    # Días desde la última compra
    customer_days_since_last_purchase = (
        today_ts - group["purchase_timestamp"].max()
    ).dt.days

    # Cantidad de compras por rango de días
    df["days_since_purchase"] = (
        today_ts - df["purchase_timestamp"]
    ).dt.days

    customer_purchase_count_30d = (
        df[df["days_since_purchase"] <= 30]
        .groupby("customer_id")
        .size()
        .reindex(group.size().index, fill_value=0)
    )

    customer_purchase_count_30_90d = (
        df[
            (df["days_since_purchase"] > 30)
            & (df["days_since_purchase"] <= 90)
        ]
        .groupby("customer_id")
        .size()
        .reindex(group.size().index, fill_value=0)
    )

    customer_purchase_count_90_180d = (
        df[
            (df["days_since_purchase"] > 90)
            & (df["days_since_purchase"] <= 180)
        ]
        .groupby("customer_id")
        .size()
        .reindex(group.size().index, fill_value=0)
    )

    # Precio mínimo y máximo de compra
    customer_min_price = group["item_price"].min()
    customer_max_price = group["item_price"].max()

    # Frecuencia promedio de compra en los últimos 90 días(se mide en días).
    recent_purchases_90d = df[df["days_since_purchase"] <= 90].copy()

    recent_purchases_90d = recent_purchases_90d.sort_values(
        ["customer_id", "purchase_timestamp"]
    )

    recent_purchases_90d["days_between_purchases_90d"] = (
        recent_purchases_90d
        .groupby("customer_id")["purchase_timestamp"]
        .diff()
        .dt.days
    )

    customer_purchase_frequency_90d = (
        recent_purchases_90d
        .groupby("customer_id")["days_between_purchases_90d"]
        .mean()
        .reindex(group.size().index, fill_value=90)
        .fillna(90)
    )

    # ── Construir DataFrame final ───────────────────────────────────────────

    # NOTA: para los features del cliente usa la convencion customer_[FEATURE_NAME] ya que esto facilitará el trabajo del preprocessing
    customer_feat = pd.concat(
        {
            "customer_id": group["customer_id"].first(),
            "customer_age_years": customer_age_years,
            "customer_tenure_months": customer_tenure_months,
            # Agrega aquí las features que calculaste arriba, por ejemplo:
            # "customer_avg_price": customer_avg_price,
            "customer_avg_price": customer_avg_price,
            "customer_purchase_count": customer_purchase_count_total,
            "customer_days_since_last_purchase": customer_days_since_last_purchase,
            "customer_purchase_count_30d": customer_purchase_count_30d,
            "customer_purchase_count_30_90d": customer_purchase_count_30_90d,
            "customer_purchase_count_90_180d": customer_purchase_count_90_180d,
            "customer_min_price": customer_min_price,
            "customer_max_price": customer_max_price,
            "customer_purchase_frequency_90d": customer_purchase_frequency_90d,
            **customer_category_pct.to_dict("series"),
        },
        axis=1,
    ).reset_index(drop=True)

    # Persistir — read_test_data() carga este archivo en lugar de recomputar
    save_path = os.path.abspath(os.path.join(DATA_DIR, "customer_features.csv"))
    customer_feat.to_csv(save_path, index=False)
    print(f"Customer features saved -> {save_path}")
    return customer_feat
