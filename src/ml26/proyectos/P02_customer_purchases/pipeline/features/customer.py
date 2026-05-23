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

    # ── Construir DataFrame final ───────────────────────────────────────────

    # NOTA: para los features del cliente usa la convencion customer_[FEATURE_NAME] ya que esto facilitará el trabajo del preprocessing
    customer_feat = pd.concat(
        {
            "customer_id": group["customer_id"].first(),
            "customer_age_years": customer_age_years,
            "customer_tenure_months": customer_tenure_months,
            # Agrega aquí las features que calculaste arriba, por ejemplo:
            # "customer_avg_price": customer_avg_price,
        },
        axis=1,
    ).reset_index(drop=True)

    # Persistir — read_test_data() carga este archivo en lugar de recomputar
    save_path = os.path.abspath(os.path.join(DATA_DIR, "customer_features.csv"))
    customer_feat.to_csv(save_path, index=False)
    print(f"Customer features saved -> {save_path}")
    return customer_feat
