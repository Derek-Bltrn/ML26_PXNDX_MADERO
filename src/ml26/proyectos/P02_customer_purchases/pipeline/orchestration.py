"""
Capa 3 — Orquestacion. NO MODIFICAR.

Lee, combina y preprocesa los datos en el orden correcto:
  - read_train_data: calcula features de cliente, genera negativos, ajusta el preprocessor.
  - read_test_data : carga features de cliente pre-calculadas, aplica el preprocessor guardado.

Modificar este archivo puede romper la separacion train/test o causar data leakage.
"""

import pandas as pd

from ml26.proyectos.P02_customer_purchases.pipeline.features.customer import (
    extract_customer_features,
)
from ml26.proyectos.P02_customer_purchases.pipeline.features.image import (
    extract_image_features,
)
from ml26.proyectos.P02_customer_purchases.pipeline.io import (
    df_to_numeric,
    read_csv,
)
from ml26.proyectos.P02_customer_purchases.pipeline.negatives import (
    gen_final_dataset,
    gen_smart_negatives,
)
from ml26.proyectos.P02_customer_purchases.pipeline.preprocessing import (
    preprocess,
)


def _add_image_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae features de imagen por ítem y las une al DataFrame principal."""
    img_feat = extract_image_features(df)
    return pd.merge(df, img_feat, on="item_id", how="left")


def _add_customer_features(
    df: pd.DataFrame, customer_feat: pd.DataFrame
) -> pd.DataFrame:
    """Une las features agregadas de cliente al DataFrame principal."""
    raw_cols = {
        "customer_id",
        "customer_gender",
        "customer_date_of_birth",
        "customer_signup_date",
    }
    agg_cols = [c for c in customer_feat.columns if c not in raw_cols]
    return pd.merge(
        df, customer_feat[["customer_id"] + agg_cols], on="customer_id", how="left"
    )


def read_train_data():
    """Carga y preprocesa los datos de entrenamiento.

    Flujo:
      1. Carga el CSV de compras (solo positivos).
      2. Calcula y persiste features por cliente.
      3. Genera ejemplos negativos.
      4. Combina positivos y negativos (gen_final_dataset).
      5. Agrega features de cliente al dataset combinado.
      6. Extrae y agrega features de imagen por ítem.  [opcional]
      7. Aplica preprocess (ajusta y guarda el preprocessor).

    Returns
    -------
    X : pd.DataFrame -- features de entrenamiento.
    y : pd.Series   -- etiquetas (0 / 1).
    """
    # 1. Carga el CSV de compras (solo positivos).
    train_df = read_csv("customer_purchases_train")

    # 2. Calcula y persiste features por cliente.
    customer_feat = extract_customer_features(train_df)

    # 3. Genera ejemplos negativos.
    # Revisa el código de negatives.py para seleccionar tu estrategia
    negatives = gen_smart_negatives(train_df, n_per_positive=1)

    # 4. Combina positivos y negativos (gen_final_dataset).
    full_df = gen_final_dataset(train_df, negatives)

    # 5. Agrega features de cliente al dataset combinado.
    full_df = _add_customer_features(full_df, customer_feat)

    # 6. Extrae y agrega features de imagen por ítem.
    full_df = _add_image_features(full_df)

    # 7. Aplica preprocess (ajusta y guarda el preprocessor).
    processed = preprocess(full_df, training=True)
    processed = df_to_numeric(processed)
    processed = pd.concat([processed, full_df["label"].reset_index(drop=True)], axis=1)

    # Regresar conjunto separando atributos de etiqueta para entrenar
    y = processed["label"]
    X = processed.drop(columns=["label", "customer_id", "item_id"], errors="ignore")
    return X, y


def read_test_data():
    """Carga y preprocesa el conjunto de test.

    Flujo:
      1. Carga el CSV de test (pares cliente × ítem nuevo, sin historial de compra).
      2. Carga las features de cliente calculadas por read_train_data()
         desde customer_features.csv — no se recalculan porque el historial
         de compras no está disponible en test.
      3. Agrega features de cliente al dataset.
      4. Extrae y agrega features de imagen por ítem.  [opcional]
      5. Aplica preprocess cargando el preprocessor guardado (training=False)
         — nunca se ajusta sobre test para evitar data leakage.

    A diferencia de read_train_data, no hay generación de negativos ni separación de etiquetas: el CSV de test ya contiene los pares a predecir y el label está oculto.

    Returns
    -------
    X : pd.DataFrame -- features del test (sin label).
    """
    # 1. Carga el CSV de test.
    df = read_csv("customer_purchases_test")

    # 2. Carga las features de cliente calculadas por read_train_data()
    customer_feat = read_csv("customer_features")

    # 3. Agrega features de cliente al dataset.
    merged = _add_customer_features(df, customer_feat)

    # 4. Extrae y agrega features de imagen por ítem.
    merged = _add_image_features(merged)

    # 5. Aplica preprocess cargando el preprocessor guardado (training=False)
    processed = preprocess(merged, training=False)
    processed = df_to_numeric(processed)
    processed = processed.drop(
        columns=["customer_id", "item_id", "item_days_since_release_cutoff"],
        errors="ignore",
    )
    return processed
