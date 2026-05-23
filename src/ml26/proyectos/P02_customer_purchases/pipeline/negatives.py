"""
Estrategias de generación de negativos para el pipeline de entrenamiento.

El CSV de entrenamiento contiene SOLO compras confirmadas (label=1). Para
entrenar un clasificador binario necesitamos también ejemplos negativos:
pares (cliente, ítem) donde el cliente NO compró el ítem.

Estrategias disponibles
-----------------------
    gen_random_negatives     — N negativos fijos por cliente, muestreo uniforme. Ignora frecuencia de compra por cliente.
    gen_uniform_random       — negativos proporcionales a la actividad del cliente.
    gen_popularity_weighted  — ítems populares tienen más probabilidad de ser negativos para un cliente dado.

Ideas para explorar (gen_smart_negatives)
-----------------------------------------
Una vez que entiendas las estrategias anteriores, piensa en cómo el
comportamiento del cliente puede guiar la selección de negativos:

  - Mismatch de categoría: seleccionar ítems de categorías que el cliente raramente compra. Intuición: si alguien solo compra zapatos, un vestido es un negativo plausible.

  - Mismatch de precio: seleccionar ítems fuera del rango de precios histórico del cliente. Intuición: un cliente de presupuesto bajo probablemente no comprará un ítem premium.

  - Ítems recientes no vistos: priorizar ítems lanzados después de la última compra del cliente. Estos son negativos "honestos" para el problema cold-start.

  - Mix de estrategias: combinar varias estrategias en distintas proporciones para generar negativos fáciles y difíciles al mismo tiempo.

Todas las funciones reciben el DataFrame de positivos (customer_id, item_id más columnas del CSV) y devuelven un DataFrame con columnas
customer_id, item_id, label (= 0). Úsalas junto con gen_final_dataset para construir el dataset de entrenamiento completo.
"""

import numpy as np
import pandas as pd


def get_negatives(df: pd.DataFrame) -> dict:
    """Construye un mapa de ítems no comprados por cada cliente.

    Parameters
    ----------
    df : DataFrame de compras positivas con columnas customer_id e item_id.

    Returns
    -------
    dict : {customer_id -> set(item_ids no comprados por ese cliente)}
    """
    unique_items = set(df["item_id"].unique())
    negatives = {}
    for customer in df["customer_id"].unique():
        purchased = set(df[df["customer_id"] == customer]["item_id"].unique())
        negatives[customer] = unique_items - purchased
    return negatives


def gen_random_negatives(df: pd.DataFrame, n_per_positive: int = 1) -> pd.DataFrame:
    """Genera negativos muestreando aleatoriamente ítems no comprados por cliente.

    Por cada cliente, selecciona hasta n_per_positive ítems al azar de los
    que no compró. Si el cliente tiene menos ítems no comprados que
    n_per_positive, se usan todos los disponibles.

    Nota: el muestreo es por cliente, no global. Cada cliente aporta el mismo
    número de negativos independientemente de cuántas compras tenga. Esto
    difiere de un muestreo uniforme global (gen_uniform_random), donde clientes
    con más compras aportan más negativos porque el muestreo es proporcional a
    la actividad.

    Parameters
    ----------
    df              : DataFrame de compras positivas con columnas customer_id e item_id.
    n_per_positive  : cuántos negativos generar por cliente (default 1).

    Returns
    -------
    pd.DataFrame con columnas customer_id, item_id, label (= 0).
    """
    negatives = get_negatives(df)
    rows = []
    for cid, item_set in negatives.items():
        sample_size = min(len(item_set), n_per_positive)
        sampled = np.random.choice(list(item_set), size=sample_size, replace=False)
        rows.extend({"customer_id": cid, "item_id": iid, "label": 0} for iid in sampled)
    return pd.DataFrame(rows)


def gen_uniform_random(df: pd.DataFrame, n_per_positive: int = 1) -> pd.DataFrame:
    """Genera negativos de forma global, proporcional a la actividad del cliente.

    A diferencia de gen_random_negatives —que da exactamente n negativos por
    cliente—, aquí el número de negativos por cliente escala con sus compras:
    un cliente con 10 compras aporta 10 × n_per_positive negativos, mientras
    que uno con 2 compras aporta solo 2 × n_per_positive. El dataset resultante
    refleja la actividad real de cada cliente en lugar de igualarla.

    Cuándo usar cada una:
        gen_random_negatives → distribución balanceada por cliente; útil si
                               quieres que todos los clientes tengan igual peso.
        gen_uniform_random   → distribución proporcional a compras; más cercano
                               a lo que vería el modelo en producción, donde los
                               clientes activos generan más señal.

    Parameters
    ----------
    df              : DataFrame de compras positivas con columnas customer_id e item_id.
    n_per_positive  : cuántos negativos generar por compra positiva (default 1).

    Returns
    -------
    pd.DataFrame con columnas customer_id, item_id, label (= 0).
    """
    purchase_counts = df.groupby("customer_id").size()
    negatives_map = get_negatives(df)
    rows = []
    for cid, n_purchases in purchase_counts.items():
        item_set = negatives_map.get(cid, set())
        target = min(len(item_set), n_purchases * n_per_positive)
        if target == 0:
            continue
        sampled = np.random.choice(list(item_set), size=target, replace=False)
        rows.extend({"customer_id": cid, "item_id": iid, "label": 0} for iid in sampled)
    return pd.DataFrame(rows)


def gen_popularity_weighted(df: pd.DataFrame, n_per_positive: int = 1) -> pd.DataFrame:
    """Genera negativos favoreciendo ítems populares.

    Igual que gen_random_negatives en estructura (N negativos por cliente), pero en lugar de muestrear ítems no comprados de forma uniforme, los pondera por su frecuencia global de compra: ítems que otros clientes compran mucho tienen más probabilidad de ser seleccionados como negativos.

    Intuición: si un ítem es muy popular y este cliente NO lo compró, eso es una señal informativa — el modelo tiene que aprender qué hace diferente a este cliente. Negativos puramente aleatorios no siempre capturan esa señal.

    Parametros
    ----------
    df              : DataFrame de compras positivas con columnas customer_id e item_id.
    n_per_positive  : cuántos negativos generar por cliente (default 1).

    Returns
    -------
    pd.DataFrame con columnas customer_id, item_id, label (= 0).
    """
    item_counts = df["item_id"].value_counts()
    weight_map = (item_counts / item_counts.sum()).to_dict()

    negatives_map = get_negatives(df)
    rows = []
    for cid, item_set in negatives_map.items():
        candidates = [iid for iid in item_set if iid in weight_map]
        if not candidates:
            continue
        w = np.array([weight_map[iid] for iid in candidates])
        w /= w.sum()
        sample_size = min(len(candidates), n_per_positive)
        sampled = np.random.choice(candidates, size=sample_size, replace=False, p=w)
        rows.extend({"customer_id": cid, "item_id": iid, "label": 0} for iid in sampled)
    return pd.DataFrame(rows)


def gen_smart_negatives(df: pd.DataFrame, n_per_positive: int = 1) -> pd.DataFrame:
    """Placeholder para tu propia estrategia de negativos.

    Aquí puedes implementar cualquier lógica que considere el comportamiento
    del cliente para seleccionar negativos más informativos. Algunas ideas:

    - Mismatch de categoría: para cada cliente, calcular sus top-k categorías
      más compradas y seleccionar ítems de las categorías restantes.
      (Requiere la columna item_category en df.)

    - Mismatch de precio: calcular el precio mediano histórico del cliente y
      seleccionar ítems cuyo precio esté fuera de un rango [median*(1-p),
      median*(1+p)]. (Requiere item_price en df.)

    - Negativos recientes: priorizar ítems lanzados después de la última compra
      del cliente, que son los más relevantes para el problema cold-start.
      (Requiere item_release_date y purchase_timestamp en df.)

    - Mix de estrategias: llamar a varias de las funciones anteriores y
      concatenar sus resultados con distintas proporciones.

    Parameters
    ----------
    df              : DataFrame de compras positivas con todas las columnas del CSV.
    n_per_positive  : cuántos negativos generar por cliente.

    Returns
    -------
    pd.DataFrame con columnas customer_id, item_id, label (= 0).
    """
    raise NotImplementedError("Implementa tu propia estrategia aquí.")


def gen_final_dataset(train_df: pd.DataFrame, negatives: pd.DataFrame) -> pd.DataFrame:
    """Combina positivos y negativos en un dataset listo para entrenar.

    Enriquece los negativos con las features de cliente e ítem extraídas
    del historial positivo, concatena, y mezcla el resultado.

    Parameters
    ----------
    train_df  : DataFrame de compras positivas con todas las columnas del CSV.
    negatives : DataFrame de negativos con columnas customer_id, item_id, label.

    Returns
    -------
    pd.DataFrame con las mismas columnas que train_df, label in {0, 1}, mezclado.
    """
    customer_columns = [
        "customer_date_of_birth",
        "customer_gender",
        "customer_signup_date",
    ]
    item_columns = [
        "item_title",
        "item_category",
        "item_price",
        "item_img_filename",
        "item_avg_rating",
        "item_num_ratings",
        "item_release_date",
    ]
    purchase_columns = [
        "purchase_id",
        "purchase_timestamp",
        "customer_item_views",
        "purchase_item_rating",
        "purchase_device",
    ]

    customer_feat = train_df[["customer_id"] + customer_columns].drop_duplicates("customer_id")
    item_feat = train_df[["item_id"] + item_columns].drop_duplicates("item_id")

    neg_enriched = negatives.merge(customer_feat, on="customer_id", how="left").merge(
        item_feat, on="item_id", how="left"
    )
    for col in purchase_columns:
        neg_enriched[col] = np.nan

    pos = train_df.copy()
    pos["label"] = 1
    neg_enriched = neg_enriched.reindex(columns=pos.columns)

    combined = pd.concat([pos, neg_enriched], ignore_index=True)
    return combined.sample(frac=1).reset_index(drop=True)
