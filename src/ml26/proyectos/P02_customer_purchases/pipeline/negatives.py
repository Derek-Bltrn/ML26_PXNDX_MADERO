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
    """
    Genera negativos proporcionales a la actividad del cliente usando una mezcla
    de estrategias.

    Idea general:
    - Clientes con mas compras generan mas negativos.
    - 20% son hard negatives: items no comprados pero plausibles para el cliente.
    - El resto mezcla popularidad, precio compatible y random para no sesgar demasiado.

    Returns
    -------
    pd.DataFrame con columnas customer_id, item_id, label (= 0).
    """
    rng = np.random.default_rng(42)

    item_catalog = (
        df[["item_id", "item_category", "item_price"]]
        .drop_duplicates("item_id")
        .reset_index(drop=True)
    )

    item_popularity = df["item_id"].value_counts().to_dict()

    rows = []

    def sample_items(candidates, k, selected, weights=None):
        candidates = [item for item in candidates if item not in selected]

        if k <= 0 or len(candidates) == 0:
            return []

        k = min(k, len(candidates))

        if weights is not None:
            w = np.array([weights.get(item, 0) for item in candidates], dtype=float)
            if w.sum() > 0:
                w = w / w.sum()
            else:
                w = None
        else:
            w = None

        return list(rng.choice(candidates, size=k, replace=False, p=w))

    for customer_id, customer_hist in df.groupby("customer_id"):
        purchased_items = set(customer_hist["item_id"].unique())
        candidate_df = item_catalog[~item_catalog["item_id"].isin(purchased_items)]

        if candidate_df.empty:
            continue

        n_purchases = len(customer_hist)
        target_n = min(len(candidate_df), n_purchases * n_per_positive)

        if target_n <= 0:
            continue

        selected = set()

        # Distribucion:
        # 20% hard negatives: items plausibles por categoria y precio.
        # 35% popular negatives: items populares que este cliente no compro.
        # 25% price-compatible negatives: items dentro del rango de precio historico.
        # 20% random negatives: diversidad y reduccion de sesgo.
        n_hard = int(round(target_n * 0.20))
        n_popular = int(round(target_n * 0.35))
        n_price = int(round(target_n * 0.25))
        n_random = target_n - n_hard - n_popular - n_price

        # 1. Hard negatives:
        # Items de categorias frecuentes del cliente y precio cercano a su promedio.
        category_pct = customer_hist["item_category"].value_counts(normalize=True)

        preferred_categories = category_pct[category_pct >= 0.20].index.tolist()
        if not preferred_categories:
            preferred_categories = category_pct.head(2).index.tolist()

        avg_price = customer_hist["item_price"].mean()
        std_price = customer_hist["item_price"].std()
        price_tolerance = max(avg_price * 0.25, 0 if pd.isna(std_price) else std_price)

        hard_df = candidate_df[
            candidate_df["item_category"].isin(preferred_categories)
            & (candidate_df["item_price"] >= avg_price - price_tolerance)
            & (candidate_df["item_price"] <= avg_price + price_tolerance)
        ]

        hard_sample = sample_items(
            hard_df["item_id"].tolist(),
            n_hard,
            selected,
            weights=item_popularity,
        )
        selected.update(hard_sample)

        # 2. Popular negatives:
        # Items que muchos otros compraron, pero este cliente no.
        popular_sample = sample_items(
            candidate_df["item_id"].tolist(),
            n_popular,
            selected,
            weights=item_popularity,
        )
        selected.update(popular_sample)

        # 3. Price-compatible negatives:
        # Items dentro del rango historico de precios del cliente.
        min_price = customer_hist["item_price"].min()
        max_price = customer_hist["item_price"].max()

        price_df = candidate_df[
            (candidate_df["item_price"] >= min_price)
            & (candidate_df["item_price"] <= max_price)
        ]

        price_sample = sample_items(
            price_df["item_id"].tolist(),
            n_price,
            selected,
        )
        selected.update(price_sample)

        # 4. Random negatives:
        # Mantiene variedad y evita que todos los negativos sean demasiado parecidos.
        random_sample = sample_items(
            candidate_df["item_id"].tolist(),
            n_random,
            selected,
        )
        selected.update(random_sample)

        # Fallback:
        # Si alguna estrategia no tuvo suficientes candidatos, rellenamos con random.
        remaining = target_n - len(selected)
        if remaining > 0:
            fallback_sample = sample_items(
                candidate_df["item_id"].tolist(),
                remaining,
                selected,
            )
            selected.update(fallback_sample)

        rows.extend(
            {
                "customer_id": customer_id,
                "item_id": item_id,
                "label": 0,
            }
            for item_id in selected
        )

    return pd.DataFrame(rows)


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
