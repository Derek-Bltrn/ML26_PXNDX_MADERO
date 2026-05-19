# Proyecto: Predicción de Compras de Nuevos Ítems

## Objetivo

Dado un historial de compras de clientes y las características de los productos, el objetivo es **predecir si un cliente existente comprará un nuevo ítem de moda que será lanzado próximamente**, dentro de los siguientes **60 días**.

Para fines del proyecto se asume que la fecha actual es **14 de mayo de 2026**. Los ítems del conjunto de test fueron lanzados después de esta fecha y no tienen historial de compra — esto es el **cold-start problem**.

---

## Entregables

1. **Exploración de datos (requerido)**
   - Completar el notebook `data_exploration.ipynb`.
   - Revisar valores faltantes, tipos de datos, distribuciones de features.
   - Analizar relaciones entre clientes, ítems y compras.

2. **Pipeline de ML (requerido)**
   - Implementar las capas de ingeniería de features y preprocesamiento.
   - Generar ejemplos negativos para entrenamiento.
   - Entrenar y evaluar al menos dos modelos clasificador, al menos uno tradicional/tabular.
   - Integrar al menos un atributo derivado de la imagen.

3. **Monitoreo y resultados (requerido)**
   - Usar el logger provisto (`utils.py`) para registrar experimentos.
   - Reportar métricas de evaluación adecuadas e hiperparámetros.
   - Presentar análisis comparativo entre modelos o configuraciones.

4. **Opcional / Bonus**
   - Estrategia de negativos propia (`gen_smart_negatives` en `pipeline/negatives.py`).
   - Experimentos con distintos modelos o encodings.

5. **Presentación**

   El proyecto será evaluado en una presentación corta de su blog, con una duración de 10 min máximo por equipo, con enfoque en decisiones de diseño, resultados e interpretación. Su blog/presentación debe:
      - Estar dirigida a alguien **sin conocimiento técnico** pero con conocimiento general de programación.
      - Evitar mostrar código; priorizar diagramas, gráficos y visualizaciones.
      - **Conclusión obligatoria:** resumen de lo realizado y justificación técnica de la solución elegida.

---

## Estructura del boilerplate

El directorio contiene la estructura del pipeline que deben completar. Los archivos marcados como **NO MODIFICAR** orquestan el flujo correcto entre capas.

```

├── pipeline/
│   ├── __init__.py              # NO MODIFICAR
│   ├── io.py                    # NO MODIFICAR — lectura de CSVs
│   ├── orchestration.py         # NO MODIFICAR — flujo train/test
│   ├── negatives.py             # MODIFICAR (opcional) — estrategias de negativos
│   ├── preprocessing.py         # MODIFICAR — definir features y encodings
│   └── features/
│       ├── customer.py          # MODIFICAR — features por cliente
│       └── image.py             # MODIFICAR — features de imagen
├── model.py                     # MODIFICAR — wrapper de clasificadores / implementación de su modelo
├── training.py                  # MODIFICAR (opcional) — entrenamiento y validación
├── inference.py                 # MODIFICAR — cambiar model_folder al nombre de tu modelo
└── utils.py                     # NO MODIFICAR — logger
```

### Lo que deben implementar

#### 1. `pipeline/features/customer.py` — features por cliente

La función `extract_customer_features(train_df)` agrega estadísticas por cliente a partir de su historial de compras. Se llama una sola vez sobre train y el resultado se reutiliza en test (porque en test no hay historial disponible).

Ya tiene `customer_age_years` y `customer_tenure_months` como ejemplos. Deben agregar más features:

```python
# Ideas:
# Precio histórico del cliente
customer_avg_price = group["item_price"].mean()
customer_std_price = group["item_price"].std().fillna(0)

# Top-3 categorías más compradas
# (ver comentarios en el archivo)

# Dispositivo de compra preferido
preferred_device = group["purchase_device"].agg(lambda x: x.mode().iloc[0])
```

> **Importante:** cualquier feature que calculen aquí debe ser derivable del historial de compras de train, no usen columnas que no estén disponibles en test.

#### 2. `pipeline/preprocessing.py` — features y encodings

La función `preprocess(df, training)` define qué columnas entran al modelo y cómo se transforman. `build_processor` (no modificar) aplica el `ColumnTransformer`.

Deben completar las listas de features y crear features derivadas:

```python
# Features numéricas → StandardScaler
numeric_features = [
    "customer_age_years",       # ya incluido como ejemplo
    # "customer_avg_price",
    # "item_days_since_release",
    # ...
]

# Features categóricas → OneHotEncoder
categorical_features = [
    # "customer_prefered_device",
]

# Features de texto libre → CountVectorizer
count_features = [
    # "item_title",
]
```

También pueden crear features derivadas antes del `ColumnTransformer`:

```python
# Días desde lanzamiento del ítem
df["item_days_since_release"] = (
    pd.to_datetime(DATA_COLLECTED_AT) - df["item_release_date"]
).dt.days

# Match entre categoría del ítem y top categorías del cliente
df["customer_top_1_match"] = (df["customer_top_1_cat"] == df["item_category"]).astype(int)
```

> **Nota:** `item_days_since_release_cutoff` es una columna auxiliar usada internamente para separar train/val, no la borren ni la estandaricen, se necesita íntegra (passthrough).

#### 3. `pipeline/negatives.py` — estrategias de negativos (opcional)

El CSV de entrenamiento contiene **solo compras confirmadas** (label = 1). Para entrenar un clasificador binario necesitan generar ejemplos negativos.

Ya hay tres estrategias disponibles:

| Función | Descripción |
|---|---|
| `gen_random_negatives` | N negativos por cliente, muestreo uniforme |
| `gen_uniform_random` | negativos proporcionales a la actividad del cliente |
| `gen_popularity_weighted` | ítems populares tienen más probabilidad de ser negativos |

Para cambiar la estrategia, editar `orchestration.py` línea que llama a `gen_random_negatives`. También pueden implementar `gen_smart_negatives` con su propia lógica.

#### 4. `pipeline/features/image.py` — features de imagen (opcional / bonus)

Cada ítem tiene una imagen PNG de 128×128 en `datasets/customer_purchases/images/`. La imagen codifica señales visuales que **no aparecen en el CSV**.

El archivo ya incluye `extract_mean_color` como ejemplo. Para agregar un nuevo extractor:

1. Implementar `extract_X(df) -> pd.DataFrame` con columna `item_img_filename`.
2. Agregar la función a la lista `extractors` en `extract_image_features`.
3. Registrar las columnas nuevas en `numeric_features` de `preprocessing.py`.
4. Descomentar la llamada a `_add_image_features` en `orchestration.py`.

---

### Training Dataset — `customer_purchases_train.csv`

Contiene únicamente **compras confirmadas** (label = 1). Deben generar ejemplos negativos para entrenamiento.

| Columna | Tipo | Descripción |
|---|---|---|
| `purchase_id` | int | Identificador único de compra |
| `customer_id` | str | e.g. `CUST_0042` |
| `customer_date_of_birth` | date | Fecha de nacimiento |
| `customer_gender` | str / null | male o female; puede ser nulo |
| `customer_signup_date` | date | Fecha de registro en la plataforma |
| `item_id` | str | e.g. `ITEM_0137` |
| `item_title` | str | Título del producto |
| `item_category` | str | t-shirt, blouse, dress, shoes, skirt, jeans, shirt, suit, slacks, jacket |
| `item_price` | float | Precio del producto |
| `item_img_filename` | str | Nombre del archivo de imagen |
| `item_avg_rating` | float / null | Calificación promedio |
| `item_num_ratings` | int | Número de reseñas |
| `item_release_date` | date | Fecha de lanzamiento |
| `purchase_timestamp` | datetime | Fecha y hora de compra |
| `customer_item_views` | int | Vistas previas al producto |
| `purchase_item_rating` | float / null | Calificación que el usuario otorgó |
| `purchase_device` | str | mobile o desktop |
| `label` | int | 1 = compra confirmada |

### Test Dataset — `customer_purchases_test.csv`

Contiene pares (cliente existente, ítem nuevo) para los cuales deben predecir si habrá compra. Los ítems son nuevos (lanzados después del 14-05-2026) y no tienen historial.

| Columna | Tipo | Descripción |
|---|---|---|
| `purchase_id` | int | Identificador único del par (cliente, ítem) |
| `customer_id` | str | e.g. `CUST_0042` |
| `customer_date_of_birth` | date | Fecha de nacimiento |
| `customer_gender` | str / null | male o female; puede ser nulo |
| `customer_signup_date` | date | Fecha de registro en la plataforma |
| `item_id` | str | e.g. `ITEM_0137` |
| `item_title` | str | Título del producto |
| `item_category` | str | t-shirt, blouse, dress, shoes, skirt, jeans, shirt, suit, slacks, jacket |
| `item_price` | float | Precio del producto |
| `item_img_filename` | str | Nombre del archivo de imagen |
| `item_avg_rating` | null | Ítem nuevo, aún sin calificaciones |
| `item_num_ratings` | null | Ítem nuevo, aún sin reseñas |
| `item_release_date` | date | Fecha de lanzamiento (posterior al 14-05-2026) |
| `purchase_timestamp` | null | Ítem no lanzado, compra no ocurrida |
| `customer_item_views` | null | Ítem no lanzado, sin vistas registradas |
| `purchase_item_rating` | null | Ítem no lanzado, sin calificación de compra |
| `purchase_device` | null | Ítem no lanzado, dispositivo desconocido |

---

## Imágenes de productos

Cada ítem tiene una imagen sintética en:

```
/datasets/customer_purchases/images/{item_img_filename}
```

Las imágenes son PNG de **128×128 píxeles** con fondo blanco. Cada imagen codifica señales visuales relacionadas con las preferencias de los clientes que **no aparecen como columnas en el CSV**.

### Ideas para extracción de features de imagen

| Enfoque | Dificultad | Señal capturada |
|---|---|---|
| Color promedio RGB (ya implementado) | Fácil | Paleta de color |
| Histograma de color por canal | Fácil | Distribución de color |
| LBP / HOG | Medio | Patrón / textura |
| CNN embeddings (ResNet, EfficientNet) | Avanzado | Color + patrón + forma |

---

## Consideraciones

- El **cold-start problem**: el modelo no tiene historial de compra para ítems nuevos del test. Las features deben derivarse del historial del **cliente**, no del ítem.
- Los campos nulos en test no están disponibles por diseño ya que reflejan la realidad de un lanzamiento nuevo. No los usen como features.
- El dataset de train contiene **solo positivos**, generar negativos es parte del problema.
