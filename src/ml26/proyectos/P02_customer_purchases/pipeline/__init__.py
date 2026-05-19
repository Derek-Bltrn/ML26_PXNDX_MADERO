"""
Pipeline de datos para P01_customer_purchases.

Estructura en tres capas
------------------------
  Capa 1 — Ingeniería de features (MODIFICA AQUÍ):
      features/customer.py → extract_customer_features(train_df)
          Agrega estadísticas por cliente desde el historial de compras.
          Se calcula UNA VEZ sobre train y se reutiliza en test.

      features/image.py → extract_image_features(df)          [opcional]
          Extrae señales visuales de las imágenes de cada ítem
          (color dominante, patrón, etc.).

  Capa 2 — Preprocesamiento (MODIFICA AQUÍ):
      preprocessing.py → preprocess(df, training)
          Define qué columnas escalar, codificar o vectorizar,
          y crea features derivadas (días desde lanzamiento, etc.).
          El flag training garantiza que el ColumnTransformer se ajuste
          solo sobre datos de entrenamiento.

  Capa 3 — Orquestación (NO MODIFICAR):
      orchestration.py → read_train_data() / read_test_data()
          Llaman a las capas anteriores en el orden correcto.
          Garantizan que el preprocessor no se ajuste sobre test
          y que los features de cliente vengan del historial de train.

Archivos persistidos
--------------------
  customer_features.csv : calculado una vez en train, reutilizado en test.
  preprocessor.pkl      : ajustado en train, aplicado en test.
"""

from ml26.proyectos.P02_customer_purchases.pipeline.orchestration import (
    read_train_data,
    read_test_data,
)
