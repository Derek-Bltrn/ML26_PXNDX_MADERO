import json
from sklearn.metrics import classification_report

# Custom
from ml26.proyectos.P02_customer_purchases.model import (
    PurchaseModel,
)
from ml26.proyectos.P02_customer_purchases.utils import (
    setup_logger,
)
from ml26.proyectos.P02_customer_purchases.pipeline import (
    read_train_data,
)


def split_by_days(X, y, cutoff_days=60):
    """
    Separa train y validación usando item_days_since_release_cutoff.

    Las filas con ítems lanzados hace <= cutoff_days van a validación;
    el resto va a entrenamiento. Esto imita la distribución cold-start
    del test (ítems nuevos) mejor que un split aleatorio.

    Parameters
    ----------
    X            : pd.DataFrame con columna item_days_since_release_cutoff.
    y            : pd.Series de etiquetas alineada con X.
    cutoff_days  : ítems lanzados hace <= este número de días van a val.

    Returns
    -------
    X_train, X_val, y_train, y_val
    """
    if "item_days_since_release_cutoff" not in X.columns:
        raise ValueError("X must contain an 'item_days_since_release_cutoff' column")

    X = X.copy()

    # Split into train/val usando el valor crudo (no escalado)
    val_mask = X["item_days_since_release_cutoff"] <= cutoff_days
    X = X.drop(columns=["item_days_since_release_cutoff"])
    X_val, y_val = X[val_mask], y[val_mask]
    X_train, y_train = X[~val_mask], y[~val_mask]

    # Shuffle training set
    train_idx = X_train.sample(frac=1, random_state=42).index
    X_train, y_train = X_train.loc[train_idx], y_train.loc[train_idx]

    return X_train, X_val, y_train, y_val


def run_training(X, y, classifier: str):
    # El modelo genera self.name y self.run_dir al inicializarse
    model = PurchaseModel(classifier=classifier)

    # Logger escribe directamente en la carpeta del run
    logger = setup_logger(model.name, log_dir=model.run_dir)
    logger.info(f"Run: {model.name}")
    logger.info(f"Model parameters: {model.get_config()}")

    # Separar en entrenamiento y validacion
    X_train, X_val, y_train, y_val = split_by_days(X, y, cutoff_days=60)
    logger.info(f"Split dataset: {len(X_train)} train / {len(X_val)} val")

    # Entrenamiento
    logger.info(f"Starting model training {classifier}...")
    model.fit(X_train, y_train)
    logger.info("Training completed")

    # Validacion
    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    logger.info(f"Validation metrics: {report}")

    # Guardar modelo y reporte
    model.save()
    logger.info(f"Model saved to {model.run_dir}")

    report_path = model.run_dir / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")

    return model, model.run_dir


if __name__ == "__main__":
    X, y = read_train_data()
    models = ["logistic", "rf", "xgb"]
    for model in models:
        run_training(X, y, model)
