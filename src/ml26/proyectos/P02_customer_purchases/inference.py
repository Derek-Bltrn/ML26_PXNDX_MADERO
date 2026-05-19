import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from ml26.proyectos.P02_customer_purchases.pipeline import (
    read_test_data,
)
from ml26.proyectos.P02_customer_purchases.pipeline.io import (
    read_csv,
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import numpy as np

CURRENT_FILE = Path(__file__).resolve()

RESULTS_DIR = CURRENT_FILE.parent / "test_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

MODELS_DIR = CURRENT_FILE.parent / "trained_models"


def run_inference(model_folder: str, X):
    """
    Carga el modelo guardado en trained_models/<model_folder>/model.pkl
    y genera predicciones sobre X.

    Utiliza este archivo para calcular las predicciones sobre customer_purchases_test
    y subir tus resultados a la competencia de Kaggle.
    """
    full_path = MODELS_DIR / f"{model_folder}/model.pkl"
    print(f"Loading model from {full_path}")
    model = joblib.load(full_path)

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    results = pd.DataFrame({"ID": X.index, "pred": preds, "proba": probs})
    return results


def run_inference_random(X):
    """
    Clasificador aleatorio como baseline de comparación.

    Genera predicciones al azar con probabilidad uniforme en [0, 1] y umbral
    en 0.5. Un modelo bien entrenado debe superar este baseline en todas las
    métricas; si no lo hace, hay un problema en el pipeline.
    """
    probs = np.random.uniform(0, 1, size=len(X))
    preds = (probs >= 0.5).astype(int)
    return pd.DataFrame({"ID": X.index, "pred": preds, "proba": probs})


if __name__ == "__main__":
    X = read_test_data()

    model_folder = "logistic_lbfgs_1000_YYYYMMDD_HHMMSS"  # <-- cambia por tu carpeta
    results = run_inference(model_folder, X)
    results_random = run_inference_random(X)

    # Guardar predicciones para subir a Kaggle
    basepath = RESULTS_DIR / model_folder / "predictions.csv"
    basepath.parent.mkdir(exist_ok=True, parents=True)
    results[["ID", "pred"]].to_csv(basepath, index=False)
    print(f"Saved predictions to {basepath}")
