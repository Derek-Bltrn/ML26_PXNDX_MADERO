# Data management
from datetime import datetime
from pathlib import Path
import joblib

# ML
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE

CURRENT_FILE = Path(__file__).resolve()
MODELS_DIR = CURRENT_FILE.parent / "trained_models"

MODELS_DIR.mkdir(exist_ok=True, parents=True)


class PurchaseModel:
    def __init__(self, classifier="Logistic", solver="lbfgs", max_iter=1000):
        # Hyperparameters
        self.solver = solver
        self.max_iter = max_iter
        self.model_type = classifier

        # Nombre único del run — usado para la carpeta y el logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name = f"{classifier}_{solver}_{max_iter}_{timestamp}"
        self.run_dir = MODELS_DIR / self.name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.classifier = self.get_classifier(classifier)
        self.smote = SMOTE(random_state=42)
        # Pipeline
        self.model = Pipeline(
            [
                (
                    "classifier",
                    self.classifier,
                ),
            ]
        )

    def get_classifier(self, name: str, **args):
        # Return the sklearn class instance for the classifier to use
        name = name.lower()
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def get_config(self):
        return {
            "model": self.model_type,
            "solver": self.solver,
            "max_iter": self.max_iter,
        }

    def save(self):
        """
        Guarda el modelo como model.pkl en self.run_dir.
        """
        filepath = self.run_dir / "model.pkl"
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")
        return filepath

    def load(self, filename: str):
        filepath = Path(MODELS_DIR) / filename
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
