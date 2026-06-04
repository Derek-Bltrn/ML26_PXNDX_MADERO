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

        if name in ["logistic", "logreg", "lr"]:
            return LogisticRegression(
                solver=self.solver,
                max_iter=self.max_iter,
                class_weight=None,
                random_state=42,
            )

        if name in ["rf", "random_forest", "randomforest"]:
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight=None,
                random_state=42,
                n_jobs=-1,
            )
        
        if name in ["xgb", "xgboost"]:
            return xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.05,
                subsample=1,
                colsample_bytree=1,
                objective="binary:logistic",
                eval_metric="auc",
                random_state=42,
                n_jobs=-1,
            )
        raise ValueError(f"Unknown classifier: {name}")
    

    def fit(self, X, y):
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        self.model.fit(X_resampled, y_resampled)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_config(self):
        return {
            "model": self.model_type,
            "params": self.classifier.get_params(),
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
