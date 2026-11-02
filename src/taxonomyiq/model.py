"""Actual trainable scikit-learn baselines, calibration, and safe inference adapter."""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def make_model(kind: str = "hybrid"):
    if kind == "word":
        features = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    elif kind == "hybrid":
        features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2))
        ])
    else:
        raise ValueError(f"unsupported feature type: {kind}")
    base = Pipeline([("features", features), ("clf", LogisticRegression(max_iter=500, random_state=41))])
    return CalibratedClassifierCV(base, cv=3, method="sigmoid")

class Predictor:
    def __init__(self, model, metadata: dict):
        self.model = model
        self.metadata = metadata

    def predict(self, texts: list[str], threshold: float = 0.7):
        scores = self.model.predict_proba(texts)
        classes = list(self.model.classes_)
        records = []
        for probabilities in scores:
            order = np.argsort(probabilities)[::-1]
            best = int(order[0]); confidence = float(probabilities[best])
            records.append({
                "category": classes[best], "confidence": round(confidence, 6),
                "needs_review": bool(confidence < threshold),
                "review_reason": "low_confidence" if confidence < threshold else None,
                "alternatives": [{"category": classes[int(i)],
                                  "confidence": round(float(probabilities[i]), 6)}
                                 for i in order[:3]],
                "model_version": self.metadata.get("model_version", "unversioned")
            })
        return records


def save_model(model, directory: Path, metadata: dict):
    directory.mkdir(parents=True, exist_ok=True)
    # WARNING: joblib uses pickle. Load artifacts only from trusted local builds.
    joblib.dump(model, directory / "model.joblib", compress=3)
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_model(directory: Path) -> Predictor:
    metadata = json.loads((directory / "metadata.json").read_text())
    model = joblib.load(directory / "model.joblib")
    return Predictor(model, metadata)
