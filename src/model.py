"""Train, save, load and use a small scikit-learn model."""
import os

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.preprocess import validate_features

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def train_model(random_state=42):
    """Train on Iris and return (model, test_accuracy)."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model, model.score(X_test, y_test)


def save_model(model, path):
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def load_or_train(path):
    """Load the saved model if the file exists, otherwise train a fresh one.

    Returns (model, source) where source is "file" or "trained_at_startup".
    """
    if path and os.path.exists(path):
        return load_model(path), "file"
    model, _ = train_model()
    return model, "trained_at_startup"


def predict_one(model, features):
    """Predict the class name for one flower. Returns (label, confidence)."""
    X = validate_features(features)
    probs = model.predict_proba(X)[0]
    idx = int(probs.argmax())
    return CLASS_NAMES[idx], float(probs[idx])
