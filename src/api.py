"""Iris classifier API (FastAPI)."""
import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model import CLASS_NAMES, load_or_train, predict_one
from src.preprocess import FEATURE_COUNT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iris-api")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/iris_model.joblib")

app = FastAPI(title="Iris Classifier API", version="1.0.0")
_model, _source = load_or_train(MODEL_PATH)
logger.info("Model ready (source=%s)", _source)


class IrisRequest(BaseModel):
    features: list[float]


@app.get("/")
def root():
    return {
        "service": "Iris Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/classes")
def classes():
    return {"classes": CLASS_NAMES}


@app.get("/model-info")
def model_info():
    return {"model_source": _source, "classes": CLASS_NAMES, "feature_count": FEATURE_COUNT}


@app.post("/predict")
def predict(req: IrisRequest):
    try:
        label, confidence = predict_one(_model, req.features)
    except ValueError as exc:
        logger.warning("Rejected input: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    return {"label": label, "confidence": round(confidence, 4)}
