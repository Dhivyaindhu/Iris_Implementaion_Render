# Iris Classifier API

A small FastAPI service that classifies iris flowers, built in the VistaTech AI/ML course.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service info and links |
| GET | `/health` | Health check (used by Render) |
| GET | `/classes` | Class names |
| GET | `/model-info` | Where the model came from (file or trained at startup) |
| POST | `/predict` | Body: `{"features": [sepal_len, sepal_wid, petal_len, petal_wid]}` |
| GET | `/docs` | Interactive API page |

## Run locally

```
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
python train.py
pytest
python run.py
```

Open http://127.0.0.1:8000/docs

## Deploy on Render

* Build command: `pip install -r requirements.txt && python train.py`
* Start command: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
* Environment variable: `PYTHON_VERSION = 3.11.9`

Or create it from `render.yaml` with New + > Blueprint.

## Check the live service

```
python smoke_test.py https://YOUR-SERVICE.onrender.com
```
