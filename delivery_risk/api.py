"""Local inference API with validation, readiness and in-process telemetry."""
import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from .data import FEATURES

class Order(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    distance_km: float = Field(ge=0, le=50000)
    items: int = Field(ge=1, le=10000)
    promised_hours: float = Field(gt=0, le=8760)
    backlog: int = Field(ge=0, le=1000000)
    carrier: str = Field(min_length=1, max_length=80)
    region: str = Field(min_length=1, max_length=80)

class Batch(BaseModel):
    orders: list[Order] = Field(min_length=1, max_length=100)

def create_app(model_path=None):
    app = FastAPI(title='Delivery Risk', version='1.0.0')
    path = Path(model_path or os.getenv('MODEL_PATH', 'artifacts/model.joblib'))
    # Load only trusted local artifacts: joblib deserialization can execute code.
    bundle = joblib.load(path) if path.exists() else None
    counters = {'requests': 0, 'errors': 0, 'total_latency_ms': 0.0}
    lock = Lock()

    @app.middleware('http')
    async def observe(request, call_next):
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            with lock:
                counters['requests'] += 1
                counters['errors'] += int(status >= 400)
                counters['total_latency_ms'] += elapsed
            logging.getLogger('delivery_risk').info(json.dumps({
                'path': request.url.path, 'status': status, 'latency_ms': round(elapsed, 3)}))

    @app.get('/health')
    def health():
        return {'status': 'alive'}

    @app.get('/ready')
    def ready():
        if bundle is None:
            raise HTTPException(503, 'Train a model and restart the service')
        return {'ready': True, 'model_version': bundle['version'], 'data_provenance': bundle['data_provenance']}

    @app.get('/metrics')
    def telemetry():
        with lock:
            return dict(counters)

    @app.post('/predict')
    def predict(batch: Batch):
        if bundle is None:
            raise HTTPException(503, 'Model unavailable')
        frame = pd.DataFrame([o.model_dump() for o in batch.orders])[FEATURES]
        probabilities = bundle['pipeline'].predict_proba(frame)[:, 1]
        return {'model_version': bundle['version'], 'data_provenance': bundle['data_provenance'],
                'predictions': [{'late_risk': float(p), 'flagged': bool(p >= bundle['threshold'])}
                                for p in probabilities]}

    return app

app = create_app()
