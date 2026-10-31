# Delivery Risk

AI-assisted Python portfolio prototype for predicting late deliveries. Includes chronological training/validation/test splits, dummy/logistic/random-forest comparison, validation-based model selection, error reports, and a FastAPI inference service.

## Run

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m delivery_risk.data
python -m delivery_risk.train --csv data/demo.csv --provenance synthetic-demo
python -m pytest -q
uvicorn delivery_risk.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs to try predictions. Health, readiness, and in-process request metrics are at /health, /ready, and /metrics. Restart after changing the model artifact. Load only trusted joblib files.

## Data and evaluation

The default dataset is SIMULATED. Its results do not demonstrate real delivery prediction performance. Supply a CSV containing order_id, ordered_at, distance_km, items, promised_hours, backlog, carrier, region, late. All features must be available at order time; late is the eventual binary label. Record the real dataset source and license with --provenance.

Data is split chronologically 60/20/20 by distinct timestamps. Preprocessing is fitted on training data only. Model selection uses validation average precision; the classification threshold is fixed at 0.5. The selected model is evaluated on the final partition without refitting. Review artifacts/report.json and artifacts/errors.csv. Probabilities have not been calibrated.

## Limitations

Local prototype; no AWS deployment or production traffic has been tested. Metrics reset on restart and are per process. No authentication, drift detection, or production load testing. Add a real, licensed dataset and validate performance before making resume claims.

## Attribution

Implementation generated with Codex assistance; no tutorial repository was copied. Dependencies remain under their respective licenses. Understand and extend the implementation before describing personal contributions.
