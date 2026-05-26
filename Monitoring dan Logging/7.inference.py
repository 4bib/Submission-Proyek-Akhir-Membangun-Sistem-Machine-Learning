import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from prometheus_client import make_wsgi_app
from prometheus_client import Histogram, Counter, Gauge
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__)

PREDICTION_LATENCY = Histogram(
    'mlflow_inference_latency_seconds',
    'Latency of inference requests in seconds'
)
PREDICTION_COUNT = Counter(
    'mlflow_inference_requests_total',
    'Total number of inference requests'
)
PREDICTION_ERRORS = Counter(
    'mlflow_inference_errors_total',
    'Total number of inference errors'
)
LAST_OUTPUT = Gauge(
    'mlflow_inference_last_output',
    'Last prediction output value'
)

MODEL_URI = None


def load_model():
    global MODEL_URI
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / 'mlflow_model'
    if model_dir.exists():
        MODEL_URI = str(model_dir)
    else:
        MODEL_URI = 'models:/California_Housing_Project/Production_Model'
    return mlflow.sklearn.load_model(MODEL_URI)


MODEL = load_model()


@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    PREDICTION_COUNT.inc()
    try:
        payload = request.get_json(force=True)
        features = np.array(payload['features'], dtype=float)
        predictions = MODEL.predict(features)
        result = predictions.tolist()
        LAST_OUTPUT.set(float(result[0]))
        duration = time.time() - start
        PREDICTION_LATENCY.observe(duration)
        return jsonify({'predictions': result})
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        return jsonify({'error': str(exc)}), 400


if __name__ == '__main__':
    prometheus_app = make_wsgi_app()
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {'/metrics': prometheus_app})
    app.run(host='0.0.0.0', port=5001)
