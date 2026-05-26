import time
from prometheus_client import Counter, Gauge, Histogram, start_http_server

prediction_counter = Counter(
    'mlflow_prediction_requests_total',
    'Total number of prediction requests processed by the MLflow inference service'
)
request_latency = Histogram(
    'mlflow_prediction_request_latency_seconds',
    'Latency of prediction requests in seconds'
)
model_error_counter = Counter(
    'mlflow_prediction_errors_total',
    'Number of prediction requests that returned an error'
)
output_value = Gauge(
    'mlflow_prediction_output_value',
    'Last predicted output value from the MLflow inference service'
)


def start_metrics_server(port: int = 8000):
    start_http_server(port)
    print(f"Prometheus exporter running on http://localhost:{port}/metrics")


def observe_prediction(duration: float, output: float, error: bool = False):
    prediction_counter.inc()
    request_latency.observe(duration)
    output_value.set(output)
    if error:
        model_error_counter.inc()


if __name__ == '__main__':
    start_metrics_server(8000)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('Exporter stopped.')
