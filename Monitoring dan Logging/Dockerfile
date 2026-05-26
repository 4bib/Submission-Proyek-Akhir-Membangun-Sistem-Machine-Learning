FROM python:3.12-slim

WORKDIR /app

COPY 7.inference.py /app/7.inference.py
COPY mlflow_model /app/mlflow_model

RUN pip install --no-cache-dir flask werkzeug prometheus_client mlflow scikit-learn pandas numpy

EXPOSE 5001

CMD ["python", "7.inference.py"]
