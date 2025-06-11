#!/bin/bash

# Chạy MLflow server local (chỉ dùng khi KHÔNG chạy bằng docker-compose)
export MLFLOW_BACKEND_STORE_URI="sqlite:///mlflow.db"
export MLFLOW_ARTIFACT_ROOT="./mlruns"

mlflow server \
  --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
  --default-artifact-root ${MLFLOW_ARTIFACT_ROOT} \
  --host 0.0.0.0 \
  --port 5000
