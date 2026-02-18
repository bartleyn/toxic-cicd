#!/bin/sh
set -e

echo "Downloading model artifacts..."
uv run python scripts/download_artifacts.py

echo "Starting API server..."
exec uv run uvicorn api.app:app --host 0.0.0.0 --port 8080
