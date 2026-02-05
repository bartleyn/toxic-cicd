

from __future__ import annotations

import time
import os

from typing import Dict, Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import Predictor



class ScoreRequest(BaseModel):
    texts: list[str] = Field(..., description="List of texts to classify")
    threshold: float | None = Field(None, description="Decision threshold for classification (optional)", ge=0.0, le=1.0)

class ItemResult(BaseModel):
    toxicity_score: float = Field(..., description="Predicted probability of toxic class")
    label: int = Field(..., description="Predicted class label (0 or 1)")
    sentiment_score: float = Field(..., description="Sentiment score from -1 (negative) to 1 (positive)")

    
class ScoreResponse(BaseModel):
    model_version: str = Field(..., description="Version of the model used for prediction")
    threshold: float = Field(..., description="Decision threshold used for classification")
    results: list[ItemResult] = Field(..., description="List of prediction results for each input text")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status of the API")
    uptime_seconds: float = Field(..., description="Uptime of the API in seconds")
    model_loaded: bool = Field(..., description="Indicates if the model was loaded successfully")

class InfoResponse(BaseModel):
    artifact_dir: str = Field(..., description="Directory where model artifacts are stored")
    model_version: str = Field(..., description="Version of the loaded model")
    default_threshold: float = Field(..., description="Default decision threshold for classification")

app = FastAPI(title="Toxic Comment Classification API")

_START_TIME = time.time()

def get_artifact_dir() -> str:
    artifact_dir = os.getenv('MODEL_ARTIFACT_DIR', 'artifacts/latest')
    if not os.path.exists(artifact_dir):
        raise RuntimeError(f"Artifact directory '{artifact_dir}' does not exist.")
    return artifact_dir

def create_predictor() -> Predictor:

    return Predictor(artifact_dir=get_artifact_dir())

@app.on_event("startup")
def load_model_on_startup():
    try:
        app.state.predictor = create_predictor()
        app.state.model_loaded = True
    except Exception as e:
        app.state.model_loaded = False
        raise RuntimeError(f"Failed to load model: {e}")


def get_predictor() -> Predictor:
    predictor = getattr(app.state, 'predictor', None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor


@app.get("/health", response_model=HealthResponse)
def health():
    uptime_seconds = time.time() - _START_TIME
    return HealthResponse(
        status="ok",
        uptime_seconds=uptime_seconds,
        model_loaded=bool(getattr(app.state, 'model_loaded', False))
    )

@app.get("/info", response_model=InfoResponse)
def info(predictor: Predictor = Depends(get_predictor)) -> InfoResponse:
    meta = predictor.info()
    return InfoResponse(**meta)


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest, predictor: Predictor = Depends(get_predictor)) -> ScoreResponse:
    try:
        results = predictor.predict(request.texts, threshold=request.threshold)
        return ScoreResponse(
            model_version=results['model_version'],
            threshold=results['threshold'],
            results=[ItemResult(**item) for item in results['results']]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")