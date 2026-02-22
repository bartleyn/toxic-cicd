from __future__ import annotations

import os
import time

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.explain import Explainer
from src.predict import Predictor
from src.schemas import ItemResult


class ScoreRequest(BaseModel):
    texts: list[str] = Field(..., description="List of texts to classify")
    threshold: float | None = Field(
        None, description="Decision threshold for classification (optional)", ge=0.0, le=1.0
    )

    @field_validator("texts")
    @classmethod
    def texts_must_be_nonempty(cls, v):
        if len(v) == 0:
            raise ValueError("Input text list is empty")
        for i, t in enumerate(v):
            if len(t.strip()) == 0:
                raise ValueError(f"Text at index {i} must not be empty or whitespace only")
        return v


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

class ContributionItem(BaseModel):
    token: str = Field(..., description ="Input Token")
    weight: float = Field(..., description="SHAP attribution value")

class ExplainRequest(BaseModel):
    text: str = Field(..., description="Single text to explain")
    signal_name: str = Field("toxicity", description="Name of the signal to explain")
    top_n: int = Field(10, description="Number of top token contributions to return", ge=1, le=50)

class ExplainResponse(BaseModel):
    text: str = Field(..., description="Original input text")
    signal_name: str = Field(..., description="Signal that was explained")
    score: float = Field(..., description="Model score for this text")
    contributions: list[ContributionItem] = Field(..., description="top token contribution")




app = FastAPI(title="Toxic Comment Classification API")

_START_TIME = time.time()


def get_artifact_dir() -> str:
    artifact_dir = os.getenv("MODEL_ARTIFACT_DIR", "artifacts/latest")
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
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor


@app.get("/health", response_model=HealthResponse)
def health():
    uptime_seconds = time.time() - _START_TIME
    return HealthResponse(
        status="ok", uptime_seconds=uptime_seconds, model_loaded=bool(getattr(app.state, "model_loaded", False))
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
            model_version=results["model_version"], threshold=results["threshold"], results=results["results"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest, predictor: Predictor = Depends(get_predictor)) -> ExplainResponse:
    try:
        signal = predictor._get_signal(request.signal_name)
    except StopIteration:
        raise HTTPException(
            status_code=400,
            detail=f'Unknown signal: "{request.signal_name}" Available: {[m.name for m in predictor.models]}'
        )

    try:
        explainer = Explainer(signal)
        result = explainer.explain(request.text, top_n=request.top_n)
        return ExplainResponse(
            text=result.text,
            signal_name=result.signal_name,
            score=result.score,
            contributions=[ContributionItem(token=c.token, weight=c.weight) for c in result.contributions],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Explanation failed: {e}')

