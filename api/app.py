from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
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
    token: str = Field(..., description="Input Token")
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


class LabeledPost(BaseModel):
    uri: str = Field(..., description="Post URI")
    text: str = Field(..., description="Post text content")
    author_handle: str = Field(..., description="Author handle")
    created_at: str = Field(..., description="Post creation timestamp")
    feed_uri: str = Field(..., description="Feed URI")
    feed_name: str = Field(..., description="Feed name")
    toxicity_score: float = Field(..., description="Model toxicity score")
    toxicity_label: int = Field(..., description="Model toxicity label (0 or 1)")
    sentiment_score: float = Field(..., description="Sentiment score")
    hatespeech_score: float = Field(..., description="Model hate speech score")
    corrected_toxicity_label: int = Field(..., description="Human-corrected toxicity label (0 or 1)")
    corrected_hatespeech_label: int = Field(..., description="Human-corrected hate speech label (0 or 1)")
    tags: list[str] = Field(default_factory=list, description="Optional tags")
    labeled_at: str = Field(..., description="Timestamp when label was applied")


class LabelResponse(BaseModel):
    status: str = Field(..., description="Result status")
    uri: str = Field(..., description="URI of the labeled post")


LABELS_DIR = Path(os.getenv("LABELS_DIR", "data/labels"))
GCS_LABELS_BUCKET = os.getenv("GCS_LABELS_BUCKET", "")
GCS_LABELS_PREFIX = os.getenv("GCS_LABELS_PREFIX", "labels/")
FLUSH_THRESHOLD = int(os.getenv("FLUSH_THRESHOLD", "50"))

_flush_lock = threading.Lock()
logger = logging.getLogger(__name__)


def _get_gcs_client():
    from google.cloud import storage

    return storage.Client()


def flush_to_gcs(filepath: Path) -> None:
    """Upload a local JSONL file to GCS and remove the local copy."""
    if not GCS_LABELS_BUCKET:
        logger.debug("GCS_LABELS_BUCKET not set, skipping flush")
        return

    if not filepath.exists() or filepath.stat().st_size == 0:
        return

    with _flush_lock:
        # Re-check after acquiring lock (file may have been flushed by another task)
        if not filepath.exists() or filepath.stat().st_size == 0:
            return

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        blob_name = f"{GCS_LABELS_PREFIX}{filepath.stem}_{timestamp}.jsonl"

        client = _get_gcs_client()
        bucket = client.bucket(GCS_LABELS_BUCKET)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(filepath))
        filepath.unlink()
        logger.info("Flushed %s to gs://%s/%s", filepath.name, GCS_LABELS_BUCKET, blob_name)


def flush_all_labels() -> None:
    """Flush all local JSONL label files to GCS."""
    if not LABELS_DIR.exists():
        return
    for jsonl_file in LABELS_DIR.glob("labels_*.jsonl"):
        flush_to_gcs(jsonl_file)


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


@app.on_event("shutdown")
def flush_labels_on_shutdown():
    flush_all_labels()


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
            detail=f'Unknown signal: "{request.signal_name}" Available: {[m.name for m in predictor.models]}',
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
        raise HTTPException(status_code=500, detail=f"Explanation failed: {e}")


@app.post("/labels", response_model=LabelResponse, status_code=201)
def submit_label(labeled_post: LabeledPost, background_tasks: BackgroundTasks):
    try:
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        filepath = LABELS_DIR / f"labels_{date_str}.jsonl"
        payload = labeled_post.model_dump()
        with open(filepath, "a") as f:
            f.write(json.dumps(payload) + "\n")

        line_count = sum(1 for _ in open(filepath))
        if line_count >= FLUSH_THRESHOLD:
            background_tasks.add_task(flush_to_gcs, filepath)

        return LabelResponse(status="saved", uri=labeled_post.uri)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save label: {e}")
