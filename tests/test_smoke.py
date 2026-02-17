from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.app import app


def _mock_predictor():
    predictor = MagicMock()
    predictor.info.return_value = {
        "artifact_dir": "fake/dir",
        "model_version": "1.0.0",
        "default_threshold": 0.5,
    }
    predictor.predict.return_value = {
        "model_version": "1.0.0",
        "threshold": 0.5,
        "results": [],
    }
    return predictor


@pytest.fixture()
def client():
    """Create a TestClient with a mocked predictor injected before startup."""
    with patch("api.app.create_predictor", return_value=_mock_predictor()):
        app.state.model_loaded = True
        app.state.predictor = _mock_predictor()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_info_returns_model_version(client):
    resp = client.get("/info")
    assert resp.status_code == 200
    assert resp.json()["model_version"] == "1.0.0"


def test_score_returns_200(client):
    resp = client.post("/score", json={"texts": ["hello"]})
    assert resp.status_code == 200


def test_score_empty_texts_returns_422(client):
    resp = client.post("/score", json={"texts": []})
    assert resp.status_code == 422
