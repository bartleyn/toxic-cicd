from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.app import app, get_predictor


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

    def _fake_get_signal(name):
        if name in ("toxicity", "sentiment", "hatespeech"):
            signal = MagicMock()
            signal.name = name
            return signal
        raise StopIteration

    predictor._get_signal = _fake_get_signal
    return predictor


@pytest.fixture()
def client():
    """Create a TestClient with the predictor dependency overridden."""
    mock_pred = _mock_predictor()
    app.dependency_overrides[get_predictor] = lambda: mock_pred

    with patch("api.app.create_predictor", return_value=mock_pred):
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()


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


def test_explain_returns_200(client):
    with patch("api.app.Explainer") as MockExplainer:
        mock_instance = MockExplainer.return_value
        mock_instance.explain.return_value = MagicMock(
            text="hello",
            signal_name="toxicity",
            score=0.8,
            contributions=[MagicMock(token="hello", weight=0.3)],
        )
        resp = client.post("/explain", json={"text": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["signal_name"] == "toxicity"
        assert len(data["contributions"]) == 1


def test_explain_unknown_signal_returns_400(client):
    resp = client.post("/explain", json={"text": "hello", "signal_name": "nonexistant"})
    assert resp.status_code == 400
