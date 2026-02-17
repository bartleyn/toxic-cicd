from unittest.mock import MagicMock, patch

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


@patch("api.app.create_predictor")
def test_health_returns_ok(mock_create):
    mock_create.return_value = _mock_predictor()
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@patch("api.app.create_predictor")
def test_info_returns_model_version(mock_create):
    mock_create.return_value = _mock_predictor()
    client = TestClient(app)
    resp = client.get("/info")
    assert resp.status_code == 200
    assert resp.json()["model_version"] == "1.0.0"


@patch("api.app.create_predictor")
def test_score_returns_200(mock_create):
    mock_pred = _mock_predictor()
    mock_create.return_value = mock_pred
    client = TestClient(app)
    resp = client.post("/score", json={"texts": ["hello"]})
    assert resp.status_code == 200
    mock_pred.predict.assert_called_once()


@patch("api.app.create_predictor")
def test_score_empty_texts_returns_422(mock_create):
    mock_create.return_value = _mock_predictor()
    client = TestClient(app)
    resp = client.post("/score", json={"texts": []})
    assert resp.status_code == 422
