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
    return predictor


@pytest.fixture()
def client():
    """Create a TestClient with GCS mocked out."""
    mock_pred = _mock_predictor()
    app.dependency_overrides[get_predictor] = lambda: mock_pred

    with (
        patch("api.app.create_predictor", return_value=mock_pred),
        patch("api.app.save_label_to_gcs"),
        patch("api.app.GCS_LABELS_BUCKET", "test-bucket"),
    ):
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()


def _make_label_payload(**overrides):
    base = {
        "uri": "at://did:plc:abc123/app.bsky.feed.post/xyz",
        "text": "some toxic comment",
        "author_handle": "user.bsky.social",
        "created_at": "2026-02-22T12:00:00Z",
        "feed_uri": "at://did:plc:feed/app.bsky.feed.generator/toxic",
        "feed_name": "toxic-feed",
        "toxicity_score": 0.85,
        "toxicity_label": 1,
        "sentiment_score": -0.6,
        "hatespeech_score": 0.2,
        "corrected_toxicity_label": 0,
        "corrected_hatespeech_label": 0,
        "tags": ["false_positive"],
        "labeled_at": "2026-02-22T13:00:00Z",
    }
    base.update(overrides)
    return base


# --- Basic endpoint tests ---


def test_labels_returns_201(client):
    resp = client.post("/labels", json=_make_label_payload())
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "saved"
    assert data["uri"] == "at://did:plc:abc123/app.bsky.feed.post/xyz"


def test_labels_missing_required_field_returns_422(client):
    payload = _make_label_payload()
    del payload["text"]
    resp = client.post("/labels", json=payload)
    assert resp.status_code == 422


def test_labels_empty_tags_ok(client):
    resp = client.post("/labels", json=_make_label_payload(tags=[]))
    assert resp.status_code == 201


# --- GCS save tests ---


def test_labels_calls_gcs_with_payload(client):
    with patch("api.app.save_label_to_gcs") as mock_save:
        resp = client.post("/labels", json=_make_label_payload())
        assert resp.status_code == 201
        mock_save.assert_called_once()
        payload = mock_save.call_args[0][0]
        assert payload["uri"] == "at://did:plc:abc123/app.bsky.feed.post/xyz"
        assert payload["corrected_toxicity_label"] == 0
        assert payload["tags"] == ["false_positive"]


def test_labels_returns_503_when_bucket_not_configured(client):
    with patch("api.app.save_label_to_gcs", side_effect=RuntimeError("GCS_LABELS_BUCKET is not configured")):
        resp = client.post("/labels", json=_make_label_payload())
        assert resp.status_code == 503


def test_labels_returns_500_on_gcs_failure(client):
    with patch("api.app.save_label_to_gcs", side_effect=Exception("GCS upload failed")):
        resp = client.post("/labels", json=_make_label_payload())
        assert resp.status_code == 500


# --- save_label_to_gcs unit tests ---


def test_save_label_to_gcs_uploads_blob():
    from api.app import save_label_to_gcs

    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    with (
        patch("api.app._get_gcs_client", return_value=mock_client),
        patch("api.app.GCS_LABELS_BUCKET", "my-bucket"),
        patch("api.app.GCS_LABELS_PREFIX", "labels/"),
    ):
        save_label_to_gcs({"uri": "at://post/1", "text": "hello"})

    mock_client.bucket.assert_called_once_with("my-bucket")
    mock_blob.upload_from_string.assert_called_once()
    call_kwargs = mock_blob.upload_from_string.call_args
    assert "at://post/1" in call_kwargs[0][0]
    assert call_kwargs[1]["content_type"] == "application/json"


def test_save_label_to_gcs_raises_when_bucket_empty():
    from api.app import save_label_to_gcs

    with (
        patch("api.app.GCS_LABELS_BUCKET", ""),
        pytest.raises(RuntimeError, match="GCS_LABELS_BUCKET is not configured"),
    ):
        save_label_to_gcs({"uri": "test"})
