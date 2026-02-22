from __future__ import annotations

import json
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
def client(tmp_path):
    """Create a TestClient with labels dir pointed at tmp_path."""
    mock_pred = _mock_predictor()
    app.dependency_overrides[get_predictor] = lambda: mock_pred

    with (
        patch("api.app.create_predictor", return_value=mock_pred),
        patch("api.app.LABELS_DIR", tmp_path / "labels"),
        patch("api.app.GCS_LABELS_BUCKET", ""),
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


def test_labels_creates_jsonl_file(client, tmp_path):
    with patch("api.app.LABELS_DIR", tmp_path / "labels"):
        client.post("/labels", json=_make_label_payload())

        labels_dir = tmp_path / "labels"
        jsonl_files = list(labels_dir.glob("labels_*.jsonl"))
        assert len(jsonl_files) == 1

        lines = jsonl_files[0].read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["uri"] == "at://did:plc:abc123/app.bsky.feed.post/xyz"
        assert record["corrected_toxicity_label"] == 0
        assert record["tags"] == ["false_positive"]


def test_labels_appends_multiple(client, tmp_path):
    with patch("api.app.LABELS_DIR", tmp_path / "labels"):
        client.post("/labels", json=_make_label_payload(uri="at://post/1"))
        client.post("/labels", json=_make_label_payload(uri="at://post/2"))

        labels_dir = tmp_path / "labels"
        jsonl_files = list(labels_dir.glob("labels_*.jsonl"))
        assert len(jsonl_files) == 1

        lines = jsonl_files[0].read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["uri"] == "at://post/1"
        assert json.loads(lines[1])["uri"] == "at://post/2"


def test_labels_missing_required_field_returns_422(client):
    payload = _make_label_payload()
    del payload["text"]
    resp = client.post("/labels", json=payload)
    assert resp.status_code == 422


def test_labels_empty_tags_ok(client):
    resp = client.post("/labels", json=_make_label_payload(tags=[]))
    assert resp.status_code == 201


# --- GCS flush tests ---


def test_flush_triggered_at_threshold(tmp_path):
    """When line count hits FLUSH_THRESHOLD, flush_to_gcs is scheduled."""
    mock_pred = _mock_predictor()
    app.dependency_overrides[get_predictor] = lambda: mock_pred
    labels_dir = tmp_path / "labels"

    with (
        patch("api.app.create_predictor", return_value=mock_pred),
        patch("api.app.LABELS_DIR", labels_dir),
        patch("api.app.GCS_LABELS_BUCKET", "my-bucket"),
        patch("api.app.FLUSH_THRESHOLD", 3),
        patch("api.app.flush_to_gcs") as mock_flush,
    ):
        with TestClient(app) as c:
            # First 2 posts: below threshold
            c.post("/labels", json=_make_label_payload(uri="at://post/1"))
            c.post("/labels", json=_make_label_payload(uri="at://post/2"))
            assert mock_flush.call_count == 0

            # 3rd post: hits threshold, flush scheduled
            c.post("/labels", json=_make_label_payload(uri="at://post/3"))
            assert mock_flush.call_count == 1

    app.dependency_overrides.clear()


def test_no_flush_when_bucket_not_set(tmp_path):
    """flush_to_gcs is a no-op when GCS_LABELS_BUCKET is empty."""
    from api.app import flush_to_gcs

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    filepath = labels_dir / "labels_2026-02-22.jsonl"
    filepath.write_text('{"uri": "test"}\n')

    with patch("api.app.GCS_LABELS_BUCKET", ""):
        flush_to_gcs(filepath)

    # File should still exist (not uploaded, not deleted)
    assert filepath.exists()


def test_flush_uploads_and_removes_local_file(tmp_path):
    """flush_to_gcs uploads to GCS and removes the local file."""
    from api.app import flush_to_gcs

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    filepath = labels_dir / "labels_2026-02-22.jsonl"
    filepath.write_text('{"uri": "test"}\n')

    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    with (
        patch("api.app.GCS_LABELS_BUCKET", "my-bucket"),
        patch("api.app.GCS_LABELS_PREFIX", "labels/"),
        patch("api.app._get_gcs_client", return_value=mock_client),
    ):
        flush_to_gcs(filepath)

    mock_client.bucket.assert_called_once_with("my-bucket")
    mock_blob.upload_from_filename.assert_called_once_with(str(filepath))
    assert not filepath.exists()


def test_shutdown_flushes_all_files(tmp_path):
    """Shutdown hook calls flush_to_gcs for every JSONL file."""
    from api.app import flush_all_labels

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "labels_2026-02-20.jsonl").write_text('{"uri": "a"}\n')
    (labels_dir / "labels_2026-02-21.jsonl").write_text('{"uri": "b"}\n')

    with (
        patch("api.app.LABELS_DIR", labels_dir),
        patch("api.app.flush_to_gcs") as mock_flush,
    ):
        flush_all_labels()

    assert mock_flush.call_count == 2
    flushed_names = {call.args[0].name for call in mock_flush.call_args_list}
    assert flushed_names == {"labels_2026-02-20.jsonl", "labels_2026-02-21.jsonl"}
