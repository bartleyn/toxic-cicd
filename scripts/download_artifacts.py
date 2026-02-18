"""Download model artifacts from GCS before serving.

Run as: python scripts/download_artifacts.py

Required env vars:
    GCS_ARTIFACT_BUCKET  – bucket name (no gs:// prefix)
    MODEL_ARTIFACT_DIR   – local dir to write artifacts into (default: artifacts/latest)

Optional env vars:
    MODEL_VERSION        – version to download (default: "latest", which reads the LATEST pointer)
"""

from __future__ import annotations

import os

from google.cloud import storage


def resolve_version(bucket: storage.Bucket, model_type: str, version: str) -> str:
    """If version is 'latest', read the LATEST pointer file; otherwise pass through."""
    if version.lower() != "latest":
        return version

    pointer_path = f"models/{model_type}/LATEST"
    blob = bucket.blob(pointer_path)
    if not blob.exists():
        raise FileNotFoundError(f"LATEST pointer not found at gs://{bucket.name}/{pointer_path}")

    resolved = blob.download_as_text().strip()
    print(f"Resolved LATEST -> {resolved}")
    return resolved


def download_model(bucket: storage.Bucket, model_type: str, version: str, dest_dir: str) -> int:
    """Download all blobs under models/{model_type}/{version}/ into dest_dir/{model_type}/."""
    prefix = f"models/{model_type}/{version}/{model_type}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        raise FileNotFoundError(f"No artifacts found at gs://{bucket.name}/{prefix}")

    local_model_dir = os.path.join(dest_dir, model_type)
    os.makedirs(local_model_dir, exist_ok=True)

    for blob in blobs:
        # blob.name example: models/toxicity/1.0.1/toxicity/model.joblib
        filename = blob.name.removeprefix(prefix)
        if not filename:
            continue
        local_path = os.path.join(local_model_dir, filename)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"  {blob.name} -> {local_path}")

    return len(blobs)


def main() -> None:
    bucket_name = os.environ.get("GCS_ARTIFACT_BUCKET")
    if not bucket_name:
        print("GCS_ARTIFACT_BUCKET not set — skipping artifact download")
        return

    version = os.environ.get("MODEL_VERSION", "latest")
    dest_dir = os.environ.get("MODEL_ARTIFACT_DIR", "artifacts/latest")

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)

    version = resolve_version(bucket, "toxicity", version)
    print(f"Downloading toxicity model v{version} to {dest_dir}/")
    count = download_model(bucket, "toxicity", version, dest_dir)
    print(f"Downloaded {count} artifact(s)")


if __name__ == "__main__":
    main()
