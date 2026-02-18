"""Upload trained model artifacts to GCS.

Run as: python scripts/upload_artifacts.py --version 1.1.0

Required env vars:
    GCS_ARTIFACT_BUCKET  – bucket name (no gs:// prefix)

Optional flags:
    --artifact-dir       – local artifacts root (default: artifacts)
    --version            – model version to upload (required)
    --model-type         – upload a single model type instead of all found
    --set-latest         – update the LATEST pointer for each uploaded model type
"""

from __future__ import annotations

import argparse
import os

from google.cloud import storage


def upload_model(
    bucket: storage.Bucket, artifact_dir: str, model_type: str, version: str
) -> int:
    """Upload all files under artifact_dir/{version}/{model_type}/ to GCS."""
    local_dir = os.path.join(artifact_dir, version, model_type)
    if not os.path.isdir(local_dir):
        return 0

    count = 0
    for root, _dirs, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative = os.path.relpath(local_path, local_dir)
            blob_path = f"models/{model_type}/{version}/{model_type}/{relative}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f"  {local_path} -> gs://{bucket.name}/{blob_path}")
            count += 1

    return count


def set_latest_pointer(bucket: storage.Bucket, model_type: str, version: str) -> None:
    """Write a LATEST pointer file for the given model type."""
    pointer_path = f"models/{model_type}/LATEST"
    blob = bucket.blob(pointer_path)
    blob.upload_from_string(version)
    print(f"  Set gs://{bucket.name}/{pointer_path} -> {version}")


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload model artifacts to GCS")
    parser.add_argument("--version", type=str, required=True, help="Model version to upload.")
    parser.add_argument(
        "--artifact-dir", type=str, default="artifacts", help="Local artifacts root directory."
    )
    parser.add_argument(
        "--model-type", type=str, default=None,
        help="Upload only this model type. If omitted, uploads all model types found.",
    )
    parser.add_argument(
        "--set-latest", action="store_true",
        help="Update the LATEST pointer for each uploaded model type.",
    )
    return parser


def main() -> None:
    args = arg_parser().parse_args()

    bucket_name = os.environ.get("GCS_ARTIFACT_BUCKET")
    if not bucket_name:
        raise RuntimeError("GCS_ARTIFACT_BUCKET env var is required")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    version_dir = os.path.join(args.artifact_dir, args.version)
    if not os.path.isdir(version_dir):
        raise FileNotFoundError(f"No artifacts found at {version_dir}")

    if args.model_type:
        model_types = [args.model_type]
    else:
        model_types = [
            d for d in os.listdir(version_dir)
            if os.path.isdir(os.path.join(version_dir, d))
        ]

    if not model_types:
        raise FileNotFoundError(f"No model subdirectories found in {version_dir}")

    for model_type in sorted(model_types):
        print(f"Uploading {model_type} v{args.version}")
        count = upload_model(bucket, args.artifact_dir, model_type, args.version)
        if count:
            print(f"  Uploaded {count} file(s)")
            if args.set_latest:
                set_latest_pointer(bucket, model_type, args.version)
        else:
            print(f"  No files found for {model_type} — skipping")


if __name__ == "__main__":
    main()
