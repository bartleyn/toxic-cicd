from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.features import TextFeatureExtractor
from src.signals.toxicity import ModelMetadata, ModelSpec, ToxicityModel
from src.utils import get_git_sha, load_dataset_csv

"""
Model training
"""


def train_model(
    data: pd.DataFrame, text_col: str, label_col: str, spec: ModelSpec, test_size: float = 0.2, random_state: int = 42
) -> tuple[ToxicityModel, TextFeatureExtractor, float]:

    X_texts = data[text_col].tolist()
    y = data[label_col].to_numpy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_texts, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    fe = TextFeatureExtractor()

    X_train_vec = fe.fit_transform(X_train)
    X_valid_vec = fe.transform(X_valid)

    # Initialize the model

    model = ToxicityModel(spec=spec)

    # Train the model
    model.fit(X_train_vec, y_train)

    # Get some scores
    validation_scores = model.score(X_valid_vec)
    auc = roc_auc_score(y_valid, validation_scores) if len(np.unique(y_valid)) > 1 else float("nan")

    metrics = {
        "validation_auc": auc,
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "positive_class_rate_train": float(np.mean(y_train)),
        "positive_class_rate_valid": float(np.mean(y_valid)),
    }

    return fe, model, metrics


def save_model_artifacts(
    artifact_dir: str,
    model_version: str,
    fe: TextFeatureExtractor,
    model: ToxicityModel,
    metadata: ModelMetadata,
    train_metrics: dict,
    extra_metadata: dict,
    eval_df: pd.DataFrame,
    decision_threshold: float = 0.5,
) -> None:
    """
    Saves in artifact_dir:
        model.joblib
        spec.json
        metadata.json
        vectorizer.joblib
        train_metrics.json
        eval.csv
        run.json
    """

    artifact_dir = os.path.join(artifact_dir, model_version)
    artifact_dir = os.path.join(artifact_dir, "toxicity")
    os.makedirs(artifact_dir, exist_ok=True)

    model.metadata = metadata

    model.save(artifact_dir)

    joblib.dump(fe, os.path.join(artifact_dir, "vectorizer.joblib"))

    eval_df.to_csv(os.path.join(artifact_dir, "eval.csv"), index=False)

    with open(os.path.join(artifact_dir, "train_metrics.json"), "w") as f:
        json.dump(train_metrics, f, indent=2, sort_keys=True)

    run_info = {
        "model_version": model_version,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision_threshold": decision_threshold,
        "extra_metadata": extra_metadata,
    }

    with open(os.path.join(artifact_dir, "run.json"), "w") as f:
        json.dump(run_info, f, indent=2, sort_keys=True)
    return None


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Toxicity Model")

    parser.add_argument("--data-path", type=str, required=True, help="Path to the CSV dataset file.")
    parser.add_argument("--text-col", type=str, default="text", help="Name of the text column in the dataset.")
    parser.add_argument("--label-col", type=str, default="label", help="Name of the label column in the dataset.")
    parser.add_argument("--artifact-dir", type=str, required=True, help="Directory to save model artifacts.")
    parser.add_argument("--model-version", type=str, required=True, help="Version identifier for the model.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data to use for validation.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Decision threshold for classification.")
    parser.add_argument(
        "--num-rows", type=int, default=None, help="Number of rows to read from the dataset CSV file (for debugging)."
    )

    parser.add_argument(
        "--C", type=float, default=2.0, help="Inverse of regularization strength for Logistic Regression."
    )
    parser.add_argument(
        "--max-iter", type=int, default=1000, help="Maximum number of iterations for Logistic Regression."
    )
    parser.add_argument(
        "--class-weight", type=str, default=None, help='Class weight for Logistic Regression (e.g., "balanced").'
    )
    return parser


def main() -> None:
    args = arg_parser().parse_args()

    spec = ModelSpec(C=args.C, max_iter=args.max_iter, class_weight=args.class_weight, random_state=args.random_state)

    model_version = args.model_version.strip() or get_git_sha()

    df = load_dataset_csv(args.data_path, text_col=args.text_col, label_col=args.label_col, num_rows=args.num_rows)

    y = df[args.label_col].to_numpy()
    train_df, eval_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    fe, model, train_metrics = train_model(
        data=train_df,
        text_col=args.text_col,
        label_col=args.label_col,
        spec=spec,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    extra_metadata = {
        "data_path": args.data_path,
        "text_col": args.text_col,
        "label_col": args.label_col,
        "spec": asdict(spec),
    }

    metadata = ModelMetadata(
        model_version=model_version,
        decision_threshold=args.decision_threshold,
    )

    save_model_artifacts(
        artifact_dir=args.artifact_dir,
        model_version=model_version,
        fe=fe,
        model=model,
        metadata=metadata,
        train_metrics=train_metrics,
        extra_metadata=extra_metadata,
        eval_df=eval_df,
        decision_threshold=args.decision_threshold,
    )

    print(f"[train] wrote artifacts to {args.artifact_dir}")
    print(f"[train] training metrics: {json.dumps(train_metrics, indent=2, sort_keys=True)}")


if __name__ == "__main__":
    main()
