from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from src.signals.base import BaseSignal
from src.signals.hatespeech import HateSpeechModel
from src.signals.toxicity import ToxicityModel
from src.utils import load_dataset_csv, validate_texts

MODEL_REGISTRY: dict[str, type[BaseSignal]] = {
    "toxicity": ToxicityModel,
    "hatespeech": HateSpeechModel,
}


def compute_binary_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """
    Compute binary classification metrics given true labels and predicted probabilities.

    :param y_true: True binary labels
    :type y_true: np.ndarray
    :param y_proba: Predicted probabilities for the positive class
    :type y_proba: np.ndarray
    :param threshold: Decision threshold to convert probabilities to binary predictions
    :type threshold: float
    :return: Dictionary containing AUC, precision, recall, and F1-score
    :rtype: Dict[str, float]
    """
    y_pred = (y_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    return {
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
    }


def compute_overall_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_score)
    else:
        auc = float("nan")

    avg_precision = average_precision_score(y_true, y_score)

    return {"overall_auc": auc, "overall_average_precision": avg_precision}


def evaluate(
    artifact_dir: str,
    eval_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    threshold: float,
    model_cls: type[BaseSignal] = ToxicityModel,
) -> dict:

    model = model_cls.load(artifact_dir)

    texts = eval_df[text_col].tolist()
    y_true = eval_df[label_col].to_numpy()

    y_proba = model.score(texts)

    overall = compute_overall_metrics(y_true, y_proba)
    binary = compute_binary_metrics(y_true, y_proba, threshold=threshold)

    result_dict = {
        "model_version": model.metadata.model_version if model.metadata else "unknown",
        "artifact_dir": artifact_dir,
        "n_eval": int(len(eval_df)),
        "overall_metrics": overall,
        "binary_at_threshold_metrics": binary,
    }

    return result_dict


def write_metrics(metrics: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--model-type", type=str, choices=list(MODEL_REGISTRY.keys()), default="toxicity",
        help="Type of model to evaluate.",
    )
    parser.add_argument("--artifact-dir", type=str, required=True, help="Path to the model artifact directory")
    parser.add_argument("--eval-data-path", type=str, required=True, help="Path to the evaluation dataset CSV file")
    parser.add_argument("--text-col", type=str, default="text", help="Name of the text column in the dataset")
    parser.add_argument("--label-col", type=str, default="label", help="Name of the label column in the dataset")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold for binary classification")
    parser.add_argument(
        "--out-filename", type=str, default="metrics.json", help="Output path for the evaluation metrics JSON file"
    )
    return parser


def main():

    args = build_arg_parser().parse_args()

    df = load_dataset_csv(path=args.eval_data_path, text_col=args.text_col, label_col=args.label_col)

    validate_texts(df[args.text_col].to_list())

    model_cls = MODEL_REGISTRY[args.model_type]
    model = model_cls.load(args.artifact_dir)
    model_threshold = model.metadata.decision_threshold if model.metadata else 0.5
    threshold = args.threshold if args.threshold is not None else model_threshold

    metrics = evaluate(
        artifact_dir=args.artifact_dir,
        eval_df=df,
        text_col=args.text_col,
        label_col=args.label_col,
        threshold=threshold,
        model_cls=model_cls,
    )

    write_metrics(metrics, args.out_filename)
    print(f"[evaluate] Metrics written to {args.out_filename}")
    print(f"[evaluate] Metrics: {json.dumps(metrics, indent=2, sort_keys=True)}")


if __name__ == "__main__":
    main()
