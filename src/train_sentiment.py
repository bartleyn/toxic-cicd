

from __future__ import annotations

import argparse
import datetime
import os
from attr import asdict
import json

import joblib
import subprocess

from pytz import timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from typing import Tuple


from .features import validate_texts, normalize_texts, TextFeatureExtractor
from signals.sentiment import SentimentModel, SentimentModelMetadata, SentimentModelSpec
from signals.vader_labels import vader_compound
from .utils import load_dataset_csv


def get_git_sha() -> str:
    '''
    Returns the current git commit SHA
    '''
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        sha = 'unknown'
    return sha

def build_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train a sentiment analysis model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the training dataset CSV file')
    parser.add_argument('--text_col', type=str, default='text', help='Name of the text column in the dataset')
    parser.add_argument('--label_col', type=str, default='label', help='Name of the label column in the dataset')
    parser.add_argument('--num_rows', type=int, default=None, help='Number of rows to read from the dataset (for quick testing)')
    parser.add_argument('--artifact_dir', type=str, required=True, help='Directory to save the trained model artifacts')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to use as test set')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for train/test split and model training')
    parser.add_argument('--model_version', type=str, default=None, help='Version identifier for the model (defaults to git SHA)')

    return parser

def train_model(data, text_col: str, label_col: str, spec: SentimentModelSpec, test_size: float, random_state: int) -> Tuple[SentimentModel, TextFeatureExtractor, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        data[text_col].tolist(),
        data[label_col].tolist(),
        test_size=test_size,
        random_state=random_state    )
    
    fe = TextFeatureExtractor()
    X_train_vec = fe.fit_transform(X_train)
    X_test_vec = fe.transform(X_test)

    model = SentimentModel(spec=spec)
    model.fit(X_train_vec, y_train)

    scores = model.predict(X_test_vec)
    mae = mean_absolute_error(y_test, scores)
    print(f"Validation MAE: {mae:.4f}")

    metrics = {
        'model_version': model.metadata.model_version if model.metadata else 'unknown',
        'mae': mae,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    return fe, model, metrics

def save_model_artifacts(model: SentimentModel, fe: TextFeatureExtractor, 
                         metadata: SentimentModelMetadata, 
                         artifact_dir: str, 
                         train_metrics: dict,
                         extra_metadata: dict) -> None:
    os.makedirs(artifact_dir, exist_ok=True)
    
    model.metadata = metadata
    model.save(artifact_dir=artifact_dir)


    joblib.dump(fe, os.path.join(artifact_dir, 'vectorizer.joblib'))
    metadata_path = os.path.join(artifact_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(asdict(metadata), f, indent=2, sort_keys=True)

    with open(os.path.join(artifact_dir, 'train_metrics.json'), 'w') as f:
        json.dump(asdict(train_metrics), f, indent=2, sort_keys=True)

    run_info = {
        'model_version': model.metadata.model_version if model.metadata else 'unknown',
        'created_at_utc': datetime.now(timezone.utc).isoformat(timespec="seconds"),
        'extra_metadata': extra_metadata
    }
        
    with open(os.path.join(artifact_dir, 'run.json'), 'w') as f:
        json.dump(run_info, f, indent=2, sort_keys=True)
    return None
   

def main() -> None:
    args = build_args_parser().parse_args()

    model_version = args.model_version.strip() if args.model_version else get_git_sha()

    input_df = load_dataset_csv(args.dataset_path, text_col=args.text_col, label_col=args.label_col, num_rows=args.num_rows)

    texts = input_df[args.text_col].tolist()
    validate_texts(texts)
    normalized_texts = normalize_texts(texts)
    labels = vader_compound(texts)

    fe = TextFeatureExtractor()
    X = fe.fit_transform(normalized_texts)
    spec = SentimentModelSpec(random_state=args.random_state)
    metadata = SentimentModelMetadata(model_version=model_version)

    fe, model, train_metrics = train_model(
        data=input_df,
        text_col=args.text_col,
        label_col=args.label_col,
        spec=spec,
        test_size=args.test_size,
        random_state=args.random_state
    )

    extra_metadata = {
        'data_path': args.data_path,
        'text_col': args.text_col,
        'label_col': args.label_col,
        'spec': asdict(spec)
    }

    save_model_artifacts(
            model=model,
            fe=fe,
            metadata=metadata,
            artifact_dir=os.path.join(args.artifact_dir, model_version),
            train_metrics=train_metrics,
            extra_metadata=extra_metadata
        )

    print(f"[train] wrote artifacts to {args.artifact_dir}")
    print(f"[train] training metrics: {json.dumps(train_metrics, indent=2, sort_keys=True)}")


if __name__ == '__main__':
    main()

    
