from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import yaml


def load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote Toxicity Model")

    parser.add_argument("--artifact-dir", type=str, required=True, help="Path to the model to use.")
    parser.add_argument("--config", type=str, required=True, help="Config Yaml file.")
    parser.add_argument(
        "--evaluation-results", type=str, required=True, help="Path to the evaluation results JSON file."
    )

    return parser


def main():

    args = arg_parser().parse_args()
    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
        rules = config.get("rules", {})

    eval_results = load_json(args.evaluation_results)

    RULE_TO_METRIC = {
        "min_auc": ("overall_metrics", "overall_auc"),
        "min_precision": ("binary_at_threshold_metrics", "precision"),
        "min_recall": ("binary_at_threshold_metrics", "recall"),
        "min_f1": ("binary_at_threshold_metrics", "f1_score"),
        "min_eval_samples": ("n_eval",),  # top-level key
        "max_false_positive_rate": ("binary_at_threshold_metrics", "false_positive_rate"),
    }

    def get_metric_value(rule: str) -> float | int:
        metric_path = RULE_TO_METRIC.get(rule)
        if not metric_path:
            raise ValueError(f"No metric mapping found for rule '{rule}'")
        value = eval_results
        for key in metric_path:
            value = value.get(key)
            if value is None:
                raise ValueError(f"Metric path '{'.'.join(metric_path)}' not found in evaluation results.")
        return value

    all_passed = True
    failed_rules = []
    for rule_name, threshold_value in rules.items():
        metric_value = get_metric_value(rule_name)
        if rule_name.startswith("max_"):
            if metric_value > threshold_value:
                all_passed = False
                failed_rules.append((rule_name, threshold_value, metric_value))
        if rule_name.startswith("min_"):
            if metric_value < threshold_value:
                all_passed = False
                failed_rules.append((rule_name, threshold_value, metric_value))

    if not all_passed:
        for rule in failed_rules:
            rule_name, threshold_value, metric_value = rule
            print(f"[PROMOTION FAILED] Rule '{rule_name}' not met: {metric_value} fails against {threshold_value}")
        raise RuntimeError("Model promotion failed due to unmet rules.")
    print("[PROMOTION SUCCESS] All promotion rules met.")
    # all passed -> promote

    target = os.path.abspath(args.artifact_dir)
    link_dir = os.path.dirname(target)
    link_path = os.path.join(link_dir, "latest")

    tmp_link_path = link_path + "_tmp"
    if os.path.exists(tmp_link_path):
        os.remove(tmp_link_path)
    os.symlink(target, tmp_link_path)
    os.replace(tmp_link_path, link_path)
    print(f"Promoted model at '{target}' to '{link_path}'")

    sys.exit(0)
