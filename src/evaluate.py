"""Evaluate a checkpoint on the held-out test split."""
import argparse
import os
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from .dataset import LeafDataset
from .predict import Predictor
from .utils import ROOT, default_checkpoint, plot_history, read_json, write_json


def plot_confusion(matrix, names: list[str], directory: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(9, len(names) * .8), max(7, len(names) * .7)))
    labels = [name.replace(" leaf - ", " leaf\n") for name in names]
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=ax, xticks_rotation=75 if len(names) > 8 else 20, colorbar=False,
        values_format="d", text_kw={"fontsize": 7 if len(names) > 8 else 10})
    if len(names) > 8:
        ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("Batch size must be positive.")
    predictor = Predictor(args.checkpoint, args.device)
    multicrop = "class_catalog" in predictor.metadata
    health = predictor.metadata.get("assessment_type") == "leaf_health"
    args.data = args.data or ROOT / ("dataset/leaf-health" if health else "dataset/multicrop" if multicrop else "dataset")
    args.output = args.output or ROOT / ("outputs/leaf-health" if health else "outputs/multicrop" if multicrop else "outputs")
    dataset = LeafDataset(args.data, "test", config=predictor.metadata["preprocessing"])
    if dataset.class_to_idx != predictor.metadata["class_to_idx"]:
        raise ValueError("Test dataset class mapping differs from checkpoint.")
    truth, predictions, probabilities = [], [], []
    with torch.inference_mode():
        for images, labels in DataLoader(dataset, batch_size=args.batch_size):
            logits = predictor.model(images.to(predictor.device))
            predictions.extend(logits.argmax(1).cpu().tolist())
            probabilities.extend(logits.softmax(1).cpu().tolist())
            truth.extend(labels.tolist())
    names = [predictor.catalog[name]["display_name"] for name in predictor.classes]
    indices = list(range(len(names)))
    report = classification_report(truth, predictions, labels=indices, target_names=names, output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predictions, labels=indices)
    result = {"accuracy": accuracy_score(truth, predictions), "classification_report": report,
              "confusion_matrix": matrix.tolist(), "test_samples": len(truth),
              "checkpoint": str(args.checkpoint), "smoke_test_model": predictor.metadata.get("smoke_test", False)}
    task_metrics = {}
    for task in ("vegetable", "leaf", "unknown"):
        indices_for_task = [i for i, label in enumerate(truth) if predictor.catalog[predictor.classes[label]]["task"] == task]
        if not indices_for_task:
            continue
        outcomes = {i: predictor.describe(probabilities[i], None if task == "unknown" else task) for i in indices_for_task}
        accepted = [i for i in indices_for_task if outcomes[i]["status"] == "identified"]
        task_metrics[task] = {
            "samples": len(indices_for_task),
            "raw_class_accuracy": sum(truth[i] == predictions[i] for i in indices_for_task) / len(indices_for_task),
            "accepted_count": len(accepted), "acceptance_rate": len(accepted) / len(indices_for_task),
            "accepted_class_accuracy": sum(truth[i] == predictions[i] for i in accepted) / len(accepted) if accepted else None,
            "rejected_or_uncertain_count": len(indices_for_task) - len(accepted),
        }
    result["task_metrics"] = task_metrics
    result["policy_note"] = ("Leaf-health estimates use uncalibrated class probabilities, not damaged-area percentages. Non-leaf examples are limited to real vegetable photographs."
                             if health else "Task routing and abstention use uncalibrated heuristic thresholds. Unsupported examples are limited to the prepared Other class.")
    write_json(args.output / "evaluation.json", result)
    print(classification_report(truth, predictions, labels=indices, target_names=names, zero_division=0))
    for task, metrics in task_metrics.items():
        print(f"{task}: raw class accuracy {metrics['raw_class_accuracy']:.2%}; accepted {metrics['accepted_count']}/{metrics['samples']}")
    if result["smoke_test_model"]:
        print("Smoke-test checkpoint: these metrics do not represent a fully trained model.")
    directory = args.output / "charts"
    plot_confusion(matrix, names, directory)
    history = args.checkpoint.with_suffix(".history.json")
    if history.is_file():
        plot_history(read_json(history), directory)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from exc
