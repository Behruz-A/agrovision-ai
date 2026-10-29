"""Shared paths, reproducibility, metadata, and plotting."""
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PREPROCESS = {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def default_checkpoint() -> Path:
    multicrop = ROOT / "models/agrovision_multicrop.pth"
    return multicrop if multicrop.is_file() else ROOT / "models/agrovision_cnn.pth"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_for(name: str = "auto") -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Use --device cpu.")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else
                        "cpu" if name == "auto" else name)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_history(history: list[dict], directory: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    for metric in ("loss", "accuracy"):
        fig, ax = plt.subplots()
        for split in ("train", "val"):
            ax.plot([h["epoch"] for h in history], [h[f"{split}_{metric}"] for h in history], label=split)
        ax.set(xlabel="Epoch", ylabel=metric.title(), title=f"AgroVision {metric}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f"training_{metric}.png", dpi=150)
        plt.close(fig)
