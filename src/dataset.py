"""Identical validation/inference preprocessing; augmentation only for training."""
import warnings
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import PREPROCESS, read_json


def transform_for(training: bool = False, config: dict | None = None):
    config = config or PREPROCESS
    steps = [transforms.Resize((config["size"], config["size"]))]
    if training:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
                  transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)]
    return transforms.Compose(steps + [transforms.ToTensor(),
                                      transforms.Normalize(config["mean"], config["std"])])


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


class LeafDataset(Dataset):
    def __init__(self, root: Path, split: str, training: bool = False,
                 config: dict | None = None) -> None:
        self.class_to_idx = read_json(root / "class_to_idx.json")
        self.transform = transform_for(training, config)
        self.samples: list[tuple[Path, int]] = []
        for name, index in self.class_to_idx.items():
            count = 0
            for path in sorted((root / split / name).glob("*")):
                try:
                    open_rgb(path)
                    self.samples.append((path, index))
                    count += 1
                except (OSError, ValueError) as exc:
                    warnings.warn(f"Skipping invalid image {path}: {exc}")
            if not count:
                raise ValueError(f"No valid images in {root / split / name}. Run dataset preparation first.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        return self.transform(open_rgb(path)), label
