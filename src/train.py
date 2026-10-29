"""Train the custom CNN: python -m src.train --help."""
import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .dataset import LeafDataset
from .model import AgroVisionCNN
from .utils import ROOT, PREPROCESS, device_for, plot_history, read_json, seed_everything, write_json


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    model.train(optimizer is not None)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            count += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
    return total_loss / count, correct / count


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = ROOT / "dataset/multicrop" if (ROOT / "dataset/multicrop/class_catalog.json").is_file() else ROOT / "dataset"
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to the checkpoint for the selected dataset type")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--architecture", choices=["v1", "v2"], help="Default: v2 for multicrop, v1 for legacy tomato")
    parser.add_argument("--charts", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="Per split sample limit for a smoke test; 0 uses all")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    multicrop = (args.data / "class_catalog.json").is_file()
    provenance_path = args.data / "provenance.json"
    data_provenance = read_json(provenance_path) if provenance_path.is_file() else {}
    health = data_provenance.get("assessment_type") == "leaf_health"
    args.checkpoint = args.checkpoint or ROOT / ("models/agrovision_leaf_health.pth" if health else "models/agrovision_multicrop.pth" if multicrop else "models/agrovision_cnn.pth")
    args.architecture = args.architecture or ("v2" if multicrop else "v1")
    args.charts = args.charts or ROOT / ("outputs/leaf-health/charts" if health else "outputs/multicrop/charts" if multicrop else "outputs/charts")
    if min(args.epochs, args.batch_size, args.patience, args.threads) < 1 or args.lr <= 0 or args.max_samples < 0 or args.workers < 0:
        parser.error("Epochs, batch size, patience, threads and lr must be positive; sample limit/workers nonnegative.")
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = device_for(args.device)
    train = LeafDataset(args.data, "train", True)
    val = LeafDataset(args.data, "val")
    mapping = train.class_to_idx
    if len(mapping) < 2 or sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Class mapping requires at least two contiguous class indices.")
    if args.max_samples:
        generator = torch.Generator().manual_seed(args.seed)
        train = Subset(train, torch.randperm(len(train), generator=generator)[:args.max_samples].tolist())
        val = Subset(val, torch.randperm(len(val), generator=generator)[:args.max_samples].tolist())
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, generator=generator)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers)
    model = AgroVisionCNN(len(mapping), first_stride=2 if args.architecture == "v2" else 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    best, stale, history = float("inf"), 0, []
    metadata = {"architecture": f"AgroVisionCNN-{args.architecture}", "class_to_idx": mapping,
                "preprocessing": PREPROCESS, "seed": args.seed,
                "smoke_test": bool(args.max_samples), "training_samples": len(train),
                "validation_samples": len(val),
                "training_config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    catalog_path = args.data / "class_catalog.json"
    if catalog_path.is_file():
        metadata["class_catalog"] = read_json(catalog_path)
        if set(metadata["class_catalog"]) != set(mapping):
            raise ValueError("Class catalog does not match the training class mapping.")
        metadata["confidence_threshold"] = 0.65
        metadata["margin_threshold"] = 0.15
        metadata["threshold_policy"] = "Conservative heuristic, not calibrated on unknown objects"
    provenance = args.data / "provenance.json"
    if provenance.is_file():
        metadata["dataset_provenance"] = read_json(provenance)
        if metadata["dataset_provenance"].get("assessment_type"):
            metadata["assessment_type"] = metadata["dataset_provenance"]["assessment_type"]
    print(f"Device: {device}; train={len(train)} val={len(val)}; epochs <= {args.epochs}", flush=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        tl, ta = run_epoch(model, train_loader, criterion, device, optimizer)
        vl, va = run_epoch(model, val_loader, criterion, device)
        scheduler.step(vl)
        row = dict(epoch=epoch, train_loss=tl, train_accuracy=ta, val_loss=vl, val_accuracy=va,
                   seconds=time.perf_counter() - started)
        history.append(row)
        print(f"Epoch {epoch}: loss {tl:.4f}/{vl:.4f} accuracy {ta:.3%}/{va:.3%} ({row['seconds']:.1f}s)", flush=True)
        if vl < best:
            best, stale = vl, 0
            temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
            torch.save({"state_dict": model.state_dict(), "metadata": metadata, "epoch": epoch,
                        "val_loss": vl}, temporary)
            temporary.replace(args.checkpoint)
            write_json(args.checkpoint.with_suffix(".json"), metadata)
        else:
            stale += 1
        write_json(args.checkpoint.with_suffix(".history.json"), history)
        if stale >= args.patience:
            print("Early stopping.", flush=True)
            break
    plot_history(history, args.charts)
    print(f"Best checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Training failed: {exc}") from exc
