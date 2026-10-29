"""Prepare a species-agnostic healthy/diseased/not-a-leaf training set."""
import argparse
import hashlib
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

from .prepare_dataset import REPO, download, ensure_empty_splits, fetch, prepare
from .prepare_multicrop import acquire_archive
from .utils import ROOT, read_json, write_json


def list_leaf_classes() -> list[str]:
    def tree(ref):
        result = json.loads(fetch(REPO + ref))
        if result.get("truncated"):
            raise ValueError("Upstream directory listing was truncated.")
        return result
    def child(result, name):
        return next(item["sha"] for item in result["tree"] if item["path"] == name)
    root = tree("master")
    raw = tree(child(root, "raw"))
    color = tree(child(raw, "color"))
    return sorted(item["path"] for item in color["tree"] if item["type"] == "tree" and "___" in item["path"])


def build(source: Path, output: Path, cache: Path, archive: Path, per_source: int, per_state: int, seed: int, network: bool) -> dict:
    ensure_empty_splits(output)
    source.mkdir(parents=True, exist_ok=True)
    settings = {"per_source_class": per_source, "per_health_state": per_state, "seed": seed}
    config = source / "settings.json"
    if config.exists() and read_json(config) != settings:
        raise ValueError("Staging settings differ. Choose a new --source directory.")
    write_json(config, settings)
    if network:
        names = list_leaf_classes()
        write_json(source / "leaf_classes.json", names)
        download(cache, per_source, {name: name for name in names})
        acquire_archive(archive)
    elif (source / "leaf_classes.json").exists():
        names = read_json(source / "leaf_classes.json")
    else:
        names = sorted(p.name for p in cache.iterdir() if p.is_dir() and "___" in p.name)
    if not names or not archive.is_file():
        raise ValueError("Real leaf classes or the vegetable archive are missing. Use --download or supply the cache/archive.")
    rng = random.Random(seed)
    candidates = defaultdict(list)
    classes = {}
    for name in names:
        state = "Healthy" if name.split("___", 1)[1].lower() == "healthy" else "Diseased"
        classes[name] = state
        (source / name).mkdir(exist_ok=True)
        files = sorted(p for p in (cache / name).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            raise ValueError(f"Missing real leaf images: {cache / name}")
        rng.shuffle(files)
        candidates[state].extend((name, path) for path in files[:per_source])
    for state in ("Healthy", "Diseased"):
        rng.shuffle(candidates[state])
        for name, path in candidates[state][:per_state]:
            shutil.copy2(path, source / name / path.name)
    with zipfile.ZipFile(archive) as bundle:
        items = sorted((item for item in bundle.infolist() if Path(item.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}), key=lambda item: item.filename)
        rng.shuffle(items)
        folder = source / "Non_leaf_objects"
        folder.mkdir(exist_ok=True)
        classes[folder.name] = "Not_leaf"
        for item in items[:per_state]:
            name = hashlib.sha256(item.filename.encode()).hexdigest() + Path(item.filename).suffix.lower()
            (folder / name).write_bytes(bundle.read(item))
    manifest = prepare(source, output, seed, cache / "leaf-map.json", classes)
    catalog = {
        "Healthy": {"task": "leaf", "crop": None, "condition": "Healthy", "health_state": "healthy", "display_name": "Healthy leaf"},
        "Diseased": {"task": "leaf", "crop": None, "condition": "Diseased", "health_state": "diseased", "display_name": "Diseased leaf"},
        "Not_leaf": {"task": "unknown", "crop": None, "condition": None, "health_state": "not_leaf", "display_name": "Not a leaf"}}
    represented = sorted({Path(record["source"]).parent.name for record in manifest["images"] if record["class"] != "Not_leaf"})
    provenance = {"assessment_type": "leaf_health", "sampling": settings,
        "source_leaf_classes": represented, "plant_groups": sorted({name.split("___")[0] for name in represented}),
        "plantvillage": read_json(cache / "download.json") if (cache / "download.json").exists() else {"local_cache": True},
        "non_leaf_examples": "Real vegetable photographs from the previously attributed Vegetable Image Dataset v1; not comprehensive everyday-object negatives.",
        "limitations": "Species-agnostic prediction is trained on represented PlantVillage crops, not every tree or wild plant. Percentages estimate class likelihood, not damaged leaf area or a diagnosis."}
    write_json(output / "class_catalog.json", catalog)
    write_json(output / "provenance.json", provenance)
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "dataset/leaf-health-source")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset/leaf-health")
    parser.add_argument("--cache", type=Path, default=ROOT / "dataset/source")
    parser.add_argument("--archive", type=Path, default=ROOT / "dataset/vegetables-v1.zip")
    parser.add_argument("--per-source-class", type=int, default=80)
    parser.add_argument("--per-health-state", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    if min(args.per_source_class, args.per_health_state) < 10:
        parser.error("Use at least ten real images per source class/state.")
    result = build(args.source, args.output, args.cache, args.archive, args.per_source_class, args.per_health_state, args.seed, args.download)
    print(f"Prepared {len(result['images'])} real health-assessment images. {result['counts']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        raise SystemExit(f"Leaf-health preparation failed: {exc}") from exc
