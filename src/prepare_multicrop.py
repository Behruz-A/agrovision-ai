"""Build a real-image dataset for ten vegetables, leaf health, and unsupported inputs."""
import argparse
import hashlib
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .prepare_dataset import download, ensure_empty_splits, prepare
from .utils import ROOT, read_json, write_json

VEGETABLES = {"Tomato": "Tomato", "Potato": "Potato", "Capsicum": "Bell pepper",
              "Cucumber": "Cucumber", "Carrot": "Carrot", "Cabbage": "Cabbage",
              "Cauliflower": "Cauliflower", "Broccoli": "Broccoli", "Brinjal": "Eggplant", "Radish": "Radish"}
LEAVES = {"Tomato___Early_blight": ("Tomato", "Early blight"),
          "Tomato___Late_blight": ("Tomato", "Late blight"),
          "Tomato___healthy": ("Tomato", "Healthy"),
          "Potato___Early_blight": ("Potato", "Early blight"),
          "Potato___Late_blight": ("Potato", "Late blight"),
          "Potato___healthy": ("Potato", "Healthy"),
          "Pepper,_bell___Bacterial_spot": ("Bell pepper", "Bacterial spot"),
          "Pepper,_bell___healthy": ("Bell pepper", "Healthy")}
OTHER_LEAVES = ("Apple___healthy", "Grape___healthy")
ARCHIVE_URL = "https://www.kaggle.com/api/v1/datasets/download/misrakahmed/vegetable-image-dataset?datasetVersionNumber=1"


def acquire_archive(path: Path) -> None:
    if path.is_file() and zipfile.is_zipfile(path):
        print("Reusing cached vegetable archive.", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".zip.part")
    print("Downloading public Vegetable Image Dataset v1 (about 560 MB)...", flush=True)
    request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "AgroVision-AI/2"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if not zipfile.is_zipfile(temporary):
        temporary.unlink(missing_ok=True)
        raise ValueError("Kaggle did not return a ZIP archive. Download the public dataset manually and use --archive PATH.")
    temporary.replace(path)


def build(source: Path, output: Path, cache: Path, archive: Path, limit: int, seed: int, network: bool) -> dict:
    ensure_empty_splits(output)
    # Staged source is reused only when its sampling configuration is identical.
    stage_config = {"samples_per_class": limit, "seed": seed}
    config_path = source / "sampling.json"
    if config_path.exists() and read_json(config_path) != stage_config:
        raise ValueError("Staging folder uses another sample configuration. Choose a different --source.")
    source.mkdir(parents=True, exist_ok=True)
    write_json(config_path, stage_config)
    if network:
        acquire_archive(archive)
        download(cache, limit, {name: name for name in (*LEAVES, *OTHER_LEAVES)})
    if not archive.is_file():
        raise FileNotFoundError(f"Vegetable archive missing: {archive}. Use --download or --archive PATH.")
    classes, catalog = {}, {}
    rng = random.Random(seed)
    with zipfile.ZipFile(archive) as bundle:
        by_class = {}
        for item in bundle.infolist():
            parts = Path(item.filename).parts
            if len(parts) >= 2 and Path(parts[-1]).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                by_class.setdefault(parts[-2], []).append(item)
        missing = set(VEGETABLES) - set(by_class)
        if missing:
            raise ValueError(f"Archive lacks expected classes: {sorted(missing)}")
        other_names = sorted(set(by_class) - set(VEGETABLES))
        other_budget = max(limit // (len(other_names) + len(OTHER_LEAVES)), 3)
        for name, items in sorted(by_class.items()):
            folder_name = "Produce___" + name
            canonical = "Vegetable___" + VEGETABLES[name].replace(" ", "_") if name in VEGETABLES else "Other"
            classes[folder_name] = canonical
            if canonical != "Other":
                catalog[canonical] = {"task": "vegetable", "crop": VEGETABLES[name], "condition": None, "display_name": VEGETABLES[name]}
            choices = sorted(items, key=lambda item: item.filename)
            rng.shuffle(choices)
            folder = source / folder_name
            folder.mkdir(exist_ok=True)
            for item in choices[:limit if name in VEGETABLES else other_budget]:
                # Hash the archive member path so matching basenames in old splits cannot overwrite.
                filename = hashlib.sha256(item.filename.encode()).hexdigest() + Path(item.filename).suffix.lower()
                target = folder / filename
                if not target.exists():
                    target.write_bytes(bundle.read(item))  # CRC checked; never extract untrusted archive paths.
    for upstream in (*LEAVES, *OTHER_LEAVES):
        if upstream in LEAVES:
            crop, condition = LEAVES[upstream]
            canonical = "Leaf___" + crop.replace(" ", "_") + "___" + condition.replace(" ", "_")
            catalog[canonical] = {"task": "leaf", "crop": crop, "condition": condition,
                                  "display_name": f"{crop} leaf - {condition}"}
            count = limit
        else:
            canonical, count = "Other", other_budget
        classes[upstream] = canonical
        folder = source / upstream
        folder.mkdir(exist_ok=True)
        files = sorted(p for p in (cache / upstream).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            raise ValueError(f"Missing PlantVillage source folder: {cache / upstream}. Use --download.")
        rng.shuffle(files)
        for path in files[:count]:
            shutil.copy2(path, folder / path.name)
    catalog["Other"] = {"task": "unknown", "crop": None, "condition": None, "display_name": "Other / unsupported"}
    manifest = prepare(source, output, seed, cache / "leaf-map.json", classes)
    write_json(output / "class_catalog.json", catalog)
    provenance = {"vegetables": {"url": ARCHIVE_URL, "version": 1,
                   "creators": "M. Israk Ahmed, Shahriyar Mahmud Mamun, Asif Uz Zaman Asif", "license": "CC BY-SA 4.0"},
                  "plantvillage": read_json(cache / "download.json") if (cache / "download.json").is_file() else {"local_source": True},
                  "sampling": stage_config, "notes": "Real-image capped subset; not a full dataset. Exact duplicates and known leaf IDs are grouped. Vegetable specimen IDs are unavailable; near-duplicate/background correlations may remain.",
                  "negative_examples": "Five excluded vegetable types plus healthy apple and grape leaves; not comprehensive unknown-object coverage."}
    write_json(output / "provenance.json", provenance)
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "dataset/multicrop-source")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset/multicrop")
    parser.add_argument("--cache", type=Path, default=ROOT / "dataset/source")
    parser.add_argument("--archive", type=Path, default=ROOT / "dataset/vegetables-v1.zip")
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    if args.samples_per_class < 30:
        parser.error("Use at least 30 real source images per class.")
    result = build(args.source, args.output, args.cache, args.archive, args.samples_per_class, args.seed, args.download)
    print(f"Prepared {len(result['images'])} unique images across {len(result['counts']['train'])} classes.", flush=True)
    print(result["counts"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Multicrop preparation failed: {exc}") from exc
