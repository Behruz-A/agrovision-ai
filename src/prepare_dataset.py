"""Download/import PlantVillage RGB images and make reproducible grouped splits."""
import argparse
import hashlib
import random
import re
import shutil
import urllib.parse
import urllib.request
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .dataset import open_rgb
from .utils import ROOT, read_json, write_json

CLASSES = {"Tomato___Early_blight": "Tomato_Early_Blight",
           "Tomato___Late_blight": "Tomato_Late_Blight", "Tomato___healthy": "Tomato_Healthy"}
REPO = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/git/trees/"
RAW = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/"


def fetch(url: str) -> bytes:
    import time
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "AgroVision-AI-dataset-preparation"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("Download failed")


def download(source: Path, limit: int, classes: dict | None = None) -> Path:
    classes = classes or CLASSES
    import json
    def tree(ref):
        data = json.loads(fetch(REPO + ref))
        if data.get("truncated"):
            raise ValueError("Upstream file listing truncated; use a local dataset instead.")
        return data
    root = tree("master")
    revision = root["sha"]
    def child(entries, name):
        return next(item["sha"] for item in entries["tree"] if item["path"] == name)
    color = tree(child(tree(child(root, "raw")), "color"))
    source.mkdir(parents=True, exist_ok=True)
    grouping_path = source / "leaf-map.json"
    grouping_path.write_bytes(fetch(RAW + revision + "/leaf_grouping/leaf-map.json"))
    tasks = []
    for upstream in classes:
        entries = tree(child(color, upstream))["tree"]
        files = sorted([item["path"] for item in entries if item["type"] == "blob"])
        if limit:
            random.Random(42).shuffle(files)
            files = files[:limit]
        folder = source / upstream
        folder.mkdir(parents=True, exist_ok=True)
        for filename in files:
            tasks.append((folder / filename, RAW + revision + "/" + urllib.parse.quote(f"raw/color/{upstream}/{filename}")))
    def save(task):
        path, url = task
        if path.is_file():
            try:
                open_rgb(path)
                return
            except (OSError, ValueError):
                pass
        data = fetch(url)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(data)
        try:
            open_rgb(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Downloading/reusing {len(tasks)} real images (up to 12 concurrent requests)...", flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        for count, _ in enumerate(pool.map(save, tasks), 1):
            if count % 200 == 0:
                print(f"  {count}/{len(tasks)} checked", flush=True)
    write_json(source / "download.json", {"repository": REPO, "revision": revision,
                                         "limit_per_class": limit, "images_requested": len(tasks)})
    return grouping_path


def ensure_empty_splits(output: Path) -> None:
    if any(any(p.is_file() and p.name != ".gitkeep" for p in (output / split).rglob("*"))
           for split in ("train", "val", "test")):
        raise ValueError("Prepared split directories already contain files. Choose a new --output directory; no data was overwritten.")


def prepare(source: Path, output: Path, seed: int, group_map: Path | None = None,
            classes: dict | None = None) -> dict:
    classes = classes or CLASSES
    ensure_empty_splits(output)
    # Accept a repository root, raw root, color root, or folders with canonical names.
    original_source = source
    for candidate in (source / "raw/color", source / "color", source):
        if any((candidate / name).is_dir() for name in classes):
            source = candidate
            break
    if group_map is None:
        group_map = next((p for p in (original_source / "leaf-map.json", original_source / "leaf_grouping/leaf-map.json",
                                      source / "leaf-map.json") if p.is_file()), None)
    groups_by_name = read_json(group_map) if group_map else {}
    # Some upstream map entries retain a conflicted-copy suffix that the RGB files omit.
    normalized_map = defaultdict(list)
    for filename, memberships in groups_by_name.items():
        normalized = re.sub(r" \([^)]*conflicted[^)]*\)", "", filename.lower())
        normalized_map[normalized].extend([memberships] if isinstance(memberships, str) else memberships)
    groups_by_name = normalized_map
    if not groups_by_name:
        warnings.warn("No leaf-map.json found. Exact decoded duplicates are grouped, but same-leaf grouping cannot be guaranteed. Supply --group-map from upstream.")
    records, seen_pixels = [], {}
    parent = {}
    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]
    def union(a, b):
        parent[find(a)] = find(b)
    invalid, duplicates, unknown, filename_grouped = 0, 0, 0, 0
    for upstream, canonical in classes.items():
        directory = source / upstream
        if not directory.is_dir():
            directory = source / canonical
        if not directory.is_dir():
            raise ValueError(f"Missing class folder {upstream} (or {canonical}) in {source}")
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue
            try:
                image = open_rgb(path)
                digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            except (OSError, ValueError) as exc:
                warnings.warn(f"Skipping {path}: {exc}")
                invalid += 1
                continue
            # PlantVillage prepends a UUID followed by ___ to the original leaf filename.
            stem = path.stem.split("___", 1)[-1].lower()
            stem = re.sub(r" \([^)]*conflicted[^)]*\)", "", stem)
            memberships = groups_by_name.get(stem, [])
            if isinstance(memberships, str):
                memberships = [memberships]
            memberships = [str(item) for item in memberships if str(item).startswith(upstream + ":::")]
            group = canonical + ":file:" + stem
            # Greenhouse filenames encode a leaf number, view suffix, and sometimes
            # a capture day. Group all views/days of that explicitly named leaf.
            leaf_match = re.fullmatch(r"(.+?)\s+leaf\s+(\d+)(?:\.\d+)?(?:\s+day\s+\d+)?", stem)
            if leaf_match:
                union(group, canonical + ":named_leaf:" + leaf_match[1] + ":" + leaf_match[2])
                filename_grouped += 1
            for membership in memberships:
                union(group, membership)
            if not memberships:
                unknown += 1
            if digest in seen_pixels:
                previous = records[seen_pixels[digest]]
                if previous["class"] != canonical:
                    raise ValueError(f"Identical image has conflicting labels: {path} and {previous['source']}")
                union(group, previous["group"])
                duplicates += 1
                continue
            seen_pixels[digest] = len(records)
            records.append({"source": str(path.resolve()), "class": canonical, "sha256_pixels": digest,
                            "sha256_file": hashlib.sha256(path.read_bytes()).hexdigest(), "group": group})
    mapping = {name: i for i, name in enumerate(sorted(set(classes.values())))}
    counts = {split: {} for split in ("train", "val", "test")}
    assignments = []
    rng = random.Random(seed)
    for name in mapping:
        grouped = defaultdict(list)
        for record in records:
            if record["class"] == name:
                grouped[find(record["group"])].append(record)
        if len(grouped) < 3:
            raise ValueError(f"Need at least three independent leaf/image groups for {name}.")
        chunks = sorted(grouped.items())
        rng.shuffle(chunks)
        chunks.sort(key=lambda item: -len(item[1]))
        total = sum(len(items) for _, items in chunks)
        targets = {"train": total * .70, "val": total * .15, "test": total * .15}
        sizes = dict.fromkeys(targets, 0)
        for group_id, items in chunks:
            split = max(targets, key=lambda key: (targets[key] - sizes[key]) / targets[key])
            sizes[split] += len(items)
            for record in items:
                target = output / split / name / (record["sha256_file"] + Path(record["source"]).suffix.lower())
                assignments.append({**record, "group": group_id, "split": split,
                                    "destination": str(target.relative_to(output))})
        for split in sizes:
            if sizes[split] == 0:
                raise ValueError(f"Empty {split} for {name}; supply more independent leaves.")
            counts[split][name] = sizes[split]
    # Validate all assignments before writing any split files.
    for record in assignments:
        target = output / record["destination"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["source"], target)
    write_json(output / "class_to_idx.json", mapping)
    manifest = {"seed": seed, "ratios": [.70, .15, .15], "counts": counts,
                "invalid_skipped": invalid, "duplicates_removed": duplicates,
                "images_without_leaf_metadata": unknown, "group_map": str(group_map) if group_map else None,
                "images_with_explicit_leaf_filenames": filename_grouped,
                "download": read_json(original_source / "download.json") if (original_source / "download.json").is_file() else None,
                "images": assignments}
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "dataset/source")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset")
    parser.add_argument("--download", action="store_true", help="Download the three RGB classes from upstream GitHub")
    parser.add_argument("--limit-per-class", type=int, default=0, help="Download only a real-image subset; 0 downloads all")
    parser.add_argument("--group-map", type=Path, help="Upstream leaf-map.json for local imports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.limit_per_class < 0:
        parser.error("Image limit must be nonnegative.")
    ensure_empty_splits(args.output)
    if args.download:
        downloaded_map = download(args.source, args.limit_per_class)
        args.group_map = args.group_map or downloaded_map
    manifest = prepare(args.source, args.output, args.seed, args.group_map)
    print("Prepared split counts:", manifest["counts"])
    print(f"Removed {manifest['duplicates_removed']} exact pixel duplicates; skipped {manifest['invalid_skipped']} invalid images.")
    print(f"Images without leaf metadata: {manifest['images_without_leaf_metadata']}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        raise SystemExit(f"Preparation failed: {exc}\nUse --source PATH with a local PlantVillage color folder if download is unavailable.") from exc
