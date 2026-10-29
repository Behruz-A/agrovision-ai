# AgroVision AI

A local PyTorch desktop application with two distinct jobs: identify ten everyday vegetables, or estimate whether a leaf looks healthy or diseased without identifying its species. Images stay on your computer. Separate custom CNNs are trained from scratch; no pretrained classifier is used.

## Supported inputs

| Mode | Coverage |
|---|---|
| Vegetable identification | Tomato, potato, bell pepper, cucumber, carrot, cabbage, cauliflower, broccoli, eggplant, radish |
| General leaf health | Healthy leaf, Diseased leaf, Not a leaf; trained with examples from 14 PlantVillage crop groups |
| Unsupported examples | The vegetable model has an Other class; the health model uses real vegetable photos as Not a leaf examples |

These are ten selected common vegetables, not a statistical ranking of the world's ten most popular vegetables. Identification concerns the vegetable itself; it does not establish freshness or disease. General leaf assessment does not name the plant or a specific disease, and has not been validated on every tree or wild plant.

The vegetable checkpoint retains its 19-class multicrop training. The Leaf health tab now routes to a separate three-class checkpoint rather than requiring a supported crop name. It always displays Healthy leaf, Diseased leaf, and Not a leaf estimates. These sum to 100% across the three model outcomes; they are **not the percentage of leaf area damaged**, and they do not certify health. Low confidence or a small top-two margin gives an uncertain assessment. Unknown objects and unfamiliar plants can still produce confident errors.

## General leaf assessment

```powershell
python -m src.prepare_leaf_health --download
python -m src.train --data dataset/leaf-health --epochs 10 --patience 4
python -m src.evaluate --checkpoint models/agrovision_leaf_health.pth
python main.py --task leaf
```

Data preparation is already completed in this workspace and refuses to overwrite populated splits. It uses up to 80 real images from each of PlantVillage's 38 source label groups, samples 800 healthy and 800 diseased examples, and adds 800 real vegetable photos as non-leaf examples. The 2,400-image balanced subset contains 1,680 training, 360 validation, and 360 test images, with leaf grouping and duplicate isolation. Training data spans 14 crop groups, including apple, cherry, grape, peach, corn, and others; this is broader than the former three-crop disease mode, but not all possible leaves.

The health checkpoint, embedded health-state mapping, and training history are saved as `models/agrovision_leaf_health.*`. Charts and measured evaluation are in `outputs/leaf-health/`. The workbench loads this model for the Leaf health tab and keeps vegetable identification on its existing checkpoint. Missing health weights produce a training-required message instead of quietly substituting tomato disease classification.

The display uses a fixed order: Healthy leaf (green), Diseased leaf (amber), Not a leaf (gray). It shows all three raw model probabilities even if the assessment is uncertain. No disease severity or damaged-area measurement is implemented. A Not a leaf prediction is learned from vegetable photographs, not comprehensive everyday-object negatives. Test-set results and limitations are in VERIFICATION.md.

## Run the desktop app

The existing `.venv` is reused. No new packages were installed. Tkinter ships with standard Windows Python.

```powershell
.\.venv\Scripts\python.exe main.py
```

Use **Vegetable identification** for a vegetable photo and **Leaf health** for a leaf photo. Then click **Upload image** or the preview area. The interface is entirely English.

- Preview the selected image and its dimensions.
- See the vegetable name, or a general leaf-health assessment without a crop name.
- Review the top three matches, or open **View all matches** for the full probability table.
- **Save report** exports JSON including the task, outcome status, probabilities, warnings, and image filename.
- **Clear result** resets the image and results without deleting saved files. It also discards late results from an earlier analysis.
- Switch modes to analyze the same image under the other task.
- **Model details** lists the coverage of the checkpoint actually loaded.
- Keyboard shortcuts: Ctrl+O to upload, Ctrl+S to save, Escape to clear.

Inference runs in a worker thread, leaving the window responsive. Images can be replaced or cleared during analysis. The results panel scrolls while Save report and Clear result stay visible. Error states allow another image to be selected.

The app prefers `models/agrovision_multicrop.pth` when present. Otherwise it opens the legacy tomato-only checkpoint and explicitly reports that limited coverage. To select a checkpoint manually:

```powershell
.\.venv\Scripts\python.exe main.py --checkpoint models/agrovision_multicrop.pth
```

Close and reopen the app after training/replacing a checkpoint; a running window keeps its loaded weights.

## Environment and technologies

Python 3.11+ where compatible dependency wheels exist; PyTorch, torchvision, OpenCV, NumPy, Pillow, Matplotlib, scikit-learn, and standard-library Tkinter. The current environment was verified with Python 3.13.5 and CPU-only PyTorch 2.15.0.

Optionally activate the existing environment to use the shorter commands below:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

If activation is blocked by execution policy, use `.\.venv\Scripts\python.exe` instead of `python`. For a new checkout only, install dependencies into your chosen environment with `python -m pip install -r requirements.txt`. No environment creation or dependency reinstallation is required in this workspace.

## Architecture

```text
main.py                         # Desktop, image CLI, or webcam entry point
src/
  model.py                      # Custom CNN; v1 legacy and v2 CPU-efficient stride
  dataset.py                    # Real image loading and preprocessing
  prepare_dataset.py            # Shared grouped splits; original tomato preparation
  prepare_multicrop.py           # Reproducible combined real-image dataset
  prepare_leaf_health.py         # Broad healthy/diseased/not-a-leaf data
  train.py                      # Adam, scheduling, early stopping, checkpointing
  predict.py                    # Metadata-driven inference and task routing
  evaluate.py                   # Held-out metrics and charts
  gui.py                        # English Tkinter workbench
  utils.py                      # Paths, seeds, JSON, plotting
  __init__.py
dataset/
  multicrop/                    # New train/val/test, mapping, catalog, provenance
  multicrop-source/             # Sampled, original-image staging folders
  source/                       # PlantVillage download cache
  leaf-health/                  # General health train/val/test, mapping, provenance
  vegetables-v1.zip             # Pinned public vegetable archive
  train/ val/ test/              # Preserved legacy tomato split
models/
  agrovision_multicrop.pth       # New best checkpoint
  agrovision_multicrop.json      # Readable metadata
  agrovision_multicrop.history.json
  agrovision_leaf_health.*       # General three-state health model and history
  agrovision_cnn.*               # Preserved legacy tomato artifacts
outputs/
  multicrop/evaluation.json
  multicrop/charts/
  leaf-health/                  # General health evaluation and charts
  predictions/
tests/
  test_project.py
  test_multicrop.py
  test_gui.py
  test_leaf_health.py
requirements.txt
.gitignore
README.md
VERIFICATION.md                  # Historical run and update verification
```

The v2 custom CNN accepts 224 x 224 RGB images, downsamples in its first convolution, and uses four convolution/batch-normalization/ReLU/max-pooling blocks, adaptive pooling, dropout, and a fully connected classifier. Its output size comes from the dataset mapping; it is not fixed to three classes. Legacy v1 checkpoints retain their original stride and remain loadable. Softmax is applied only for inference.

## Data preparation

The multicrop dataset combines two sources:

1. [Vegetable Image Dataset v1](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset), by M. Israk Ahmed, Shahriyar Mahmud Mamun, and Asif Uz Zaman Asif. Original 15-class dataset; CC BY-SA 4.0. We select ten classes for recognition and use examples from the remaining five as unsupported inputs. Dataset paper: *DCNN-Based Vegetable Image Classification Using Transfer Learning: A Comparative Study*, ICCCSP 2021.
2. [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset), including its [leaf-group metadata](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/leaf_grouping). Tomato, potato, and bell-pepper leaf states supply leaf-health examples; healthy apple/grape leaves add unsupported examples. Follow upstream attribution and licensing terms when redistributing data.

Prepare the default 300-source-images-per-supported-class subset:

```powershell
python -m src.prepare_multicrop --download
```

This downloads the public vegetable ZIP (about 560 MB) and needed PlantVillage files, reuses cached data, then builds approximately 70/15/15 splits. No credentials were needed in the executed download. It never fabricates training images. The initial dataset contains 5,543 unique real images; this is a capped subset, not the full datasets. Potato healthy has fewer available source images, so it remains smaller than the other supported classes.

Preparation refuses to overwrite populated train/val/test directories. It is already complete in this workspace; skip preparation and open the app or train. To change the cap/seed, choose new staging/output paths:

```powershell
python -m src.prepare_multicrop --download --samples-per-class 600 --source dataset/multicrop-source-600 --output dataset/multicrop-600
```

For offline preparation, provide the original vegetable ZIP with `--archive PATH` and an existing PlantVillage cache with `--cache PATH`; omit `--download`. The cache needs the selected PlantVillage class folders and `leaf-map.json`. If a public download becomes unavailable, download the source data manually and use these options.

The preparation pipeline:

- Records pinned vegetable archive version, PlantVillage content revision, source paths, seed, sampling cap, class catalog, and per-image hashes/assignments.
- Groups known PlantVillage leaf IDs and explicit `Leaf N` filename variants across views/days.
- Removes exact decoded-RGB duplicates, joins their related groups, and refuses conflicting labels.
- Skips invalid images with a warning and keeps class indices stable and contiguous.
- Keeps all known groups and duplicate images within one split.

Vegetable source images lack specimen/session IDs. Exact duplicates cannot cross splits, but different views of an unrecorded specimen and near-duplicate/background correlations can still remain. This limitation means an independent field dataset is required to establish real-world generalization. Train transforms alone apply horizontal flip, small rotation, and color jitter. Validation, test, and inference use the identical deterministic resize/normalization stored in the checkpoint.

## Train from scratch

```powershell
python -m src.train
```

When multicrop data is prepared, defaults select it, the v2 architecture, `models/agrovision_multicrop.pth`, and `outputs/multicrop/charts/`. Defaults are at most 25 epochs, batch size 32, Adam at 0.001, early-stopping patience 5, seed 42, four CPU threads, zero Windows-safe DataLoader workers, and automatic CPU/CUDA selection. Best validation loss selects the saved weights; test data never select the checkpoint. Starting another run starts from scratch and replaces artifacts at the selected checkpoint path.

Explicit multicrop command:

```powershell
python -m src.train --data dataset/multicrop --checkpoint models/agrovision_multicrop.pth --architecture v2 --epochs 25 --batch-size 32 --charts outputs/multicrop/charts
```

At the initial benchmark the full multicrop epoch took about one CPU minute; use the measured history to estimate longer jobs on your machine. The initial update run uses a shorter, explicitly reported training budget; see VERIFICATION.md for actual results, rather than assuming training reached convergence.

Smoke-test a separate checkpoint:

```powershell
python -m src.train --data dataset/multicrop --checkpoint models/smoke_multicrop.pth --architecture v2 --epochs 1 --max-samples 96 --batch-size 16 --charts outputs/multicrop/smoke-charts
```

Smoke-test checkpoints are marked visibly and are not practical models. The checkpoint embeds architecture, weights, class mapping, class catalog, preprocessing, seed, configuration, and source provenance. Inference uses this embedded metadata, not a hardcoded UI label list. Readable metadata/history JSON and measured loss/accuracy charts are saved separately.

## Evaluate the held-out split

```powershell
python -m src.evaluate
```

The default evaluator chooses the preferred checkpoint and the matching multicrop or legacy dataset/output directory. To be explicit:

```powershell
python -m src.evaluate --data dataset/multicrop --checkpoint models/agrovision_multicrop.pth --output outputs/multicrop
```

Outputs include measured accuracy, per-class/macro/weighted precision, recall and F1, support, a classification report, and confusion matrix in JSON/console, plus training loss/accuracy and confusion-matrix PNGs. Classification metrics use the raw class argmax. Abstention/task-routing behavior is a separate product policy; raw classifier accuracy must not be interpreted as unknown-image rejection accuracy. Its confidence threshold (0.65) and top-two margin (0.15) are conservative heuristics, not calibrated guarantees.

## Image CLI and webcam

```powershell
python main.py --mode image --task vegetable --image "path/to/tomato.jpg"
python main.py --mode image --task leaf --image "path/to/leaf.jpg" --output outputs/predictions/leaf.json
python main.py --mode webcam --task vegetable --interval 0.5
python main.py --mode webcam --task leaf --camera 0 --device cpu
```

Omitting `--task` in the image/webcam CLI allows automatic routing: primary leaf/unknown outcomes are checked by the general health model when available. Use `--task leaf` to bypass crop recognition and assess leaf health directly. Webcam capture classifies the guided center crop; it does not locate objects or leaves. Position one subject inside the visible guide, and press Q to quit. OpenCV releases the camera on exit/errors. Check Windows camera permissions and camera index if unavailable. The physical webcam needs local validation; do not assume simulated tests verify the device.

## Legacy tomato workflow

The original data/checkpoint are preserved, and can be used explicitly:

```powershell
python -m src.prepare_dataset --download
python -m src.train --data dataset --architecture v1 --checkpoint models/agrovision_cnn.pth --charts outputs/charts
python -m src.evaluate --data dataset --checkpoint models/agrovision_cnn.pth --output outputs
```

The first command is for a fresh legacy split only; it refuses to overwrite the already prepared data. Legacy weights and evaluation remain available for reproducibility. The current workbench's Leaf health mode uses the separate general-health checkpoint.

## Verification and limitations

```powershell
python -m compileall -q src main.py tests
python -m unittest discover -s tests -v
python main.py --help
python -m src.prepare_multicrop --help
python -m src.train --help
python -m src.evaluate --help
```

Tests cover both CNN output sizes, real-data split isolation and mapping, deterministic preparation, corrupt-image handling, checkpoint inference, probability routing, camera failure cleanup, and the GUI upload/export/error-recovery/clear/mode-switch workflow. Unavailable real-data/checkpoint tests skip explicitly. The GUI test uses real Tk event processing and real prediction; file dialogs are mocked. Policy tests use deterministic probability vectors and do not measure model accuracy. Actual executed results are recorded in [VERIFICATION.md](VERIFICATION.md).

Controlled dataset photos differ from field images, product cutouts, complex scenes, and unfamiliar plants. This is one-image classification, not object detection, freshness grading, damaged-area measurement, or a validated agronomic diagnostic system. Unsupported-input examples cannot cover all possible objects. The next engineering steps are independent field and unseen-species validation, broader negative examples, and validated confidence calibration. Do not repeatedly tune against a previously observed test set; reserve new independent data for a final unbiased assessment.

Datasets, checkpoints, downloaded archives, generated outputs, virtual environments, caches, and secrets are excluded by `.gitignore`. A fresh clone contains source code and documentation; prepare the datasets and train the models using the commands above to generate the local model files.
