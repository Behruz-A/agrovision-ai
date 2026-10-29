# AgroVision AI: executed verification

This report records measured workspace results on October 29, 2026 (Asia/Seoul). Dataset accuracy is not a claim of field reliability.

## Latest update: general leaf-health percentages

The user clarified that leaf species identification is unnecessary: the app should estimate Healthy versus Diseased. The Leaf health tab now uses a **separate three-state CNN** and no longer requires a recognized crop. Vegetable identification keeps its existing model and weights. The UI shows Healthy leaf (green), Diseased leaf (amber), and Not a leaf (gray), in fixed order, including uncertain cases. Progress is hidden when idle.

These percentages estimate relative model class probabilities. They do **not** measure damaged leaf area, prove health, identify species, or diagnose a specific disease. Scores are not calibrated certainty. The model is trained across 14 represented crop groups, not every tree/wild-plant species.

Real training data were sampled from all 38 PlantVillage source classes and the existing attributed vegetable archive. The balanced set has **2,400 images**: 800 healthy leaves, 800 diseased leaves, and 800 real non-leaf vegetable photos. Split: **1,680 train / 360 validation / 360 test**, with 560/120/120 examples per state. Group/hash isolation tests passed. Data/catalog/provenance are in `dataset/leaf-health/`; original datasets are preserved.

Executed:

```powershell
python -m src.prepare_leaf_health --download
python -m src.train --data dataset/leaf-health --epochs 10 --batch-size 32 --patience 4
python -m src.evaluate --checkpoint models/agrovision_leaf_health.pth
```

Ten epochs completed from scratch in **267.3 seconds** of epoch computation. Best checkpoint: **epoch 10**, selected by validation loss; validation accuracy **94.44%**. Checkpoint: `models/agrovision_leaf_health.pth`; SHA-256 `ca3c671c80ca3f1be81cebd224fefe922f2b32be43c4aaec60024b90ee33ed90`.

Held-out results on 360 images:

| Measure | Result |
|---|---:|
| Overall three-state accuracy | 92.22% (332/360) |
| Healthy/diseased leaf-image raw accuracy | 89.17% (214/240) |
| Diseased-leaf recall | 87.50% (105/120) |
| Healthy-leaf recall | 90.83% (109/120) |
| Non-leaf raw recall | 98.33% (118/120) |
| Macro F1 | 92.21% |

The app accepted 216/240 leaf examples with 93.06% accuracy among those accepted; 24 leaf examples were withheld. All 120 non-leaf examples were withheld by the policy (118 explicit non-leaf predictions, two uncertain), but those negatives are vegetable photos and do not establish universal object rejection. Twelve diseased test leaves were misclassified as healthy by raw argmax, so a healthy estimate must not be treated as a guarantee.

**22 tests passed**, with no skips in the final run. Added tests verify percentage/state mapping without a crop name, no fabricated disease names, non-leaf rejection, uncertainty without hiding raw estimates, real split integrity, and actual workbench routing into the health checkpoint. The GUI integration verifies the fixed Healthy/Diseased/Not-a-leaf rows after mode switching. Syntax compilation and preparation help passed.

Actual image CLI/export checks ran for one held-out Healthy, Diseased, and Not_leaf example; all returned three scores summing to one, with no crop name. Example estimates were Healthy 75.28%, Diseased 99.96%, and Not a leaf 99.99% for their respective images. These are illustrative executed predictions, not extra aggregate metrics. JSON outputs are in `outputs/leaf-health/examples/`; evaluation and charts are in `outputs/leaf-health/`.

The app was restarted directly in Leaf health mode using `python main.py --mode gui --task leaf`. Native screenshot inspection remains unavailable as described below; real Tk integration and layout tests passed. CUDA, physical camera, unseen-species generalization, independent field performance, and probability calibration remain unverified.

## Previous multicrop update

The application now has two English modes: Vegetable identification and Leaf health. The new checkpoint contains ten vegetable labels, eight leaf-health labels for tomato/potato/bell pepper, and an Other / unsupported label. No pretrained weights or fabricated training images were used. The original tomato-only data/checkpoint remain available separately.

Implemented UI improvements: Upload image and clickable preview, separate crop/leaf-condition fields, Clear result, Save report, top-three/full probability views, model coverage details, mode switching with reanalysis, keyboard shortcuts, dimensions/filename display, progress/error states, a scrollable analysis panel, and persistent Save/Clear actions. Clear/replace invalidates old jobs and closes the old full-probability window. Window shutdown joins local worker activity and cancels callbacks before destroying Tk.

## Environment and data

- Existing environment reused; no new dependencies installed.
- Python 3.13.5; PyTorch 2.15.0+cpu; torchvision 0.29.2+cpu; Tk 8.6.
- CPU training, four threads; CUDA unavailable and untested.
- Custom CNN v2: 224 x 224 RGB input, first-convolution stride 2, 19 output classes.
- Vegetable Image Dataset v1 downloaded from its public Kaggle endpoint (560,031,432-byte ZIP); creators and license are attributed in README.md and provenance metadata.
- PlantVillage images downloaded/reused with content revision and leaf metadata recorded in `dataset/multicrop/provenance.json`.
- Up to 300 real source images per supported class; potato healthy has fewer available images. The unsupported label combines excluded vegetable types and healthy apple/grape leaves.
- Prepared **5,543 unique images**: **3,877 train / 833 validation / 833 test**. Removed **3** exact decoded-RGB duplicates; skipped **0** invalid source images.
- The split audit passed for known leaf groups, file hashes, and decoded-pixel hashes. Vegetable specimen/session IDs are unavailable: near-duplicate views and background correlations cannot be ruled out. This is a capped subset, not the full source datasets.

## Executed training

A separate v2 smoke test ran one epoch on 96 training/96 validation images in 2.4 seconds. It checked the pipeline only; its low preliminary accuracy is not a practical-model claim.

The full capped-data run executed:

```powershell
python -m src.train --data dataset/multicrop --checkpoint models/agrovision_multicrop.pth --architecture v2 --epochs 12 --batch-size 32 --patience 5 --charts outputs/multicrop/charts
```

All **12 epochs** completed. Epoch computation totaled **727.2 seconds** (about 12.1 minutes), excluding initialization, downloads, and plotting. **Epoch 11** had the lowest validation loss (**0.4250**) and was saved. Its validation accuracy was **85.47%**. Test data were not used to select weights.

Checkpoint SHA-256: `9baa1d01062e1415c74552e852def2dcd8d000c136e71f5281797de972aa15d5`.

Checkpoint metadata embeds the class catalog/mapping, architecture, preprocessing, training configuration, provenance, and heuristic abstention thresholds. Human-readable metadata and complete measured history are saved alongside it.

## Held-out evaluation

`python -m src.evaluate` executed on the preferred multicrop checkpoint and all **833 test images**. Raw classifier accuracy was **87.39% (728/833)**. Macro precision/recall/F1: **87.72% / 87.24% / 86.82%**. Weighted F1: **86.95%**.

| Task | Test images | Raw class accuracy | Accepted by app policy | Correct among accepted |
|---|---:|---:|---:|---:|
| Vegetable identification | 450 | 92.89% | 387 (86.00%) | 97.67% |
| Leaf health (crop + condition) | 339 | 85.84% | 243 (71.68%) | 93.00% |
| Unsupported examples | 44 | 43.18% classified explicitly as Other | 7 falsely accepted | 0 correct supported identifications |

The policy withheld **37/44 unsupported examples (84.09%)** via explicit Other or uncertainty; **7/44 (15.91%) were still falsely accepted**. These are limited dataset negatives, not arbitrary unknown objects. Neither high class probability nor rejection on this set guarantees correct behavior on a new photo. Thresholds (0.65 probability / 0.15 top-two margin) are uncalibrated heuristics, not tuned using the test outcomes.

Lower-recall supported classes include tomato early blight (30/45) and tomato late blight (31/45). More independent field examples, broader negatives, and calibration are needed before practical diagnosis. The app does not claim disease status for a whole vegetable.

Saved `outputs/multicrop/evaluation.json`, a 19-class confusion-matrix PNG, and measured training loss/accuracy charts. Large-matrix tick labels were made readable after inspection; only the plot was regenerated, without reevaluating or selecting model weights.

## Functional verification

- `python -m compileall -q src main.py tests`: passed.
- Project imports and real checkpoint loading: passed for new 19-class v2 and preserved 3-class v1.
- CLI help executed for the app, multicrop preparation, training, and evaluation.
- `python -m unittest discover -s tests -v`: **16 tests passed**, no skips in the final suite run.
- Tests cover both CNN output sizes and backward passes, mapping/coverage and split integrity, reproducible preparation, corrupt-image skipping, duplicate removal, overwrite protection, legacy inference, both task-routing policies, abstention, simulated camera failure cleanup, real GUI upload/export/recovery, small-window action visibility, clear during prediction, stale-event rejection, and switching modes on the selected image.
- Routing-policy tests use deterministic probability vectors; they are not claimed as measured model performance.
- GUI integration uses real Tk event processing and actual checkpoint inference; file dialogs are mocked in the tests.
- CLI prediction and JSON export executed in both task modes on real held-out images. The first tomato example was correctly withheld as uncertain (best class Tomato, probability 56.86%); the potato-leaf example was identified as healthy (87.05%). Each exported all 19 probabilities summing to one within tolerance. These examples are functional checks, not additional accuracy estimates.
- The initial shutdown/layout test exposed a Tk worker-finalization error. The corrected implementation and the complete suite subsequently passed.
- The old app processes launched by this session were stopped before launching the redesigned app.

## Limits and remaining checks

- Native screenshot inspection of the desktop app could not be completed: Computer Use reported `Computer Use native pipe is unavailable`; the browser access host was also unavailable. Widget-layout checks and real Tk integration passed, but a manual visual/native-file-dialog review remains useful.
- The physical webcam and CUDA were not exercised for this update. Camera failure cleanup is simulated in tests.
- No independent field evaluation, universal unknown-image rejection, freshness grading, object localization, or probability calibration is claimed.
- Leaf health covers **three crops**, while vegetable identification covers **ten**. Additional crops require real labeled disease data and retraining.
- The default 25-epoch run was not executed for this update; a 12-epoch initial model was trained. At measured speed, 25 epochs would take roughly 25 minutes of CPU epoch computation before early stopping.
- No Git repository was initialized and nothing was pushed or published.

## Historical tomato-only baseline

Before this update, the original three-class tomato CNN was trained for three full-data epochs on 4,486 unique PlantVillage images (3,135 train / 676 validation / 675 test). Epoch 2 was selected by validation loss and measured **87.56% test accuracy**, macro F1 **84.90%**, with early-blight recall **59.87%**. The checkpoint is preserved as `models/agrovision_cnn.pth`; its metrics must not be confused with the new multicrop model's results.
