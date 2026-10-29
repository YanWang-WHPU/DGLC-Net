# DGLC-Net

**DGLC-Net: A Dual-Teacher Group-Memory Local-Consensus Network for Cross-Date Hyperspectral Rice Exposure Mapping**

Code release for the accepted paper. Publication metadata will be added when available.

The implementation studies cross-date hyperspectral rice exposure mapping using complementary teachers, reliability-filtered Rice/Background prototypes, boundary correction, and local-consensus routing. The network retains four internal classes; classes 1 and 2 are merged for binary Rice evaluation.

This repository contains source code, configuration examples, documentation, and synthetic tests. Datasets, labels, normalization statistics, trained weights, predictions, experiment logs, and manuscript files are not distributed.

## Setup

Use Python 3.10 or later. Install a PyTorch build suitable for your CPU/CUDA environment, then install the remaining requirements:

```bash
python -m venv .venv
# Activate .venv in your shell.
python -m pip install -r requirements.txt
```

The optional raw-data preparation script additionally requires an ArcGIS Pro Python environment with ArcPy and GDAL. These GIS dependencies are unnecessary when using prepared NPZ patches. See [the data format](docs/DATA_FORMAT.md).

## Code map

| File | Purpose |
| --- | --- |
| `scripts/train_hsi_uda.py` | U-Net backbone, patch loader, losses, source training, and experimental adaptation routines |
| `scripts/eval_ensemble_proto_refine.py` | Teacher fusion, source/target prototypes, group memory, boundary gate, and ablations |
| `scripts/evaluate_hsi_ensemble.py` | Ensemble evaluation |
| `scripts/apply_local_consensus_router.py` | Spatial routing of saved predictions, with optional labels for evaluation |
| `scripts/evaluate_predictions.py` | Binary evaluation, including replacement with aligned validation masks |
| `scripts/prepare_hsi_uda_dataset.py` | Optional ArcPy/GDAL preprocessing for the original scene naming convention |
| `scripts/build_aligned_uav_val.py` | Align UAV labels to hyperspectral input grids |
| `scripts/run_config.py` | Inspect or execute a JSON command recipe |

## Usage

Run commands from the repository root. Supply your own prepared data and checkpoints; all paths in the recipes are local placeholders.

```bash
# Inspect a command before running it.
python scripts/run_config.py configs/source_example.json --dry-run

# Example source training; this is not a recovered historical training recipe.
python scripts/run_config.py configs/source_example.json

# Evaluate the two teachers with target group memory.
python scripts/run_config.py configs/dual_teacher_group_memory.json

# Apply the archived local-router parameters to a saved probability archive.
python scripts/apply_local_consensus_router.py --base outputs/base/predictions.npz --window 51 --restore-top-k 4400 --flip-top-k 4100 --ignore-zero-eval

# Score predictions against separately aligned masks if available.
python scripts/evaluate_predictions.py --prediction outputs/local_consensus/local_consensus_w51_r4400_b4100_prob.npz --uav-dir processed_hsi_uda_aligned/uav_val
```

The original records associate the final local-router result with an `fgpst_raw_prob.npz` archive. The dual-teacher/group-memory experiments have separate records. These paths are documented separately in [reproduction notes](docs/REPRODUCTION.md); the repository does not assert a verified end-to-end reproduction of the paper's final result. Historical training arguments are incomplete, and some original adaptation routines consume labeled source batches. Read those notes before selecting a protocol.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic arrays to check binary metrics, routing behavior and label independence, patch normalization, model output shapes, and gradient flow. They require no research data or weights.

## Paper and reuse

The paper has been accepted and is awaiting publication. A final citation and DOI are not yet provided. No software license has been selected for this initial release; contact the repository owner about reuse permissions.
