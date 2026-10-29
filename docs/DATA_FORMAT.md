# Local data interface

No data, labels, scene manifests, or statistics are included in this repository.

## Prepared patches

The original loaders expect this local directory structure:

```text
processed_hsi_uda/
  manifest.json
  source_train/*.npz
  source_val/*.npz
  target/*.npz
  uav_val/*.npz
```

Each NPZ archive contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `image` | `[32, H, W]` | Hyperspectral image, channel first; converted to float32 |
| `mask` | `[H, W]` | Integer labels; required for labeled splits, ignored for target adaptation patches |

The original patch size is 128. Patches batched together must have the same shape. The preparation scripts also write `source`, `x`, and `y` for alignment; these are not required by the training loader.

Labels are `0` (original background/unlabeled category), `1` and `2` (rice subclasses), `3` (original lotus/non-rice category), and `255` (ignore). Binary Rice is the union of 1 and 2. With `--ignore-zero-eval`, labels 0 and 255 are excluded and class 3 forms the Background reference. Predictions of 0 or 3 both count as non-rice. Training retains the original four-class label encoding.

`manifest.json` must contain `stats.mean` and `stats.std` with 32 values each, plus `stats.class_counts` mapping class IDs to source pixel counts for training. Create statistics from your own source training patches. The loader applies `(image - mean) / max(std, 1e-6)` separately for each band. A full GIS-generated manifest contains additional provenance fields, but the model loader does not need them.

Training/evaluation entry points use the original shared loader, which constructs all four splits even when only some are used. Provide nonempty source and target splits and labeled validation splits for these entry points. Training uses `drop_last=True`, so source/target training splits must each contain at least one batch.

## Saved prediction archives

The local router reads `pred` of shape `[N,H,W]` and `prob` of shape `[N,4,H,W]`. `target` of shape `[N,H,W]` is optional. The router uses predictions and probabilities to route pixels; labels are used only to score the output. Its output `prob` is one-hot encoding of the routed classes, not calibrated confidence.

`evaluate_predictions.py` reads `pred` or takes `argmax` of `prob`. It uses embedded `target` unless `--uav-dir` supplies replacement masks. Patches must have exactly the same order and extent. External mask files are sorted by filename; the shape check cannot detect reordered scenes.

## Raw GIS preparation

`prepare_hsi_uda_dataset.py` preserves the original scene-discovery conventions under `data/`, including 32 band files, source polygon names, and UAV label patterns. It is an experiment-specific utility, not a general satellite-data reader. Review and adapt its filename patterns and label field to your own inputs. The `--clean` flag deletes the selected output directory before preparation; omit it when preserving previous outputs.

`build_aligned_uav_val.py` uses GDAL to warp the label raster onto the hyperspectral grid with nearest-neighbor resampling, then extracts masks using the patch coordinates. Original intermediate stacks and label rasters must exist locally.
