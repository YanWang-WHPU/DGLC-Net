# Implementation and reproduction notes

## Release scope

This is a curated source release from the original experiment workspace. The model/training, prototype/refinement, ensemble, and GIS preparation implementations retain their original algorithms. The release adds documentation, CLI recipes, a configuration launcher, a generic saved-prediction evaluator, and synthetic tests. The router additionally accepts archives without labels and validates input shapes and routing parameters; its ranking and pixel update rules are unchanged for valid original inputs.

Third-party repositories downloaded for comparison, parameter-search scripts, learned post-hoc stacked selectors, manuscript sources, result files, and all data/weights are excluded. No license is assigned on the author's behalf.

## Two-teacher and group-memory path

The recovered no-boundary-gate record specifies teacher order `mix`, then `proto`, weights `0.57, 0.43`, temperature `0.68`, source refinement strength `0.008`, target reliability quantile `0.9`, per-group sampling cap `4096`, target alpha `0.1`, and target group strength `0.2`. The corresponding recipe is `configs/dual_teacher_group_memory.json`.

The evaluator exposes `--proto-mode boundary_gate` for the boundary-gate variant and additional source/group/multi-prototype ablations. Gate thresholds and smoothing settings are CLI options. The recovered result JSON does not serialize every CLI option; recipe choices for unrecorded arguments are examples, not verified historical settings.

The target group builder collects reliability-filtered, normalized features over the target loader and computes a reliability-weighted group center. It is an offline bank construction in this implementation. The original evaluator also builds source anchor prototypes from source images and labels. Thus this evaluator requires local source patches as well as trained checkpoints; it is not a deployment-only, source-data-free interface.

## Archived local-consensus path

The saved final router record references `runs_hsi_uda_class_entropy_gate_v1/fgpst_raw_prob.npz`, with window `51`, restore budget `4400`, and flip budget `4100`, excluding labels 0 and 255 during scoring. The base archive metadata references a single `fgpst` checkpoint and a `source_v4` anchor, with source refinement strength `0.0` and temperature `1.0`.

`configs/router_base_evaluation.json` illustrates how to generate an archive with those checkpoints; unrecorded smoothing choices are explicitly set in the recipe and require verification against the original run. `configs/local_consensus.json` then applies the recovered routing parameters to that archive. This archived input is distinct from the dual-teacher/group-memory output. Connecting the latter to the router creates another experiment and must be evaluated separately.

The restore budget selects Background predictions with the greatest local Rice support and changes them to the more probable rice subclass. The flip budget selects Rice predictions with the greatest local Background support and changes them to class 3. Both candidate masks are computed from the original input prediction. Budgets are global over the complete patch stack, including locations excluded from metric evaluation. They depend on image extent, patch order, and tie ordering; the released values are not scene-independent proportions.

## Training and evaluation protocol

`source_example.json` demonstrates source training with current CLI defaults. Exact historical hyperparameters for all trained teacher checkpoints were not recovered. Do not treat the example as a bit-for-bit teacher retraining recipe.

The original training file includes several exploratory adaptation routines. These routines use source batches/labels, and checkpoint selection uses the UAV validation metric specified by `--selection-metric`. The evaluator's prototype path also uses source labels. Inspect the selected routine and provide an independent held-out evaluation before making claims about source-free operation or target-label-independent model selection. This release preserves those behaviors rather than silently changing the original experiment protocol.

The workspace contains both earlier ignore-zero UAV evaluation and later spatially aligned-label recalculation. The same predictions can receive different scores under those masks. Keep the selected masks, patch order, class mapping, and tuning/evaluation split explicit. No performance table is included here because a complete retraining and unified reproduction have not been established for this code-only release.

## Running checks

Install `requirements.txt` and run `python -m unittest discover -s tests -v`. Use `python scripts/run_config.py CONFIG --dry-run` to inspect recipes without loading data. For training, use `python scripts/train_hsi_uda.py --help`; for fusion/refinement, use `python scripts/eval_ensemble_proto_refine.py --help`.
