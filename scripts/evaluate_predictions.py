"""Evaluate saved predictions, optionally replacing masks with aligned UAV labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from apply_local_consensus_router import score_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--uav-dir", type=Path, help="Sorted NPZ files must match prediction patch order")
    parser.add_argument("--include-zero", action="store_true", help="Count label 0 as Background")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with np.load(args.prediction) as data:
        pred = data["pred"] if "pred" in data else data["prob"].argmax(axis=1)
        target = data["target"] if "target" in data else None
    if args.uav_dir is not None:
        files = sorted(args.uav_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No UAV patches found in {args.uav_dir}")
        masks = []
        for path in files:
            with np.load(path) as data:
                masks.append(data["mask"])
        target = np.stack(masks)
    if target is None:
        raise ValueError("Provide target in the prediction archive or use --uav-dir")
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {target.shape}")
    if not np.isin(target, [0, 1, 2, 3, 255]).all():
        raise ValueError("Target labels must be 0, 1, 2, 3, or 255")
    if not np.isin(pred, [0, 1, 2, 3]).all():
        raise ValueError("Prediction labels must be 0, 1, 2, or 3")
    result = {
        "ignore_zero_eval": not args.include_zero,
        "metrics": score_prediction(pred, target, ignore_zero_eval=not args.include_zero),
    }
    serialized = json.dumps(result, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
