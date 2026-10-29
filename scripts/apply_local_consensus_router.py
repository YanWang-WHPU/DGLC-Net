from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


def score_prediction(pred: np.ndarray, target: np.ndarray, ignore_zero_eval: bool) -> dict:
    valid = target != 255
    if ignore_zero_eval:
        valid &= target != 0
    gt_rice = ((target == 1) | (target == 2)) & valid
    pred_rice = ((pred == 1) | (pred == 2)) & valid
    gt_bg = (target == 3) & valid if ignore_zero_eval else (~gt_rice) & valid
    pred_bg = (~pred_rice) & valid

    tp = int(np.count_nonzero(gt_rice & pred_rice))
    fp = int(np.count_nonzero(gt_bg & pred_rice))
    fn = int(np.count_nonzero(gt_rice & pred_bg))
    tn = int(np.count_nonzero(gt_bg & pred_bg))

    def div(num: float, den: float) -> float:
        return float(num / den) if den else 0.0

    recall = div(tp, tp + fn)
    precision = div(tp, tp + fp)
    f1 = div(2 * tp, 2 * tp + fp + fn)
    rice_iou = div(tp, tp + fp + fn)
    bg_iou = div(tn, tn + fp + fn)
    bg_f1 = div(2 * tn, 2 * tn + fp + fn)
    oa = div(tp + tn, int(valid.sum()))
    return {
        "oa": oa,
        "mf1": 0.5 * (f1 + bg_f1),
        "miou": 0.5 * (rice_iou + bg_iou),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "rice_iou": rice_iou,
        "bg_iou": bg_iou,
        "bg_f1": bg_f1,
        "binary_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def topk_mask(values: np.ndarray, candidate: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(candidate.shape, dtype=bool)
    coords = np.argwhere(candidate)
    if k <= 0 or coords.size == 0:
        return out
    k = min(k, len(coords))
    ranked = np.argsort(values[candidate])[::-1][:k]
    chosen = coords[ranked]
    out[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
    return out


def apply_router(base_npz: Path, window: int, restore_top_k: int, flip_top_k: int) -> tuple[np.ndarray, np.ndarray | None, dict]:
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if restore_top_k < 0 or flip_top_k < 0:
        raise ValueError("routing budgets must be nonnegative")
    with np.load(base_npz) as data:
        if "pred" not in data or "prob" not in data:
            raise ValueError(f"{base_npz} must contain pred and prob")
        pred = data["pred"].copy()
        prob = data["prob"].astype(np.float32)
        target = data["target"].copy() if "target" in data else None
    if pred.ndim != 3 or prob.shape != (pred.shape[0], 4, *pred.shape[1:]):
        raise ValueError("Expected pred [N,H,W] and prob [N,4,H,W]")
    if not np.isin(pred, [0, 1, 2, 3]).all() or not np.isfinite(prob).all():
        raise ValueError("Invalid prediction labels or non-finite probabilities")
    if target is not None and (target.shape != pred.shape or not np.isin(target, [0, 1, 2, 3, 255]).all()):
        raise ValueError("target must match pred shape and use labels 0, 1, 2, 3, 255")
    pred = pred.astype(np.uint8)
    if target is not None:
        target = target.astype(np.uint8)

    base_rice = (pred == 1) | (pred == 2)
    base_bg = ~base_rice
    local_rice = np.stack(
        [ndi.uniform_filter(base_rice[i].astype(np.float32), size=window, mode="nearest") for i in range(pred.shape[0])],
        axis=0,
    )
    local_bg = np.stack(
        [ndi.uniform_filter(base_bg[i].astype(np.float32), size=window, mode="nearest") for i in range(pred.shape[0])],
        axis=0,
    )

    restore = topk_mask(local_rice, base_bg, restore_top_k)
    flip = topk_mask(local_bg, base_rice, flip_top_k)

    rice_class = np.where(prob[:, 1] >= prob[:, 2], 1, 2).astype(np.uint8)
    pred[restore] = rice_class[restore]
    pred[flip] = 3

    debug = {
        "base_npz": str(base_npz),
        "window": window,
        "restore_top_k": int(restore_top_k),
        "flip_top_k": int(flip_top_k),
        "restored_pixels": int(restore.sum()),
        "flipped_pixels": int(flip.sum()),
        "restore_fraction_of_base_bg": float(restore.sum() / max(base_bg.sum(), 1)),
        "flip_fraction_of_base_rice": float(flip.sum() / max(base_rice.sum(), 1)),
    }
    return pred, target, debug


def one_hot_prob(pred: np.ndarray, num_classes: int = 4) -> np.ndarray:
    prob = np.zeros((pred.shape[0], num_classes, pred.shape[1], pred.shape[2]), dtype=np.float32)
    for cls in range(num_classes):
        prob[:, cls] = pred == cls
    return prob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="NPZ with pred and prob; target is optional")
    parser.add_argument("--window", type=int, default=51)
    parser.add_argument("--restore-top-k", type=int, default=4400)
    parser.add_argument("--flip-top-k", type=int, default=4100)
    parser.add_argument("--ignore-zero-eval", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/local_consensus"))
    parser.add_argument("--name", type=str, default="local_consensus_w51_r4400_b4100")
    args = parser.parse_args()

    pred, target, debug = apply_router(args.base, args.window, args.restore_top_k, args.flip_top_k)
    metrics = score_prediction(pred, target, args.ignore_zero_eval) if target is not None else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / f"{args.name}_prob.npz"
    json_path = args.out_dir / f"{args.name}.json"
    arrays = {"pred": pred, "prob": one_hot_prob(pred)}
    if target is not None:
        arrays["target"] = target
    np.savez_compressed(npz_path, **arrays)
    result = {"method": "Local consensus router", "debug": debug, "metrics": metrics, "npz": str(npz_path)}
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if metrics is not None:
        print(
            f"{args.name}: OA={metrics['oa']:.6f} mF1={metrics['mf1']:.6f} mIoU={metrics['miou']:.6f} "
            f"Recall={metrics['recall']:.6f} Precision={metrics['precision']:.6f} "
            f"F1={metrics['f1']:.6f} RiceIoU={metrics['rice_iou']:.6f}"
        )
    print(json_path)


if __name__ == "__main__":
    main()
