from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
import train_hsi_uda as hsi  # noqa: E402


def load_model(ckpt_path: Path, device: torch.device, base_ch: int, feat_dim: int) -> hsi.SimpleUNet:
    model = hsi.SimpleUNet(in_ch=32, num_classes=4, base=base_ch, feat_dim=feat_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def score_prediction(pred: np.ndarray, target: np.ndarray, ignore_zero_eval: bool = False) -> dict:
    valid = target != 255
    if ignore_zero_eval:
        valid &= target != 0

    pred_v = pred[valid]
    target_v = target[valid]

    meter = hsi.ConfusionMeter(4)
    if target_v.size:
        meter.add(pred_v, target_v)
    class_metrics = meter.compute()

    gt_rice = (target_v == 1) | (target_v == 2)
    pred_rice = (pred_v == 1) | (pred_v == 2)
    gt_bg = target_v == 3 if ignore_zero_eval else ~gt_rice
    pred_bg = ~pred_rice

    tp = int(np.count_nonzero(gt_rice & pred_rice))
    fp = int(np.count_nonzero(gt_bg & pred_rice))
    fn = int(np.count_nonzero(gt_rice & pred_bg))
    tn = int(np.count_nonzero(gt_bg & pred_bg))

    def safe_div(num: int, den: int) -> float:
        return float(num / den) if den else 0.0

    rice_iou = safe_div(tp, tp + fp + fn)
    background_iou = safe_div(tn, tn + fp + fn)
    rice_f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    background_f1 = safe_div(2 * tn, 2 * tn + fp + fn)

    metrics = {
        "ignore_zero_eval": bool(ignore_zero_eval),
        "valid_pixels": int(valid.sum()),
        "oa": safe_div(tp + tn, valid.sum()),
        "miou": 0.5 * (rice_iou + background_iou),
        "mf1": 0.5 * (rice_f1 + background_f1),
        "rice_iou": rice_iou,
        "background_iou": background_iou,
        "rice_f1": rice_f1,
        "background_f1": background_f1,
        "binary_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "pred_counts": {str(k): int(v) for k, v in zip(*np.unique(pred_v, return_counts=True))} if pred_v.size else {},
        "class_metrics": class_metrics,
    }
    return metrics


@torch.no_grad()
def eval_model_on_loader(model: hsi.SimpleUNet, loader, device: torch.device) -> dict:
    meter = hsi.ConfusionMeter(4)
    with torch.no_grad():
        for batch in loader:
            img = batch["img"].to(device)
            target = batch["mask"].numpy()
            pred = model(img, return_feat=True)[0].argmax(dim=1).cpu().numpy()
            for p, t in zip(pred, target):
                meter.add(p, t)
    return meter.compute()


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(args.ckpts) != len(args.weights):
        raise ValueError("--ckpts and --weights must have the same length")
    weights = np.array(args.weights, dtype=np.float32)
    weights = weights / np.clip(weights.sum(), 1e-6, None)
    models = [load_model(Path(p), device, args.base_ch, args.feat_dim) for p in args.ckpts]
    _, _, _, _, _, source_val_loader, _, _, uav_loader, _ = hsi.make_loaders(
        args.data_root, args.batch_size, args.num_workers, augment=False, target_augment=False
    )
    class_weights = None
    if args.fusion_mode == "classwise":
        ious = []
        for model in models:
            metrics = eval_model_on_loader(model, source_val_loader, device)
            ious.append(np.array(metrics["iou"], dtype=np.float32))
        priors = np.stack(ious, axis=0)
        priors = np.power(np.clip(priors, 1e-4, None), args.classwise_alpha)
        class_weights = priors / np.clip(priors.sum(axis=0, keepdims=True), 1e-6, None)
    elif args.fusion_mode == "expert":
        expert_idx = []
        for cls in range(4):
            best = 0
            best_iou = -1.0
            for i, model in enumerate(models):
                metrics = eval_model_on_loader(model, source_val_loader, device)
                iou = float(metrics["iou"][cls])
                if iou > best_iou:
                    best_iou = iou
                    best = i
            expert_idx.append(best)
        class_weights = np.zeros((len(models), 4), dtype=np.float32)
        for cls, idx in enumerate(expert_idx):
            class_weights[idx, cls] = 1.0
    all_pred = []
    all_target = []
    for batch in uav_loader:
        img = batch["img"].to(device)
        target = batch["mask"].numpy()
        probs = []
        for weight, model in zip(weights, models):
            logits, _ = model(img, return_feat=True)
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy() * float(weight))
        if class_weights is None:
            prob_sum = np.sum(probs, axis=0)
        else:
            prob_sum = np.zeros_like(probs[0])
            if args.fusion_mode == "classwise":
                for mi, prob in enumerate(probs):
                    prob_sum += class_weights[mi][None, :, None, None] * prob
            else:
                for mi, prob in enumerate(probs):
                    prob_sum += class_weights[mi][None, :, None, None] * prob
        prob_sum = torch.from_numpy(prob_sum).to(device)
        all_pred.append(prob_sum.argmax(dim=1).cpu().numpy())
        all_target.append(target)
    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)
    metrics = score_prediction(pred, target, ignore_zero_eval=args.ignore_zero_eval)
    return {
        "name": args.name,
        "ckpts": [str(Path(p)) for p in args.ckpts],
        "weights": weights.tolist(),
        "metrics": metrics,
    }


def write_outputs(out_dir: Path, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{result['name']}.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = out_dir / "ensemble_ablation.csv"
    row = {
        "name": result["name"],
        "rice_iou": result["metrics"]["rice_iou"],
        "background_iou": result["metrics"]["background_iou"],
        "oa": result["metrics"]["oa"],
        "mf1": result["metrics"]["mf1"],
        "miou": result["metrics"]["miou"],
        "weights": json.dumps(result["weights"]),
        "ckpts": json.dumps(result["ckpts"]),
    }
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--data-root", type=Path, default=hsi.ROOT / "processed_hsi_uda")
    parser.add_argument("--out-dir", type=Path, default=hsi.ROOT / "runs_hsi_uda_ensemble_eval")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-ch", type=int, default=48)
    parser.add_argument("--feat-dim", type=int, default=96)
    parser.add_argument("--fusion-mode", choices=["average", "classwise", "expert"], default="average")
    parser.add_argument("--classwise-alpha", type=float, default=1.0)
    parser.add_argument("--ignore-zero-eval", action="store_true")
    args = parser.parse_args()
    result = evaluate(args)
    write_outputs(args.out_dir, result)
    m = result["metrics"]
    print(
        f"{result['name']}: riceIoU={m['rice_iou']:.4f} backgroundIoU={m['background_iou']:.4f} "
        f"OA={m['oa']:.4f} mF1={m['mf1']:.4f} mIoU={m['miou']:.4f}"
    )


if __name__ == "__main__":
    main()
