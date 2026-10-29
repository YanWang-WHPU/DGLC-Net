from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_history(out_dir: Path, record: dict) -> None:
    with (out_dir / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class PatchNPZDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        mean: np.ndarray,
        std: np.ndarray,
        labeled: bool = True,
        augment: bool = False,
        band_dropout: float = 0.0,
    ):
        self.files = files
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.labeled = labeled
        self.augment = augment
        self.band_dropout = band_dropout

    def __len__(self) -> int:
        return len(self.files)

    def _augment(self, img: np.ndarray, mask: np.ndarray | None):
        if random.random() < 0.5:
            img = np.flip(img, axis=2).copy()
            if mask is not None:
                mask = np.flip(mask, axis=1).copy()
        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            if mask is not None:
                mask = np.flip(mask, axis=0).copy()
        if random.random() < 0.2:
            img = np.transpose(img, (0, 2, 1)).copy()
            if mask is not None:
                mask = np.transpose(mask, (1, 0)).copy()
        if self.band_dropout > 0 and random.random() < 0.5:
            c = img.shape[0]
            n_drop = max(1, int(c * self.band_dropout))
            drop_idx = np.random.choice(c, size=n_drop, replace=False)
            img[drop_idx] = 0
        return img, mask

    def __getitem__(self, idx: int):
        item = np.load(self.files[idx])
        img = item["image"].astype(np.float32)
        img = (img - self.mean[:, None, None]) / self.std[:, None, None]
        mask = item["mask"].astype(np.int64) if self.labeled and "mask" in item else None
        if mask is not None:
            mask[~np.isin(mask, [0, 1, 2, 3])] = 255
        if self.augment:
            img, mask = self._augment(img, mask)
        img = torch.from_numpy(img.copy())
        if mask is None:
            return {"img": img, "path": str(self.files[idx]), "idx": idx}
        return {"img": img, "mask": torch.from_numpy(mask.copy()), "path": str(self.files[idx]), "idx": idx}


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_ch: int = 32, num_classes: int = 4, base: int = 48, feat_dim: int = 96):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.up3 = nn.Conv2d(base * 8, base * 4, 1)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.Conv2d(base * 2, base, 1)
        self.dec1 = ConvBlock(base * 2, base)
        self.proj = nn.Conv2d(base, feat_dim, 1)
        self.classifier = nn.Conv2d(feat_dim, num_classes, 1)
        self.rice_classifier = nn.Conv2d(feat_dim, 1, 1)

    def forward(self, x, return_feat: bool = False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = torch.nn.functional.interpolate(self.up3(b), size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = torch.nn.functional.interpolate(self.up2(d3), size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = torch.nn.functional.interpolate(self.up1(d2), size=e1.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.dec1(torch.cat([d1, e1], dim=1))
        proj = self.proj(feat)
        logits = self.classifier(proj)
        if return_feat:
            return logits, proj
        return logits


class ConfusionMeter:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def add(self, pred: np.ndarray, target: np.ndarray):
        mask = target != 255
        pred = pred[mask].reshape(-1)
        target = target[mask].reshape(-1)
        for t, p in zip(target, pred):
            self.mat[int(t), int(p)] += 1

    def compute(self):
        tp = np.diag(self.mat).astype(np.float64)
        gt = self.mat.sum(axis=1).astype(np.float64)
        pred = self.mat.sum(axis=0).astype(np.float64)
        union = gt + pred - tp
        iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
        dice = np.divide(2 * tp, gt + pred, out=np.zeros_like(tp), where=(gt + pred) > 0)
        oa = float(tp.sum() / max(self.mat.sum(), 1))
        fg_mask = union[1:] > 0 if len(union) > 1 else np.array([], dtype=bool)
        miou_fg = float(iou[1:][fg_mask].mean()) if fg_mask.any() else 0.0
        rice_tp = float(self.mat[1:3, 1:3].sum()) if self.num_classes >= 3 else 0.0
        rice_gt = gt[1:3].sum() if self.num_classes >= 3 else 0.0
        rice_pred = pred[1:3].sum() if self.num_classes >= 3 else 0.0
        rice_union = rice_gt + rice_pred - rice_tp
        rice_iou = float(rice_tp / rice_union) if rice_union > 0 else 0.0
        return {
            "oa": oa,
            "miou": float(iou.mean()),
            "miou_fg": miou_fg,
            "rice_exposure_iou": rice_iou,
            "mdice": float(dice.mean()),
            "iou": iou.tolist(),
            "dice": dice.tolist(),
        }


def weighted_ce_loss(logits, target, class_weights=None):
    return nn.functional.cross_entropy(logits, target, weight=class_weights, ignore_index=255)


def dice_loss(logits, target, class_weights=None, eps: float = 1e-6):
    num_classes = logits.shape[1]
    prob = torch.softmax(logits, dim=1)
    valid = (target != 255).unsqueeze(1)
    target_clamped = target.clamp(0, num_classes - 1)
    one_hot = nn.functional.one_hot(target_clamped, num_classes=num_classes).permute(0, 3, 1, 2).float()
    prob = prob * valid
    one_hot = one_hot * valid
    inter = (prob * one_hot).sum(dim=(0, 2, 3))
    denom = (prob + one_hot).sum(dim=(0, 2, 3))
    dice = 1.0 - (2.0 * inter + eps) / (denom + eps)
    if class_weights is not None:
        dice = (dice * class_weights).sum() / class_weights.sum()
    else:
        dice = dice.mean()
    return dice


def rice_exposure_loss(logits, target, mask: torch.Tensor | None = None, target_is_binary: bool = False, eps: float = 1e-6):
    prob = torch.softmax(logits, dim=1)
    rice_prob = (prob[:, 1] + prob[:, 2]).clamp(eps, 1.0 - eps)
    if target_is_binary:
        target_bin = target.float()
        valid = torch.ones_like(target_bin, dtype=torch.bool)
    else:
        target_bin = ((target == 1) | (target == 2)).float()
        valid = target != 255
    if mask is not None:
        valid = valid & mask
    if valid.sum() == 0:
        return logits.new_tensor(0.0)
    bce = nn.functional.binary_cross_entropy(rice_prob, target_bin, reduction="none")
    bce = bce[valid].mean()
    pred_v = rice_prob[valid]
    target_v = target_bin[valid]
    inter = (pred_v * target_v).sum()
    dice = 1.0 - (2.0 * inter + eps) / (pred_v.sum() + target_v.sum() + eps)
    return bce + dice


def rice_head_loss(rice_logits: torch.Tensor, target: torch.Tensor, neg_weight: float = 1.0, eps: float = 1e-6):
    valid = target != 255
    if valid.sum() == 0:
        return rice_logits.new_tensor(0.0)
    target_bin = ((target == 1) | (target == 2)).float()
    bce = nn.functional.binary_cross_entropy_with_logits(rice_logits, target_bin, reduction="none")
    weights = torch.where(target_bin > 0.5, torch.ones_like(target_bin), target_bin.new_full(target_bin.shape, neg_weight))
    bce = (bce[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(eps)
    pred = torch.sigmoid(rice_logits)[valid]
    gt = target_bin[valid]
    inter = (pred * gt).sum()
    dice = 1.0 - (2.0 * inter + eps) / (pred.sum() + gt.sum() + eps)
    return bce + dice


def binary_tversky_loss(
    rice_logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 1.0,
    eps: float = 1e-6,
):
    valid = target != 255
    if mask is not None:
        valid = valid & mask
    if valid.sum() == 0:
        return rice_logits.new_tensor(0.0)
    target_bin = ((target == 1) | (target == 2)).float()
    prob = torch.sigmoid(rice_logits)
    prob = prob[valid]
    target_bin = target_bin[valid]
    tp = (prob * target_bin).sum()
    fp = (prob * (1.0 - target_bin)).sum()
    fn = ((1.0 - prob) * target_bin).sum()
    tversky = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
    loss = 1.0 - tversky
    if gamma != 1.0:
        loss = loss.pow(gamma)
    return loss


def confident_binary_pseudo_loss(
    student_rice_logits: torch.Tensor,
    teacher_rice_prob: torch.Tensor,
    pos_thr: float = 0.7,
    neg_thr: float = 0.3,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
):
    valid = (teacher_rice_prob >= pos_thr) | (teacher_rice_prob <= neg_thr)
    if valid.sum() == 0:
        return student_rice_logits.new_tensor(0.0)
    pseudo = (teacher_rice_prob >= 0.5).float()
    prob = torch.sigmoid(student_rice_logits)
    prob = prob[valid]
    pseudo = pseudo[valid]
    confidence = (teacher_rice_prob[valid] - 0.5).abs().mul(2.0).clamp_min(0.25)
    bce = nn.functional.binary_cross_entropy(prob, pseudo, reduction="none")
    bce = (bce * confidence).sum() / confidence.sum().clamp_min(eps)
    tp = (prob * pseudo * confidence).sum()
    fp = (prob * (1.0 - pseudo) * confidence).sum()
    fn = ((1.0 - prob) * pseudo * confidence).sum()
    tversky = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
    return bce + (1.0 - tversky)


def rice_head_soft_distill_loss(student_rice_logits: torch.Tensor, teacher_rice_logits: torch.Tensor, eps: float = 1e-6):
    student_prob = torch.sigmoid(student_rice_logits).clamp(eps, 1.0 - eps)
    with torch.no_grad():
        teacher_prob = torch.sigmoid(teacher_rice_logits).clamp(eps, 1.0 - eps)
        teacher_weight = torch.maximum(teacher_prob, 1.0 - teacher_prob)
    bce = nn.functional.binary_cross_entropy(student_prob, teacher_prob, reduction="none")
    bce = (bce * teacher_weight).sum() / teacher_weight.sum().clamp_min(eps)
    inter = (student_prob * teacher_prob * teacher_weight).sum()
    dice = 1.0 - (2.0 * inter + eps) / (
        (student_prob * teacher_weight).sum() + (teacher_prob * teacher_weight).sum() + eps
    )
    return bce + dice


def rice_soft_distill_loss(student_logits, teacher_logits, eps: float = 1e-6):
    student_prob = torch.softmax(student_logits, dim=1)
    student_rice = (student_prob[:, 1] + student_prob[:, 2]).clamp(eps, 1.0 - eps)
    with torch.no_grad():
        teacher_prob = torch.softmax(teacher_logits, dim=1)
        teacher_rice = (teacher_prob[:, 1] + teacher_prob[:, 2]).clamp(eps, 1.0 - eps)
        teacher_weight = torch.maximum(teacher_rice, 1.0 - teacher_rice)
    bce = nn.functional.binary_cross_entropy(student_rice, teacher_rice, reduction="none")
    bce = (bce * teacher_weight).sum() / teacher_weight.sum().clamp_min(eps)
    inter = (student_rice * teacher_rice * teacher_weight).sum()
    dice = 1.0 - (2.0 * inter + eps) / (
        (student_rice * teacher_weight).sum() + (teacher_rice * teacher_weight).sum() + eps
    )
    return bce + dice


def prototype_probs(feat: torch.Tensor, proto: torch.Tensor, temperature: float = 0.1):
    feat_n = nn.functional.normalize(feat, dim=1)
    proto_n = nn.functional.normalize(proto, dim=1)
    sim = torch.einsum("bchw,kc->bkhw", feat_n, proto_n) / max(temperature, 1e-6)
    return torch.softmax(sim, dim=1)


def soft_kl_loss(logits: torch.Tensor, target_prob: torch.Tensor, mask: torch.Tensor | None = None):
    log_prob = torch.log_softmax(logits, dim=1)
    loss = nn.functional.kl_div(log_prob, target_prob, reduction="none").sum(dim=1)
    if mask is not None:
        loss = loss * mask.float()
        denom = mask.float().sum().clamp_min(1.0)
    else:
        denom = loss.numel()
    return loss.sum() / denom


def weighted_soft_kl_loss(
    logits: torch.Tensor,
    target_prob: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
):
    log_prob = torch.log_softmax(logits, dim=1)
    loss = nn.functional.kl_div(log_prob, target_prob, reduction="none").sum(dim=1)
    weight = mask.float()
    if sample_weight is not None:
        sw = sample_weight.float()
        if sw.dim() == 1:
            sw = sw.view(-1, 1, 1)
        weight = weight * sw
    denom = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denom


def neighbor_consistency_loss(feat: torch.Tensor, prob: torch.Tensor, eps: float = 1e-6):
    feat = nn.functional.normalize(feat, dim=1)
    total = feat.new_tensor(0.0)
    count = 0
    for dy, dx in ((0, 1), (1, 0)):
        f1 = feat[:, :, :-dy or None, :-dx or None]
        f2 = feat[:, :, dy:, dx:]
        p1 = prob[:, :-dy or None, :-dx or None]
        p2 = prob[:, dy:, dx:]
        if f1.numel() == 0 or f2.numel() == 0:
            continue
        sim = torch.exp(-((f1 - f2) ** 2).mean(dim=1))
        total = total + (sim * (p1 - p2).abs()).mean()
        count += 1
    if count == 0:
        return feat.new_tensor(0.0)
    return total / count


def spectral_augment_batch(
    img: torch.Tensor,
    gain_std: float = 0.04,
    bias_std: float = 0.02,
    noise_std: float = 0.01,
    band_dropout: float = 0.05,
):
    gain = torch.randn(img.shape[0], img.shape[1], 1, 1, device=img.device) * gain_std + 1.0
    bias = torch.randn(img.shape[0], img.shape[1], 1, 1, device=img.device) * bias_std
    out = img * gain + bias
    if band_dropout > 0:
        drop = (torch.rand(img.shape[0], img.shape[1], 1, 1, device=img.device) < band_dropout).float()
        out = out * (1.0 - drop)
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    return out


def masked_ce_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask & (target != 255)
    if valid.sum() == 0:
        return logits.new_tensor(0.0)
    loss = nn.functional.cross_entropy(logits, target, reduction="none", ignore_index=255)
    return loss[valid].mean()


def masked_dice_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, class_weights=None, eps: float = 1e-6):
    masked_target = target.clone()
    masked_target[~mask] = 255
    return dice_loss(logits, masked_target, class_weights=class_weights, eps=eps)


@torch.no_grad()
def update_prototypes_soft(proto: torch.Tensor, feat: torch.Tensor, prob: torch.Tensor, momentum: float = 0.99):
    feat = nn.functional.normalize(feat, dim=1)
    for cls in range(prob.shape[1]):
        cls_prob = prob[:, cls]
        weight = cls_prob.sum()
        if float(weight) <= 0:
            continue
        weighted_feat = (feat * cls_prob.unsqueeze(1)).sum(dim=(0, 2, 3)) / weight.clamp_min(1e-6)
        proto[cls].mul_(momentum).add_(weighted_feat, alpha=1 - momentum)


def entropy_from_prob(prob: torch.Tensor) -> torch.Tensor:
    return -(prob * (prob.clamp_min(1e-8)).log()).sum(dim=1)


@torch.no_grad()
def update_ema(teacher: nn.Module, student: nn.Module, alpha: float = 0.99) -> None:
    t_state = teacher.state_dict()
    s_state = student.state_dict()
    for k in t_state.keys():
        if t_state[k].dtype.is_floating_point:
            t_state[k].mul_(alpha).add_(s_state[k], alpha=1 - alpha)
        else:
            t_state[k].copy_(s_state[k])


def build_proto_loss(feat: torch.Tensor, label: torch.Tensor, proto: torch.Tensor, mask: torch.Tensor | None = None):
    if mask is None:
        mask = label != 255
    if mask.sum() == 0:
        return feat.new_tensor(0.0)
    feat_n = nn.functional.normalize(feat, dim=1)
    proto_n = nn.functional.normalize(proto, dim=1)
    total = feat.new_tensor(0.0)
    count = 0
    for cls in torch.unique(label[mask]):
        if int(cls) < 0 or int(cls) >= proto.shape[0]:
            continue
        cls_mask = mask & (label == cls)
        if cls_mask.sum() == 0:
            continue
        f = feat_n.permute(0, 2, 3, 1)[cls_mask]
        p = proto_n[int(cls)]
        total = total + (1 - (f * p).sum(dim=-1)).mean()
        count += 1
    if count == 0:
        return feat.new_tensor(0.0)
    return total / count


def prototype_contrast_loss(
    feat: torch.Tensor,
    label: torch.Tensor,
    proto: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 0.1,
):
    if mask is None:
        mask = label != 255
    if mask.sum() == 0:
        return feat.new_tensor(0.0)
    feat_n = nn.functional.normalize(feat, dim=1)
    proto_n = nn.functional.normalize(proto, dim=1)
    sim = torch.einsum("bchw,kc->bkhw", feat_n, proto_n) / max(temperature, 1e-6)
    target = label.clamp(0, proto.shape[0] - 1)
    loss = nn.functional.cross_entropy(sim, target, reduction="none", ignore_index=255)
    weight = mask.float()
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def refine_probs_with_prototypes(
    base_prob: torch.Tensor,
    feat: torch.Tensor,
    proto: torch.Tensor,
    temperature: float = 0.1,
    refine_strength: float = 0.5,
):
    proto_prob = prototype_probs(feat, proto, temperature=temperature)
    base_ent = entropy_from_prob(base_prob) / math.log(base_prob.shape[1])
    proto_conf = proto_prob.max(dim=1).values
    refine_gate = (base_ent * (1.0 - proto_conf)).clamp(0.0, 1.0)
    refine_gate = (refine_strength * refine_gate).clamp(0.0, 1.0).unsqueeze(1)
    refined = base_prob * (1.0 - refine_gate) + proto_prob * refine_gate
    return refined / refined.sum(dim=1, keepdim=True).clamp_min(1e-6)


def complementary_class_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight_map: torch.Tensor | None = None,
    eps: float = 1e-6,
):
    valid = mask & (target != 255)
    if valid.sum() == 0:
        return logits.new_tensor(0.0)
    prob = torch.softmax(logits, dim=1)
    comp_idx = torch.zeros_like(target)
    rice_mask = ((target == 1) | (target == 2)) & valid
    nonrice_mask = ((target == 0) | (target == 3)) & valid
    if rice_mask.any():
        rice_comp = torch.where(
            prob[:, 0] >= prob[:, 3],
            torch.zeros_like(target),
            torch.full_like(target, 3),
        )
        comp_idx = torch.where(rice_mask, rice_comp, comp_idx)
    if nonrice_mask.any():
        nonrice_comp = torch.where(
            prob[:, 1] >= prob[:, 2],
            torch.ones_like(target),
            torch.full_like(target, 2),
        )
        comp_idx = torch.where(nonrice_mask, nonrice_comp, comp_idx)
    comp_prob = prob.gather(1, comp_idx.unsqueeze(1)).squeeze(1).clamp_min(eps)
    loss = -torch.log((1.0 - comp_prob).clamp_min(eps))
    weight = valid.float()
    if weight_map is not None:
        weight = weight * weight_map.float()
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def extrapolate_features(
    feat: torch.Tensor,
    label: torch.Tensor,
    proto: torch.Tensor,
    gamma: float = 0.2,
    mask: torch.Tensor | None = None,
):
    if mask is None:
        mask = label != 255
    if mask.sum() == 0:
        return feat
    label_idx = label.clamp(0, proto.shape[0] - 1)
    proto_map = proto[label_idx].permute(0, 3, 1, 2)
    if mask.dtype != torch.bool:
        mask = mask.bool()
    feat_n = nn.functional.normalize(feat, dim=1)
    proto_n = nn.functional.normalize(proto_map, dim=1)
    hardness = (1.0 - (feat_n * proto_n).sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
    gamma_map = gamma * hardness
    gamma_map = gamma_map * mask.unsqueeze(1).float()
    feat_ex = feat + gamma_map * (feat - proto_map)
    return torch.where(mask.unsqueeze(1), feat_ex, feat)


@torch.no_grad()
def update_prototypes(proto: torch.Tensor, feat: torch.Tensor, label: torch.Tensor, momentum: float = 0.99):
    feat = nn.functional.normalize(feat, dim=1)
    for cls in torch.unique(label[label != 255]):
        cls = int(cls)
        cls_mask = label == cls
        if cls_mask.sum() == 0:
            continue
        f = feat.permute(0, 2, 3, 1)[cls_mask].mean(dim=0)
        proto[cls].mul_(momentum).add_(f, alpha=1 - momentum)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
    lambda_rice_head_source: float = 0.0,
    rice_head_neg_weight: float = 1.0,
    rice_tversky_alpha: float = 0.3,
    rice_tversky_beta: float = 0.7,
    rice_tversky_gamma: float = 1.0,
):
    model.train()
    total = 0.0
    for batch in loader:
        img = batch["img"].to(device)
        mask = batch["mask"].to(device)
        logits, feat = model(img, return_feat=True)
        loss = weighted_ce_loss(logits, mask, class_weights) + dice_loss(logits, mask, class_weights)
        if lambda_rice_head_source > 0:
            rice_logits = model.rice_classifier(feat).squeeze(1)
            loss = loss + lambda_rice_head_source * binary_tversky_loss(
                rice_logits,
                mask,
                alpha=rice_tversky_alpha,
                beta=rice_tversky_beta,
                gamma=rice_tversky_gamma,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


def eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 4,
    rice_threshold: float = 0.5,
):
    model.eval()
    meter = ConfusionMeter(num_classes)
    rice_tp = 0
    rice_gt = 0
    rice_pred = 0
    nonrice_tp = 0
    nonrice_gt = 0
    nonrice_pred = 0
    main_rice_tp = 0
    main_rice_gt = 0
    main_rice_pred = 0
    main_nonrice_tp = 0
    main_nonrice_gt = 0
    main_nonrice_pred = 0
    with torch.no_grad():
        for batch in loader:
            img = batch["img"].to(device)
            mask = batch["mask"].cpu().numpy()
            logits, feat = model(img, return_feat=True)
            pred = logits.argmax(dim=1).cpu().numpy()
            rice_prob = torch.sigmoid(model.rice_classifier(feat).squeeze(1))
            rice_pred_batch = (rice_prob > rice_threshold).cpu().numpy()
            for p, t in zip(pred, mask):
                meter.add(p, t)
                valid = t != 255
                gt_rice = ((t == 1) | (t == 2)) & valid
                pred_rice = ((p == 1) | (p == 2)) & valid
                main_rice_tp += int(np.count_nonzero(gt_rice & pred_rice))
                main_rice_gt += int(np.count_nonzero(gt_rice))
                main_rice_pred += int(np.count_nonzero(pred_rice))
                gt_nonrice = (~gt_rice) & valid
                pred_nonrice = (~pred_rice) & valid
                main_nonrice_tp += int(np.count_nonzero(gt_nonrice & pred_nonrice))
                main_nonrice_gt += int(np.count_nonzero(gt_nonrice))
                main_nonrice_pred += int(np.count_nonzero(pred_nonrice))
            for p_rice, t in zip(rice_pred_batch, mask):
                valid = t != 255
                gt_rice = ((t == 1) | (t == 2)) & valid
                pred_rice = p_rice & valid
                rice_tp += int(np.count_nonzero(gt_rice & pred_rice))
                rice_gt += int(np.count_nonzero(gt_rice))
                rice_pred += int(np.count_nonzero(pred_rice))
                gt_nonrice = (~gt_rice) & valid
                pred_nonrice = (~p_rice) & valid
                nonrice_tp += int(np.count_nonzero(gt_nonrice & pred_nonrice))
                nonrice_gt += int(np.count_nonzero(gt_nonrice))
                nonrice_pred += int(np.count_nonzero(pred_nonrice))
    metrics = meter.compute()
    rice_union = rice_gt + rice_pred - rice_tp
    nonrice_union = nonrice_gt + nonrice_pred - nonrice_tp
    main_rice_union = main_rice_gt + main_rice_pred - main_rice_tp
    main_nonrice_union = main_nonrice_gt + main_nonrice_pred - main_nonrice_tp
    rice_iou = float(rice_tp / rice_union) if rice_union > 0 else 0.0
    nonrice_iou = float(nonrice_tp / nonrice_union) if nonrice_union > 0 else 0.0
    main_rice_iou = float(main_rice_tp / main_rice_union) if main_rice_union > 0 else 0.0
    main_nonrice_iou = float(main_nonrice_tp / main_nonrice_union) if main_nonrice_union > 0 else 0.0
    metrics["main_binary_iou"] = main_rice_iou
    metrics["main_binary_nonrice_iou"] = main_nonrice_iou
    metrics["main_binary_miou"] = 0.5 * (main_rice_iou + main_nonrice_iou)
    metrics["rice_head_iou"] = rice_iou
    metrics["rice_head_nonrice_iou"] = nonrice_iou
    metrics["rice_head_miou"] = 0.5 * (rice_iou + nonrice_iou)
    metrics["rice_head_precision"] = float(rice_tp / rice_pred) if rice_pred > 0 else 0.0
    metrics["rice_head_recall"] = float(rice_tp / rice_gt) if rice_gt > 0 else 0.0
    return metrics


def load_manifest(data_root: Path):
    return json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))


def make_loaders(data_root: Path, batch_size: int, num_workers: int, augment: bool = True, target_augment: bool = True):
    manifest = load_manifest(data_root)
    mean = np.array(manifest["stats"]["mean"], dtype=np.float32)
    std = np.array(manifest["stats"]["std"], dtype=np.float32)
    source_train = sorted((data_root / "source_train").glob("*.npz"))
    source_val = sorted((data_root / "source_val").glob("*.npz"))
    uav_val = sorted((data_root / "uav_val").glob("*.npz"))
    target = sorted((data_root / "target").glob("*.npz"))
    source_ds = PatchNPZDataset(source_train, mean, std, labeled=True, augment=augment, band_dropout=0.1)
    source_val_ds = PatchNPZDataset(source_val, mean, std, labeled=True, augment=False)
    target_ds = PatchNPZDataset(target, mean, std, labeled=False, augment=target_augment, band_dropout=0.15 if target_augment else 0.0)
    uav_val_ds = PatchNPZDataset(uav_val, mean, std, labeled=True, augment=False)
    return (
        source_ds,
        source_val_ds,
        target_ds,
        uav_val_ds,
        torch.utils.data.DataLoader(source_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True),
        torch.utils.data.DataLoader(source_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        torch.utils.data.DataLoader(target_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True),
        torch.utils.data.DataLoader(target_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        torch.utils.data.DataLoader(uav_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        manifest,
    )


def compute_class_weights(manifest: dict, device: torch.device):
    counts = manifest["stats"]["class_counts"]
    weights = np.ones(4, dtype=np.float32)
    total = sum(counts.values())
    for cls in range(4):
        freq = counts.get(str(cls), counts.get(cls, 1))
        weights[cls] = 1.0 / math.sqrt(max(freq / total, 1e-8))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def collect_target_bank(model: nn.Module | list[nn.Module], loader: DataLoader, device: torch.device, temperature: float):
    models = model if isinstance(model, list) else [model]
    for m in models:
        m.eval()
    feat_bank = None
    prob_bank = None
    for batch in loader:
        img = batch["img"].to(device)
        idx = batch["idx"].cpu().numpy()
        prob_sum = None
        feat_sum = None
        for m in models:
            logits, feat = m(img, return_feat=True)
            prob = torch.softmax(logits / max(temperature, 1e-6), dim=1)
            feat_g = nn.functional.normalize(feat.mean(dim=(2, 3)), dim=1)
            prob_sum = prob if prob_sum is None else prob_sum + prob
            feat_sum = feat_g if feat_sum is None else feat_sum + feat_g
        prob = (prob_sum / len(models)).cpu().to(torch.float16).numpy()
        feat_g = (feat_sum / len(models)).cpu().numpy().astype(np.float32)
        if feat_bank is None:
            n = len(loader.dataset)
            feat_bank = np.zeros((n, feat_g.shape[1]), dtype=np.float32)
            prob_bank = np.zeros((n, prob.shape[1], prob.shape[2], prob.shape[3]), dtype=np.float16)
        feat_bank[idx] = feat_g
        prob_bank[idx] = prob
    return feat_bank, prob_bank


def denoise_target_bank(
    feat_bank: np.ndarray,
    prob_bank: np.ndarray,
    neighbor_k: int = 5,
    sim_temperature: float = 0.2,
    prior_gamma: float = 1.0,
    stable_conf_thr: float = 0.7,
    stable_ent_thr: float = 0.9,
):
    n, c, h, w = prob_bank.shape
    feat_norm = feat_bank / np.clip(np.linalg.norm(feat_bank, axis=1, keepdims=True), 1e-6, None)
    sim = feat_norm @ feat_norm.T
    np.fill_diagonal(sim, -np.inf)
    k = max(1, min(neighbor_k, n - 1))
    topk = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    target_soft = np.zeros((n, c, h, w), dtype=np.float16)
    target_pseudo = np.zeros((n, h, w), dtype=np.uint8)
    target_stable = np.zeros((n, h, w), dtype=np.bool_)
    target_weight = np.zeros((n,), dtype=np.float32)
    sample_prior = prob_bank.mean(axis=(2, 3)).astype(np.float32)
    for i in range(n):
        neigh = topk[i]
        neigh_sim = sim[i, neigh]
        neigh_sim = neigh_sim - np.max(neigh_sim)
        neigh_w = np.exp(neigh_sim / max(sim_temperature, 1e-6))
        neigh_w = neigh_w / np.clip(neigh_w.sum(), 1e-6, None)
        prior = (sample_prior[neigh] * neigh_w[:, None]).sum(axis=0)
        prior = np.power(np.clip(prior, 1e-6, 1.0), prior_gamma)
        p = prob_bank[i].astype(np.float32)
        p_adj = p * prior[:, None, None]
        p_adj = p_adj / np.clip(p_adj.sum(axis=0, keepdims=True), 1e-6, None)
        p_t = torch.from_numpy(p_adj[None]).float()
        p_smooth = 0.5 * p_t + 0.5 * torch.nn.functional.avg_pool2d(p_t, 3, 1, 1)
        p_smooth = p_smooth[0].numpy()
        pseudo = p_smooth.argmax(axis=0).astype(np.uint8)
        conf = p_smooth.max(axis=0)
        ent = -(p_smooth * np.log(np.clip(p_smooth, 1e-8, 1.0))).sum(axis=0)
        stable = (conf >= stable_conf_thr) & (ent <= stable_ent_thr) & (pseudo == p_adj.argmax(axis=0))
        target_soft[i] = p_smooth.astype(np.float16)
        target_pseudo[i] = pseudo
        target_stable[i] = stable
        target_weight[i] = float(stable.mean())
    return target_soft, target_pseudo, target_stable, target_weight, sample_prior


def build_classwise_reliability_bank(
    prob_bank: np.ndarray,
    class_quantiles: list[float] | tuple[float, ...],
    min_reliable_frac: float = 0.01,
):
    prob = torch.from_numpy(prob_bank.astype(np.float32))
    pred = prob.argmax(dim=1)
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=1)
    entropy = entropy / math.log(prob.shape[1])
    one_hot = torch.nn.functional.one_hot(pred, num_classes=prob.shape[1]).permute(0, 3, 1, 2).float()
    local_vote = torch.nn.functional.avg_pool2d(one_hot, 3, stride=1, padding=1)
    local_consistency = local_vote.max(dim=1).values
    reliability = 0.5 * (1.0 - entropy) + 0.5 * local_consistency
    reliable = torch.zeros_like(pred, dtype=torch.bool)
    class_thr = {}
    class_stats = {}
    for cls in range(prob.shape[1]):
        cls_mask = pred == cls
        cls_rel = reliability[cls_mask]
        if cls_rel.numel() == 0:
            class_thr[cls] = 1.0
            class_stats[cls] = {"count": 0, "thr": 1.0, "mean_rel": 0.0}
            continue
        q = class_quantiles[min(cls, len(class_quantiles) - 1)]
        thr = float(torch.quantile(cls_rel, q).item())
        cls_reliable = cls_mask & (reliability >= thr)
        if float(cls_reliable.float().mean()) < min_reliable_frac:
            thr = float(torch.quantile(cls_rel, max(0.0, q - 0.15)).item())
            cls_reliable = cls_mask & (reliability >= thr)
        reliable = reliable | cls_reliable
        class_thr[cls] = thr
        class_stats[cls] = {
            "count": int(cls_reliable.sum().item()),
            "thr": thr,
            "mean_rel": float(reliability[cls_reliable].mean().item()) if cls_reliable.any() else 0.0,
        }
    target_soft = prob.cpu().to(torch.float16).numpy()
    target_pseudo = pred.cpu().numpy().astype(np.uint8)
    target_reliable = reliable.cpu().numpy().astype(np.bool_)
    target_weight = reliability.cpu().numpy().astype(np.float32)
    return target_soft, target_pseudo, target_reliable, target_weight, class_thr, class_stats


def lotus_margin_contrast_loss(
    feat: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lotus_proto: torch.Tensor,
    rice_proto: torch.Tensor,
    reliability: torch.Tensor | None = None,
    rice_weight: float = 0.1,
    temperature: float = 0.1,
    eps: float = 1e-6,
):
    valid = mask & (target != 255)
    if valid.sum() == 0:
        return feat.new_tensor(0.0)
    feat_n = nn.functional.normalize(feat, dim=1)
    lotus_proto = nn.functional.normalize(lotus_proto.view(1, -1), dim=1)[0]
    rice_proto = nn.functional.normalize(rice_proto.view(1, -1), dim=1)[0]
    sim_lotus = (feat_n * lotus_proto.view(1, -1, 1, 1)).sum(dim=1)
    sim_rice = (feat_n * rice_proto.view(1, -1, 1, 1)).sum(dim=1)
    margin = (sim_lotus - sim_rice) / max(temperature, eps)
    target_lotus = (target == 3) & valid
    target_rice = ((target == 1) | (target == 2)) & valid
    if reliability is None:
        reliability = torch.ones_like(target, dtype=feat.dtype)
    rel = reliability.float().clamp(0.25, 1.0)
    loss = feat.new_tensor(0.0)
    count = 0
    if target_lotus.any():
        w = rel[target_lotus]
        loss = loss + (nn.functional.softplus(-margin[target_lotus]) * w).sum() / w.sum().clamp_min(eps)
        count += 1
    if target_rice.any():
        w = rel[target_rice]
        loss = loss + rice_weight * (nn.functional.softplus(margin[target_rice]) * w).sum() / w.sum().clamp_min(eps)
        count += 1
    if count == 0:
        return feat.new_tensor(0.0)
    return loss / count


def hard_nonrice_margin_loss(
    feat: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    nonrice_proto: torch.Tensor,
    rice_proto: torch.Tensor,
    reliability: torch.Tensor | None = None,
    rice_prob: torch.Tensor | None = None,
    rice_prob_thr: float = 0.6,
    temperature: float = 0.1,
    eps: float = 1e-6,
):
    valid = mask & (target != 255)
    if rice_prob is not None:
        valid = valid & (rice_prob >= rice_prob_thr)
    hard_nonrice = valid & ((target == 0) | (target == 3))
    if hard_nonrice.sum() == 0:
        return feat.new_tensor(0.0)
    feat_n = nn.functional.normalize(feat, dim=1)
    nonrice_proto = nn.functional.normalize(nonrice_proto.view(1, -1), dim=1)[0]
    rice_proto = nn.functional.normalize(rice_proto.view(1, -1), dim=1)[0]
    sim_nonrice = (feat_n * nonrice_proto.view(1, -1, 1, 1)).sum(dim=1)
    sim_rice = (feat_n * rice_proto.view(1, -1, 1, 1)).sum(dim=1)
    margin = (sim_nonrice - sim_rice) / max(temperature, eps)
    if reliability is None:
        reliability = torch.ones_like(target, dtype=feat.dtype)
    w = reliability.float().clamp(0.25, 1.0)
    w = w[hard_nonrice]
    return (nn.functional.softplus(-margin[hard_nonrice]) * w).sum() / w.sum().clamp_min(eps)


def class_ratio_prior_loss(
    logits: torch.Tensor,
    prior: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
):
    prob = torch.softmax(logits, dim=1)
    if mask is not None:
        weight = mask.float()
        if weight.sum() == 0:
            return logits.new_tensor(0.0)
        mean_prob = (prob * weight.unsqueeze(1)).sum(dim=(0, 2, 3)) / weight.sum().clamp_min(1.0)
    else:
        mean_prob = prob.mean(dim=(0, 2, 3))
    mean_prob = mean_prob.clamp_min(eps)
    prior = prior.to(device=logits.device, dtype=logits.dtype).clamp_min(eps)
    prior = prior / prior.sum().clamp_min(eps)
    return (mean_prob * (mean_prob.log() - prior.log())).sum()


@torch.no_grad()
def update_group_prototype(
    proto: torch.Tensor,
    feat: torch.Tensor,
    label: torch.Tensor,
    class_ids: tuple[int, ...] | list[int],
    mask: torch.Tensor | None = None,
    momentum: float = 0.99,
):
    if mask is None:
        mask = label != 255
    else:
        mask = mask & (label != 255)
    class_mask = torch.zeros_like(mask)
    for cls in class_ids:
        class_mask = class_mask | (label == cls)
    class_mask = class_mask & mask
    if class_mask.sum() == 0:
        return
    feat = nn.functional.normalize(feat, dim=1)
    pooled = feat.permute(0, 2, 3, 1)[class_mask].mean(dim=0)
    proto.mul_(momentum).add_(pooled, alpha=1 - momentum)


def set_trainable_params(model: nn.Module, head_only: bool = True):
    for p in model.parameters():
        p.requires_grad_(False)
    modules = [model.proj, model.classifier, model.rice_classifier]
    if not head_only:
        modules = [model.enc1, model.enc2, model.enc3, model.bottleneck, model.up3, model.dec3, model.up2, model.dec2, model.up1, model.dec1, model.proj, model.classifier, model.rice_classifier]
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(True)


def run_source_only(args, data_root: Path, device: torch.device, out_dir: Path):
    _, source_val_ds, _, uav_val_ds, source_loader, source_val_loader, _, _, uav_loader, manifest = make_loaders(
        data_root, args.batch_size, args.num_workers, augment=True
    )
    model = SimpleUNet(in_ch=32, num_classes=4, base=args.base_ch, feat_dim=args.feat_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = compute_class_weights(manifest, device)
    best = -1.0
    best_path = out_dir / "source_best.pt"
    for epoch in range(1, args.epochs_source + 1):
        loss = train_epoch(
            model,
            source_loader,
            optimizer,
            device,
            class_weights,
            args.lambda_rice_head_source,
            args.rice_head_neg_weight,
            args.rice_tversky_alpha,
            args.rice_tversky_beta,
            args.rice_tversky_gamma,
        )
        val = eval_model(model, source_val_loader, device, rice_threshold=args.rice_threshold)
        test = eval_model(model, uav_loader, device, rice_threshold=args.rice_threshold)
        print(
            f"[source] epoch={epoch} loss={loss:.4f} "
            f"source_fg_mIoU={val['miou_fg']:.4f} uav_fg_mIoU={test['miou_fg']:.4f} "
            f"uav_riceIoU={test['rice_exposure_iou']:.4f} uav_lateIoU={test['iou'][2]:.4f} "
            f"uav_lotusIoU={test['iou'][3]:.4f} uav_mainBinary_mIoU={test['main_binary_miou']:.4f} "
            f"uav_riceHeadIoU={test['rice_head_iou']:.4f} uav_nonRiceHeadIoU={test['rice_head_nonrice_iou']:.4f} "
            f"uav_riceHead_mIoU={test['rice_head_miou']:.4f}"
        )
        score = test[args.selection_metric]
        write_history(
            out_dir,
            {
                "mode": "source",
                "epoch": epoch,
                "loss": float(loss),
                "source": val,
                "uav": test,
                "selection_metric": f"uav.{args.selection_metric}",
            },
        )
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "manifest": manifest, "best_uav": test}, best_path)
    return best_path


def self_train(
    args,
    data_root: Path,
    device: torch.device,
    out_dir: Path,
    init_ckpt: Path,
):
    _, source_val_ds, target_ds, uav_val_ds, source_loader, source_val_loader, target_loader, target_eval_loader, uav_loader, manifest = make_loaders(
        data_root, args.batch_size, args.num_workers, augment=True
    )
    student = SimpleUNet(in_ch=32, num_classes=4, base=args.base_ch, feat_dim=args.feat_dim).to(device)
    teacher = SimpleUNet(in_ch=32, num_classes=4, base=args.base_ch, feat_dim=args.feat_dim).to(device)
    ckpt = torch.load(init_ckpt, map_location=device)
    student.load_state_dict(ckpt["model"], strict=False)
    teacher.load_state_dict(ckpt["model"], strict=False)
    class_weights = compute_class_weights(manifest, device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    proto = torch.zeros(4, args.feat_dim, device=device)
    student.eval()
    with torch.no_grad():
        for batch in source_loader:
            s_img = batch["img"].to(device)
            s_mask = batch["mask"].to(device)
            _, s_feat = student(s_img, return_feat=True)
            update_prototypes(proto, s_feat, s_mask, momentum=0.0)
    student.train()
    best = -1.0
    best_path = out_dir / "uda_best.pt"
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    for epoch in range(1, args.epochs_uda + 1):
        student.train()
        total_loss = 0.0
        steps = max(len(source_loader), len(target_loader))
        for _ in range(steps):
            try:
                sb = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                sb = next(source_iter)
            try:
                tb = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tb = next(target_iter)
            s_img = sb["img"].to(device)
            s_mask = sb["mask"].to(device)
            t_img = tb["img"].to(device)
            with torch.no_grad():
                t_logits_w, t_feat_w = teacher(t_img, return_feat=True)
                t_prob = torch.softmax(t_logits_w / args.temperature, dim=1)
                t_conf, t_pseudo = t_prob.max(dim=1)
                t_ent = entropy_from_prob(t_prob)
                t_mask = (t_conf > args.tau) & (t_ent < args.entropy_thr)
                p_proto = prototype_probs(t_feat_w, proto, temperature=args.proto_temp)
                mix_w = t_conf.unsqueeze(1)
                target_soft = mix_w * t_prob + (1.0 - mix_w) * p_proto
                target_soft = target_soft / target_soft.sum(dim=1, keepdim=True).clamp_min(1e-6)
            s_logits, s_feat = student(s_img, return_feat=True)
            t_logits, t_feat = student(t_img, return_feat=True)
            sup_loss = weighted_ce_loss(s_logits, s_mask, class_weights) + dice_loss(s_logits, s_mask, class_weights)
            sup_rice_loss = rice_exposure_loss(s_logits, s_mask)
            s_rice_head_logits = student.rice_classifier(s_feat).squeeze(1)
            t_rice_head_logits = student.rice_classifier(t_feat).squeeze(1)
            t_rice_head_logits_w = teacher.rice_classifier(t_feat_w).squeeze(1)
            t_rice_prob_w = torch.sigmoid(t_rice_head_logits_w)
            t_rice_prob_main = torch.softmax(t_logits_w, dim=1)[:, 1] + torch.softmax(t_logits_w, dim=1)[:, 2]
            t_rice_prob_teacher = 0.5 * t_rice_prob_w + 0.5 * t_rice_prob_main
            sup_rice_head_loss = binary_tversky_loss(
                s_rice_head_logits,
                s_mask,
                alpha=args.rice_tversky_alpha,
                beta=args.rice_tversky_beta,
                gamma=args.rice_tversky_gamma,
            )
            if t_mask.any():
                unsup_ce = nn.functional.cross_entropy(t_logits, t_pseudo, reduction="none")
                unsup_ce = (unsup_ce * t_mask.float()).sum() / t_mask.float().sum()
                pseudo_masked = t_pseudo.clone()
                pseudo_masked[~t_mask] = 255
                unsup_dice = dice_loss(t_logits, pseudo_masked, class_weights)
                unsup_loss = unsup_ce + unsup_dice
            else:
                unsup_loss = t_logits.new_tensor(0.0)
            proto_loss = build_proto_loss(s_feat, s_mask, proto)
            if t_mask.any():
                proto_loss = proto_loss + build_proto_loss(t_feat, t_pseudo, proto, t_mask)
            target_rice_loss = confident_binary_pseudo_loss(
                t_rice_head_logits,
                t_rice_prob_teacher,
                pos_thr=args.rice_pos_thr,
                neg_thr=args.rice_neg_thr,
                alpha=args.rice_tversky_alpha,
                beta=args.rice_tversky_beta,
            )
            target_rice_head_loss = 0.5 * rice_head_soft_distill_loss(t_rice_head_logits, t_rice_head_logits_w)
            target_soft_loss = soft_kl_loss(t_logits, target_soft)
            t_rice_prob = torch.sigmoid(t_rice_head_logits)
            smooth_loss = neighbor_consistency_loss(t_feat, t_rice_prob)
            loss = (
                sup_loss
                + args.lambda_rice_source * sup_rice_loss
                + args.lambda_rice_head_source * sup_rice_head_loss
                + args.lambda_u * unsup_loss
                + args.lambda_rice_target * target_rice_loss
                + args.lambda_rice_head_target * target_rice_head_loss
                + args.lambda_target_soft * target_soft_loss
                + args.lambda_smooth_target * smooth_loss
                + args.lambda_proto * proto_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update_ema(teacher, student, alpha=args.ema)
            update_prototypes(proto, s_feat.detach(), s_mask, momentum=args.proto_momentum)
            update_prototypes_soft(proto, t_feat.detach(), target_soft.detach(), momentum=args.proto_momentum)
            total_loss += float(loss.item())
        val = eval_model(student, source_val_loader, device, rice_threshold=args.rice_threshold)
        test = eval_model(student, uav_loader, device, rice_threshold=args.rice_threshold)
        print(
            f"[uda] epoch={epoch} loss={total_loss / max(steps,1):.4f} "
            f"source_fg_mIoU={val['miou_fg']:.4f} uav_fg_mIoU={test['miou_fg']:.4f} "
            f"uav_riceIoU={test['rice_exposure_iou']:.4f} uav_lateIoU={test['iou'][2]:.4f} "
            f"uav_lotusIoU={test['iou'][3]:.4f} uav_mainBinary_mIoU={test['main_binary_miou']:.4f} "
            f"uav_riceHeadIoU={test['rice_head_iou']:.4f} uav_nonRiceHeadIoU={test['rice_head_nonrice_iou']:.4f} "
            f"uav_riceHead_mIoU={test['rice_head_miou']:.4f}"
        )
        score = test[args.selection_metric]
        write_history(
            out_dir,
            {
                "mode": "uda",
                "epoch": epoch,
                "loss": float(total_loss / max(steps, 1)),
                "source": val,
                "uav": test,
                "selection_metric": f"uav.{args.selection_metric}",
            },
        )
        if score > best:
            best = score
            torch.save(
                {
                    "model": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "proto": proto.detach().cpu(),
                    "manifest": manifest,
                    "best_uav": test,
                },
                best_path,
            )
    return best_path


def self_train_snd(
    args,
    data_root: Path,
    device: torch.device,
    out_dir: Path,
    init_ckpt: Path,
):
    _, source_val_ds, target_ds, uav_val_ds, source_loader, source_val_loader, target_loader, target_eval_loader, uav_loader, manifest = make_loaders(
        data_root, args.batch_size, args.num_workers, augment=True, target_augment=False
    )
    student = SimpleUNet(in_ch=32, num_classes=4, base=args.base_ch, feat_dim=args.feat_dim).to(device)
    teacher = SimpleUNet(in_ch=32, num_classes=4, base=args.base_ch, feat_dim=args.feat_dim).to(device)
    ckpt = torch.load(init_ckpt, map_location=device)
    student.load_state_dict(ckpt["model"], strict=False)
    teacher.load_state_dict(ckpt["model"], strict=False)
    bank_teachers: list[nn.Module] = []
    for teacher_ckpt in args.teacher_ckpts:
        tm = SimpleUNet(in_ch=32, num_classes=4, base=args.base_ch, feat_dim=args.feat_dim).to(device)
        tm_ckpt = torch.load(teacher_ckpt, map_location=device)
        tm.load_state_dict(tm_ckpt["model"], strict=False)
        tm.eval()
        for p in tm.parameters():
            p.requires_grad_(False)
        bank_teachers.append(tm)
    bank_models = bank_teachers if bank_teachers else [teacher]
    if args.adapt_head_only:
        set_trainable_params(student, head_only=True)
    class_weights = compute_class_weights(manifest, device)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    proto = torch.zeros(4, args.feat_dim, device=device)
    student.eval()
    with torch.no_grad():
        for batch in source_loader:
            s_img = batch["img"].to(device)
            s_mask = batch["mask"].to(device)
            _, s_feat = student(s_img, return_feat=True)
            update_prototypes(proto, s_feat, s_mask, momentum=0.0)
    student.train()
    best = -1.0
    best_path = out_dir / "uda_snd_best.pt"
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    for epoch in range(1, args.epochs_uda + 1):
        feat_bank, prob_bank = collect_target_bank(bank_models, target_eval_loader, device, args.temperature)
        target_soft_bank, target_pseudo_bank, target_stable_bank, target_weight_bank, _ = denoise_target_bank(
            feat_bank,
            prob_bank,
            neighbor_k=args.neighbor_k,
            sim_temperature=args.bank_temp,
            prior_gamma=args.prior_gamma,
            stable_conf_thr=args.stable_conf_thr,
            stable_ent_thr=args.stable_ent_thr,
        )
        student.train()
        total_loss = 0.0
        steps = max(len(source_loader), len(target_loader))
        for _ in range(steps):
            try:
                sb = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                sb = next(source_iter)
            try:
                tb = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tb = next(target_iter)
            s_img = sb["img"].to(device)
            s_mask = sb["mask"].to(device)
            t_img = tb["img"].to(device)
            t_idx = tb["idx"].cpu().numpy()
            t_img_strong = spectral_augment_batch(t_img)
            with torch.no_grad():
                t_teacher_logits, t_teacher_feat = teacher(t_img, return_feat=True)
                t_teacher_rice_logits = teacher.rice_classifier(t_teacher_feat).squeeze(1)
            t_logits, t_feat = student(t_img_strong, return_feat=True)
            s_logits, s_feat = student(s_img, return_feat=True)
            sup_loss = weighted_ce_loss(s_logits, s_mask, class_weights) + dice_loss(s_logits, s_mask, class_weights)
            s_feat_ex = extrapolate_features(s_feat, s_mask, proto, gamma=args.feature_extrap_gamma)
            s_logits_ex = student.classifier(s_feat_ex)
            sup_ex_loss = weighted_ce_loss(s_logits_ex, s_mask, class_weights) + dice_loss(s_logits_ex, s_mask, class_weights)
            sup_rice_loss = rice_exposure_loss(s_logits, s_mask)
            s_rice_head_logits = student.rice_classifier(s_feat).squeeze(1)
            sup_rice_head_loss = binary_tversky_loss(
                s_rice_head_logits,
                s_mask,
                alpha=args.rice_tversky_alpha,
                beta=args.rice_tversky_beta,
                gamma=args.rice_tversky_gamma,
            )
            target_soft = torch.from_numpy(target_soft_bank[t_idx]).to(device=device, dtype=torch.float32)
            target_pseudo = torch.from_numpy(target_pseudo_bank[t_idx]).to(device=device, dtype=torch.long)
            target_stable = torch.from_numpy(target_stable_bank[t_idx]).to(device=device)
            target_weight = torch.from_numpy(target_weight_bank[t_idx]).to(device=device)
            quality = target_weight.mean().clamp(0.05, 1.0)
            pseudo_masked = target_pseudo.clone()
            pseudo_masked[~target_stable] = 255
            target_soft_loss = weighted_soft_kl_loss(t_logits, target_soft, target_stable, target_weight)
            target_ce = masked_ce_loss(t_logits, pseudo_masked, target_stable)
            target_dice = masked_dice_loss(t_logits, pseudo_masked, target_stable, class_weights)
            target_rice_prob = (target_soft[:, 1] + target_soft[:, 2]).clamp(0.0, 1.0)
            target_rice_loss = confident_binary_pseudo_loss(
                student.rice_classifier(t_feat).squeeze(1),
                target_rice_prob,
                pos_thr=args.rice_pos_thr,
                neg_thr=args.rice_neg_thr,
                alpha=args.rice_tversky_alpha,
                beta=args.rice_tversky_beta,
            )
            target_rice_head_loss = 0.5 * rice_head_soft_distill_loss(
                student.rice_classifier(t_feat).squeeze(1),
                t_teacher_rice_logits,
            )
            proto_loss = build_proto_loss(s_feat, s_mask, proto)
            proto_loss = proto_loss + build_proto_loss(t_feat, pseudo_masked, proto, target_stable)
            t_rice_prob = torch.sigmoid(student.rice_classifier(t_feat).squeeze(1))
            smooth_loss = neighbor_consistency_loss(t_feat, t_rice_prob)
            loss = (
                sup_loss
                + args.lambda_rice_source * sup_rice_loss
                + args.lambda_rice_head_source * sup_rice_head_loss
                + quality * (
                    args.lambda_u * (target_ce + target_dice)
                    + args.lambda_target_soft * target_soft_loss
                    + args.lambda_rice_target * target_rice_loss
                    + args.lambda_rice_head_target * target_rice_head_loss
                    + args.lambda_smooth_target * smooth_loss
                    + args.lambda_proto * proto_loss
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if not bank_teachers:
                update_ema(teacher, student, alpha=args.ema)
            update_prototypes(proto, s_feat.detach(), s_mask, momentum=args.proto_momentum)
            update_prototypes_soft(proto, t_feat.detach(), target_soft.detach(), momentum=args.proto_momentum)
            total_loss += float(loss.item())
        val = eval_model(student, source_val_loader, device, rice_threshold=args.rice_threshold)
        test = eval_model(student, uav_loader, device, rice_threshold=args.rice_threshold)
        print(
            f"[snd] epoch={epoch} loss={total_loss / max(steps,1):.4f} "
            f"source_fg_mIoU={val['miou_fg']:.4f} uav_fg_mIoU={test['miou_fg']:.4f} "
            f"uav_riceIoU={test['rice_exposure_iou']:.4f} uav_lateIoU={test['iou'][2]:.4f} "
            f"uav_lotusIoU={test['iou'][3]:.4f} uav_mainBinary_mIoU={test['main_binary_miou']:.4f} "
            f"uav_riceHeadIoU={test['rice_head_iou']:.4f} uav_nonRiceHeadIoU={test['rice_head_nonrice_iou']:.4f} "
            f"uav_riceHead_mIoU={test['rice_head_miou']:.4f}"
        )
        score = test[args.selection_metric]
        write_history(
            out_dir,
            {
                "mode": "uda_snd",
                "epoch": epoch,
                "loss": float(total_loss / max(steps, 1)),
                "source": val,
                "uav": test,
                "selection_metric": f"uav.{args.selection_metric}",
                "target_quality_mean": float(target_weight_bank.mean()),
            },
        )
        if score > best:
            best = score
            torch.save(
                {
                    "model": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "proto": proto.detach().cpu(),
                    "manifest": manifest,
                    "best_uav": test,
                    "target_quality_mean": float(target_weight_bank.mean()),
                },
                best_path,
            )
    return best_path


def self_train_reliable(
    args,
    data_root: Path,
    device: torch.device,
    out_dir: Path,
    init_ckpt: Path,
):
    _, source_val_ds, target_ds, uav_val_ds, source_loader, source_val_loader, target_loader, target_eval_loader, uav_loader, manifest = make_loaders(
        data_root, args.batch_size, args.num_workers, augment=True, target_augment=False
    )
    student = SimpleUNet(in_ch=32, num_classes=4, base=args.base_ch, feat_dim=args.feat_dim).to(device)
    teacher = SimpleUNet(in_ch=32, num_classes=4, base=args.base_ch, feat_dim=args.feat_dim).to(device)
    ckpt = torch.load(init_ckpt, map_location=device)
    student.load_state_dict(ckpt["model"], strict=False)
    teacher.load_state_dict(ckpt["model"], strict=False)
    bank_teachers: list[nn.Module] = []
    for teacher_ckpt in args.teacher_ckpts:
        tm = SimpleUNet(in_ch=32, num_classes=4, base=args.base_ch, feat_dim=args.feat_dim).to(device)
        tm_ckpt = torch.load(teacher_ckpt, map_location=device)
        tm.load_state_dict(tm_ckpt["model"], strict=False)
        tm.eval()
        for p in tm.parameters():
            p.requires_grad_(False)
        bank_teachers.append(tm)
    bank_models = bank_teachers if bank_teachers else [teacher]
    if args.adapt_head_only:
        set_trainable_params(student, head_only=True)
    class_weights = compute_class_weights(manifest, device)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    proto = torch.zeros(4, args.feat_dim, device=device)
    lotus_proto = torch.zeros(args.feat_dim, device=device)
    rice_proto = torch.zeros(args.feat_dim, device=device)
    nonrice_proto = torch.zeros(args.feat_dim, device=device)
    student.eval()
    with torch.no_grad():
        for batch in source_loader:
            s_img = batch["img"].to(device)
            s_mask = batch["mask"].to(device)
            _, s_feat = student(s_img, return_feat=True)
            update_prototypes(proto, s_feat, s_mask, momentum=0.0)
            update_group_prototype(lotus_proto, s_feat, s_mask, (3,), momentum=0.0)
            update_group_prototype(rice_proto, s_feat, s_mask, (1, 2), momentum=0.0)
            update_group_prototype(nonrice_proto, s_feat, s_mask, (0, 3), momentum=0.0)
    student.train()
    best = -1.0
    best_path = out_dir / "uda_reliable_best.pt"
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    for epoch in range(1, args.epochs_uda + 1):
        feat_bank, prob_bank = collect_target_bank(bank_models, target_eval_loader, device, args.temperature)
        class_prior = prob_bank.mean(axis=(0, 2, 3)).astype(np.float32)
        class_prior = np.power(np.clip(class_prior, 1e-6, 1.0), args.prior_gamma)
        class_prior = class_prior / np.clip(class_prior.sum(), 1e-6, None)
        class_prior_t = torch.from_numpy(class_prior).to(device=device)
        target_soft_bank, target_pseudo_bank, target_reliable_bank, target_weight_bank, class_thr, class_stats = build_classwise_reliability_bank(
            prob_bank,
            class_quantiles=[0.80, 0.70, 0.70, 0.80],
        )
        student.train()
        total_loss = 0.0
        steps = max(len(source_loader), len(target_loader))
        for _ in range(steps):
            try:
                sb = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                sb = next(source_iter)
            try:
                tb = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tb = next(target_iter)
            s_img = sb["img"].to(device)
            s_mask = sb["mask"].to(device)
            t_img = tb["img"].to(device)
            t_idx = tb["idx"].cpu().numpy()
            t_img_strong = spectral_augment_batch(t_img)
            with torch.no_grad():
                t_teacher_logits, t_teacher_feat = teacher(t_img, return_feat=True)
                t_teacher_rice_logits = teacher.rice_classifier(t_teacher_feat).squeeze(1)
            t_logits, t_feat = student(t_img_strong, return_feat=True)
            s_logits, s_feat = student(s_img, return_feat=True)
            sup_loss = weighted_ce_loss(s_logits, s_mask, class_weights) + dice_loss(s_logits, s_mask, class_weights)
            s_feat_ex = extrapolate_features(s_feat, s_mask, proto, gamma=args.feature_extrap_gamma)
            s_logits_ex = student.classifier(s_feat_ex)
            sup_ex_loss = weighted_ce_loss(s_logits_ex, s_mask, class_weights) + dice_loss(s_logits_ex, s_mask, class_weights)
            sup_rice_loss = rice_exposure_loss(s_logits, s_mask)
            s_rice_head_logits = student.rice_classifier(s_feat).squeeze(1)
            sup_rice_head_loss = binary_tversky_loss(
                s_rice_head_logits,
                s_mask,
                alpha=args.rice_tversky_alpha,
                beta=args.rice_tversky_beta,
                gamma=args.rice_tversky_gamma,
            )
            target_soft = torch.from_numpy(target_soft_bank[t_idx]).to(device=device, dtype=torch.float32)
            target_pseudo = torch.from_numpy(target_pseudo_bank[t_idx]).to(device=device, dtype=torch.long)
            target_reliable = torch.from_numpy(target_reliable_bank[t_idx]).to(device=device)
            target_weight = torch.from_numpy(target_weight_bank[t_idx]).to(device=device)
            target_soft = refine_probs_with_prototypes(
                target_soft,
                t_feat,
                proto,
                temperature=args.proto_temp,
                refine_strength=args.proto_refine_strength,
            )
            target_pseudo = target_soft.argmax(dim=1)
            target_weight = 0.5 * target_weight + 0.5 * target_soft.max(dim=1).values
            reliable_frac = target_reliable.float().mean().clamp(0.01, 1.0)
            pseudo_masked = target_pseudo.clone()
            pseudo_masked[~target_reliable] = 255
            target_soft_loss = weighted_soft_kl_loss(t_logits, target_soft, target_reliable, target_weight)
            target_ce = masked_ce_loss(t_logits, pseudo_masked, target_reliable)
            target_dice = masked_dice_loss(t_logits, pseudo_masked, target_reliable, class_weights)
            target_rice_prob = (target_soft[:, 1] + target_soft[:, 2]).clamp(0.0, 1.0)
            target_rice_loss = confident_binary_pseudo_loss(
                student.rice_classifier(t_feat).squeeze(1),
                target_rice_prob,
                pos_thr=args.rice_pos_thr,
                neg_thr=args.rice_neg_thr,
                alpha=args.rice_tversky_alpha,
                beta=args.rice_tversky_beta,
            )
            target_rice_head_loss = 0.5 * rice_head_soft_distill_loss(
                student.rice_classifier(t_feat).squeeze(1),
                t_teacher_rice_logits,
            )
            t_rice_prob = torch.sigmoid(student.rice_classifier(t_feat).squeeze(1))
            proto_loss = build_proto_loss(s_feat, s_mask, proto)
            proto_loss = proto_loss + build_proto_loss(t_feat, pseudo_masked, proto, target_reliable)
            proto_contrast_loss = prototype_contrast_loss(s_feat, s_mask, proto)
            proto_contrast_loss = proto_contrast_loss + prototype_contrast_loss(
                t_feat,
                pseudo_masked,
                proto,
                target_reliable,
                temperature=args.proto_temp,
            )
            t_feat_ex = extrapolate_features(t_feat, pseudo_masked, proto, gamma=args.feature_extrap_gamma, mask=target_reliable)
            t_logits_ex = student.classifier(t_feat_ex)
            target_ex_loss = weighted_soft_kl_loss(t_logits_ex, target_soft, target_reliable, target_weight)
            complementary_loss = complementary_class_loss(
                t_logits,
                pseudo_masked,
                target_reliable,
                weight_map=target_weight,
            )
            lotus_margin_loss = lotus_margin_contrast_loss(
                t_feat,
                pseudo_masked,
                target_reliable,
                lotus_proto,
                rice_proto,
                reliability=target_weight,
                rice_weight=args.lotus_rice_weight,
                temperature=args.lotus_margin_temp,
            )
            hard_nonrice_loss = hard_nonrice_margin_loss(
                t_feat,
                pseudo_masked,
                target_reliable,
                nonrice_proto,
                rice_proto,
                reliability=target_weight,
                rice_prob=t_rice_prob,
                rice_prob_thr=args.hard_nonrice_thr,
                temperature=args.hard_nonrice_temp,
            )
            ratio_prior_loss = class_ratio_prior_loss(t_logits, class_prior_t, mask=target_reliable)
            smooth_loss = neighbor_consistency_loss(t_feat, t_rice_prob)
            loss = (
                sup_loss
                + args.lambda_feature_extrap * sup_ex_loss
                + args.lambda_rice_source * sup_rice_loss
                + args.lambda_rice_head_source * sup_rice_head_loss
                + reliable_frac * (
                    args.lambda_u * (target_ce + target_dice)
                    + args.lambda_feature_extrap * target_ex_loss
                    + args.lambda_target_soft * target_soft_loss
                    + args.lambda_rice_target * target_rice_loss
                    + args.lambda_rice_head_target * target_rice_head_loss
                    + args.lambda_smooth_target * smooth_loss
                    + args.lambda_proto * proto_loss
                    + args.lambda_proto_contrast * proto_contrast_loss
                    + args.lambda_complementary * complementary_loss
                    + args.lambda_lotus_margin * lotus_margin_loss
                    + args.lambda_hard_nonrice * hard_nonrice_loss
                    + args.lambda_ratio_prior * ratio_prior_loss
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if not bank_teachers:
                update_ema(teacher, student, alpha=args.ema)
            update_prototypes(proto, s_feat.detach(), s_mask, momentum=args.proto_momentum)
            update_prototypes_soft(proto, t_feat.detach(), target_soft.detach(), momentum=args.proto_momentum)
            update_group_prototype(
                lotus_proto,
                t_feat.detach(),
                pseudo_masked,
                (3,),
                mask=target_reliable,
                momentum=args.lotus_memory_momentum,
            )
            update_group_prototype(
                nonrice_proto,
                t_feat.detach(),
                pseudo_masked,
                (0, 3),
                mask=target_reliable,
                momentum=args.nonrice_memory_momentum,
            )
            total_loss += float(loss.item())
        val = eval_model(student, source_val_loader, device, rice_threshold=args.rice_threshold)
        test = eval_model(student, uav_loader, device, rice_threshold=args.rice_threshold)
        print(
            f"[reliable] epoch={epoch} loss={total_loss / max(steps,1):.4f} "
            f"source_fg_mIoU={val['miou_fg']:.4f} uav_fg_mIoU={test['miou_fg']:.4f} "
            f"uav_riceIoU={test['rice_exposure_iou']:.4f} uav_lateIoU={test['iou'][2]:.4f} "
            f"uav_lotusIoU={test['iou'][3]:.4f} uav_mainBinary_mIoU={test['main_binary_miou']:.4f} "
            f"uav_riceHeadIoU={test['rice_head_iou']:.4f} uav_nonRiceHeadIoU={test['rice_head_nonrice_iou']:.4f}"
        )
        score = test[args.selection_metric]
        write_history(
            out_dir,
            {
                "mode": "uda_reliable",
                "epoch": epoch,
                "loss": float(total_loss / max(steps, 1)),
                "source": val,
                "uav": test,
                "selection_metric": f"uav.{args.selection_metric}",
                "class_thr": class_thr,
                "class_stats": class_stats,
                "lotus_margin_loss": float(lotus_margin_loss.item()) if torch.is_tensor(lotus_margin_loss) else float(lotus_margin_loss),
                "hard_nonrice_loss": float(hard_nonrice_loss.item()) if torch.is_tensor(hard_nonrice_loss) else float(hard_nonrice_loss),
                "ratio_prior_loss": float(ratio_prior_loss.item()) if torch.is_tensor(ratio_prior_loss) else float(ratio_prior_loss),
                "proto_contrast_loss": float(proto_contrast_loss.item()) if torch.is_tensor(proto_contrast_loss) else float(proto_contrast_loss),
                "complementary_loss": float(complementary_loss.item()) if torch.is_tensor(complementary_loss) else float(complementary_loss),
                "target_ex_loss": float(target_ex_loss.item()) if torch.is_tensor(target_ex_loss) else float(target_ex_loss),
            },
        )
        if score > best:
            best = score
            torch.save(
                {
                    "model": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "proto": proto.detach().cpu(),
                    "manifest": manifest,
                    "best_uav": test,
                    "class_thr": class_thr,
                    "class_stats": class_stats,
                },
                best_path,
            )
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "processed_hsi_uda")
    parser.add_argument("--out", type=Path, default=ROOT / "runs_hsi_uda")
    parser.add_argument("--mode", choices=["source", "uda"], default="source")
    parser.add_argument("--target-strategy", choices=["standard", "snd", "reliable"], default="standard")
    parser.add_argument("--adapt-head-only", action="store_true", default=False)
    parser.add_argument(
        "--selection-metric",
        choices=["miou_fg", "rice_exposure_iou", "main_binary_miou", "rice_head_iou", "rice_head_miou"],
        default="miou_fg",
    )
    parser.add_argument("--init-ckpt", type=Path, default=None)
    parser.add_argument("--teacher-ckpts", type=Path, nargs="*", default=[])
    parser.add_argument("--epochs-source", type=int, default=20)
    parser.add_argument("--epochs-uda", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-ch", type=int, default=48)
    parser.add_argument("--feat-dim", type=int, default=96)
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--entropy-thr", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambda-u", type=float, default=1.0)
    parser.add_argument("--lambda-proto", type=float, default=0.1)
    parser.add_argument("--lambda-lotus-margin", type=float, default=0.0)
    parser.add_argument("--lotus-rice-weight", type=float, default=0.1)
    parser.add_argument("--lambda-hard-nonrice", type=float, default=0.0)
    parser.add_argument("--lambda-ratio-prior", type=float, default=0.0)
    parser.add_argument("--lambda-rice-source", type=float, default=0.0)
    parser.add_argument("--lambda-rice-target", type=float, default=0.0)
    parser.add_argument("--lambda-rice-head-source", type=float, default=0.0)
    parser.add_argument("--lambda-rice-head-target", type=float, default=0.0)
    parser.add_argument("--rice-head-neg-weight", type=float, default=1.0)
    parser.add_argument("--lambda-target-soft", type=float, default=0.0)
    parser.add_argument("--lambda-smooth-target", type=float, default=0.0)
    parser.add_argument("--lambda-proto-contrast", type=float, default=0.0)
    parser.add_argument("--lambda-complementary", type=float, default=0.0)
    parser.add_argument("--lambda-feature-extrap", type=float, default=0.0)
    parser.add_argument("--proto-temp", type=float, default=0.10)
    parser.add_argument("--proto-refine-strength", type=float, default=0.50)
    parser.add_argument("--feature-extrap-gamma", type=float, default=0.20)
    parser.add_argument("--lotus-margin-temp", type=float, default=0.10)
    parser.add_argument("--hard-nonrice-thr", type=float, default=0.60)
    parser.add_argument("--hard-nonrice-temp", type=float, default=0.10)
    parser.add_argument("--rice-tau", type=float, default=0.90)
    parser.add_argument("--rice-threshold", type=float, default=0.55)
    parser.add_argument("--rice-pos-thr", type=float, default=0.70)
    parser.add_argument("--rice-neg-thr", type=float, default=0.30)
    parser.add_argument("--rice-tversky-alpha", type=float, default=0.30)
    parser.add_argument("--rice-tversky-beta", type=float, default=0.70)
    parser.add_argument("--rice-tversky-gamma", type=float, default=1.00)
    parser.add_argument("--ema", type=float, default=0.99)
    parser.add_argument("--proto-momentum", type=float, default=0.99)
    parser.add_argument("--lotus-memory-momentum", type=float, default=0.99)
    parser.add_argument("--nonrice-memory-momentum", type=float, default=0.99)
    parser.add_argument("--neighbor-k", type=int, default=5)
    parser.add_argument("--bank-temp", type=float, default=0.20)
    parser.add_argument("--prior-gamma", type=float, default=1.0)
    parser.add_argument("--stable-conf-thr", type=float, default=0.70)
    parser.add_argument("--stable-ent-thr", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.target_strategy == "snd":
        args.adapt_head_only = True

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    if not (args.data_root / "manifest.json").exists():
        raise FileNotFoundError(f"Missing dataset manifest: {args.data_root / 'manifest.json'}")

    if args.mode == "source":
        run_source_only(args, args.data_root, device, args.out)
    else:
        ckpt = args.init_ckpt if args.init_ckpt is not None else run_source_only(args, args.data_root, device, args.out)
        if args.target_strategy == "snd":
            self_train_snd(args, args.data_root, device, args.out, ckpt)
        elif args.target_strategy == "reliable":
            self_train_reliable(args, args.data_root, device, args.out, ckpt)
        else:
            self_train(args, args.data_root, device, args.out, ckpt)


if __name__ == "__main__":
    main()
