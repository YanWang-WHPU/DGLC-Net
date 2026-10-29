from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

import train_hsi_uda as hsi


@torch.no_grad()
def build_anchor_prototypes(model, loader, device):
    feat_dim = model.proj.out_channels
    proto = torch.zeros(4, feat_dim, device=device)
    rice_proto = torch.zeros(feat_dim, device=device)
    nonrice_proto = torch.zeros(feat_dim, device=device)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            img = batch["img"].to(device)
            mask = batch["mask"].to(device)
            _, feat = model(img, return_feat=True)
            hsi.update_prototypes(proto, feat, mask, momentum=0.0)
            hsi.update_group_prototype(rice_proto, feat, mask, (1, 2), momentum=0.0)
            hsi.update_group_prototype(nonrice_proto, feat, mask, (0, 3), momentum=0.0)
    return proto, rice_proto, nonrice_proto


@torch.no_grad()
def build_anchor_multi_prototypes(model, loader, device, num_subprotos: int = 4, sample_cap: int = 4096):
    feat_dim = model.proj.out_channels
    proto = torch.zeros(4, num_subprotos, feat_dim, device=device)
    samples = [[] for _ in range(4)]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            img = batch["img"].to(device)
            mask = batch["mask"].to(device)
            _, feat = model(img, return_feat=True)
            feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
            mask_flat = mask.reshape(-1)
            for cls in range(4):
                cls_mask = mask_flat == cls
                if not cls_mask.any():
                    continue
                cls_feat = feat_flat[cls_mask]
                if cls_feat.shape[0] > sample_cap:
                    idx = torch.linspace(0, cls_feat.shape[0] - 1, steps=sample_cap, device=device).long()
                    cls_feat = cls_feat[idx]
                samples[cls].append(cls_feat.detach())
    for cls in range(4):
        if not samples[cls]:
            continue
        feats = torch.cat(samples[cls], dim=0)
        feats = torch.nn.functional.normalize(feats, dim=1)
        center = feats.mean(dim=0, keepdim=True)
        first = torch.argmax((feats * center).sum(dim=1))
        chosen = [feats[first]]
        while len(chosen) < num_subprotos and len(chosen) < feats.shape[0]:
            sim = torch.stack([feats @ p for p in chosen], dim=1).max(dim=1).values
            idx = torch.argmin(sim)
            chosen.append(feats[idx])
        while len(chosen) < num_subprotos:
            chosen.append(chosen[-1].clone())
        proto[cls] = torch.stack(chosen[:num_subprotos], dim=0)
    return proto


def class_prototype_scores(
    feat: torch.Tensor,
    proto: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Return per-class scores from a multi-prototype memory."""
    feat_n = torch.nn.functional.normalize(feat, dim=1)
    proto_n = torch.nn.functional.normalize(proto, dim=-1)
    scores = []
    for cls in range(proto_n.shape[0]):
        cls_score = torch.einsum("bchw,mc->bmhw", feat_n, proto_n[cls])
        scores.append(torch.logsumexp(cls_score / max(temperature, 1e-6), dim=1))
    return torch.stack(scores, dim=1)


@torch.no_grad()
def build_target_evolved_prototypes(
    models,
    anchor_model,
    source_proto: torch.Tensor,
    loader,
    device,
    weights,
    temperature: float = 0.68,
    quantile: float = 0.80,
    per_class_cap: int = 2048,
    target_subprotos: int = 2,
):
    """Mine reliable target features and form a target-adaptive prototype memory.

    The reliability score follows the source-free adaptation pattern used by
    prototype-memory methods: confidence, teacher agreement, low entropy, and
    source-prototype margin are combined before features are admitted.
    """
    samples = [[] for _ in range(4)]
    scores = [[] for _ in range(4)]
    source_proto = torch.nn.functional.normalize(source_proto, dim=-1)
    for batch in loader:
        img = batch["img"].to(device)
        teacher_probs = []
        teacher_preds = []
        for model in models:
            logits, _ = model(img, return_feat=True)
            prob = torch.softmax(logits / max(temperature, 1e-6), dim=1)
            teacher_probs.append(prob)
            teacher_preds.append(prob.argmax(dim=1))
        ens_prob = sum(float(w) * p for w, p in zip(weights, teacher_probs))
        ens_prob = ens_prob / max(float(sum(weights)), 1e-6)
        pred = ens_prob.argmax(dim=1)
        conf = ens_prob.max(dim=1).values
        entropy = entropy_from_prob(ens_prob) / math.log(ens_prob.shape[1])
        agreement = torch.stack([(p == pred).float() for p in teacher_preds], dim=0).mean(dim=0)
        _, feat = anchor_model(img, return_feat=True)
        source_scores = class_prototype_scores(feat, source_proto, temperature=0.20)
        source_top2 = torch.topk(source_scores, k=2, dim=1).values
        source_margin = (source_top2[:, 0] - source_top2[:, 1]).sigmoid()
        reliability = (
            0.35 * conf
            + 0.25 * agreement
            + 0.25 * (1.0 - entropy)
            + 0.15 * source_margin
        ).clamp(0.0, 1.0)
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        pred_flat = pred.reshape(-1)
        rel_flat = reliability.reshape(-1)
        for cls in range(4):
            cls_mask = pred_flat == cls
            if not cls_mask.any():
                continue
            cls_feat = feat_flat[cls_mask]
            cls_rel = rel_flat[cls_mask]
            # Class-wise quantiles avoid the rice class dominating the memory.
            threshold = torch.quantile(cls_rel, min(max(quantile, 0.50), 0.98))
            keep = cls_rel >= threshold
            cls_feat = cls_feat[keep]
            cls_rel = cls_rel[keep]
            if cls_feat.numel() == 0:
                continue
            if cls_rel.numel() > per_class_cap:
                topk = torch.topk(cls_rel, per_class_cap).indices
                cls_feat = cls_feat[topk]
                cls_rel = cls_rel[topk]
            samples[cls].append(torch.nn.functional.normalize(cls_feat.detach(), dim=1))
            scores[cls].append(cls_rel.detach())

    evolved = torch.zeros(
        source_proto.shape[0],
        source_proto.shape[1] + target_subprotos,
        source_proto.shape[2],
        device=source_proto.device,
        dtype=source_proto.dtype,
    )
    evolved[:, : source_proto.shape[1]] = source_proto
    stats = {}
    for cls in range(4):
        if not samples[cls]:
            evolved[cls, source_proto.shape[1] :] = source_proto[cls, :target_subprotos]
            stats[str(cls)] = {"selected": 0, "target_prototypes": 0}
            continue
        feats = torch.cat(samples[cls], dim=0)
        rel = torch.cat(scores[cls], dim=0)
        # Reliability-weighted target centroid gives a stable domain-shifted
        # anchor while farthest-point samples preserve class multimodality.
        weighted_center = (feats * rel.unsqueeze(1)).sum(dim=0)
        weighted_center = torch.nn.functional.normalize(weighted_center.unsqueeze(0), dim=1)[0]
        chosen = [weighted_center]
        while len(chosen) < min(target_subprotos, feats.shape[0]):
            sim = torch.stack([feats @ p for p in chosen], dim=1).max(dim=1).values
            chosen.append(feats[torch.argmin(sim)])
        target_bank = torch.stack(chosen, dim=0)
        while target_bank.shape[0] < target_subprotos:
            target_bank = torch.cat([target_bank, target_bank[-1:].clone()], dim=0)
        evolved[cls, source_proto.shape[1] :] = target_bank[:target_subprotos]
        stats[str(cls)] = {
            "selected": int(feats.shape[0]),
            "target_prototypes": int(target_bank.shape[0]),
            "mean_reliability": float(rel.mean().item()),
        }
    return evolved, stats


def fuse_target_evolved_refine(
    ens_prob: torch.Tensor,
    feat: torch.Tensor,
    source_proto: torch.Tensor,
    evolved_proto: torch.Tensor,
    strength: float = 0.5,
    temperature: float = 0.68,
    target_alpha: float = 0.25,
):
    """Fuse source and target prototype scores with entropy-gated refinement."""
    source_scores = class_prototype_scores(feat, source_proto, temperature=temperature)
    evolved_scores = class_prototype_scores(feat, evolved_proto, temperature=temperature)
    # The evolved memory contains source and target prototypes.  We use the
    # source score as a stable reference and only inject the target component
    # through a bounded score residual.
    target_scores = evolved_scores - source_scores
    class_scores = source_scores + float(target_alpha) * target_scores
    proto_prob = torch.softmax(class_scores, dim=1)
    rice_score = torch.logsumexp(class_scores[:, [1, 2]], dim=1)
    nonrice_score = torch.logsumexp(class_scores[:, [0, 3]], dim=1)
    group_prob = torch.softmax(torch.stack([nonrice_score, rice_score], dim=1), dim=1)

    rice_prob = (ens_prob[:, 1] + ens_prob[:, 2]).clamp(1e-6, 1.0 - 1e-6)
    ent = entropy_from_prob(ens_prob) / math.log(ens_prob.shape[1])
    blend = (float(strength) * ent).clamp(0.0, 1.0)
    final_rice = ((1.0 - blend) * rice_prob + blend * group_prob[:, 1]).clamp(1e-6, 1.0 - 1e-6)
    cond_alpha = min(max(float(target_alpha), 0.0), 0.50)
    rice_cond = ((1.0 - cond_alpha) * ens_prob[:, 1:3] + cond_alpha * proto_prob[:, 1:3])
    rice_cond = rice_cond / rice_cond.sum(dim=1, keepdim=True).clamp_min(1e-6)
    nonrice_cond = ((1.0 - cond_alpha) * ens_prob[:, [0, 3]] + cond_alpha * proto_prob[:, [0, 3]])
    nonrice_cond = nonrice_cond / nonrice_cond.sum(dim=1, keepdim=True).clamp_min(1e-6)
    final = torch.zeros_like(ens_prob)
    final[:, 1] = final_rice * rice_cond[:, 0]
    final[:, 2] = final_rice * rice_cond[:, 1]
    final[:, 0] = (1.0 - final_rice) * nonrice_cond[:, 0]
    final[:, 3] = (1.0 - final_rice) * nonrice_cond[:, 1]
    return final / final.sum(dim=1, keepdim=True).clamp_min(1e-6)


@torch.no_grad()
def build_target_group_prototypes(
    models,
    anchor_model,
    loader,
    device,
    weights,
    temperature: float = 0.68,
    quantile: float = 0.80,
    per_group_cap: int = 4096,
):
    """Build two target prototypes directly for rice exposure vs non-rice."""
    samples = [[], []]
    scores = [[], []]
    for batch in loader:
        img = batch["img"].to(device)
        probs = []
        preds = []
        for model in models:
            logits, _ = model(img, return_feat=True)
            prob = torch.softmax(logits / max(temperature, 1e-6), dim=1)
            probs.append(prob)
            preds.append(prob.argmax(dim=1))
        ens_prob = sum(float(w) * p for w, p in zip(weights, probs))
        ens_prob = ens_prob / max(float(sum(weights)), 1e-6)
        pred = ens_prob.argmax(dim=1)
        group_pred = ((pred == 1) | (pred == 2)).long()
        conf = ens_prob.max(dim=1).values
        entropy = entropy_from_prob(ens_prob) / math.log(ens_prob.shape[1])
        agreement = torch.stack([(p == pred).float() for p in preds], dim=0).mean(dim=0)
        reliability = (0.45 * conf + 0.30 * agreement + 0.25 * (1.0 - entropy)).clamp(0.0, 1.0)
        _, feat = anchor_model(img, return_feat=True)
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        group_flat = group_pred.reshape(-1)
        rel_flat = reliability.reshape(-1)
        for group in range(2):
            mask = group_flat == group
            if not mask.any():
                continue
            group_feat = feat_flat[mask]
            group_rel = rel_flat[mask]
            threshold = torch.quantile(group_rel, min(max(quantile, 0.50), 0.98))
            keep = group_rel >= threshold
            group_feat = group_feat[keep]
            group_rel = group_rel[keep]
            if group_feat.numel() == 0:
                continue
            if group_rel.numel() > per_group_cap:
                topk = torch.topk(group_rel, per_group_cap).indices
                group_feat = group_feat[topk]
                group_rel = group_rel[topk]
            samples[group].append(torch.nn.functional.normalize(group_feat.detach(), dim=1))
            scores[group].append(group_rel.detach())
    proto = []
    stats = {}
    for group in range(2):
        if not samples[group]:
            raise ValueError("Target group prototype bank is empty.")
        feat = torch.cat(samples[group], dim=0)
        rel = torch.cat(scores[group], dim=0)
        center = (feat * rel.unsqueeze(1)).sum(dim=0)
        center = torch.nn.functional.normalize(center.unsqueeze(0), dim=1)[0]
        proto.append(center)
        stats[str(group)] = {
            "selected": int(feat.shape[0]),
            "mean_reliability": float(rel.mean().item()),
        }
    return torch.stack(proto, dim=0), stats


def fuse_target_group_refine(
    ens_prob: torch.Tensor,
    feat: torch.Tensor,
    source_rice_proto: torch.Tensor,
    source_nonrice_proto: torch.Tensor,
    target_group_proto: torch.Tensor,
    strength: float = 0.15,
    temperature: float = 0.68,
    target_alpha: float = 0.20,
):
    feat_n = torch.nn.functional.normalize(feat, dim=1)
    src_rice = torch.nn.functional.normalize(source_rice_proto.view(1, -1), dim=1)
    src_nonrice = torch.nn.functional.normalize(source_nonrice_proto.view(1, -1), dim=1)
    tgt = torch.nn.functional.normalize(target_group_proto, dim=1)
    src_scores = torch.stack(
        [
            (feat_n * src_nonrice.view(1, -1, 1, 1)).sum(dim=1),
            (feat_n * src_rice.view(1, -1, 1, 1)).sum(dim=1),
        ],
        dim=1,
    )
    tgt_scores = torch.stack(
        [
            (feat_n * tgt[0].view(1, -1, 1, 1)).sum(dim=1),
            (feat_n * tgt[1].view(1, -1, 1, 1)).sum(dim=1),
        ],
        dim=1,
    )
    alpha = min(max(float(target_alpha), 0.0), 0.50)
    group_prob = torch.softmax(((1.0 - alpha) * src_scores + alpha * tgt_scores) / max(temperature, 1e-6), dim=1)
    rice_prob = (ens_prob[:, 1] + ens_prob[:, 2]).clamp(1e-6, 1.0 - 1e-6)
    rel = normalized_reliability(ens_prob)
    gate = (float(strength) * (1.0 - rel)).clamp(0.0, 1.0)
    final_rice = ((1.0 - gate) * rice_prob + gate * group_prob[:, 1]).clamp(1e-6, 1.0 - 1e-6)
    rice_cond = ens_prob[:, 1:3] / ens_prob[:, 1:3].sum(dim=1, keepdim=True).clamp_min(1e-6)
    nonrice_cond = ens_prob[:, [0, 3]] / ens_prob[:, [0, 3]].sum(dim=1, keepdim=True).clamp_min(1e-6)
    final = torch.zeros_like(ens_prob)
    final[:, 1] = final_rice * rice_cond[:, 0]
    final[:, 2] = final_rice * rice_cond[:, 1]
    final[:, 0] = (1.0 - final_rice) * nonrice_cond[:, 0]
    final[:, 3] = (1.0 - final_rice) * nonrice_cond[:, 1]
    return final / final.sum(dim=1, keepdim=True).clamp_min(1e-6)


@torch.no_grad()
def build_target_bank(models, anchor_model, loader, device, weights, min_conf: float = 0.0, per_class_cap: int = 2048):
    feat_bank = [[] for _ in range(4)]
    prob_bank = [[] for _ in range(4)]
    for batch in loader:
        img = batch["img"].to(device)
        prob_sum = None
        for model, weight in zip(models, weights):
            logits, _ = model(img, return_feat=True)
            prob = torch.softmax(logits, dim=1)
            prob_sum = prob * float(weight) if prob_sum is None else prob_sum + prob * float(weight)
        ens_prob = prob_sum / max(float(sum(weights)), 1e-6)
        _, feat = anchor_model(img, return_feat=True)
        conf, pred = ens_prob.max(dim=1)
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        prob_flat = ens_prob.permute(0, 2, 3, 1).reshape(-1, ens_prob.shape[1])
        pred_flat = pred.reshape(-1)
        conf_flat = conf.reshape(-1)
        for cls in range(4):
            cls_mask = (pred_flat == cls) & (conf_flat >= min_conf)
            if not cls_mask.any():
                continue
            cls_conf = conf_flat[cls_mask]
            cls_feat = feat_flat[cls_mask]
            cls_prob = prob_flat[cls_mask]
            if cls_conf.numel() > per_class_cap:
                topk = torch.topk(cls_conf, per_class_cap).indices
                cls_feat = cls_feat[topk]
                cls_prob = cls_prob[topk]
            feat_bank[cls].append(cls_feat.detach())
            prob_bank[cls].append(cls_prob.detach())
    bank_feat = []
    bank_prob = []
    for cls in range(4):
        if feat_bank[cls]:
            bank_feat.append(torch.cat(feat_bank[cls], dim=0))
            bank_prob.append(torch.cat(prob_bank[cls], dim=0))
    if not bank_feat:
        raise ValueError("Target bank is empty; lower min_conf or check target loader.")
    return torch.cat(bank_feat, dim=0), torch.cat(bank_prob, dim=0)


@torch.no_grad()
def build_labeled_bank(anchor_model, loader, device, per_class_cap: int = 2048):
    feat_bank = [[] for _ in range(4)]
    prob_bank = [[] for _ in range(4)]
    eye = torch.eye(4, device=device)
    for batch in loader:
        img = batch["img"].to(device)
        mask = batch["mask"].to(device)
        _, feat = anchor_model(img, return_feat=True)
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        mask_flat = mask.reshape(-1)
        for cls in range(4):
            cls_mask = mask_flat == cls
            if not cls_mask.any():
                continue
            cls_feat = feat_flat[cls_mask]
            if cls_feat.shape[0] > per_class_cap:
                idx = torch.linspace(0, cls_feat.shape[0] - 1, steps=per_class_cap, device=device).long()
                cls_feat = cls_feat[idx]
            cls_prob = eye[mask_flat[cls_mask]]
            if cls_prob.shape[0] > per_class_cap:
                idx = torch.linspace(0, cls_prob.shape[0] - 1, steps=per_class_cap, device=device).long()
                cls_prob = cls_prob[idx]
            feat_bank[cls].append(cls_feat.detach())
            prob_bank[cls].append(cls_prob.detach())
    bank_feat = []
    bank_prob = []
    for cls in range(4):
        if feat_bank[cls]:
            bank_feat.append(torch.cat(feat_bank[cls], dim=0))
            bank_prob.append(torch.cat(prob_bank[cls], dim=0))
    if not bank_feat:
        raise ValueError("Labeled source bank is empty.")
    return torch.cat(bank_feat, dim=0), torch.cat(bank_prob, dim=0)


def soft_knn_refine(query_feat: torch.Tensor, bank_feat: torch.Tensor, bank_prob: torch.Tensor, k: int = 7, chunk_size: int = 8192):
    query = torch.nn.functional.normalize(query_feat, dim=1)
    bank = torch.nn.functional.normalize(bank_feat, dim=1)
    refined = []
    for start in range(0, query.shape[0], chunk_size):
        q = query[start : start + chunk_size]
        sim = q @ bank.t()
        topv, topi = torch.topk(sim, k=min(k, bank.shape[0]), dim=1)
        weights = torch.softmax(topv, dim=1).unsqueeze(-1)
        knn_prob = (bank_prob[topi] * weights).sum(dim=1)
        refined.append(knn_prob)
    return torch.cat(refined, dim=0)


def entropy_from_prob(prob: torch.Tensor) -> torch.Tensor:
    return -(prob * prob.clamp_min(1e-8).log()).sum(dim=1)


def normalized_reliability(prob: torch.Tensor) -> torch.Tensor:
    ent = entropy_from_prob(prob) / math.log(prob.shape[1])
    one_hot = torch.nn.functional.one_hot(prob.argmax(dim=1), num_classes=prob.shape[1]).permute(0, 3, 1, 2).float()
    local_vote = torch.nn.functional.avg_pool2d(one_hot, 3, stride=1, padding=1)
    local_consistency = local_vote.max(dim=1).values
    return (0.5 * (1.0 - ent) + 0.5 * local_consistency).clamp(0.0, 1.0)


def fuse_group_refine(
    ens_prob: torch.Tensor,
    feat: torch.Tensor,
    rice_proto: torch.Tensor,
    nonrice_proto: torch.Tensor,
    strength: float = 0.5,
    temperature: float = 0.1,
):
    feat_n = torch.nn.functional.normalize(feat, dim=1)
    if rice_proto.dim() <= 2:
        rice_sim = (feat_n * torch.nn.functional.normalize(rice_proto.view(1, -1), dim=1).view(1, -1, 1, 1)).sum(dim=1)
        nonrice_sim = (feat_n * torch.nn.functional.normalize(nonrice_proto.view(1, -1), dim=1).view(1, -1, 1, 1)).sum(dim=1)
        group_prob = torch.softmax(torch.stack([nonrice_sim, rice_sim], dim=1) / max(temperature, 1e-6), dim=1)
    else:
        proto_bank = torch.nn.functional.normalize(rice_proto, dim=2)
        class_scores = []
        for cls in range(proto_bank.shape[0]):
            class_scores.append(
                torch.logsumexp(
                    torch.einsum("bchw,mc->bmhw", feat_n, proto_bank[cls]) / max(temperature, 1e-6),
                    dim=1,
                )
            )
        rice_sim = torch.logsumexp(torch.stack([class_scores[1], class_scores[2]], dim=1), dim=1)
        nonrice_sim = torch.logsumexp(torch.stack([class_scores[0], class_scores[3]], dim=1), dim=1)
        group_prob = torch.softmax(torch.stack([nonrice_sim, rice_sim], dim=1), dim=1)
    rice_prob = (ens_prob[:, 1] + ens_prob[:, 2]).clamp(1e-6, 1.0 - 1e-6)
    nonrice_prob = (ens_prob[:, 0] + ens_prob[:, 3]).clamp(1e-6, 1.0 - 1e-6)
    ent = entropy_from_prob(ens_prob) / math.log(ens_prob.shape[1])
    blend = (strength * ent).clamp(0.0, 1.0).unsqueeze(1)
    final_rice = (1.0 - blend) * rice_prob.unsqueeze(1) + blend * group_prob[:, 1:2]
    final_rice = final_rice[:, 0].clamp(1e-6, 1.0 - 1e-6)
    final_nonrice = 1.0 - final_rice
    rice_cond = ens_prob[:, 1:3] / ens_prob[:, 1:3].sum(dim=1, keepdim=True).clamp_min(1e-6)
    nonrice_cond = ens_prob[:, [0, 3]] / ens_prob[:, [0, 3]].sum(dim=1, keepdim=True).clamp_min(1e-6)
    final = torch.zeros_like(ens_prob)
    final[:, 1] = final_rice * rice_cond[:, 0]
    final[:, 2] = final_rice * rice_cond[:, 1]
    final[:, 0] = final_nonrice * nonrice_cond[:, 0]
    final[:, 3] = final_nonrice * nonrice_cond[:, 1]
    final = final / final.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return final


def fuse_conflict_refine(
    ens_prob: torch.Tensor,
    feat: torch.Tensor,
    rice_proto: torch.Tensor,
    nonrice_proto: torch.Tensor,
    strength: float = 0.5,
    temperature: float = 0.1,
    gate_quantile: float = 0.8,
):
    feat_n = torch.nn.functional.normalize(feat, dim=1)
    rice_sim = (feat_n * torch.nn.functional.normalize(rice_proto.view(1, -1), dim=1).view(1, -1, 1, 1)).sum(dim=1)
    nonrice_sim = (feat_n * torch.nn.functional.normalize(nonrice_proto.view(1, -1), dim=1).view(1, -1, 1, 1)).sum(dim=1)
    group_prob = torch.softmax(torch.stack([nonrice_sim, rice_sim], dim=1) / max(temperature, 1e-6), dim=1)

    rice_prob = (ens_prob[:, 1] + ens_prob[:, 2]).clamp(1e-6, 1.0 - 1e-6)
    nonrice_prob = (ens_prob[:, 0] + ens_prob[:, 3]).clamp(1e-6, 1.0 - 1e-6)
    rel = normalized_reliability(ens_prob)
    pred_cls = ens_prob.argmax(dim=1)
    class_q = [
        max(gate_quantile - 0.10, 0.50),
        min(gate_quantile + 0.10, 0.98),
        min(gate_quantile + 0.10, 0.98),
        max(gate_quantile - 0.05, 0.50),
    ]
    thr = torch.empty_like(rel)
    for cls in range(4):
        cls_mask = pred_cls == cls
        if cls_mask.any():
            thr_val = torch.quantile(rel[cls_mask], class_q[cls])
            thr[cls_mask] = thr_val
        else:
            thr[pred_cls == cls] = gate_quantile
    low_rel = rel < thr
    ens_rice_pred = rice_prob > 0.5
    group_rice_pred = group_prob[:, 1] > 0.5
    conflict = low_rel & (ens_rice_pred != group_rice_pred)
    blend = (strength * (1.0 - rel)).clamp(0.0, 1.0)
    blend = torch.where(conflict, blend, torch.zeros_like(blend))

    final_rice = ((1.0 - blend) * rice_prob + blend * group_prob[:, 1]).clamp(1e-6, 1.0 - 1e-6)
    final_nonrice = 1.0 - final_rice
    rice_cond = ens_prob[:, 1:3] / ens_prob[:, 1:3].sum(dim=1, keepdim=True).clamp_min(1e-6)
    nonrice_cond = ens_prob[:, [0, 3]] / ens_prob[:, [0, 3]].sum(dim=1, keepdim=True).clamp_min(1e-6)
    final = torch.zeros_like(ens_prob)
    final[:, 1] = final_rice * rice_cond[:, 0]
    final[:, 2] = final_rice * rice_cond[:, 1]
    final[:, 0] = final_nonrice * nonrice_cond[:, 0]
    final[:, 3] = final_nonrice * nonrice_cond[:, 1]
    final = final / final.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return final


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

    return {
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


def remove_small_rice_components(pred: np.ndarray, min_area: int, open_kernel: int = 0, close_kernel: int = 0) -> np.ndarray:
    if min_area <= 0:
        open_kernel = 0 if open_kernel <= 1 else open_kernel
        close_kernel = 0 if close_kernel <= 1 else close_kernel
        if open_kernel <= 1 and close_kernel <= 1:
            return pred
    out = pred.copy()
    if open_kernel > 1:
        structure = np.ones((open_kernel, open_kernel), dtype=bool)
    if close_kernel > 1:
        close_structure = np.ones((close_kernel, close_kernel), dtype=bool)
    for i in range(out.shape[0]):
        original_rice = (out[i] == 1) | (out[i] == 2)
        rice = original_rice.copy()
        if open_kernel > 1:
            rice = ndi.binary_opening(rice, structure=structure)
        if close_kernel > 1:
            rice = ndi.binary_closing(rice, structure=close_structure)
        labeled, n = ndi.label(rice)
        if n == 0:
            out[i][original_rice] = 0
            continue
        counts = np.bincount(labeled.ravel())
        keep = np.zeros_like(rice, dtype=bool)
        for comp_id in range(1, len(counts)):
            if counts[comp_id] >= min_area:
                keep |= labeled == comp_id
        out[i][original_rice & ~keep] = 0
        out[i][(~original_rice) & keep] = 1
    return out


def boundary_reassign_rice(
    pred: np.ndarray,
    prob: torch.Tensor,
    rice_prob_thr: float,
    local_rice_thr: float,
    local_kernel: int = 3,
) -> np.ndarray:
    if rice_prob_thr <= 0.0 or local_rice_thr <= 0.0:
        return pred
    out = pred.copy()
    rice_prob = (prob[:, 1] + prob[:, 2]).cpu().numpy()
    nonrice_pref = prob[:, [0, 3]].argmax(dim=1).cpu().numpy()
    local = torch.nn.functional.avg_pool2d(
        (prob[:, 1] + prob[:, 2]).unsqueeze(1),
        local_kernel,
        stride=1,
        padding=local_kernel // 2,
    ).squeeze(1).cpu().numpy()
    for i in range(out.shape[0]):
        rice = (out[i] == 1) | (out[i] == 2)
        flip = rice & (rice_prob[i] < rice_prob_thr) & (local[i] < local_rice_thr)
        out[i][flip] = nonrice_pref[i][flip]
    return out


def boundary_reliability_gate(
    pred: np.ndarray,
    base_prob: torch.Tensor,
    target_prob: torch.Tensor,
    base_thr: float = 0.65,
    target_thr: float = 0.65,
    local_thr: float = 0.53,
    kernel: int = 5,
    auto_quantile: float = -1.0,
) -> np.ndarray:
    """Remove rice pixels jointly rejected by source teachers and target memory."""
    out = pred.copy()
    base_rice = base_prob[:, 1] + base_prob[:, 2]
    target_rice = target_prob[:, 1] + target_prob[:, 2]
    pad = max(int(kernel), 1) // 2
    kernel = max(int(kernel), 1)
    base_pad = torch.nn.functional.pad(base_rice.unsqueeze(1), (pad, pad, pad, pad), mode="replicate")
    target_pad = torch.nn.functional.pad(target_rice.unsqueeze(1), (pad, pad, pad, pad), mode="replicate")
    base_local = torch.nn.functional.avg_pool2d(base_pad, kernel, stride=1, padding=0).squeeze(1)
    target_local = torch.nn.functional.avg_pool2d(target_pad, kernel, stride=1, padding=0).squeeze(1)
    current_rice = torch.from_numpy((pred == 1) | (pred == 2)).to(base_prob.device)
    if auto_quantile > 0 and current_rice.any():
        q = float(np.clip(auto_quantile, 0.05, 0.95))
        base_thr = float(torch.quantile(base_rice[current_rice], q).item())
        target_thr = float(torch.quantile(target_rice[current_rice], q).item())
        local_score = torch.minimum(base_local, target_local)
        local_thr = float(
            torch.quantile(local_score[current_rice], q).item()
        )
    nonrice_pref = torch.where(base_prob[:, 0] >= base_prob[:, 3], 0, 3)
    flip = (
        current_rice.cpu().numpy()
        & (base_rice < base_thr).cpu().numpy()
        & (target_rice < target_thr).cpu().numpy()
        & (base_local < local_thr).cpu().numpy()
        & (target_local < local_thr).cpu().numpy()
    )
    nonrice_np = nonrice_pref.cpu().numpy()
    for i in range(out.shape[0]):
        out[i][flip[i]] = nonrice_np[i][flip[i]]
    return out


@torch.no_grad()
def evaluate(
    models,
    anchor_model,
    loader,
    device,
    weights,
    proto,
    rice_proto,
    nonrice_proto,
    bank_feat,
    bank_prob,
    evolved_proto,
    target_group_proto,
    source_group_rice_proto,
    source_group_nonrice_proto,
    strength: float,
    temperature: float,
    min_rice_area: int,
    rice_open_kernel: int,
    rice_close_kernel: int,
    binary_threshold: float,
    threshold_gamma: float,
    boundary_flip_rice_thr: float,
    boundary_flip_local_thr: float,
    nonrice_boost: float,
    smooth_kernel: int,
    smooth_mode: str,
    proto_mode: str,
    entropy_quantile: float,
    target_alpha: float,
    target_group_strength: float,
    boundary_gate_base_thr: float,
    boundary_gate_target_thr: float,
    boundary_gate_local_thr: float,
    boundary_gate_kernel: int,
    boundary_gate_auto_quantile: float,
    ignore_zero_eval: bool,
    dump_path: Path | None = None,
):
    all_pred = []
    all_target = []
    all_prob = []
    all_gate_prob = []
    for batch in loader:
        img = batch["img"].to(device)
        target = batch["mask"].numpy()
        prob_sum = None
        for model, weight in zip(models, weights):
            logits, _ = model(img, return_feat=True)
            prob = torch.softmax(logits / max(temperature, 1e-6), dim=1)
            prob_sum = prob * float(weight) if prob_sum is None else prob_sum + prob * float(weight)
        ens_prob = prob_sum / max(float(sum(weights)), 1e-6)
        _, feat = anchor_model(img, return_feat=True)
        gate_reference = None
        if proto_mode == "boundary_gate":
            refined = fuse_group_refine(
                ens_prob,
                feat,
                proto,
                proto,
                strength=strength,
                temperature=temperature,
            )
            gate_reference = fuse_target_group_refine(
                ens_prob,
                feat,
                source_group_rice_proto,
                source_group_nonrice_proto,
                target_group_proto,
                strength=target_group_strength,
                temperature=temperature,
                target_alpha=target_alpha,
            )
        elif proto_mode == "target_group":
            refined = fuse_target_group_refine(
                ens_prob,
                feat,
                rice_proto,
                nonrice_proto,
                target_group_proto,
                strength=target_group_strength,
                temperature=temperature,
                target_alpha=target_alpha,
            )
        elif proto_mode == "evolve":
            refined = fuse_target_evolved_refine(
                ens_prob,
                feat,
                proto,
                evolved_proto,
                strength=strength,
                temperature=temperature,
                target_alpha=target_alpha,
            )
        elif proto_mode == "selective":
            proto_prob = torch.softmax(
                torch.einsum(
                    "bchw,kc->bkhw",
                    torch.nn.functional.normalize(feat, dim=1),
                    torch.nn.functional.normalize(proto, dim=1),
                ) / max(temperature, 1e-6),
                dim=1,
            )
            ent = entropy_from_prob(ens_prob) / math.log(ens_prob.shape[1])
            one_hot = torch.nn.functional.one_hot(ens_prob.argmax(dim=1), num_classes=ens_prob.shape[1]).permute(0, 3, 1, 2).float()
            boundary = torch.nn.functional.avg_pool2d(one_hot, 3, 1, 1).max(dim=1).values < 0.999
            if entropy_quantile > 0:
                thr = torch.quantile(ent.flatten(1), entropy_quantile, dim=1, keepdim=True).unsqueeze(-1)
                ent_mask = ent.unsqueeze(1) >= thr
            else:
                ent_mask = ent.unsqueeze(1) > 0.5
            info_mask = boundary.unsqueeze(1) | ent_mask
            refined = ens_prob * (1.0 - strength * info_mask.float()) + proto_prob * (strength * info_mask.float())
            refined = refined / refined.sum(dim=1, keepdim=True).clamp_min(1e-6)
        elif proto_mode == "conflict":
            refined = fuse_conflict_refine(
                ens_prob,
                feat,
                rice_proto,
                nonrice_proto,
                strength=strength,
                temperature=temperature,
                gate_quantile=entropy_quantile,
            )
        elif proto_mode in ("knn", "source_knn", "source_conflict"):
            query_feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
            knn_prob = soft_knn_refine(query_feat, bank_feat, bank_prob, k=7)
            knn_prob = knn_prob.view(img.shape[0], img.shape[2], img.shape[3], -1).permute(0, 3, 1, 2)
            rel = normalized_reliability(ens_prob)
            if proto_mode == "source_conflict":
                ens_rice = ens_prob[:, 1] + ens_prob[:, 2]
                knn_rice = knn_prob[:, 1] + knn_prob[:, 2]
                conflict = (ens_rice > 0.5) != (knn_rice > 0.5)
                blend = (strength * (1.0 - rel)).clamp(0.0, 1.0)
                blend = torch.where(conflict, blend, torch.zeros_like(blend))
                final_rice = ((1.0 - blend) * ens_rice + blend * knn_rice).clamp(1e-6, 1.0 - 1e-6)
                rice_cond = ens_prob[:, 1:3] / ens_prob[:, 1:3].sum(dim=1, keepdim=True).clamp_min(1e-6)
                nonrice_cond = ens_prob[:, [0, 3]] / ens_prob[:, [0, 3]].sum(dim=1, keepdim=True).clamp_min(1e-6)
                refined = torch.zeros_like(ens_prob)
                refined[:, 1] = final_rice * rice_cond[:, 0]
                refined[:, 2] = final_rice * rice_cond[:, 1]
                refined[:, 0] = (1.0 - final_rice) * nonrice_cond[:, 0]
                refined[:, 3] = (1.0 - final_rice) * nonrice_cond[:, 1]
                refined = refined / refined.sum(dim=1, keepdim=True).clamp_min(1e-6)
            else:
                rel = rel.unsqueeze(1)
                refined = (1.0 - strength * (1.0 - rel)) * ens_prob + (strength * (1.0 - rel)) * knn_prob
                refined = refined / refined.sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            refined = fuse_group_refine(ens_prob, feat, rice_proto, nonrice_proto, strength=strength, temperature=temperature)
        smooth_kernel = max(int(smooth_kernel), 1)
        if smooth_mode == "class":
            if smooth_kernel > 1:
                pad = smooth_kernel // 2
                refined = 0.5 * refined + 0.5 * torch.nn.functional.avg_pool2d(refined, smooth_kernel, 1, pad)
                if gate_reference is not None:
                    gate_reference = 0.5 * gate_reference + 0.5 * torch.nn.functional.avg_pool2d(
                        gate_reference, smooth_kernel, 1, pad
                    )
            refined = refined / refined.sum(dim=1, keepdim=True).clamp_min(1e-6)
            if binary_threshold >= 0.0:
                rice_prob = refined[:, 1] + refined[:, 2]
                rice_cond = refined[:, 1:3] / refined[:, 1:3].sum(dim=1, keepdim=True).clamp_min(1e-6)
                nonrice_cond = refined[:, [0, 3]] / refined[:, [0, 3]].sum(dim=1, keepdim=True).clamp_min(1e-6)
                tau = binary_threshold
                if threshold_gamma > 0.0:
                    rel_map = normalized_reliability(refined)
                    tau = tau + threshold_gamma * (1.0 - rel_map)
                rice_mask = rice_prob > tau
                refined = torch.zeros_like(refined)
                refined[:, 1] = rice_mask.float() * rice_cond[:, 0]
                refined[:, 2] = rice_mask.float() * rice_cond[:, 1]
                refined[:, 0] = (~rice_mask).float() * nonrice_cond[:, 0]
                refined[:, 3] = (~rice_mask).float() * nonrice_cond[:, 1]
                refined = refined / refined.sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            rice_prob = refined[:, 1] + refined[:, 2]
            if smooth_kernel > 1:
                pad = smooth_kernel // 2
                rice_smooth = torch.nn.functional.avg_pool2d(rice_prob.unsqueeze(1), smooth_kernel, 1, pad).squeeze(1)
                rice_prob = 0.5 * rice_prob + 0.5 * rice_smooth
            rice_prob = rice_prob.clamp(1e-6, 1.0 - 1e-6)
            nonrice_prob = 1.0 - rice_prob
            rice_cond = refined[:, 1:3] / refined[:, 1:3].sum(dim=1, keepdim=True).clamp_min(1e-6)
            nonrice_cond = refined[:, [0, 3]] / refined[:, [0, 3]].sum(dim=1, keepdim=True).clamp_min(1e-6)
            refined = torch.zeros_like(refined)
            if binary_threshold >= 0.0:
                tau = binary_threshold
                if threshold_gamma > 0.0:
                    rel_map = normalized_reliability(refined)
                    tau = tau + threshold_gamma * (1.0 - rel_map)
                rice_mask = rice_prob > tau
                refined[:, 1] = rice_mask.float() * rice_cond[:, 0]
                refined[:, 2] = rice_mask.float() * rice_cond[:, 1]
                refined[:, 0] = (~rice_mask).float() * nonrice_cond[:, 0]
                refined[:, 3] = (~rice_mask).float() * nonrice_cond[:, 1]
            else:
                refined[:, 1] = rice_prob * rice_cond[:, 0]
                refined[:, 2] = rice_prob * rice_cond[:, 1]
                refined[:, 0] = nonrice_prob * nonrice_cond[:, 0]
                refined[:, 3] = nonrice_prob * nonrice_cond[:, 1]
            refined = refined / refined.sum(dim=1, keepdim=True).clamp_min(1e-6)
        if nonrice_boost != 1.0:
            refined = refined.clone()
            refined[:, [0, 3]] = refined[:, [0, 3]] * nonrice_boost
            refined = refined / refined.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pred = refined.argmax(dim=1).cpu().numpy()
        pred = boundary_reassign_rice(
            pred,
            refined,
            rice_prob_thr=boundary_flip_rice_thr,
            local_rice_thr=boundary_flip_local_thr,
            local_kernel=3,
        )
        pred = remove_small_rice_components(
            pred,
            min_rice_area,
            open_kernel=rice_open_kernel,
            close_kernel=rice_close_kernel,
        )
        if proto_mode == "boundary_gate" and gate_reference is not None:
            pred = boundary_reliability_gate(
                pred,
                refined,
                gate_reference,
                base_thr=boundary_gate_base_thr,
                target_thr=boundary_gate_target_thr,
                local_thr=boundary_gate_local_thr,
                kernel=boundary_gate_kernel,
                auto_quantile=boundary_gate_auto_quantile,
            )
        all_pred.append(pred)
        all_target.append(target)
        all_prob.append(refined.cpu().numpy())
        if gate_reference is not None:
            all_gate_prob.append(gate_reference.cpu().numpy())
    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)
    metrics = score_prediction(pred, target, ignore_zero_eval=ignore_zero_eval)
    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dump_path,
            pred=pred.astype(np.uint8),
            target=target.astype(np.uint8),
            prob=np.concatenate(all_prob, axis=0).astype(np.float16),
            gate_prob=(
                np.concatenate(all_gate_prob, axis=0).astype(np.float16)
                if all_gate_prob
                else np.empty((0,), dtype=np.float16)
            ),
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=hsi.ROOT / "processed_hsi_uda")
    parser.add_argument("--uav-dir", type=Path, default=None, help="Optional UAV validation patch directory.")
    parser.add_argument("--teacher-ckpts", type=Path, nargs="+", required=True)
    parser.add_argument("--teacher-weights", type=float, nargs="*", default=None)
    parser.add_argument("--anchor-ckpt", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-rice-area", type=int, default=0)
    parser.add_argument("--rice-open-kernel", type=int, default=0)
    parser.add_argument("--rice-close-kernel", type=int, default=0)
    parser.add_argument("--binary-threshold", type=float, default=-1.0)
    parser.add_argument("--threshold-gamma", type=float, default=0.0)
    parser.add_argument("--boundary-flip-rice-thr", type=float, default=0.0)
    parser.add_argument("--boundary-flip-local-thr", type=float, default=0.0)
    parser.add_argument("--nonrice-boost", type=float, default=1.0)
    parser.add_argument("--smooth-kernel", type=int, default=3)
    parser.add_argument("--smooth-mode", choices=["class", "binary"], default="binary")
    parser.add_argument("--proto-mode", choices=["group", "multi", "evolve", "target_group", "boundary_gate", "selective", "conflict", "knn", "source_knn", "source_conflict"], default="group")
    parser.add_argument("--entropy-quantile", type=float, default=0.8)
    parser.add_argument("--bank-min-conf", type=float, default=0.0)
    parser.add_argument("--bank-cap", type=int, default=2048)
    parser.add_argument("--target-proto-quantile", type=float, default=0.80)
    parser.add_argument("--target-proto-cap", type=int, default=2048)
    parser.add_argument("--target-proto-subprotos", type=int, default=2)
    parser.add_argument("--target-alpha", type=float, default=0.25)
    parser.add_argument("--target-group-strength", type=float, default=0.15)
    parser.add_argument("--boundary-gate-base-thr", type=float, default=0.65)
    parser.add_argument("--boundary-gate-target-thr", type=float, default=0.65)
    parser.add_argument("--boundary-gate-local-thr", type=float, default=0.53)
    parser.add_argument("--boundary-gate-kernel", type=int, default=5)
    parser.add_argument("--boundary-gate-auto-quantile", type=float, default=-1.0)
    parser.add_argument("--ignore-zero-eval", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dump", type=Path, default=None)
    args = parser.parse_args()

    hsi.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, source_val_ds, target_ds, uav_val_ds, source_loader, source_val_loader, target_loader, target_eval_loader, uav_loader, manifest = hsi.make_loaders(
        args.data_root, args.batch_size, args.num_workers, augment=False, target_augment=False
    )
    if args.uav_dir is not None:
        mean = np.array(manifest["stats"]["mean"], dtype=np.float32)
        std = np.array(manifest["stats"]["std"], dtype=np.float32)
        uav_files = sorted(args.uav_dir.glob("*.npz"))
        if not uav_files:
            raise FileNotFoundError(f"No UAV patches found in {args.uav_dir}")
        uav_val_ds = hsi.PatchNPZDataset(uav_files, mean, std, labeled=True, augment=False)
        uav_loader = torch.utils.data.DataLoader(
            uav_val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    teachers = []
    for ckpt_path in args.teacher_ckpts:
        model = hsi.SimpleUNet(in_ch=32, num_classes=4, base=48, feat_dim=96).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        teachers.append(model)
    if not teachers:
        raise ValueError("Need at least one teacher checkpoint")

    if args.teacher_weights is None or len(args.teacher_weights) == 0:
        weights = [1.0 / len(teachers)] * len(teachers)
    else:
        if len(args.teacher_weights) != len(teachers):
            raise ValueError("--teacher-weights must match --teacher-ckpts length")
        weights = args.teacher_weights

    anchor_model = hsi.SimpleUNet(in_ch=32, num_classes=4, base=48, feat_dim=96).to(device)
    anchor_ckpt = torch.load(args.anchor_ckpt, map_location=device)
    anchor_model.load_state_dict(anchor_ckpt["model"], strict=False)
    anchor_model.eval()
    for p in anchor_model.parameters():
        p.requires_grad_(False)

    evolved_proto = None
    evolve_stats = None
    target_group_proto = None
    target_group_stats = None
    if args.proto_mode == "boundary_gate":
        hsi.set_seed(args.seed)
        _, rice_proto, nonrice_proto = build_anchor_prototypes(anchor_model, source_loader, device)
        hsi.set_seed(args.seed)
        proto = build_anchor_multi_prototypes(anchor_model, source_loader, device)
    elif args.proto_mode in ("multi", "evolve"):
        proto = build_anchor_multi_prototypes(anchor_model, source_loader, device)
        rice_proto = proto
        nonrice_proto = proto
    else:
        proto, rice_proto, nonrice_proto = build_anchor_prototypes(anchor_model, source_loader, device)
    source_group_rice_proto = rice_proto
    source_group_nonrice_proto = nonrice_proto
    bank_feat = None
    bank_prob = None
    if args.proto_mode == "knn":
        bank_feat, bank_prob = build_target_bank(
            teachers,
            anchor_model,
            target_loader,
            device,
            weights,
            min_conf=args.bank_min_conf,
            per_class_cap=args.bank_cap,
        )
    elif args.proto_mode in ("source_knn", "source_conflict"):
        bank_feat, bank_prob = build_labeled_bank(anchor_model, source_loader, device, per_class_cap=args.bank_cap)
    if args.proto_mode == "evolve":
        evolved_proto, evolve_stats = build_target_evolved_prototypes(
            teachers,
            anchor_model,
            proto,
            target_eval_loader,
            device,
            weights,
            temperature=args.temperature,
            quantile=args.target_proto_quantile,
            per_class_cap=args.target_proto_cap,
            target_subprotos=args.target_proto_subprotos,
        )
    elif args.proto_mode in ("target_group", "boundary_gate"):
        target_group_proto, target_group_stats = build_target_group_prototypes(
            teachers,
            anchor_model,
            target_eval_loader,
            device,
            weights,
            temperature=args.temperature,
            quantile=args.target_proto_quantile,
            per_group_cap=args.target_proto_cap,
        )
    metrics = evaluate(
        teachers,
        anchor_model,
        uav_loader,
        device,
        weights,
        proto,
        rice_proto,
        nonrice_proto,
        bank_feat,
        bank_prob,
        evolved_proto,
        target_group_proto,
        source_group_rice_proto,
        source_group_nonrice_proto,
        strength=args.strength,
        temperature=args.temperature,
        min_rice_area=args.min_rice_area,
        rice_open_kernel=args.rice_open_kernel,
        rice_close_kernel=args.rice_close_kernel,
        binary_threshold=args.binary_threshold,
        threshold_gamma=args.threshold_gamma,
        boundary_flip_rice_thr=args.boundary_flip_rice_thr,
        boundary_flip_local_thr=args.boundary_flip_local_thr,
        nonrice_boost=args.nonrice_boost,
        smooth_kernel=args.smooth_kernel,
        smooth_mode=args.smooth_mode,
        proto_mode=args.proto_mode,
        entropy_quantile=args.entropy_quantile,
        target_alpha=args.target_alpha,
        target_group_strength=args.target_group_strength,
        boundary_gate_base_thr=args.boundary_gate_base_thr,
        boundary_gate_target_thr=args.boundary_gate_target_thr,
        boundary_gate_local_thr=args.boundary_gate_local_thr,
        boundary_gate_kernel=args.boundary_gate_kernel,
        boundary_gate_auto_quantile=args.boundary_gate_auto_quantile,
        ignore_zero_eval=args.ignore_zero_eval,
        dump_path=args.dump,
    )
    print(
        f"strength={args.strength:.2f} riceIoU={metrics['rice_iou']:.4f} "
        f"backgroundIoU={metrics['background_iou']:.4f} OA={metrics['oa']:.4f} "
        f"mF1={metrics['mf1']:.4f} mIoU={metrics['miou']:.4f}"
    )
    result = {
        "anchor_ckpt": str(args.anchor_ckpt),
        "teacher_ckpts": [str(p) for p in args.teacher_ckpts],
        "teacher_weights": weights,
        "strength": args.strength,
        "temperature": args.temperature,
        "proto_mode": args.proto_mode,
        "target_proto_quantile": args.target_proto_quantile,
        "target_proto_cap": args.target_proto_cap,
        "target_proto_subprotos": args.target_proto_subprotos,
        "target_alpha": args.target_alpha,
        "target_group_strength": args.target_group_strength,
        "ignore_zero_eval": bool(args.ignore_zero_eval),
        "metrics": metrics,
    }
    if evolve_stats is not None:
        result["evolve_stats"] = evolve_stats
    if target_group_stats is not None:
        result["target_group_stats"] = target_group_stats
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
