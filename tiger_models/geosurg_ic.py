"""Training-only geometry-routed mixed responses; no feature affinity target."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mixed_response(margins):
    """View order: clean, side A, side B, both. Preserves additive effects."""
    return margins[3] - margins[1] - margins[2] + margins[0]


def interface_probe(target, depth, seed, min_jump=0.03, fraction=0.25):
    """Pick a GT class pair independently of depth, then route its boundary.

    Boundary scores are differences of within-class local depth means divided
    by image depth IQR (10--90%). Shuffling is within the same class pair and
    preserves the number of selected pixels exactly. No validation labels or
    estimated depth enter inference.
    """
    gen = torch.Generator(device=target.device).manual_seed(seed)
    h, w = target.shape
    pairs = []
    for a, b in ((target[:, :-1], target[:, 1:]),
                 (target[:-1, :], target[1:, :])):
        valid = (a != b) & (a != 0) & (b != 0)
        pairs.append(torch.minimum(a[valid], b[valid]) * 31 + torch.maximum(a[valid], b[valid]))
    ids, counts = torch.unique(torch.cat(pairs), return_counts=True)
    ids = ids[counts >= 8]
    if len(ids) == 0:
        return None
    pair = int(ids[torch.randint(len(ids), (), generator=gen, device=target.device)])
    a, b = pair // 31, pair % 31
    ma, mb = (target == a).float(), (target == b).float()
    def dilate(x):
        return F.max_pool2d(x[None, None], 5, 1, 2)[0, 0] > 0
    band = ((ma.bool() & dilate(mb)) | (mb.bool() & dilate(ma)))
    def avg(x):
        return F.avg_pool2d(x[None, None], 9, 1, 4)[0, 0]
    low = torch.quantile(depth.float(), .1)
    scale = (torch.quantile(depth.float(), .9) - low).clamp_min(1e-6)
    normalized = (depth.float() - low) / scale
    da = avg(normalized * ma) / avg(ma).clamp_min(1e-6)
    db = avg(normalized * mb) / avg(mb).clamp_min(1e-6)
    score = (da - db).abs()
    loc = band.flatten().nonzero().flatten()
    values = score.flatten()[loc]
    k = min(max(1, int(len(loc) * fraction)), int((values >= min_jump).sum()))
    weights = {}
    if k:
        top = values.topk(k).indices
        permutation = torch.randperm(len(loc), generator=gen, device=target.device)
        random_order = torch.randperm(len(loc), generator=gen, device=target.device)
        for name, selected in (("geo", top), ("shuffle", permutation[top]), ("gt", random_order[:k])):
            mask = torch.zeros(h * w, device=target.device)
            mask[loc[selected]] = 1.0 / k
            weights[name] = mask.reshape(h, w)
    else:
        weights = {name: torch.zeros_like(depth) for name in ("geo", "shuffle", "gt")}
    return {"a": a, "b": b, "side_a": ma.bool(), "side_b": mb.bool(),
            "weights": weights, "pixels": k, "boundary_pixels": len(loc),
            "mean_jump": float(values.mean())}


def appearance_views(image, probe, seed, strength=0.2):
    """Two independent, disjoint RGB gain/offset probes; geometry unchanged."""
    gen = torch.Generator(device=image.device).manual_seed(seed)
    mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
    std = image.new_tensor([.229, .224, .225])[None, :, None, None]
    rgb = image * std + mean
    deltas = []
    for side in ("side_a", "side_b"):
        gain = 1 + strength * (2 * torch.rand((1, 3, 1, 1), generator=gen, device=image.device) - 1)
        bias = strength * .25 * (2 * torch.rand((1, 3, 1, 1), generator=gen, device=image.device) - 1)
        mask = F.interpolate(probe[side][None, None].float(), image.shape[-2:], mode="nearest")
        changed = (rgb * gain + bias).clamp(0, 1)
        deltas.append((changed - rgb) / std * mask)
    return [image, image + deltas[0], image + deltas[1], image + deltas[0] + deltas[1]]


def log_margin(probability, probe, size):
    # log p_a - log p_b is the implied semantic logit margin. No presence
    # modulation: avoid regularizing the global presence head indirectly.
    p = F.interpolate(probability.float(), size, mode="bilinear", align_corners=False)
    return p[0, probe["a"]].clamp_min(1e-6).log() - p[0, probe["b"]].clamp_min(1e-6).log()
