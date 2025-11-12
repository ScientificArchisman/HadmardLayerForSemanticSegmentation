import torch
import torch.nn.functional as F


def brier_score_segmentation(
    probs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int | None = None,
    sample_weight: torch.Tensor | None = None,
    reduction: str = "mean",   # "mean" | "sum" | "none"
    scale_by_2: bool = True   # set True for score in [0,1]
) -> torch.Tensor:
    """
    probs:   (B,K,H,W) class probabilities (softmaxed)
    targets: (B,H,W) class indices 0..K-1  OR  (B,K,H,W) one-hot/soft labels
    sample_weight: optional per-pixel weights, (B,H,W) or (B,1,H,W)
    returns: scalar if reduction!='none', else (B,H,W) per-pixel scores (invalid pixels set to 0)
    """
    P = probs.float()
    B, K, H, W = P.shape

    if targets.ndim == 3:  
        T = targets.long()
        if ignore_index is not None:
            valid = T != ignore_index
            T_safe = torch.where(valid, T, torch.zeros_like(T))
        else:
            valid = torch.ones_like(T, dtype=torch.bool)
            T_safe = T
        Y = F.one_hot(T_safe, num_classes=K).permute(0, 3, 1, 2).to(P.dtype)
    elif targets.ndim == 4:  
        Y = targets.to(P.dtype)
        valid = torch.ones((B, H, W), dtype=torch.bool, device=P.device)
    else:
        raise ValueError("targets must be (B,H,W) indices or (B,K,H,W) label maps")

    per_pixel = ((P - Y) ** 2).sum(dim=1)  # (B,H,W)

    if sample_weight is not None:
        w = sample_weight.float()
        if w.ndim == 4 and w.size(1) == 1:
            w = w[:, 0]
        if w.ndim != 3:
            raise ValueError("sample_weight must be (B,H,W) or (B,1,H,W)")
        per_pixel = per_pixel * w
        valid = valid & (w > 0)

    if reduction == "mean":
        denom = valid.sum().clamp_min(1).to(per_pixel.dtype)
        out = per_pixel[valid].sum() / denom
    elif reduction == "sum":
        out = per_pixel[valid].sum()
    elif reduction == "none":
        out = per_pixel.clone()
        out[~valid] = 0
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    if scale_by_2:
        return out / 2
    return out
