import torch
from typing import Optional, Literal, Dict




@torch.no_grad()
def expected_calibration_error(
    logits_or_probs: torch.Tensor,   # (B, C, H, W)
    targets: torch.Tensor,           # (B, H, W) 
    n_bins: int = 15,
    ignore_index: Optional[int] = None,
    from_logits: bool = True,
    norm: Literal["l1", "l2"] = "l1",   # ECE with |.| (l1) or squared (l2) gap
) -> Dict[str, torch.Tensor]:
    """
    Computes pixel-wise multiclass ECE for semantic segmentation.
    Uses max-softmax confidence per pixel and compares it to empirical accuracy
    within confidence bins across all non-ignored pixels.

    Returns a dict with:
      - 'ece': scalar tensor
      - 'bin_acc': (n_bins,) per-bin accuracy
      - 'bin_conf': (n_bins,) per-bin mean confidence
      - 'bin_counts': (n_bins,) number of pixels per bin
      - 'edges': (n_bins+1,) bin edges in [0,1]
    """
    assert logits_or_probs.dim() == 4, "Expected (B, C, H, W)"
    B, C, H, W = logits_or_probs.shape
    device = logits_or_probs.device

    if from_logits:
        probs = torch.softmax(logits_or_probs, dim=1)
    else:
        probs = logits_or_probs

    conf, pred = probs.max(dim=1)

    conf = conf.reshape(-1)
    pred = pred.reshape(-1)
    tgt  = targets.reshape(-1).to(pred.dtype)

    mask = torch.ones_like(tgt, dtype=torch.bool)
    if ignore_index is not None:
        mask &= (tgt != ignore_index)
    mask &= (tgt >= 0) & (tgt < C)

    if mask.sum() == 0:
        edges = torch.linspace(0, 1, n_bins + 1, device=device)
        zeros = torch.zeros(n_bins, device=device, dtype=torch.float32)
        return dict(ece=zeros.sum(), bin_acc=zeros, bin_conf=zeros, bin_counts=zeros, edges=edges)

    conf = conf[mask].to(torch.float32)
    pred = pred[mask]
    tgt  = tgt[mask]

    # Correctness indicator
    correct = (pred == tgt).to(torch.float32)

    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=device, dtype=conf.dtype)
    bin_ids = torch.bucketize(conf, edges[1:-1], right=True)

    bin_counts  = torch.bincount(bin_ids, minlength=n_bins).to(torch.float32)
    sum_conf    = torch.bincount(bin_ids, weights=conf, minlength=n_bins)
    sum_correct = torch.bincount(bin_ids, weights=correct, minlength=n_bins)

    denom = bin_counts.clamp_min(1.0)
    bin_conf = sum_conf / denom
    bin_acc  = sum_correct / denom

    if norm == "l1":
        gap = (bin_acc - bin_conf).abs()
    elif norm == "l2":
        gap = (bin_acc - bin_conf).pow(2)
    else:
        raise ValueError("norm must be 'l1' or 'l2'")

    # Weighted average by bin occupancy
    N = bin_counts.sum().clamp_min(1.0)
    ece = (bin_counts / N * gap).sum()

    return {
        "ece": ece,
        "bin_acc": bin_acc,
        "bin_conf": bin_conf,
        "bin_counts": bin_counts,
        "edges": edges,
    }
