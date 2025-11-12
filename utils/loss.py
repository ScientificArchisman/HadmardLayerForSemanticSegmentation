import torch
import torch.nn.functional as F

def multiclass_dice_loss(
    logits: torch.Tensor,          
    target: torch.Tensor,          
    ignore_index: int = 255,
    smooth: float = 1e-6,
    average: str = "macro",        
):
    """
    Soft Dice loss for multiclass segmentation with ignore support.

    - logits: raw scores (B,C,H,W)
    - target: class indices in [0..C-1] or ignore_index
    - average:
        "none"     -> returns per-class Dice loss, shape (C,)
        "macro"    -> mean over classes that are present (have any target pixels)
        "weighted" -> class-frequency-weighted mean
    """
    B, C, H, W = logits.shape
    target = target.long()
    probs = F.softmax(logits, dim=1)                 

    if ignore_index is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        valid = (target != ignore_index)             

    if not valid.any():
        return logits.new_zeros(())

    tgt = target.clone()
    tgt[~valid] = 0
    one_hot = F.one_hot(tgt, num_classes=C).permute(0, 3, 1, 2).float()  # (B,C,H,W)

    valid = valid.unsqueeze(1)                       # (B,1,H,W)
    probs = probs * valid
    one_hot = one_hot * valid

    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)                  
    cardinality  = probs.sum(dims) + one_hot.sum(dims)          
    dice_per_cls = (2 * intersection + smooth) / (cardinality + smooth)  
    loss_per_cls = 1.0 - dice_per_cls                           

    if average == "none":
        return loss_per_cls

    present = (one_hot.sum(dims) > 0)                           

    if average == "macro":
        if present.any():
            return loss_per_cls[present].mean()
        else:
            return logits.new_zeros(())  

    if average == "weighted":
        weights = one_hot.sum(dims)                             
        if weights.sum() == 0:
            return logits.new_zeros(())
        weights = weights / (weights.sum() + 1e-12)
        return (loss_per_cls * weights).sum()

    raise ValueError(f"Unknown average='{average}'")



def l1_multiclass_from_logits(logits, targets, ignore_index=255, eps=1e-8):
    """
    logits:  [N, C, H, W]
    targets: [N, H, W] with ints in [0, C-1] or = ignore_index
    """
    N, C, H, W = logits.shape
    probs = torch.softmax(logits, dim=1)                

    valid = (targets >= 0) & (targets < C)
    if ignore_index is not None:
        valid = valid & (targets != ignore_index)

    idx = targets.clone()
    idx[~valid] = 0
    idx = idx.long().unsqueeze(1)                       

    one_hot = torch.zeros_like(probs)                   
    one_hot.scatter_(1, idx, 1.0)                       
    one_hot = one_hot * valid.unsqueeze(1)              

    l1 = (probs - one_hot).abs()
    denom = valid.sum() * C
    return l1.sum() / (denom + eps)


