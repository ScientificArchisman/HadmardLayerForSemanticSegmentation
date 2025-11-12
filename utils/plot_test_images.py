import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scienceplots
plt.style.use(["science", "notebook", "grid"])



# Cityscapes 19-class palette (index -> RGB)
CS_NAMES = [
    "road","sidewalk","building","wall","fence","pole","traffic light","traffic sign",
    "vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"
]
CS_COLORS = np.array([
    [128,  64, 128],
    [244,  35, 232],
    [ 70,  70,  70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170,  30],
    [220, 220,   0],
    [107, 142,  35],
    [152, 251, 152],
    [ 70, 130, 180],
    [220,  20,  60],
    [255,   0,   0],
    [  0,   0, 142],
    [  0,   0,  70],
    [  0,  60, 100],
    [  0,  80, 100],
    [  0,   0, 230],
    [119,  11,  32],
], dtype=np.uint8)

IGNORE_INDEX = 255  

def denorm_img(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """x: torch.Tensor [3,H,W] or [B,3,H,W]; returns float np array in [0,1]"""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x.squeeze(0).permute(1,2,0).detach().cpu().numpy()

def logits_to_pred(logits):
    """logits: [B,C,H,W] -> pred: [B,H,W] (argmax)"""
    return logits.argmax(dim=1)

def colorize_mask(mask, palette=CS_COLORS, ignore_index=IGNORE_INDEX):
    """
    mask: torch or np [H,W] with class ids
    returns RGB uint8 [H,W,3]
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask != ignore_index)
    rgb[valid] = palette[mask[valid]]
    return rgb

def overlay(image_rgb01, mask_rgb, alpha=0.5):
    """
    image_rgb01: float [H,W,3] in [0,1]
    mask_rgb: uint8 [H,W,3]
    returns float [H,W,3] in [0,1]
    """
    overlay = image_rgb01.copy()
    m = (mask_rgb.sum(axis=-1) > 0)  # where mask is not ignore
    overlay[m] = (1 - alpha) * image_rgb01[m] + alpha * (mask_rgb[m] / 255.0)
    return overlay

def legend_handles(names=CS_NAMES, colors=CS_COLORS):
    return [Patch(facecolor=np.array(c)/255.0, edgecolor='none', label=n) for n, c in zip(names, colors)]



def plot_seg_sample(image, gt_mask=None, pred_mask=None, alpha=0.5, title=None, show=False):
    """
    image: torch.Tensor [3,H,W] (normalized or 0..1)
    gt_mask: torch/np [H,W] with class ids   (optional)
    pred_mask: torch/np [H,W] with class ids (optional)
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list[matplotlib.axes.Axes]
    legend : matplotlib.legend.Legend or None
    """
    img = denorm_img(image)

    panes = [("Image", img)]
    if gt_mask is not None:
        gt_rgb = colorize_mask(gt_mask)
        panes.append(("GT mask", gt_rgb/255.0))
        panes.append(("GT overlay", overlay(img, gt_rgb, alpha)))
    if pred_mask is not None:
        pd_rgb = colorize_mask(pred_mask)
        panes.append(("Pred mask", pd_rgb/255.0))
        panes.append(("Pred overlay", overlay(img, pd_rgb, alpha)))

    cols = len(panes)
    fig, axs = plt.subplots(1, cols, figsize=(4*cols, 4))
    ax_list = np.atleast_1d(axs).ravel().tolist()

    for ax, (name, arr) in zip(ax_list, panes):
        ax.imshow(arr)
        ax.axis('off')
        ax.set_title(name if title is None else f"{name} • {title}")

    
    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax_list


# --- Example usage ---
# logits: torch.Tensor [B,19,H,W], image: [B,3,H,W], gt: [B,H,W]
# pred = logits_to_pred(logits)
# fig, axes, leg = plot_seg_sample(image[0], gt_mask=gt[0], pred_mask=pred[0], alpha=0.6)
# fig.suptitle("My composite")             # add later
# fig.savefig("viz.png", dpi=200)          # save later
# plt.show()                              # display when you're ready