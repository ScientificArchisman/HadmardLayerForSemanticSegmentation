from pathlib import Path
import os
import torch.nn.functional as F
from datetime import datetime
from pprint import pformat
from torch.utils.data import DataLoader
from utils.load_cityscapes import CityscapesSubset, JointResize
import yaml 
import torch
from torchmetrics import JaccardIndex
from torch import nn
from main.train import get_iou_tables
from utils.plot_test_images import logits_to_pred, plot_seg_sample
from utils.logger import create_logger
from models.unet_pp import UNetplusplus
from models.local_temp_scaling import LTSHead, build_lts_input
from torchvision import transforms
from utils.calibration_diagrams import plot_reliability_diagram, plot_gap_bars, plot_confidence_histogram, plot_acc_conf_lines
from tqdm import tqdm
from utils.brier_score import brier_score_segmentation
from utils.ece import expected_calibration_error
if torch.backends.mps.is_available(): device = "mps"
elif torch.cuda.is_available(): device = "cuda"
else: device = "cpu"

dir_path = Path(os.getcwd())
config_path = dir_path / "configs" / "config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)



def test_segmentation(model, val_loader, num_classes: int, device: torch.device,  class_names: list, log_dir : str, ignore_index: int = 255,
    lambda_ce : float = 1.0, class_weights=None, lts_head = None,  **kwargs):

    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    else:
        class_weights = None

    if not lts_head:
        logger.warning("No LTS head provided, proceeding without calibration.")
        lts_head = nn.Identity()

    lts_head.to(device).eval()
    iou_metric = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=ignore_index, average="none").to(device)
    iou_metric.reset()

    model.to(device)
    model.eval()
    val_loss = 0.0
    with torch.no_grad(), torch.amp.autocast(device_type=device):
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            out = model(images)
            logits, logits_no_hadamard = out["logits"], out["logits_no_hadamard"]

            lts_input = build_lts_input(logits=logits, last_feats=logits_no_hadamard, mode="all").to(device)   # (B,2,H,W)
            T = lts_head(lts_input)                # (B,1,H,W)
            logits = logits / T
            val_loss += lambda_ce * F.cross_entropy(input=logits, target=targets.long(), ignore_index=ignore_index, 
                                        weight=class_weights).item() * images.size(0)
            iou_metric.update(logits, targets)


    per_class_iou = iou_metric.compute()                 
    miou = torch.nanmean(per_class_iou).item()            
    val_loss = val_loss / len(val_loader.dataset)

    

    iou_table = get_iou_tables(per_class_iou.cpu().numpy(), class_names=class_names, **kwargs)
    logger.info("Validation IoU Table:\n%s\n\n", iou_table)
    line = f"Val_loss {val_loss:.4f} | mIoU {miou:.4f}"
    bar  = "-" * (len(line) + 2)
    box  = f"\n\n+{bar}+\n| {line} |\n+{bar}+\n\n"
    logger.info("%s", box)

    probs = F.softmax(logits, dim=1)
    brier = brier_score_segmentation(probs=probs, targets=targets, ignore_index=ignore_index, reduction="mean", scale_by_2=True)
    logger.info(f"Brier score: {brier:.4f}")


    ece_out = expected_calibration_error(logits_or_probs=logits, targets=targets, n_bins=15, 
                                                                           ignore_index=ignore_index, from_logits=True, norm="l1")
    ece, bin_acc, bin_conf, bin_counts, edges = ece_out["ece"], ece_out["bin_acc"], ece_out["bin_conf"], ece_out["bin_counts"], ece_out["edges"]
    logger.info(f"ECE: {ece:.4f}")
    for i in range(len(bin_acc)):
        logger.info(f" Bin {i+1:2d}: Count={bin_counts[i]:7.0f} | Acc={bin_acc[i]:.4f} | Conf={bin_conf[i]:.4f} | Edge=({edges[i]:.2f},{edges[i+1]:.2f})")

    plot_reliability_diagram(ece_out, title_prefix="", output_path=f"{log_dir}/reliability_diagram.png")
    plot_gap_bars(ece_out, title_prefix="", output_path=f"{log_dir}/gap_bars.png")
    plot_confidence_histogram(ece_out, title_prefix="", output_path=f"{log_dir}/confidence_histogram.png")
    plot_acc_conf_lines(ece_out, title_prefix="", output_path=f"{log_dir}/acc_conf_lines.png")

    preds = logits_to_pred(logits=logits)
    iou_metric = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=ignore_index, average="none").to(device)
    with torch.no_grad(), torch.amp.autocast(device_type=device):
        for idx, (logit, target, image, pred) in tqdm(enumerate(zip(logits, targets, images, preds))):
            fig, _ = plot_seg_sample(image, gt_mask=target, pred_mask=pred, alpha=0.6, title = f"Sample {idx + 1}")
      
            logit = logit.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)
            iou_metric.update(logit, target)
            per_class_iou = iou_metric.compute()  
            miou = torch.nanmean(per_class_iou).item() 
            iou_metric.reset()

            fig.set_constrained_layout(False)     
            w, h = fig.get_size_inches()
            fig.set_size_inches(w, h + 4, forward=True)
            fig.subplots_adjust(bottom=0.32)     

            bar_ax = fig.add_axes([0.07, 0.15, 0.86, 0.20])  
            values, labels = per_class_iou.cpu().numpy(), class_names
            bars = bar_ax.bar(range(len(values)), values)
            bar_ax.bar_label(bars, labels=[f"{v*100:.1f}" for v in values], padding=2, fontweight = "bold")
            bar_ax.set_title(f"Per-class IoU. Mean IoU = {miou:.3f}", fontweight = "bold")
            bar_ax.set_xticks(range(len(labels)))
            bar_ax.set_xticklabels(labels, rotation=45, ha='right')

            fig.savefig(f"{log_dir}/sample_{idx + 1}.png", dpi=200)

    return model



if __name__ == "__main__":
    # Setup all hyperparameters
    LOG_DIR = Path(config["log_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    DATASET_PATH = dir_path / Path(config["cityscapes"]["data_dir"])
    IGNORE_IDX = config["cityscapes"]["ignore_index"]
    VALID_CLASSES = config["cityscapes"]["valid_classes"] + [IGNORE_IDX]
    OUTPUT_SHAPE = config["cityscapes"]["output_hw"]
    MODE = config["cityscapes"]["mode"]
    TARGET_TYPE = config["cityscapes"]["target_type"]

    DATA_MEAN = config["cityscapes"]["mean"]
    DATA_STD = config["cityscapes"]["std"]

    TEST_BATCH_SIZE = config["test_loader"]["batch_size"]
    TEST_NUM_WORKERS = config["test_loader"]["num_workers"]
    TEST_SHUFFLE = config["test_loader"]["shuffle"]
    TEST_PIN_MEMORY = config["test_loader"]["pin_memory"]

    MODEL_WEIGHTS_PATH = "weights/best_model_hadamard.pt"

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found at {MODEL_WEIGHTS_PATH}")


    # setup logger
    logger = create_logger("test_" + config["experiment_name"], log_dir=LOG_DIR)
    logger.info("\n\n\nCONFIGURATION:\n\n%s\n\n", pformat(config, width=100))

    # get data
    join_transform = JointResize(size_hw=OUTPUT_SHAPE)
    cityscapes_test = CityscapesSubset(root=DATASET_PATH, split="val", mode=MODE, target_type=TARGET_TYPE, valid_classes=VALID_CLASSES[:-1], 
                                        ignore_idx=IGNORE_IDX, remap_to_compact=True, joint_transform=join_transform, image_transform=transforms.Normalize(DATA_MEAN, DATA_STD))
    test_loader = DataLoader(dataset=cityscapes_test, batch_size=TEST_BATCH_SIZE, shuffle=TEST_SHUFFLE, num_workers=TEST_NUM_WORKERS, pin_memory=TEST_PIN_MEMORY)

    logger.info(
    "Split sizes: test=%d", len(cityscapes_test))

    model = UNetplusplus(in_channels=3, num_classes=len(VALID_CLASSES) - 1, use_pretrained=False, use_hadamard=True)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, weights_only=True, map_location=device)["model"])

    lts_head = LTSHead(in_ch=40, hidden=64, t_min=0.1, t_max=5.0).to(device)
    lts_head.load_state_dict(torch.load("weights/best_lts_model.pt", weights_only=True, map_location=device)["lts_model"])
    logger.info("Model: %s", model)
    logger.info("LTS Head: %s", lts_head)
    
    # Test the model
    logger.info("Using device: %s", device)
    model = test_segmentation(model=model, lts_head=lts_head, val_loader=test_loader, num_classes=len(VALID_CLASSES) - 1, device=device, 
                              class_names = cityscapes_test.class_names, class_ids=cityscapes_test.keep_ids, log_dir=LOG_DIR)
    


    