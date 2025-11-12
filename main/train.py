from pathlib import Path
import os
import pandas as pd 
import numpy as np
import torch.nn.functional as F
from utils.logger import create_logger
from pprint import pformat
from torch.utils.data import DataLoader
from utils.load_cityscapes import CityscapesSubset, JointResize
import yaml 
import torch
from utils.loss import multiclass_dice_loss, l1_multiclass_from_logits
from torchmetrics import JaccardIndex
# from models.unet_pp import UNetplusplus
from models.resnet import ResNetModel
# from models.segformer import SegFormer
from torchvision import transforms
import gc
if torch.backends.mps.is_available(): device = "mps"
elif torch.cuda.is_available(): device = "cuda"
else: device = "cpu"

dir_path = Path(os.getcwd())
config_path = dir_path / "configs" / "config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


def get_iou_tables(per_class_iou, class_names, class_ids):
    return pd.DataFrame(
    {
        "Class": class_names,
        "IoU": np.round(per_class_iou * 100, 3),
    }, index=class_ids)


def train_segmentation(model, train_loader, val_loader, opt, scheduler,
    num_classes: int, device: torch.device, epochs: int = 50, ignore_index: int = 255,
    lambda_ce : float = 1.0, lambda_dice: float = 0.5, lambda_l1: float = 1.0,
    grad_clip_norm: float = 1.0, class_weights=None, results_dir: str = "results", **kwargs):

    results_path = Path(os.path.join(os.getcwd(), results_dir))
    results_path.mkdir(parents=True, exist_ok=True)
    
    model.to(device)
    if os.path.exists(results_path / "best_model.pt"):
        try:
            model.load_state_dict(torch.load(results_path / "best_model.pt", weights_only=True)["model"])
            logger.info("Loaded existing best model from %s", results_path / "best_model.pt")
        except Exception as e:
            logger.error("Failed to load existing best model: %s", e)
            logger.info("Starting training from scratch.")
            model = model.to(device) 
            torch.cuda.empty_cache()
    else:
        logger.info("No existing best model found, starting training from scratch.")


    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    else:
        class_weights = None

    iou_metric = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=ignore_index, average="none").to(device)
    best_miou = -1.0
    step = 0
    total_steps = epochs * len(train_loader)

    logger.info("Starting training for %d epochs with %d total steps", epochs, total_steps)
    logger.info("Using device: %s", device)


    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            step += 1

            images, targets = images.to(device), targets.to(device).contiguous()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device):
                out = model(images)  # (B, C, H, W)
                logits, _ = out["out"], out["aux"]
                logits = logits.contiguous()
                loss = 0

                loss = (
                    F.cross_entropy(logits, targets.long(), ignore_index=ignore_index, weight=class_weights) * lambda_ce
                    + multiclass_dice_loss(logits, targets, ignore_index=ignore_index) * lambda_dice
                    + l1_multiclass_from_logits(logits, targets, ignore_index=ignore_index) * lambda_l1
                )


            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            opt.step()
            running_loss += loss.item() * images.size(0)
            
            current_lr = opt.param_groups[0]['lr']
            if step % 1 == 0:
                logger.info(f"Epoch {epoch:03d} | Step {step:05d}/{total_steps:05d} | batch_loss {loss.item():.4f} | lr {current_lr:.6f}")

            torch.cuda.empty_cache()
            gc.collect()

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        iou_metric.reset()
        val_loss = 0.0
        with torch.no_grad(), torch.amp.autocast(device_type=device):
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                out = model(images)
                logits, _ = out["out"], out["aux"]
                val_loss += lambda_ce * F.cross_entropy(input=logits, target=targets.long(), ignore_index=ignore_index, 
                                            weight=class_weights).item() * images.size(0)
                iou_metric.update(logits, targets)


        per_class_iou = iou_metric.compute()                 
        miou = torch.nanmean(per_class_iou).item()            
        val_loss = val_loss / len(val_loader.dataset)

        iou_table = get_iou_tables(per_class_iou.cpu().numpy(), **kwargs)
        logger.info("Validation IoU Table:\n%s\n\n", iou_table)
        line = f"Epoch {epoch:03d} | Step {step:05d}/{total_steps:05d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | mIoU {miou:.4f}"
        bar  = "-" * (len(line) + 2)
        box  = f"\n\n+{bar}+\n| {line} |\n+{bar}+\n\n"
        logger.info("%s", box)

        # save best model based on mIoU
        if miou > best_miou:
            best_miou = miou
            logger.info("New best mIoU: %.4f, saving model...", best_miou)
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "miou": miou},
                       results_path / "best_model.pt")
        

    logger.info(f"Best mIoU: {best_miou:.4f}")
    return model



if __name__ == "__main__":
    # Setup all hyperparameters
    LOG_DIR = config["log_dir"]
    EXPT_NAME = config["experiment_name"]
    DATASET_PATH = dir_path / Path(config["cityscapes"]["data_dir"])
    OUTPUT_SHAPE = config["cityscapes"]["output_hw"]
    MODE = config["cityscapes"]["mode"]
    TARGET_TYPE = config["cityscapes"]["target_type"]
    IGNORE_IDX = config["cityscapes"]["ignore_index"]
    VALID_CLASSES = config["cityscapes"]["valid_classes"] + [IGNORE_IDX]

    DATA_MEAN = config["cityscapes"]["mean"]
    DATA_STD = config["cityscapes"]["std"]

    TRAIN_BATCH_SIZE = config["train_loader"]["batch_size"]
    TRAIN_NUM_WORKERS = config["train_loader"]["num_workers"]
    TRAIN_SHUFFLE = config["train_loader"]["shuffle"]
    TRAIN_PIN_MEMORY = config["train_loader"]["pin_memory"]

    VAL_BATCH_SIZE = config["train_loader"]["batch_size"]
    VAL_NUM_WORKERS = config["train_loader"]["num_workers"]
    VAL_SHUFFLE = config["train_loader"]["shuffle"]
    VAL_PIN_MEMORY = config["train_loader"]["pin_memory"]

    TEST_BATCH_SIZE = config["train_loader"]["batch_size"]
    TEST_NUM_WORKERS = config["train_loader"]["num_workers"]
    TEST_SHUFFLE = config["train_loader"]["shuffle"]
    TEST_PIN_MEMORY = config["train_loader"]["pin_memory"]

    LEARNING_RATE = config["training"]["optimizer"]["lr"]
    NUM_EPOCHS = config["training"]["epochs"]
    WEIGHT_DECAY = config["training"]["optimizer"]["weight_decay"]
    WEIGHTS_DIR = config["training"]["weights_dir"]
    OPTIMIZER_NAME = config["training"]["optimizer"]["type"]


    LAMBDA_CE = config["training"]["loss"]["lambda_ce"]
    LAMBDA_DICE = config["training"]["loss"]["lambda_dice"]
    LAMBDA_L1 = config["training"]["loss"]["lambda_l1"]
    SCHEDULER = config["training"]["scheduler"]["type"]



    # setup logger
    logger = create_logger(EXPT_NAME, log_dir=LOG_DIR)
    logger.info("\n\n\nCONFIGURATION:\n\n%s\n\n", pformat(config, width=100))



    # get data
    join_transform = JointResize(size_hw=OUTPUT_SHAPE)
    cityscapes_train = CityscapesSubset(root=DATASET_PATH, split="train", mode=MODE, target_type=TARGET_TYPE, valid_classes=VALID_CLASSES[:-1], 
                                        ignore_idx=IGNORE_IDX, remap_to_compact=True, joint_transform=join_transform, image_transform=transforms.Normalize(DATA_MEAN, DATA_STD))
    cityscapes_test = CityscapesSubset(root=DATASET_PATH, split="test", mode=MODE, target_type=TARGET_TYPE, valid_classes=VALID_CLASSES[:-1], 
                                        ignore_idx=IGNORE_IDX, remap_to_compact=True, joint_transform=join_transform, image_transform=transforms.Normalize(DATA_MEAN, DATA_STD))
    cityscapes_val = CityscapesSubset(root=DATASET_PATH, split="val", mode=MODE, target_type=TARGET_TYPE, valid_classes=VALID_CLASSES[:-1], 
                                        ignore_idx=IGNORE_IDX, remap_to_compact=True, joint_transform=join_transform, image_transform=transforms.Normalize(DATA_MEAN, DATA_STD))
    train_loader = DataLoader(dataset=cityscapes_train, batch_size=TRAIN_BATCH_SIZE, shuffle=TRAIN_SHUFFLE, num_workers=TRAIN_NUM_WORKERS, pin_memory=TRAIN_PIN_MEMORY)
    val_loader = DataLoader(dataset=cityscapes_val, batch_size=VAL_BATCH_SIZE, shuffle=VAL_SHUFFLE, num_workers=VAL_NUM_WORKERS, pin_memory=VAL_PIN_MEMORY)
    test_loader = DataLoader(dataset=cityscapes_test, batch_size=TEST_BATCH_SIZE, shuffle=TEST_SHUFFLE, num_workers=TEST_NUM_WORKERS, pin_memory=TEST_PIN_MEMORY)

    logger.info(
    "Split sizes: train=%d, val=%d, test=%d",
    len(cityscapes_train), len(cityscapes_val), len(cityscapes_test))


    # model = UNetplusplus(in_channels=3, num_classes=len(VALID_CLASSES) - 1, use_pretrained=False, use_hadamard=True)
    model = ResNetModel(num_classes=len(VALID_CLASSES) - 1, output_stride=16, aux_loss=True, name='resnet50', pretrained_backbone=True, use_hadamard=True)
    # model = SegFormer(in_channels=3, num_classes=len(VALID_CLASSES) - 1, use_pretrained=True, use_hadamard=True)

    if OPTIMIZER_NAME == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=float(LEARNING_RATE), weight_decay=float(WEIGHT_DECAY))
    elif OPTIMIZER_NAME == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=float(LEARNING_RATE), momentum=0.9, weight_decay=float(WEIGHT_DECAY))

    if SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config["training"]["scheduler"]["T_max"])
        
    elif SCHEDULER == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=config["training"]["scheduler"]["step_size"], 
                                                    gamma=config["training"]["scheduler"]["gamma"])
    elif SCHEDULER == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=config["training"]["scheduler"]["milestones"], 
                                                          gamma=config["training"]["scheduler"]["gamma"])
    elif SCHEDULER == "poly":
        scheduler = torch.optim.lr_scheduler.PolynomialLR(opt, total_iters=NUM_EPOCHS, power=0.9)
        
    elif SCHEDULER == "cyclic":
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=1e-6, max_lr=LEARNING_RATE, step_size_up=5, mode="triangular2", cycle_momentum=False)

    logger.info("Model: %s", model)
    
    # Train the model
    logger.info("Using device: %s", device)
    model = train_segmentation(model, train_loader, val_loader, opt, scheduler=scheduler, num_classes=len(VALID_CLASSES) - 1,
                               device=device, epochs=NUM_EPOCHS, ignore_index=IGNORE_IDX, results_dir=WEIGHTS_DIR,
                               lambda_ce=LAMBDA_CE, lambda_dice=LAMBDA_DICE, lambda_l1=LAMBDA_L1,
                               class_names=cityscapes_train.class_names,
                               class_ids=cityscapes_train.keep_ids)
    


    


