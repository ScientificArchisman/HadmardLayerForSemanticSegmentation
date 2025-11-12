import gc
import torch
import torch.nn.functional as F
from torch import optim
from models.local_temp_scaling import LTSHead, build_lts_input
from models.unet_pp import UNetplusplus
import numpy as np 
from pprint import pformat
from pathlib import Path
from main.train import create_logger
from torch.utils.data import DataLoader
from utils.load_cityscapes import CityscapesSubset, JointResize
from torchvision import transforms
import yaml 
import os 

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"



def laplacian_smoothness(T):
    # T: (B,1,H,W)
    dx = T[:, :, :, 1:] - T[:, :, :, :-1]
    dy = T[:, :, 1:, :] - T[:, :, :-1, :]
    return (dx.pow(2).mean() + dy.pow(2).mean())

def train_lts(segmentor, lts_head, calib_loader, device,
              results_dir = "results", epochs=5, lr=1e-3, weight_decay=1e-4,
              lambda_smooth=1e-3, ignore_index=255):
    
    results_path = Path(os.path.join(os.getcwd(), results_dir))
    results_path.mkdir(parents=True, exist_ok=True)

    segmentor.to(device).eval()                    # freeze model for deterministic logits
    for p in segmentor.parameters():
        p.requires_grad_(False)

    lts_head.to(device).train()
    opt = optim.AdamW(lts_head.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(calib_loader)
    logger.info("Starting calibration for %d epochs with %d total steps", epochs, total_steps)
    logger.info("Using device: %s", device)
    logger.info("Calibration Model: %s", lts_head)

    best_loss, step = np.inf, 0
    for epoch in range(epochs):
        running_loss = 0.0
        for images, targets in calib_loader:
            step += 1
            images, targets = images.to(device), targets.to(device)
            with torch.no_grad():
                out = segmentor(images)
                logits, logits_no_hadamard = out["logits"], out["logits_no_hadamard"]

            # Build LTS input x
            lts_input = build_lts_input(logits=logits, last_feats=logits_no_hadamard, mode="all").to(device)   # (B,40,H,W)

            with torch.autocast(device_type=device, dtype=torch.float16):
                T = lts_head(lts_input)               # (B,1,H,W)
                calib_logits = logits / T     
                loss = F.cross_entropy(calib_logits, targets, ignore_index=ignore_index) + lambda_smooth * laplacian_smoothness(T)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += loss.item() * images.size(0)

            if step % 1 == 0:
                logger.info(
                    f"Epoch {epoch:03d} | Step {step:05d}/{total_steps:05d} | "
                    f"Loss {loss.item():.4f}")

            torch.cuda.empty_cache()
            gc.collect()

        epoch_loss = running_loss / len(calib_loader)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            logger.info(f"New best loss: {best_loss}, Saving model..............")
            torch.save({
                "lts_model": lts_head.state_dict(), 
                "epoch" : epoch}, results_path / "best_lts_model.pt")
        logger.info(f"Epoch {epoch+1}/{epochs} | loss={epoch_loss}")

    lts_head.eval()
    return lts_head




if __name__ == "__main__":
    dir_path = Path(os.getcwd())
    config_path = dir_path / "configs" / "config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Setup all hyperparameters
    LOG_DIR = config["log_dir"]
    DATASET_PATH = dir_path / Path(config["cityscapes"]["data_dir"])
    OUTPUT_SHAPE = config["cityscapes"]["output_hw"]
    MODE = config["cityscapes"]["mode"]
    TARGET_TYPE = config["cityscapes"]["target_type"]
    IGNORE_IDX = config["cityscapes"]["ignore_index"]
    VALID_CLASSES = config["cityscapes"]["valid_classes"] + [IGNORE_IDX]

    DATA_MEAN = config["cityscapes"]["mean"]
    DATA_STD = config["cityscapes"]["std"]

    VAL_BATCH_SIZE = config["val_loader"]["batch_size"]
    VAL_NUM_WORKERS = config["val_loader"]["num_workers"]
    VAL_SHUFFLE = config["val_loader"]["shuffle"]
    VAL_PIN_MEMORY = config["val_loader"]["pin_memory"]


    NUM_EPOCHS = config["calibration"]["training"]["epochs"]
    WEIGHT_DECAY = config["calibration"]["optimizer"]["weight_decay"]
    LEARNING_RATE = config["calibration"]["optimizer"]["lr"]
    OPTIMIZER_NAME = config["calibration"]["optimizer"]["type"]
    WEIGHTS_DIR = config["training"]["weights_dir"]
    SEGMENTOR_WEIGHTS_PATH = "weights/best_model_hadamard.pt"

    LAMDA_SMOOTH = config["calibration"]["training"]["lambda_smooth"]


    lts_model = LTSHead(in_ch=40, hidden=64, t_min=0.1, t_max=5.0).to(device)
    lts_model.load_state_dict(torch.load("weights/best_lts_model.pt", weights_only=True, map_location=device)["lts_model"])
    segmentor = UNetplusplus(in_channels=3, num_classes=len(VALID_CLASSES) - 1, use_pretrained=False, use_hadamard=True).to(device)
    segmentor.load_state_dict(torch.load(SEGMENTOR_WEIGHTS_PATH, weights_only=True, map_location=device)["model"])

    if OPTIMIZER_NAME == "adamw":
        opt = torch.optim.AdamW(lts_model.parameters(), lr=float(LEARNING_RATE), weight_decay=float(WEIGHT_DECAY))
    elif OPTIMIZER_NAME == "sgd":
        opt = torch.optim.SGD(lts_model.parameters(), lr=float(LEARNING_RATE), momentum=0.9, weight_decay=float(WEIGHT_DECAY))

    
    logger = create_logger("calib_" + config["experiment_name"], log_dir=LOG_DIR)
    logger.info("\n\n\nCONFIGURATION:\n\n%s\n\n", pformat(config, width=100))

    # Get data
    join_transform = JointResize(size_hw=OUTPUT_SHAPE)
    cityscapes_val = CityscapesSubset(root=DATASET_PATH, split="val", mode=MODE, target_type=TARGET_TYPE, valid_classes=VALID_CLASSES[:-1], 
                                        ignore_idx=IGNORE_IDX, remap_to_compact=True, joint_transform=join_transform, image_transform=transforms.Normalize(DATA_MEAN, DATA_STD))
    val_loader = DataLoader(dataset=cityscapes_val, batch_size=VAL_BATCH_SIZE, shuffle=VAL_SHUFFLE, num_workers=VAL_NUM_WORKERS, pin_memory=VAL_PIN_MEMORY)

    # train LTS model
    lts_model = train_lts(segmentor=segmentor, lts_head=lts_model, calib_loader=val_loader, device=device,
                          results_dir=WEIGHTS_DIR, epochs=NUM_EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                          lambda_smooth=LAMDA_SMOOTH, ignore_index=IGNORE_IDX)