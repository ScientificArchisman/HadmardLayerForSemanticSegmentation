<div style="
  display:flex; flex-direction:column; align-items:center; gap:14px;
  padding:22px; border-radius:16px;
  background: transparent;
  border:1px solid rgba(127,127,127,0.25);
">
  <div style="text-align:center; font-size:26px; font-weight:800; line-height:1.25;
              mix-blend-mode:difference; color:white;">
    Autonomous Vehicles Semantic Segmentation with Uncertainty Calibration
  </div>
  <div style="display:flex; align-items:center; justify-content:center; gap:36px; margin-top:8px;">
    <img src="data/assets/ifn_logo.png" alt="IfN Logo" style="height:140px; object-fit:contain;" />
    <img src="data/assets/tu_bs_logo.svg.png" alt="TU Braunschweig Logo" style="height:120px; object-fit:contain;" />
  </div>
</div>


## 1. Project Overview
- End-to-end semantic segmentation pipeline for autonomous driving scenarios built around the Cityscapes dataset (19 trainId classes, ignore index 255).
- Encoder-decoder segmentation backbones (DeepLab-style ResNet, UNet++, SegFormer) instrumented with an optional Hadamard coding layer for error-correcting logit projections.
- Local Temperature Scaling (LTS) head for spatially-aware uncertainty calibration with smoothness regularisation.
- Training, calibration, and evaluation entrypoints with per-class IoU, Expected Calibration Error (ECE), Brier score, qualitative visualisations, and experiment logging.
- Reusable utilities for dataset handling, loss computation, logging, plotting, and automated Cityscapes downloads.

The current default experiment (`configs/config.yaml`) trains a DeepLab-style ResNet-50 with auxiliary classifier, Hadamard coding, and a multi-component loss (cross-entropy, Dice, L1). Calibrated logits are produced by the LTS head that consumes both decoder logits and intermediate features.

## 2. Repository Layout
- `main/train.py` – training loop, optimizer/scheduler wiring, metric tracking, checkpointing.
- `main/calibrate.py` – fine-tunes the LTS head on validation data with Laplacian smoothness.
- `main/test.py` – evaluation with calibration metrics, reliability diagrams, qualitative plots.
- `configs/` – experiment hyperparameters (`config.yaml`) and model-specific settings (`unetpp.yaml`, `segformer.yaml`).
- `models/` – segmentation backbones, Hadamard layer, LTS head, architecture docs.
- `utils/` – dataset wrappers, loss functions, calibration metrics, plotting helpers, download script, logging utilities.
- `weights/` – checkpoints saved by training and calibration; pre-populated with baseline weights.
- `logs/` – timestamped run folders containing `.log` files, plots, and exported figures.
- `setup.sh` – convenience script that provisions a Conda environment and installs PyTorch + dependencies.

## 3. Environment Setup
### Prerequisites
- Python 3.10+
- Conda (Miniconda/Anaconda) recommended for environment isolation
- macOS or Linux; NVIDIA GPU optional (CPU and Apple MPS paths are supported)

### Using the provided script
```bash
chmod +x setup.sh
./setup.sh torch_env   # choose your preferred environment name
conda activate torch_env
```
The script selects the appropriate PyTorch wheels (CUDA, CPU, or MPS), upgrades `pip`, and installs `matplotlib`, `timm`, and `segmentation-models-pytorch`.

### Manual installation (if you prefer pip/venv)
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0  # choose wheels matching your platform
python -m pip install matplotlib timm segmentation-models-pytorch pyyaml pandas torchmetrics tqdm scienceplots
```
Add any extra packages your experiments require (for example `tensorboard`).

## 4. Dataset Preparation
1. Create `data/cityscapes/` and populate it with `leftImg8bit/` and `gtFine/` folders from the official Cityscapes downloads.
2. Alternatively use the helper script (requires Cityscapes credentials in `.env`):
   ```bash
   export CITYSCAPES_USERNAME="<username>"
   export CITYSCAPES_PASSWORD="<password>"
   python utils/download_cityscapes.py 1 3
   ```
   Package IDs match the official download portal (1 = ground-truth fine annotations, 3 = leftImg8bit train/val/test).
3. Ensure the directory referenced by `configs/config.yaml::cityscapes.data_dir` matches the location (default `data/cityscapes`).

Images are resized to `640x1152` by default through `utils.load_cityscapes.JointResize`. Adjust the shape if you want higher resolution (be mindful of GPU memory).

## 5. Configuration System
All experiment hyperparameters live in `configs/config.yaml`:
- `experiment_name`, `log_dir`, `pytorch_hub.weights_dir` – run naming and caching.
- `cityscapes` – dataset root, valid classes, normalization stats, output resolution.
- `train_loader`, `val_loader`, `test_loader` – batch size, workers, shuffle, pinned memory.
- `training` – epochs, optimizer type/params, loss weights, scheduler settings, checkpoint directory (`weights_dir`).
- `calibration` – epochs, optimizer, smoothness weight for LTS fine-tuning.

Model-specific knobs are in `configs/unetpp.yaml` and `configs/segformer.yaml`. They are consumed by the corresponding model constructors and expose encoder choice, depth, decoder channels, and activation.

Update configs via regular YAML editing. Example: switch to cosine annealing and increase batch size
```yaml
train_loader:
  batch_size: 6
  num_workers: 4

training:
  optimizer:
    type: "adamw"
    lr: 0.0005
  scheduler:
    type: "cosine"
    T_max: 100
```

## 6. Experiment Workflow
1. **Configure** – adapt `configs/config.yaml` (and optional model configs) to your experiment.
2. **Train the segmentor**
  ```bash
  python -m main.train
  ```
  - Loads Cityscapes splits with compact label remapping.
  - Applies mixed loss (CrossEntropy + Dice + L1) with class weighting support.
  - Tracks per-class IoU using `torchmetrics.JaccardIndex` and saves the best model as `weights/best_model.pt`.
3. **Calibrate** (optional but recommended)
  ```bash
  python -m main.calibrate
  ```
  - Loads the segmentation checkpoint (default `weights/best_model_hadamard.pt`).
  - Trains `models.local_temp_scaling.LTSHead` on the validation split with Laplacian smoothness (`lambda_smooth`).
  - Persists LTS weights to `weights/best_lts_model.pt`.
4. **Evaluate**
  ```bash
  python -m main.test
  ```
  - Restores the segmentor and LTS head.
  - Reports mIoU, per-class IoU, Brier score, ECE bins, and qualitative overlays saved under `logs/YYYYMMDD_HHMMSS/`.
  - Generates `reliability_diagram.png`, `gap_bars.png`, `confidence_histogram.png`, `acc_conf_lines.png`, and `sample_*.png` composites.

Use the log files in `logs/<timestamp>/<run>.log` to monitor training dynamics. Each script emits configuration dumps for reproducibility.

## 7. Modifying Experiments
### Switching models or backbones
`main/train.py` instantiates the segmentation model near the bottom. Swap to a different backbone and toggle Hadamard coding:
```python
from models.unet_pp import UNetplusplus
# from models.segformer import SegFormer

model = UNetplusplus(
   in_channels=3,
   num_classes=len(VALID_CLASSES) - 1,
   use_pretrained=True,
   use_hadamard=False,
)
```
The Hadamard layer wraps logits with encode/decode projections (see `models/hadamard.py`). Setting `use_hadamard=False` disables it.

For DeepLab-style ResNet options, adjust the constructor:
```python
model = ResNetModel(
   num_classes=len(VALID_CLASSES) - 1,
   output_stride=8,
   aux_loss=True,
   name="resnet101",
   pretrained_backbone=True,
   use_hadamard=True,
)
```

### Custom loss and metrics
- Loss weights live under `training.loss` (lambda_ce/dice/l1).
- Functions are defined in `utils/loss.py` and are composed in the training loop. Extend or replace them there (e.g., add focal loss).
- Additional metrics can be tracked by instantiating new `torchmetrics` objects alongside `JaccardIndex`.

### Data augmentations and resolution
- `JointResize` in `utils/load_cityscapes.py` currently rescales to a fixed size. Replace it with RandomCrop/Flip pipelines as needed (ensure masks use nearest-neighbour interpolation).
- Normalisation statistics are set in `configs/config.yaml::cityscapes.mean/std`.

### Scheduler/optimizer tweaks
The scheduler is selected via `training.scheduler.type`. Supported options: `cosine`, `step`, `multistep`, `poly`, `cyclic`. Add custom schedulers by extending the conditional block in `main/train.py`.

### Running multiple experiments
- Change `experiment_name` to create separate log files.
- Use distinct `weights_dir` folders (e.g. `weights/exp1`, `weights/exp2`) to keep checkpoints isolated.
- Consider copying config files per experiment (`configs/exp_resnet.yaml`) and loading them manually if you need parallel configurations.

## 8. Component Reference
| Path | Description |
| --- | --- |
| `main/train.py` | Training driver, dataset setup, logging, checkpoint saver (`train_segmentation`). |
| `main/calibrate.py` | Calibration loop for spatial LTS (builds inputs via `build_lts_input`). |
| `main/test.py` | Evaluates segmentor+LTS, exports metrics and visualisations. |
| `models/resnet.py` | DeepLab-style head over dilated ResNet-50/101/152 with ASPP and optional Hadamard layer. |
| `models/unet_pp.py` | Wrapper around `segmentation_models_pytorch.UnetPlusPlus`. |
| `models/segformer.py` | SegFormer implementation using SMP and optional Hadamard coding. |
| `models/hadamard.py` | Sylvester-construction codebook, orthogonal encode/decode layer (no learnable params). |
| `models/local_temp_scaling.py` | LTS head definition and feature fusion helper. |
| `utils/load_cityscapes.py` | Cityscapes dataset wrapper, class metadata, joint transforms. |
| `utils/loss.py` | Multi-class Dice loss and L1-from-logits helper. |
| `utils/ece.py` | Pixel-wise expected calibration error with configurable norm. |
| `utils/brier_score.py` | Per-pixel multi-class Brier score. |
| `utils/calibration_diagrams.py` | Reliability diagrams, gap bars, histograms, accuracy-confidence plots. |
| `utils/plot_test_images.py` | Colour palette, overlays, and per-sample IoU bar charts. |
| `utils/logger.py` | Timestamped file + console logging helper. |
| `utils/download_cityscapes.py` | Automated downloader with resume support and archive extraction. |
| `setup.sh` | Conda environment bootstrapper with PyTorch wheel selection. |

## 9. Outputs and Checkpoints
- Best segmentation checkpoint: `weights/best_model.pt` (dict with `model`, `epoch`, `miou`).
- Pre-shared baselines: `weights/best_model_hadamard.pt`, `weights/best_model_non_hadamard.pt`, `weights/best_lts_model.pt`, `weights/best_model_resnet50_no_pretraining_no_hadamard_cosine_scheduler.pt`, etc.
- Calibration head checkpoint: `weights/best_lts_model.pt` (key `lts_model`).
- Run artefacts: `logs/<timestamp>/` contains `*.log`, calibration plots, qualitative samples, and per-sample IoU bar charts.

To resume or continue training:
```python
checkpoint = torch.load("weights/best_model.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
```
Keep in mind that optimizer state is not saved in the default trainer; add it to `torch.save` if you require full-state resumption.

## 10. Extending the Project
- **New datasets:** Implement a dataset class similar to `CityscapesSubset` with remapping and pass it into the loaders. Update `valid_classes`, `ignore_index`, and transforms accordingly.
- **Alternative calibration:** Replace `LTSHead` with your calibrator, ensuring it exposes a `forward(logits_features) -> temperatures`. Reuse `build_lts_input` or provide your own feature builder.
- **Additional metrics:** Add utilities under `utils/` and call them inside `main/test.py` to log new diagnostics.
- **Model documentation:** See `models/docs/RESNET.md` for an architectural overview of the DeepLab-style heads.

## 11. Troubleshooting
- CUDA OOM: reduce `train_loader.batch_size`, lower `output_hw`, or switch to gradient accumulation.
- Cityscapes path errors: verify `configs/config.yaml::cityscapes.data_dir` and ensure the folder contains `leftImg8bit/` and `gtFine/`.
- Missing weights: either run training or download the provided checkpoints into `weights/`.
- Mixed-precision issues on CPU: the training loop uses `torch.amp.autocast`. On pure CPU builds, set `autocast` context to `device_type="cpu"` or remove it.

## 12. References
- Cordts et al., "The Cityscapes Dataset for Semantic Urban Scene Understanding," CVPR 2016.
- Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation," DLMIA 2018.
- Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation" (DeepLab v3), arXiv 2017.
- Seo et al., "Learning where to look while tracking," CVPR 2022 (spatial temperature scaling inspiration).
- Standard Hadamard code constructions (Sylvester, 1867).
- Guo et al., "On Calibration of Modern Neural Networks," ICML 2017 (ECE, reliability diagrams).
