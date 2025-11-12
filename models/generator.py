import torch 
from torch import nn
from resnet import Resnet_FCN
from unet import UNet
from hadamard import HadamardLayer
from pathlib import Path
import yaml 
ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config.yaml"
with open(CFG_PATH, 'r') as file:
    config = yaml.safe_load(file)



IF_RESNET_PRETRAINED = config["resnet_encoder"]["pretrained"]
UNET_BILINEAR = config["unet"]["bilinear"]
HADAMARD_USE_SOFTMAX = config["hadamard_layer"]["use_softmax"]
HADAMARD_SOFTMAX_DIM = config["hadamard_layer"]["softmax_dim"]
HADAMARD_DIMS = tuple(config["hadamard_layer"]["dims"])
HADAMARD_NORMALIZE = config["hadamard_layer"]["normalize"]



class Generator(nn.Module):
    def __init__(self, backbone_name: str, in_channels: int, num_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if num_classes & (num_classes - 1): # Checks if num_classes is a power of 2
            raise ValueError("Number of output classes must be a power of 2")
        
        available_models = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "unet"]
        if backbone_name not in available_models:
            raise AssertionError(f"Backbone name must be in {available_models}")
        
        if "unet" in backbone_name:
            self.backbone = UNet(in_channels=in_channels, num_classes=num_classes, bilinear=UNET_BILINEAR)
        elif "resnet" in backbone_name:
            self.backbone = Resnet_FCN(model_name=backbone_name, num_classes=num_classes, pretrained=IF_RESNET_PRETRAINED)

        self.hadamard_layer = HadamardLayer(use_softmax=HADAMARD_USE_SOFTMAX, softmax_dim=HADAMARD_SOFTMAX_DIM, 
                                            dims=HADAMARD_DIMS, normalize=HADAMARD_NORMALIZE)

    def forward(self, x):
        output_gen = self.backbone(x)
        output_hadamard = self.hadamard_layer(output_gen)
        return torch.concat((output_hadamard, x), dim = 1)


if __name__ == "__main__":
    in_channels = config["generator"]["in_channels"]
    num_classes = config["generator"]["num_classes"]
    backbone_name = config["generator"]["backbone_name"]

    x = torch.rand((2, 3, 32, 32))
    generator = Generator(backbone_name=backbone_name, in_channels=in_channels, num_classes=num_classes)    
    generator.eval()
    with torch.no_grad():
        output = generator(x)
        print(output.shape)
