from segmentation_models_pytorch import UnetPlusPlus
from pathlib import Path
import os 
import yaml 
from torch import nn
import torch
from models.hadamard import HadamardLayer


dir_path = Path(os.getcwd())
config_path = dir_path / "configs" / "unetpp.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


ENCODER_NAME = config["encoder_name"]
ENCODER_DEPTH = config["encoder_depth"]
ENCODER_WEIGHTS = config["encoder_weights"]
DECODER_CHANNELS = config["decoder_channels"]
DECODER_ATTENTION_TYPE = config["decoder_attention_type"]
ACTIVATION = config["activation"]


class UNetplusplus(nn.Module):
    def __init__(self, in_channels, num_classes, use_pretrained = True, use_hadamard: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = UnetPlusPlus(encoder_name=ENCODER_NAME, 
                                  encoder_depth=ENCODER_DEPTH, 
                                  encoder_weights=ENCODER_WEIGHTS if use_pretrained else None, 
                                  decoder_channels=DECODER_CHANNELS, 
                                  decoder_attention_type=DECODER_ATTENTION_TYPE, 
                                  activation=None, #ACTIVATION, 
                                  decoder_interpolation="bilinear", 
                                  in_channels=in_channels, 
                                  classes=num_classes)
        self.tanh = nn.Tanh()
        if use_hadamard:
            self.hadamard_layer = HadamardLayer(k=num_classes.bit_length(), n_rows=num_classes)
        else:
            self.hadamard_layer = nn.Identity()

    def forward(self, x):
        x_model = self.model(x)
        x_hadamard = self.hadamard_layer(x_model)
        return {"out": x_hadamard, "aux": x_model}

if __name__ == "__main__":
    x = torch.randn(size=(1, 3, 128, 128))
    with torch.no_grad():
        model = UNetplusplus(in_channels=3, num_classes=19, use_pretrained=False, use_hadamard=True)
        output = model(x) 
        print(output["out"].shape)
