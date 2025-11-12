from torch import nn 
import torch.nn.functional as F
import torch
from models.unet_pp import UNetplusplus
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class LTSHead(nn.Module):
    def __init__(self, in_ch=3, hidden=64, t_min=0.5, t_max=5.0):
        super().__init__()
        self.t_min, self.t_max = t_min, t_max
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),

            nn.Conv2d(hidden, 1, 3, padding=1))
        self.softplus = nn.Softplus()

    def forward(self, x):
        t_logits = self.net(x)
        T = self.softplus(t_logits) + 1e-6  
        T = torch.clamp(T, min=self.t_min, max=self.t_max)
        return T  
    


def build_lts_input(logits: torch.Tensor, last_feats: torch.Tensor = None, mode = "logits") -> torch.Tensor:
    """ Build the input for the LTS Head
    Args:
        logits (torch.tensor): Logits (output) of the segmentor head
        last_feats (torch.tensor): Features of the last layer of the segmentor head
        mode (str): Mode of the calibration. Options are "logits", "logits_feat", "conf_ent".
    Returns:
        The input to the LTS head."""
    
    probs = logits.softmax(dim = 1)
    conf = logits.max(dim = 1, keepdim=True).values # (B, 1, H, W)
    ent   = -(probs * (probs.clamp_min(1e-12)).log()).sum(1, keepdim=True)  # (B,1,H,W)

    if last_feats is None:
        last_feats = logits 
    if mode == "logits":
        return logits
    elif mode == "logits_feats" and last_feats:
        last_feats = F.interpolate(last_feats, size=logits.shape[-2:], mode='bilinear', align_corners=False)
        return torch.cat([logits, last_feats], dim = 1)
    elif mode == "conf_ent":
        return torch.cat([conf, ent], dim = 1)
    return torch.cat([logits, last_feats, conf, ent], dim=1)



if __name__ == "__main__":
    x = torch.rand(size=(1, 3, 128, 128))

    with torch.no_grad():
        segmentor = UNetplusplus(in_channels=3, num_classes=19, use_pretrained=False, use_hadamard=True)
        calibrator = LTSHead(in_ch=40)

        out = segmentor(x)
        logits, logits_no_hadamard = out["logits"], out["logits_no_hadamard"]
        calibration_input = build_lts_input(logits=logits, last_feats=logits_no_hadamard, mode="all")
        temp = calibrator(calibration_input)
        calibrated_logits = logits / temp

        print(logits.shape)
        print(calibration_input.shape)
        print(temp.shape)
        print(calibrated_logits.shape)

