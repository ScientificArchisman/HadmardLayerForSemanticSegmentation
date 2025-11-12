import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights
from models.hadamard import HadamardLayer



def make_dilated_resnet(name, output_stride=16, pretrained=True, norm_layer=nn.BatchNorm2d):
    assert output_stride in (8, 16)
    replace = (False, output_stride == 8, True) if output_stride == 8 else (False, False, True)

    if name == 'resnet50':
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,
                     norm_layer=norm_layer, replace_stride_with_dilation=replace)
    elif name == 'resnet101':
        m = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2 if pretrained else None,
                      norm_layer=norm_layer, replace_stride_with_dilation=replace)
    elif name == 'resnet152':
        m = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2 if pretrained else None,
                      norm_layer=norm_layer, replace_stride_with_dilation=replace)
    else:
        raise ValueError(f"Unsupported ResNet model: {name}. Can be 'resnet50', 'resnet101', or 'resnet152'.")

    m.avgpool = nn.Identity()
    m.fc = nn.Identity()
    return m


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates, use_gn_for_pool=True, gn_groups=32):
        super().__init__()

        def Norm(c):  # regular branches keep BN
            return nn.BatchNorm2d(c)

        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          Norm(out_ch), nn.ReLU(inplace=True))
        ])
        for r in rates[1:]:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                    Norm(out_ch), nn.ReLU(inplace=True)
                )
            )

        if use_gn_for_pool:
            pool_norm = nn.GroupNorm(num_groups=min(gn_groups, out_ch), num_channels=out_ch)
        else:
            pool_norm = nn.Identity()

        self.img_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            pool_norm,                     
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(self.branches) + 1), out_ch, 1, bias=False),
            Norm(out_ch), nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        pooled = self.img_pool(x)
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode='bilinear', align_corners=False)
        feats.append(pooled)
        y = torch.cat(feats, dim=1)
        return self.project(y)


class DeepLabHead(nn.Module):
    def __init__(self, in_ch=2048, aspp_ch=256, num_classes=19, output_stride=16):
        super().__init__()
        rates = [1, 6, 12, 18] if output_stride == 16 else [1, 12, 24, 36]
        self.aspp = ASPP(in_ch, aspp_ch, rates=rates)
        self.classifier = nn.Sequential(
            nn.Conv2d(aspp_ch, aspp_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_ch), nn.ReLU(inplace=True),
            nn.Conv2d(aspp_ch, num_classes, 1)
        )

    def forward(self, x):
        x = self.aspp(x)
        return self.classifier(x)

class ResNetModel(nn.Module):
    def __init__(self, num_classes, output_stride=16, aux_loss=False, name='resnet50', pretrained_backbone=True, use_hadamard: bool = False):
        super().__init__()
        self.backbone = make_dilated_resnet(name=name, output_stride=output_stride, pretrained=pretrained_backbone)
        self.aux_loss = aux_loss
        in_ch = self.backbone.layer4[-1].conv3.out_channels  
        self.head = DeepLabHead(in_ch=in_ch, aspp_ch=256, num_classes=num_classes, output_stride=output_stride)
        if aux_loss:
            mid_ch = self.backbone.layer3[-1].conv3.out_channels  
            self.aux_head = nn.Sequential(
                nn.Conv2d(mid_ch, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, 1)
            )

        if use_hadamard:
            self.hadamard_layer = HadamardLayer(k=num_classes.bit_length(), n_rows=num_classes)
        else:
            self.hadamard_layer = nn.Identity()


    def forward(self, x):
        h, w = x.shape[-2:]
        # Forward through ResNet stages
        x = self.backbone.conv1(x); x = self.backbone.bn1(x); x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x1 = self.backbone.layer1(x)  # /4
        x2 = self.backbone.layer2(x1) # /8
        x3 = self.backbone.layer3(x2) # /16 or /8
        x4 = self.backbone.layer4(x3) # /16 or /8 (final)

        logits = self.head(x4)
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        logits = self.hadamard_layer(logits)

        if self.aux_loss:
            aux = self.aux_head(x3)
            aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=False)
            return {"out": logits, "aux": aux}
        return {"out": logits}

# Example usage
if __name__ == "__main__":
    model = ResNetModel(num_classes=19, output_stride=16, aux_loss=True, name='resnet101', pretrained_backbone=True, use_hadamard=True)
    out = model(torch.randn(2, 3, 512, 1024))

    print({k: v.shape for k, v in out.items()})
