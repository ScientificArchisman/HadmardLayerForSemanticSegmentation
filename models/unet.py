import torch 
from torch import nn 

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_maxpool_dropout = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding="same", 
                               kernel_size=(3, 3), stride=1, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                               padding="same", kernel_size=(3, 3), stride=1, bias = False, **kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.use_maxpool_dropout = use_maxpool_dropout

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.bn(x)
        x = self.relu(self.conv2(x))

        if self.use_maxpool_dropout:
            x_out = self.dropout(self.maxpool(x))
            return x, x_out
        return x



class Encoder(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_block1 = DownsampleBlock(in_channels=in_channels, out_channels=64, use_maxpool_dropout=True)
        self.conv_block2 = DownsampleBlock(in_channels=64, out_channels=128, use_maxpool_dropout=True)
        self.conv_block3 = DownsampleBlock(in_channels=128, out_channels=256, use_maxpool_dropout=True)
        self.conv_block4 = DownsampleBlock(in_channels=256, out_channels=512, use_maxpool_dropout=True)
        self.conv_block5 = DownsampleBlock(in_channels=512, out_channels=1024, use_maxpool_dropout=False)

    def forward(self, x):
        f2, d4 = self.conv_block1(x) # first downsample
        f6, d8 = self.conv_block2(d4) # second downsample
        f10, d12 = self.conv_block3(d8) # third downsample
        f14, d16 = self.conv_block4(d12) # fourth downsample
        f18 = self.conv_block5(d16) # fifth downsample
        return f18, f14, f10, f6, f2


class DecoderConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, 
                               kernel_size=(3, 3), padding="same", stride=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=mid_channels)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, 
                               kernel_size=(3, 3), padding="same", stride=1, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.bn(x)
        return self.relu(self.conv2(x))


class UpsampleBlock(nn.Module):
    def __init__(self, scale_factor: int = 2, dropout_prob: float = 0.2):
        super().__init__() 
        self.upsamle = nn.Upsample(scale_factor=scale_factor)
        self.dropout = nn.Dropout(p = dropout_prob)
    
    def forward(self, x):
        x = self.upsamle(x)
        return self.dropout(x) 
       

class Decoder(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_channels = [1536, 768, 384, 192]
        mid_channels = [512, 256, 128, 64]
        out_channels = [512, 256, 128, num_classes]
        self.upsample_layers = nn.Sequential(*[UpsampleBlock() for _ in range(4)])
        self.decoder_layers = nn.Sequential(*[DecoderConvBlock(in_channels=in_ch, mid_channels=mid_ch, out_channels=out_ch)
                                            for in_ch, mid_ch, out_ch in zip(in_channels, mid_channels, out_channels)])
        
    def _upsample_concat_decode(self, input_tensor, tensor_to_concat, upsample_layer, decoder_layer):
        upsampled_tensor = upsample_layer(input_tensor)
        concat_tensor = torch.concat((upsampled_tensor, tensor_to_concat), dim=1)
        decoded_tensor = decoder_layer(concat_tensor)
        return decoded_tensor

    def forward(self, f18, f14, f10, f6, f2):
        f22 = self._upsample_concat_decode(f18, f14, self.upsample_layers[0], self.decoder_layers[0]) # upsample 1
        f26 = self._upsample_concat_decode(f22, f10, self.upsample_layers[1], self.decoder_layers[1]) # upsample 2
        f30 = self._upsample_concat_decode(f26, f6, self.upsample_layers[2], self.decoder_layers[2]) # upsample 3
        f34 = self._upsample_concat_decode(f30, f2, self.upsample_layers[3], self.decoder_layers[3]) # upsample 4
        return f34



class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, x):
        f18, f14, f10, f6, f2 = self.encoder(x)
        x = self.decoder(f18, f14, f10, f6, f2)
        return x




if __name__ == "__main__":
    x = torch.rand(size = (1, 3, 1024, 1024))
    model = UNet(in_channels=3, num_classes=20)
    out = model(x)
    print(out.shape)

    