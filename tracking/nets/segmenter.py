from torch import nn
import torch


class Concatenation(nn.Module):
    
    def __init__(self):
        super(Concatenation, self).__init__()
        
    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), dim=1)
        return x
    
    
class EncodingBlock(nn.Module):
    """
    Ecoder Part (Down Sampling)
    Input (H, W) then output (H/2, W/2)
    1. 2 Convolution Laysers
    2. 1 Max Pooling / 2-Strides Convolution
    
    downsample: 'conv', 'maxpool'
    """
    
    def __init__(self, in_channels:int, out_channels:int, downsample='conv'):
        super(EncodingBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU()
        )
        
        assert downsample in ['conv', 'maxpool']
        if downsample == 'conv':
            self.downsample = nn.Sequential(
                nn.Conv2d(self.out_channels, self.out_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.LeakyReLU()
            )
        elif downsample == 'maxpool':
            self.downsample = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        r = x  # residual
        x = self.downsample(x) 
        return x, r
            
        
class DecodingBlock(nn.Module):
    """
    Ecoder Part (Up Sampling)
    Input (H, W) and (H, W) then output (2*H, 2*W)
    1. 1 Linear Upsample
    2. 1 Concatenation of Encoder and Decoder Layers
    3. 2 Convolution Laysers
    
    upsample: 'transpose_conv', 'bilinear'
    """
    
    def __init__(self, in_channels:int, out_channels:int, upsample='transpose_conv'):
        super(DecodingBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        assert upsample in ['transpose_conv', 'bilinear']
        if upsample == 'transpose_conv':
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.out_channels, self.out_channels,
                    kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.LeakyReLU()
            )
        elif upsample == 'bilinear':
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            
        self.concat = Concatenation()
        
        # conv layers
        self.conv0 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU()
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU()
        )
        
    def forward(self, decoder_layer, encoder_layer):
        x = self.conv0(decoder_layer)  # change channels
        x = self.upsample(x)  # change size
        
        x = self.concat(x, encoder_layer)
        
        x = self.conv1(x)
        x = self.conv2(x)
        return x
        
        
    
class UNet(nn.Module):
    """
    downsample: 'conv', 'maxpool'
    upsample: 'transpose_conv', 'bilinear'
    """
    in_channels = 1
    out_channels =  1
    downsample = 'conv'
    upsample = 'transpose_conv'
    
    def __init__(self):
        super(UNet, self).__init__()
        
        # encoders
        self.encoder1 = EncodingBlock(self.in_channels, 32, self.downsample)
        self.encoder2 = EncodingBlock(32, 64, self.downsample)
        self.encoder3 = EncodingBlock(64, 128, self.downsample)
        self.encoder4 = EncodingBlock(128, 256, self.downsample)
        
        # bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )

        # decoders
        self.decoder4 = DecodingBlock(512, 256, self.upsample)
        self.decoder3 = DecodingBlock(256, 128, self.upsample)
        self.decoder2 = DecodingBlock(128, 64, self.upsample)
        self.decoder1 = DecodingBlock(64, 32, self.upsample)
        
        # output
        self.conv_out = nn.Conv2d(32, self.out_channels, 1, stride=1, padding=0)
    
    def forward(self, x):
        x, encode1 = self.encoder1(x)
        x, encode2 = self.encoder2(x)
        x, encode3 = self.encoder3(x)
        x, encode4 = self.encoder4(x)
        x = self.bridge(x)
        x = self.decoder4(x, encode4)
        x = self.decoder3(x, encode3)
        x = self.decoder2(x, encode2)
        x = self.decoder1(x, encode1)
        x = self.conv_out(x)
        return x
    
    
def official_model():
    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=1, out_channels=1, init_features=32, pretrained=False
    )
    return model