import torch.nn as nn
import torch.nn.functional as fun
import torch


# ------
# Mish activation
# ------
class Mish(nn.Module):
    
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(fun.softplus(x))

    
# ------
# Unit of Convolution Layer
# Conv2d + BatchNormalization + Mish/LeakyReLU
# ------
class Conv(nn.Module):
    """
    activation: leaky_relu, mish, linear
    """
    
    def __init__(
        self, in_channels, out_channels, 
        kernel_size, stride=1, activation='leaky_relu'
    ):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=kernel_size//2, bias=False
        )
        self.batchnrom = nn.BatchNorm2d(out_channels)
        
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'mish':
            self.activation = Mish()
        elif activation == 'linear':
            self.activation = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnrom(x)
        x = self.activation(x)
        return x

    
# ------
# Residual Block
# ------
class ResBlock(nn.Module):
    
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            Conv(channels, channels, 1, activation='mish'),
            Conv(channels, channels, 3, activation='linear')
        )
        self.activation = Mish()

    def forward(self, x):
        r = x
        x = self.block(x)
        x = r + x
        x = self.activation(x)
        return x

    
# ------
# CSP Residual block
# Downsampling by a factor of 2
# ------
class CSPResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, n_resblock):
        super(CSPResBlock, self).__init__()
        self.downsample = Conv(in_channels, out_channels, 3, stride=2, activation='mish')
        self.path1 = Conv(out_channels, out_channels//2, 1, activation='mish')
        self.path2 = Conv(out_channels, out_channels//2, 1, activation='mish')
        self.resblocks = nn.Sequential(
            *[ResBlock(out_channels//2) for _ in range(n_resblock)],
            Conv(out_channels//2, out_channels//2, 1)
        )
        self.conv_mix = Conv(out_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.downsample(x)
        # path 1 for big shortcut
        x1 = self.path1(x)
        # path 2 for resblocks
        x2 = self.path2(x)
        x2 = self.resblocks(x2)
        # stack then mix
        x = torch.cat([x1,x2], dim=1)
        x = self.conv_mix(x)
        return x
    

# ------
# CSP Darknet
# returns 3 feature maps
# ------
class CSPDarknet(nn.Module):

    def __init__(self, in_channels):
        super(CSPDarknet, self).__init__()
        self.conv1 = Conv(in_channels, 32, kernel_size=3, stride=1)
        self.stages = nn.ModuleList([
            CSPResBlock(32, 64, 2),
            CSPResBlock(64, 128, 2),
            CSPResBlock(128, 256, 8),
            CSPResBlock(256, 512, 8),
            CSPResBlock(512, 1024, 4)
        ])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        p3 = self.stages[2](x)  # pyramid 3
        p4 = self.stages[3](p3)  # pyramid 4
        p5 = self.stages[4](p4)  # pyramid 5
        return p3, p4, p5

    
# ------
# Spatial Pyramid Pooling (SPP)
# 5x5, 9x9, 13x13 maxpooling
# return a feature map with same size 
# ------
class SpatialPyramidPooling(nn.Module):
    
    def __init__(self, pool_sizes=[5,9,13]):
        super(SpatialPyramidPooling, self).__init__()
        self.maxpools = nn.ModuleList(
            [nn.MaxPool2d(ps,1,ps//2) for ps in pool_sizes]
        )

    def forward(self, x):
        # feature maps of 5x5, 9x9, 13x13
        fms = [maxpool(x) for maxpool in self.maxpools[::-1]]
        x = torch.cat(fms + [x], dim=1)
        return x
    
    
# ------
# Upsampling
# integrates channels then upsample with a factor of 2
# ------
class Upsample(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        x = self.upsample(x)
        return x
          
            
# ------
# YOLO head
# ------
class YOLOHead(nn.Module):
            
    def __init__(self, channels:list, in_channels):
        super(YOLOHead, self).__init__()
        self.head = nn.Sequential(
            Conv(in_channels, channels[0], 3),
            nn.Conv2d(channels[0], channels[1], 1)
        )
    
    def forward(self, x):
        return self.head(x)
            
         
def make_three_convs(channels:list, in_channels):
    convs = nn.Sequential(
        Conv(in_channels, channels[0], 1),
        Conv(channels[0], channels[1], 3),
        Conv(channels[1], channels[0], 1)
    )
    return convs

def make_five_convs(channels:list, in_channels):
    convs = nn.Sequential(
        Conv(in_channels, channels[0], 1),
        Conv(channels[0], channels[1], 3),
        Conv(channels[1], channels[0], 1),
        Conv(channels[0], channels[1], 3),
        Conv(channels[1], channels[0], 1)
    )
    return convs
            
            
# ------
# YOLOv4!!!
# Single Head
# ------
class YOLOv4(nn.Module):
    in_channels = 1
    num_classes = 1
            
    def __init__(self, num_anchors=3):
        super(YOLOv4, self).__init__()
        self.backbone = CSPDarknet(self.in_channels)
        # n_anchors * [obj, cx, cy, w, h]
        head_channels = num_anchors * (5 + self.num_classes)

        # necks
        # pyramid 5 to SPP to head 1
        self.conv_p5_1 = make_three_convs([512,1024], 1024)
        self.SPP = SpatialPyramidPooling()
        self.conv_p5_2 = make_three_convs([512,1024], 2048)
        self.upsample_p5 = Upsample(512, 256)  # upsampling
        
        # pyramid 4 to head 2
        self.conv_p4_1 = Conv(512, 256, 1)
        self.conv_p4_2 = make_five_convs([256, 512], 512)
        self.upsample_p4 = Upsample(256, 128)  # upsampling
        self.conv_p4_3 = make_five_convs([256, 512], 512)
        self.head_2 = YOLOHead([512, head_channels], 256)  # head 2
        
        # pyramid 3 to head 3
        self.conv_p3_1 = Conv(256, 128, 1)
        self.conv_p3_2 = make_five_convs([128,256], 256)
        self.downsample_p3 = Conv(128, 256, 3, stride=2)  # downsampling

    def forward(self, x):
        # extract p3, p4, p5 from backbone
        p3, p4, p5 = self.backbone(x)

        # p5 pipeline (head)
        p5 = self.conv_p5_1(p5)
        p5 = self.SPP(p5)
        p5 = self.conv_p5_2(p5)
        p5_up = self.upsample_p5(p5)

        # p4 pipeline (head)
        p4 = self.conv_p4_1(p4)
        p4 = torch.cat([p4,p5_up], dim=1)
        p4 = self.conv_p4_2(p4)
        p4_up = self.upsample_p4(p4)

        # p3 pipeline
        p3 = self.conv_p3_1(p3)
        p3 = torch.cat([p3,p4_up], dim=1)
        p3 = self.conv_p3_2(p3)
        p3_down = self.downsample_p3(p3)

        # p4 pipeline
        p4 = torch.cat([p3_down, p4], dim=1)
        p4 = self.conv_p4_3(p4)

        # heads
        h = self.head_2(p4)
        return h