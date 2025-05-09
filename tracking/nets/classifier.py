from torch import nn
import torchvision as tv
import torch


class Conv(nn.Module):
    
    def __init__(self, in_channels, out_channels ,kernel_size, stride=1, linear=False):
        super(Conv, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size, stride, kernel_size//2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
        if linear:
            self.acti = nn.Identity()
        else:
            self.acti = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.acti(x)
        return x


class ResBlock(nn.Module):
    """
    If down sample, size/2 and out_channels = 2*in_channels.
    """
    
    def __init__(self, in_channels, downsample=False):
        super(ResBlock, self).__init__()
        
        if downsample:
            stride = 2
            out_channels = 2*in_channels
            self.downsample = Conv(in_channels, out_channels, 1, stride, linear=True)
        else:
            stride = 1
            out_channels = in_channels
            self.downsample = None
        
        self.conv1 = Conv(in_channels, out_channels, 3)
        self.conv2 = Conv(out_channels, out_channels, 3, stride, linear=True)
        self.acti = nn.LeakyReLU()
        
    def forward(self, x):
        r = x
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        if self.downsample:
            r = self.downsample(r)
        
        return self.acti(x + r)
        
    
class ResNet(nn.Module):
    """
    Revised ResNet vesrsion.
    Fully convolutional network.
    Size reduction factor = 32.
    """
    in_channels = 1
    n_blocks = [3, 3, 3, 3]
    
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
#         input_norm = nn.InstanceNorm2d(self.in_channels)
        
        self.layer0 = Conv(self.in_channels, 64, 7, stride=2)
        self.layer1 = self._cook_layer(64, self.n_blocks[0])
        self.layer2 = self._cook_layer(128, self.n_blocks[1])
        self.layer3 = self._cook_layer(256, self.n_blocks[2])
        self.layer4 = self._cook_layer(512, self.n_blocks[3])
        self.fusion = Conv(1024, num_classes, 1, linear=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def _cook_layer(self, in_channels, n_blocks):
        blocks = []
        for n in range(n_blocks-1):
            blocks = [ResBlock(in_channels)]
        blocks.append(ResBlock(in_channels, downsample=True))
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fusion(x)
        x = self.avgpool(x)
        x = x.squeeze(-1).squeeze(-1)
        return torch.sigmoid(x)
        
        
def official_model(name, num_class, in_channels=1):
    if name == 'official-18':
        model = tv.models.resnet18()
    elif name == 'official-34':
        model = tv.models.resnet34()
    elif name == 'official-50':
        model = tv.models.resnet50()
    else:
        raise AssertionError(
            'Only supports models: "default", "official-18", "official-34", "official-50".'
        )
        
    # input
    model.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.in_channels = in_channels
    
    # output
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_class),
        nn.Sigmoid()
    )

    return model