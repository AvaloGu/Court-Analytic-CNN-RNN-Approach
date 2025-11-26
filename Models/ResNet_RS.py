import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from torchvision.ops import SqueezeExcitation

# hyperparameters
squeeze_reduction_ratio = 4
dropout_rate = 0
# TODO: EMA of weights, implement during training loop (wouldn't it be expensive?)

class Block(nn.Module):
    r"""
    ResNet RS Block
    """
    def __init__(self, channel_size, contraction=4, drop_path=0.):
        super().__init__()
        bottleneck_channel_size = channel_size // contraction
        
        self.conv1 = nn.Conv2d(channel_size, bottleneck_channel_size, kernel_size=1, stride=1, padding=0)
        # spatial batchnorm takes channel size as input
        self.bn1 = nn.BatchNorm2d(bottleneck_channel_size)

        self.conv2 = nn.Conv2d(bottleneck_channel_size, bottleneck_channel_size, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channel_size)

        self.conv3 = nn.Conv2d(bottleneck_channel_size, channel_size, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(channel_size)

        # Squeeze and Excitation layer
        self.se = SqueezeExcitation(input_channels = channel_size, 
                                    squeeze_channels = channel_size // squeeze_reduction_ratio)

        # stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.relu = nn.ReLU() # inplace=False, do not modifies the input tensor in-place, safer with autograd

    def forward(self, x):
        # x is (N, C, H, W)
        input = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.se(x) # Squeeze and Excitation
        x = input + self.drop_path(x) # residual connection
        x = self.relu(x)
        return x


class DownSamplingBlock(nn.Module):
    r"""
    ResNet RS DownSampling Block
    """
    def __init__(self, in_channel_size, out_channel_size, contraction=4, drop_path=0.):
        super().__init__()

        bottleneck_channel_size = out_channel_size // contraction

        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channel_size, bottleneck_channel_size, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(bottleneck_channel_size)

        # 3x3 conv with stride 2 pad 1 for downsampling
        self.conv2 = nn.Conv2d(bottleneck_channel_size, bottleneck_channel_size, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channel_size)

        # 1x1 conv to increase channel size
        self.conv3 = nn.Conv2d(bottleneck_channel_size, out_channel_size, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channel_size)

        # downsample the identiy mapping to match dimensions
        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2), # downsample by 2
            nn.Conv2d(in_channel_size, out_channel_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel_size)    
        )

        # Squeeze and Excitation layer
        self.se = SqueezeExcitation(input_channels = out_channel_size, 
                                    squeeze_channels = out_channel_size // squeeze_reduction_ratio)

        # stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        # x is (N, C, H, W)
        input = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.se(x) # Squeeze and Excitation
        input = self.downsample(input) # downsample the input to match dimensions
        x = input + self.drop_path(x) # residual connection
        x = self.relu(x)
        return x
    

class ResNet_RS(nn.Module):
    r"""
    ResNet RS 
    """
    def __init__(self, in_chans=3, depths=[3, 4, 6, 3], dims=[64, 256, 512, 1024, 2048], drop_path_rate=0.):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU()
        )

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # [0, 0.0118..., ..., 0.2]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(DownSamplingBlock(in_channel_size=dims[i], out_channel_size=dims[i+1], drop_path=dp_rates[cur]),
                                  *[Block(channel_size=dims[i+1], drop_path=dp_rates[cur + j]) for j in range(1, depths[i])])
            self.stages.append(stage)
            cur += depths[i]
        
        self.norm = nn.BatchNorm1d(dims[-1])
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # x is (N, 3, 224, 224)
        x = self.stem(x) # (N, 64, 112, 112)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x.mean([-2, -1])) # global average pooling and norm, (N, C, H, W) -> (N, C)
        x = self.dropout(x)
        return x

ResNet_RS_Model = ResNet_RS()

            