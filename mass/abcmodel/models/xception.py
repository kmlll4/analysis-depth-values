import torch
import torch.nn as nn
from collections import OrderedDict

from .common import Block


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, in_channels=4):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2,2,start_with_relu=True,grow_first=False)

        self.conv3 = nn.Sequential(
            SeparableConv2d(1024, 1536, 3, 1, 1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            SeparableConv2d(1536, 2048, 3, 1, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten(start_dim=1)

        self.features = nn.ModuleDict(OrderedDict([
            ('conv_1', self.conv1),
            ('conv_2', self.conv2),
            ('block_1', self.block1),
            ('block_2', self.block2),
            ('block_3', self.block3),
            ('block_4', self.block4),
            ('block_5', self.block5),
            ('block_6', self.block6),
            ('block_7', self.block7),
            ('block_8', self.block8),
            ('block_9', self.block9),
            ('block_10', self.block10),
            ('block_11', self.block11),
            ('block_12', self.block12),
            ('conv_3', self.conv3),
            ('conv_4', self.conv4), # [N, 2048, 7, 7]
            ('adaptive_avg_pool', self.adaptive_avg_pool),  # [N, 2048, 1, 1]
            ('flatten', self.flatten) # [N, 2048, 1, 1]
        ]))

    def forward(self, x):
        for name, feature in self.features.items():
            x = feature(x)
        return x
