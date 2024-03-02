import torch
import torch.nn as nn

from collections import OrderedDict

from .common import CoordConv2d, ConvBNReluBlock, _make_divisible, conv_1x1_bn, conv_3x3_bn, h_swish, h_sigmoid

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.arguments = {
			'inp': inp, 'hidden_dim': hidden_dim, 'oup': oup, 
			'kernel_size': kernel_size, 'stride': stride, 
			'use_se': use_se, 'use_hs': use_hs}

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, mode, width_mult=1., input_channels=3, include_top=False, num_classes=1000):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        assert mode in ['large', 'small']
        self.mode = mode
        self.cfgs = self.large_cfg if self.mode == 'large' else self.small_cfg
        self.include_top = bool(include_top)

        # building first layer
        num_input_features = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(input_channels, num_input_features, 2)]  # First Layer
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(num_input_features * t, 8)
            layers.append(block(num_input_features, exp_size, output_channel, k, s, use_se, use_hs))
            num_input_features = output_channel
        # self.features = nn.Sequential(OrderedDict([(f'block_{i}', layer) for i, layer in enumerate(layers)]))
        self.features = nn.ModuleDict(OrderedDict(
            [(f'block_{i}', layer) for i, layer in enumerate(layers)]))
        # building last several layers
        self.conv = conv_1x1_bn(num_input_features, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(
            output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        if self.include_top:
            self.classifier = nn.Sequential(
                nn.Linear(exp_size, output_channel),
                h_swish(),
                nn.Dropout(0.2),
                nn.Linear(output_channel, num_classes),
            )

        # self._initialize_weights()

    def forward(self, x):
        for feature in self.features.values():
            x = feature(x)
        x = self.conv(x)
        x = self.avgpool(x)
        if self.include_top:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x

    @property
    def large_cfg(self):
        """
        Constructs a MobileNetV3-Large model
        """
        _cfgs = [
            # k, t, c, SE, HS, s
            [3, 1, 16, 0, 0, 1],
            [3, 4, 24, 0, 0, 2],
            [3, 3, 24, 0, 0, 1],
            [5, 3, 40, 1, 0, 2],
            [5, 3, 40, 1, 0, 1],
            [5, 3, 40, 1, 0, 1],
            [3, 6, 80, 0, 1, 2],
            [3, 2.5, 80, 0, 1, 1],
            [3, 2.3, 80, 0, 1, 1],
            [3, 2.3, 80, 0, 1, 1],
            [3, 6, 112, 1, 1, 1],
            [3, 6, 112, 1, 1, 1],
            [5, 6, 160, 1, 1, 2],
            [5, 6, 160, 1, 1, 1],
            [5, 6, 160, 1, 1, 1]
        ]
        return _cfgs

    @property
    def small_cfg(self):
        """
                Constructs a MobileNetV3-Small model
                """
        _cfgs = [
            # k, t, c, SE, HS, s
            [3, 1, 16, 1, 0, 2],
            [3, 4.5, 24, 0, 0, 2],
            [3, 3.67, 24, 0, 0, 1],
            [5, 4, 40, 1, 1, 2],
            [5, 6, 40, 1, 1, 1],
            [5, 6, 40, 1, 1, 1],
            [5, 3, 48, 1, 1, 1],
            [5, 3, 48, 1, 1, 1],
            [5, 6, 96, 1, 1, 2],
            [5, 6, 96, 1, 1, 1],
            [5, 6, 96, 1, 1, 1],
        ]
        return _cfgs
