from typing import Tuple

import torch
from torch import nn

from .mobilenetv3 import MobileNetV3
from .xception import Xception
from .resnet import ResNet, BasicBlock, Bottleneck

from .efficientnet import Efficientnet
from .hrnet import HRNet_Posture


class MassModelV3(nn.Module):
    # def __init__(self, args, input_channels=4, num_features=16, img_out_features=16, backbone='mobilenet',
    def __init__(self, input_channels=4, num_features=16, img_out_features=16, backbone='mobilenet',
                 momentum=0.99, eps=0.001, min=70., max=135, initial_bias=None, transfer=False, dropout=False, pretrained_weight=None):
        self.min = min
        self.max = max
        super(MassModelV3, self).__init__()
        if backbone == 'mobilenet':
            self.image_feature_extractor = MobileNetV3(
                mode='large', include_top=False, input_channels=input_channels)
            in_features = 960  # mobilenet v3 large
        elif backbone == 'xception':
            self.image_feature_extractor = Xception(in_channels=input_channels)
            in_features = 2048
        elif backbone == 'resnet18':
            self.image_feature_extractor = ResNet(input_channels=input_channels, block=BasicBlock, layers=[2, 2, 2, 2])
            in_features = 512
        elif backbone == 'resnet34':
            self.image_feature_extractor = ResNet(input_channels=input_channels, block=BasicBlock, layers=[3, 4, 6, 3])
            in_features = 512
        elif backbone == 'resnet50':
            self.image_feature_extractor = ResNet(input_channels=input_channels, block=Bottleneck, layers=[3, 4, 6, 3])
            in_features = 2048
        elif backbone == 'efficientnet':
            self.image_feature_extractor = Efficientnet(input_channels=input_channels, transfer=transfer, pretrained_weight=pretrained_weight)
            in_features = 1280
        elif backbone == 'hrnet':
            self.image_feature_extractor = HRNet_Posture(input_channels=input_channels, transfer=transfer, pretrained_weight=pretrained_weight)
            in_features = 1536

        else:
            raise ValueError(f"Invalid backbone: {backbone}")
        self.img_fc1 = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=img_out_features),
            nn.BatchNorm1d(num_features=img_out_features, momentum=momentum, eps=eps),
            nn.ReLU(inplace=True)
        )
        self.img_fc2 = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=img_out_features),
            nn.BatchNorm1d(num_features=img_out_features, momentum=momentum, eps=eps),
            # nn.LayerNorm(img_out_features),
            nn.ReLU(inplace=True)
        )
        self.density_fc = nn.Sequential(
            nn.Linear(in_features=img_out_features, out_features=1),
            nn.ReLU(inplace=True)
        )
        self.volume_fc1 = nn.Sequential(
            nn.Linear(in_features=img_out_features + num_features, out_features=512),
            nn.BatchNorm1d(num_features=512, momentum=momentum, eps=eps),
            # nn.LayerNorm(512),
            nn.ReLU(inplace=True),
        )
        self.volume_fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=64),
            nn.BatchNorm1d(num_features=64, momentum=momentum, eps=eps),
            # nn.LayerNorm(64),
            nn.ReLU(inplace=True),
        )
        self.volume_fc3 = nn.Sequential(
            # nn.Dropout(0.2), # 追加
            nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, x, features) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: torch.Tensor (b, 4, 224, 224)
            features: torch.Tensor (b, num_feats)
        Returns:
            mass: predicted mass (product of volume and density)
            volume: output from feature tower
            density: output from density tower
        """
        img_feature = self.image_feature_extractor(x)
        img_feature = img_feature.flatten(1)  # (b, features, 1, 1) -> (b, features)
        # img_feature_left = self.img_fc1(img_feature)
        img_feature_right = self.img_fc2(img_feature)
        # density = self.density_fc(img_feature_left)

        features = torch.cat([img_feature_right, features], dim=1)
        # features = img_feature_right
        features = self.volume_fc1(features)
        features = self.volume_fc2(features)
        mass = self.volume_fc3(features)  # activated by custom tanh
        # mass = torch.clamp(mass, min=self.min, max=self.max)

        return mass


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList()
        for m in models:
            self.models.append(m)

    def forward(self, *args, **kwargs):
        return torch.stack([model(*args, **kwargs) for model in self.models], dim=1)
