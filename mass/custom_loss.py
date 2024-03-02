# import torch
from torch import nn
# import torch.nn.functional as F


class MseWithPos(nn.Module):
    def __init__(self):  # パラメータの設定など初期化処理を行う
        super(MseWithPos, self).__init__()

    def forward(self, outputs, targets, pos_weights):  # モデルの出力と正解データ
        # loss = F.mse_loss(outputs, targets, reduction='mean')
        loss = (pos_weights * (outputs - targets) ** 2).mean()

        return loss
