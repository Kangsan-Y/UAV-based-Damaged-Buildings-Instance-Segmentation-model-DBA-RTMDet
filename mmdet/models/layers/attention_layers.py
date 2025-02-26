import torch.nn as nn
from mmdet.registry import MODELS
from fightingcv_attention.attention.CBAM import CBAMBlock as _CBAMBlock

@MODELS.register_module()
class CBAMBlock(nn.Module):
    
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.module = _CBAMBlock(channel = in_channels, reduction=reduction, kernel_size=kernel_size)

    def forward(self, x):
        return self.module(x)