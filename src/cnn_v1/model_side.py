
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
from .model import DualBranchCNN
from .heads.concat_side import ConcatSideHead

class DualBranchCNNWithSide(nn.Module):
    """
    复用既有 DualBranchCNN 的两支主干，替换 head 为 ConcatSideHead。
    """
    def __init__(self, base: DualBranchCNN, side_dims: dict, out_dim: int, hidden: int=256):
        super().__init__()
        self.branch5 = base.branch5
        self.branch3 = base.branch3
        # 推断 trunk 输出维度
        trunk_dim = self.branch5.out_channels + self.branch3.out_channels
        self.head = ConcatSideHead(trunk_dim, side_dims, out_dim, hidden=hidden)

    def forward(self, utr5, utr3, side: dict|None=None):
        f5 = self.branch5(utr5)
        f3 = self.branch3(utr3)
        fused = torch.cat([f5, f3], dim=-1)
        return self.head(fused, side=side)
