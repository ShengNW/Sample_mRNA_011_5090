
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn

class ConcatSideHead(nn.Module):
    """
    将 trunk 输出与若干侧表特征拼接，再经 MLP 预测多任务回归 (D_out)。
    输入：
      - trunk_dim: 序列 CNN 主干输出维度
      - dims: dict(tissue=?, rbp=?, struct=?)
      - out_dim: 标签维度
    """
    def __init__(self, trunk_dim: int, dims: dict, out_dim: int, hidden: int=256, drop: float=0.2):
        super().__init__()
        side_dim = sum([d for d in dims.values() if isinstance(d, int) and d>0])
        self.in_dim = trunk_dim + side_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, trunk_feat, side: dict|None=None):
        if side is None or len(side)==0:
            x = trunk_feat
        else:
            parts = [trunk_feat]
            for k in ('tissue','rbp','struct'):
                if (side is not None) and (k in side) and (side[k] is not None):
                    parts.append(side[k])
            x = torch.cat(parts, dim=-1)
        return self.mlp(x)
