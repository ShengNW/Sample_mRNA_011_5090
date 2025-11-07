
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Optional
import torch
from torch.utils.data import Dataset
import os, json

from .dataset import ShardedUTRDataset
from ..features.store import FeatureStore

class AugmentedUTRDataset(Dataset):
    """
    基于现有 ShardedUTRDataset 的“零拷贝”包装：仅在 __getitem__ 增补 side 特征字段。
    需求：
      - organ_id: 若底层分片或 index 表含 organ 列，则用于 tissue/rbp 查表。
      - transcript_id: 若 index 表含此列，则用于结构特征查表。
    若缺失键，返回的相应特征为 None（由 collate_fn 统一补零）。
    """
    def __init__(self, base: ShardedUTRDataset, features: FeatureStore, index_json: Optional[str]=None):
        self.base = base
        self.features = features
        self.index_rows = None
        if index_json and os.path.exists(index_json):
            with open(index_json, "r", encoding="utf-8") as f:
                self.index_rows = json.load(f)  # 每行可含 {'global_idx', 'part_id','local_idx','organ_id','transcript_id'}

    def __len__(self): return len(self.base)

    def _lookup_keys(self, global_idx: int):
        organ_id = None; transcript_id = None
        if self.index_rows:
            # 假设存在 global_idx 序：可直接按序号取
            if global_idx < len(self.index_rows):
                row = self.index_rows[global_idx]
                organ_id = row.get('organ_id', None)
                transcript_id = row.get('transcript_id', None)
        return organ_id, transcript_id

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base[idx]  # {'utr5','utr3','label'}
        organ_id, transcript_id = self._lookup_keys(idx)
        tissue = self.features.get_tissue(organ_id) if organ_id is not None else None
        rbp    = self.features.get_rbp(organ_id) if organ_id is not None else None
        xstruct= self.features.get_struct(transcript_id) if transcript_id is not None else None
        item.update({'organ_id': organ_id, 'transcript_id': transcript_id,
                     'tissue_vec': tissue, 'rbp_vec': rbp, 'x_struct': xstruct})
        return item

def collate_with_side(batch):
    """将 None 特征补零，打包为张量。"""
    import numpy as np
    import torch
    # 基础
    x5 = torch.stack([b['utr5'] for b in batch], 0)
    x3 = torch.stack([b['utr3'] for b in batch], 0)
    y  = torch.stack([b['label'] for b in batch], 0)
    # side
    def to_tensor_list(key, dim=0):
        arrs = []
        cur_dim = None
        for b in batch:
            v = b.get(key, None)
            if v is None:
                if cur_dim is None:
                    # 推迟到遇到第一个非 None 决定维度
                    arrs.append(None)
                else:
                    arrs.append(np.zeros(cur_dim, dtype=np.float32))
            else:
                vv = np.asarray(v, dtype=np.float32)
                cur_dim = vv.shape[0]
                arrs.append(vv)
        # 第二遍补零
        if cur_dim is None:
            return None
        filled = [ (a if a is not None else np.zeros(cur_dim, dtype=np.float32)) for a in arrs ]
        return torch.from_numpy(np.stack(filled, 0))

    tissue = to_tensor_list('tissue_vec')
    rbp    = to_tensor_list('rbp_vec')
    xstruct= to_tensor_list('x_struct')
    side = {}
    if tissue is not None: side['tissue'] = tissue
    if rbp    is not None: side['rbp']    = rbp
    if xstruct is not None: side['struct']= xstruct

    return {'utr5': x5, 'utr3': x3, 'label': y, 'side': side}
