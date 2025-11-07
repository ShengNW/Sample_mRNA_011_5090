
# -*- coding: utf-8 -*-
"""
FeatureStore: 统一加载三类“侧表”特征并按键回填。
- organ_id → tissue_vec / rbp_vec
- transcript_id → x_struct
文件格式尽量宽容：支持 .npy、.json、.csv（第一列为键，其余列为向量）。
"""
from __future__ import annotations
import os, json, csv
from typing import Dict, Tuple, Optional, Any
import numpy as np

def _load_kv_matrix(path: str) -> Tuple[Dict[Any, int], np.ndarray]:
    """
    返回 (key_to_row_index, matrix)；matrix shape=(N, D)
    支持：
      - .npy   : dict(keys=list, data=np.ndarray NxD)
      - .json  : {"keys": [...], "data": [[...],[...],...]}
      - .csv   : 第一列为键，其余列为浮点
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"feature file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        obj = np.load(path, allow_pickle=True).item()
        keys = obj["keys"]; data = np.asarray(obj["data"], dtype=np.float32)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        keys = obj["keys"]; data = np.asarray(obj["data"], dtype=np.float32)
    elif ext == ".csv":
        keys = []
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                keys.append(row[0])
                rows.append([float(x) for x in row[1:]])
        data = np.asarray(rows, dtype=np.float32)
    else:
        raise ValueError(f"unsupported feature file ext: {ext}")
    key2row = {k:i for i,k in enumerate(keys)}
    return key2row, data

class FeatureStore:
    def __init__(self, 
                 tissue_path: Optional[str]=None,
                 rbp_path: Optional[str]=None,
                 struct_path: Optional[str]=None,
                 organ_id_is_int: bool=True):
        self.tissue_key2row, self.tissue = ({}, None)
        self.rbp_key2row, self.rbp = ({}, None)
        self.struct_key2row, self.struct = ({}, None)
        self.organ_id_is_int = organ_id_is_int
        if tissue_path:
            self.tissue_key2row, self.tissue = _load_kv_matrix(tissue_path)
        if rbp_path:
            self.rbp_key2row, self.rbp = _load_kv_matrix(rbp_path)
        if struct_path:
            self.struct_key2row, self.struct = _load_kv_matrix(struct_path)

    def get_tissue(self, organ_id) -> Optional[np.ndarray]:
        if self.tissue is None: return None
        key = int(organ_id) if self.organ_id_is_int else str(organ_id)
        ridx = self.tissue_key2row.get(key, None)
        if ridx is None: return None
        return self.tissue[ridx]

    def get_rbp(self, organ_id) -> Optional[np.ndarray]:
        if self.rbp is None: return None
        key = int(organ_id) if self.organ_id_is_int else str(organ_id)
        ridx = self.rbp_key2row.get(key, None)
        if ridx is None: return None
        return self.rbp[ridx]

    def get_struct(self, transcript_id) -> Optional[np.ndarray]:
        if self.struct is None: return None
        key = str(transcript_id)
        ridx = self.struct_key2row.get(key, None)
        if ridx is None: return None
        return self.struct[ridx]

    @property
    def dims(self):
        t = self.tissue.shape[1] if isinstance(self.tissue, np.ndarray) else 0
        r = self.rbp.shape[1] if isinstance(self.rbp, np.ndarray) else 0
        s = self.struct.shape[1] if isinstance(self.struct, np.ndarray) else 0
        return dict(tissue=t, rbp=r, struct=s)
