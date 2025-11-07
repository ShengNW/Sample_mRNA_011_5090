
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np

def r2_per_dim(y_true: np.ndarray, y_pred: np.ndarray):
    """逐维 R²。"""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape
    num = ((y_true - y_pred)**2).sum(axis=0)
    den = ((y_true - y_true.mean(axis=0))**2).sum(axis=0) + 1e-12
    r2 = 1.0 - num/den
    return r2

def spearman_per_dim(y_true: np.ndarray, y_pred: np.ndarray):
    """逐维 Spearman（无 scipy 依赖的简化实现）。"""
    def rankdata(a):
        temp = a.argsort()
        ranks = np.empty_like(temp, dtype=np.float64)
        ranks[temp] = np.arange(len(a), dtype=np.float64)
        return ranks
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    S = []
    for j in range(y_true.shape[1]):
        rt = rankdata(y_true[:,j])
        rp = rankdata(y_pred[:,j])
        rt = (rt - rt.mean()) / (rt.std()+1e-12)
        rp = (rp - rp.mean()) / (rp.std()+1e-12)
        S.append(np.clip((rt*rp).mean(), -1.0, 1.0))
    return np.array(S, dtype=np.float32)
