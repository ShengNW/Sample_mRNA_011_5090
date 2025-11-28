"""Shared helpers for Fig2/3 predictor-side plots.

Each script keeps I/O simple: read manifest for organ vocab, standardise
prediction tables, and expose metrics helpers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_manifest(manifest_path: Path) -> Dict:
    path = manifest_path if manifest_path.is_absolute() else repo_root() / manifest_path
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    vocab = data.get("organ_vocab", {})
    data["organ_vocab"] = {int(k): v for k, v in vocab.items()}
    return data


def standardise_preds(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for key in ("y_true", "target", "label", "truth"):
        if key in cols:
            rename[cols[key]] = "y_true"
            break
    for key in ("y_pred", "prediction", "pred"):
        if key in cols:
            rename[cols[key]] = "y_pred"
            break
    if "organ_id" in df.columns:
        rename["organ_id"] = "organ_id"
    out = df.rename(columns=rename)
    missing = [c for c in ("y_true", "y_pred") if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "organ_id" not in out.columns:
        out["organ_id"] = -1
    return out


def compute_reg_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    resid = y_pred - y_true
    mse = float(np.mean(resid ** 2))
    r2 = float(1.0 - mse / (float(np.var(y_true)) + 1e-12))
    rho = float(np.corrcoef(y_true, y_pred)[0, 1])
    rmse = float(np.sqrt(mse))
    return {"r2": r2, "rho": rho, "rmse": rmse}


def select_organs(
    vocab: Mapping[int, str],
    preferred: Sequence[str] | None = None,
    top_n: int = 8,
    per_org_scores: Mapping[int, float] | None = None,
) -> list[Tuple[int, str]]:
    """Resolve organ IDs to plot."""
    name2id = {v.lower(): k for k, v in vocab.items()}
    resolved: list[int] = []
    if preferred:
        for item in preferred:
            if isinstance(item, str) and item.strip().lower() in name2id:
                resolved.append(name2id[item.strip().lower()])
            else:
                try:
                    resolved.append(int(item))
                except Exception:
                    continue
    else:
        ids = list(vocab.keys())
        if per_org_scores:
            ids = sorted(ids, key=lambda x: per_org_scores.get(x, -np.inf), reverse=True)
        resolved = ids[:top_n]
    seen = set()
    uniq = []
    for oid in resolved:
        if oid in seen:
            continue
        seen.add(oid)
        uniq.append((oid, vocab.get(int(oid), str(oid))))
    return uniq


def save_excel_sheets(tables: Mapping[str, pd.DataFrame], out_path: Path) -> None:
    out_path = out_path if out_path.is_absolute() else repo_root() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        for sheet, df in tables.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)


__all__ = [
    "load_manifest",
    "standardise_preds",
    "compute_reg_metrics",
    "select_organs",
    "save_excel_sheets",
    "repo_root",
]
