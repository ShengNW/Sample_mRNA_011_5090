#!/usr/bin/env python3
"""Fig 2G fallback: correlate aggregated RBP-hit fraction with predictions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent))
from common import load_manifest, save_excel_sheets, select_organs, standardise_preds  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
from src.side.dataset import UTRFeatureShardDataset  # type: ignore


def rbp_fraction(utr5: torch.Tensor, utr3: torch.Tensor) -> float:
    # Channel 5 (0-based) is RBP hit mask per dataset builder (A,C,G,T,N,RBP,tRNA)
    mask = torch.cat([utr5[5], utr3[5]], dim=0)
    return float(mask.mean().item())


def collect_rbp_frac(dataset_dir: Path, split: str, max_samples: int | None = None, batch_size: int = 256) -> np.ndarray:
    ds = UTRFeatureShardDataset(str(dataset_dir), split=split, max_cache_shards=1)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    fracs = []
    seen = 0
    for batch in loader:
        for u5, u3 in zip(batch["utr5"], batch["utr3"]):
            fracs.append(rbp_fraction(u5, u3))
            seen += 1
            if max_samples and seen >= max_samples:
                return np.array(fracs)
    return np.array(fracs)


def per_organ_corr(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for oid, g in df.groupby("organ_id"):
        if len(g) < 10:
            continue
        rho = np.corrcoef(g["rbp_frac"], g["y_pred"])[0, 1]
        rows.append({"organ_id": int(oid), "rho": float(rho), "count": len(g)})
    return pd.DataFrame(rows).sort_values("rho", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Fig2G RBP contribution (aggregated RBP-hit fraction)")
    parser.add_argument("--preds", type=Path, default=Path("outputs/preds/test_preds.parquet"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/seq_cnn_v1_rbp_trna"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--outdir", type=Path, default=Path("reports/figures/fig2_fig3"))
    parser.add_argument("--excel", type=Path, default=Path("outputs/fig2_fig3_data/fig2G_rbp.xlsx"))
    parser.add_argument("--max-samples", type=int, default=40000)
    args = parser.parse_args()

    manifest = load_manifest(args.dataset / "manifest.json")
    vocab = manifest["organ_vocab"]
    preds = standardise_preds(pd.read_parquet(args.preds)).iloc[: args.max_samples].copy()
    fracs = collect_rbp_frac(args.dataset, args.split, max_samples=args.max_samples)
    preds["rbp_frac"] = fracs[: len(preds)]

    corr = per_organ_corr(preds)
    corr["organ_name"] = corr["organ_id"].map(lambda x: vocab.get(int(x), str(int(x))))
    top = corr.head(10)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["organ_name"][::-1], top["rho"][::-1], color="#e15759")
    ax.set_xlabel("Correlation (RBP-hit fraction vs prediction)")
    ax.set_title("RBP channel contribution (fallback)")
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.outdir / "fig2G_rbp_contrib.png", dpi=300)
    plt.close(fig)

    preds["organ_name"] = preds["organ_id"].map(lambda x: vocab.get(int(x), str(int(x))))
    save_excel_sheets(
        {
            "sample_level": preds[["y_true", "y_pred", "organ_id", "organ_name", "rbp_frac"]],
            "per_organ_corr": corr[["organ_id", "organ_name", "rho", "count"]],
        },
        args.excel,
    )


if __name__ == "__main__":
    main()
