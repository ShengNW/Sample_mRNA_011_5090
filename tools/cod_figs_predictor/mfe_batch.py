#!/usr/bin/env python3
"""Fig 2H: MFE vs predicted expression (reads precomputed MFE CSV)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent))
from common import load_manifest, save_excel_sheets, standardise_preds  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
from src.side.dataset import UTRFeatureShardDataset  # type: ignore


def plot_mfe(df: pd.DataFrame, mfe_col: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.kdeplot(data=df, x=mfe_col, y="y_pred", fill=True, cmap="mako", ax=ax, thresh=0.05)
    ax.set_xlabel("Predicted 3'UTR minimum free energy (kcal/mol)")
    ax.set_ylabel("Predicted expression")
    ax.set_title("Structure–expression relationship (3'UTR MFE vs predicted expression)")
    rho, _ = spearmanr(df[mfe_col], df["y_pred"], nan_policy="omit")
    ax.text(
        0.02,
        0.98,
        f"ρ = {rho:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_seq_id_map(dataset_dir: Path, split: str, organ_vocab: dict) -> pd.DataFrame:
    """Return DataFrame with global_idx, seq_id, organ_id, transcript_id for the split."""
    # sequential order aligned with preds
    idx = pd.read_parquet(dataset_dir / "index" / split / "index.parquet")
    raw = pd.read_json(REPO_ROOT / "data" / "rawData" / "index" / f"{split}.json")
    raw_df = pd.DataFrame(raw)
    if "global_idx" not in raw_df.columns:
        raise RuntimeError("Fig2H: raw index missing 'global_idx'")
    raw_df = raw_df[["global_idx", "organ_id", "transcript_id"]]
    merged = idx.merge(raw_df, on="global_idx", how="left", suffixes=("", "_name"))
    organ_col = "organ_id_x" if "organ_id_x" in merged.columns else "organ_id"
    merged["organ_name"] = merged[organ_col].map(organ_vocab)
    merged["seq_id"] = merged["transcript_id"].astype(str) + "__" + merged["organ_name"].astype(str)
    return merged


def merge_mfe(preds: pd.DataFrame, dataset_dir: Path, split: str, mfe_path: Path, organ_vocab: dict) -> tuple[pd.DataFrame, str]:
    mfe_df = pd.read_csv(mfe_path)
    mfe_cols = [c for c in mfe_df.columns if c.lower().startswith("mfe")]
    if not mfe_cols:
        raise RuntimeError("Fig2H: no column starting with 'mfe' found in mfe_predict.csv")
    mfe_col = "mfe_utr3" if "mfe_utr3" in mfe_df.columns else mfe_cols[0]

    seq_map = build_seq_id_map(dataset_dir, split, organ_vocab)
    if len(seq_map) < len(preds):
        seq_map = seq_map.iloc[: len(preds)]
    preds = preds.copy()
    preds["global_idx"] = seq_map["global_idx"].values[: len(preds)]
    preds["seq_id"] = seq_map["seq_id"].values[: len(preds)]

    merged = preds.merge(mfe_df[["seq_id", mfe_col]], on="seq_id", how="left")
    if merged[mfe_col].isna().all() or merged[mfe_col].nunique(dropna=True) <= 1:
        raise RuntimeError("Fig2H: MFE column missing or constant after merge with preds")
    if len(merged) != len(preds):
        print(f"[Fig2H] Warning: merged rows {len(merged)} vs preds {len(preds)}")
    return merged, mfe_col


def main():
    parser = argparse.ArgumentParser(description="Fig2H MFE vs expression")
    parser.add_argument("--preds", type=Path, default=Path("outputs/preds/test_preds.parquet"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/seq_cnn_v1_rbp_trna"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--outdir", type=Path, default=Path("reports/figures/fig2_fig3"))
    parser.add_argument("--excel", type=Path, default=Path("outputs/fig2_fig3_data/fig2H_mfe.xlsx"))
    parser.add_argument(
        "--mfe-csv",
        type=Path,
        default=Path("/root/autodl-tmp/Sample_mRNA_011_5090-main_regpatched/Sample_mRNA_011_5090-main/Phase2Data/mfe_predict.csv"),
    )
    parser.add_argument("--max-samples", type=int, default=20000, help="subset for plotting")
    args = parser.parse_args()

    manifest = load_manifest(args.dataset / "manifest.json")
    preds = standardise_preds(pd.read_parquet(args.preds)).iloc[: args.max_samples].copy()
    merged, mfe_col = merge_mfe(preds, args.dataset, args.split, args.mfe_csv, manifest["organ_vocab"])

    plot_mfe(merged, mfe_col, args.outdir / "fig2H_mfe_density.png")
    save_excel_sheets(
        {"mfe_vs_pred": merged[["y_true", "y_pred", "organ_id", "seq_id", mfe_col]].rename(columns={mfe_col: "mfe"})},
        args.excel,
    )


if __name__ == "__main__":
    main()
