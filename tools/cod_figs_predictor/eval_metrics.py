#!/usr/bin/env python3
"""Fig 2A/2B: global Pred-vs-Obs + per-tissue correlation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __package__ is None:  # allow running as script
    sys.path.append(str(Path(__file__).resolve().parent))
from common import (  # type: ignore
    compute_reg_metrics,
    load_manifest,
    save_excel_sheets,
    select_organs,
    standardise_preds,
)


DEFAULT_ORGANS = [
    "Liver",
    "Muscle - Skeletal",
    "Heart - Left Ventricle",
    "Lung",
    "Brain - Cortex",
    "Kidney - Cortex",
    "Pancreas",
    "Spleen",
]


def per_tissue_corr(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for oid, g in df.groupby("organ_id"):
        if len(g) < 5:
            continue
        rho = np.corrcoef(g["y_true"], g["y_pred"])[0, 1]
        rmse = float(np.sqrt(np.mean((g["y_pred"] - g["y_true"]) ** 2)))
        rows.append({"organ_id": int(oid), "rho": float(rho), "rmse": rmse, "count": int(len(g))})
    return pd.DataFrame(rows)


def plot_pred_obs(df: pd.DataFrame, metrics: Dict[str, float], out_path: Path) -> None:
    sns.set_style("whitegrid")
    g = sns.jointplot(
        data=df.sample(min(len(df), 50000), random_state=0),
        x="y_true",
        y="y_pred",
        kind="hex",
        color="#4e79a7",
        height=6,
        marginal_kws={"bins": 40, "fill": True, "color": "#4e79a7"},
    )
    ax = g.ax_joint
    lo = float(min(df["y_true"].min(), df["y_pred"].min()))
    hi = float(max(df["y_true"].max(), df["y_pred"].max()))
    ax.plot([lo, hi], [lo, hi], ls="--", c="gray", lw=1)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.text(
        0.02,
        0.98,
        f"ρ = {metrics['rho']:.3f}\nRMSE = {metrics['rmse']:.3f}\nR² = {metrics['r2']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.tight_layout()
    g.fig.savefig(out_path, dpi=300)
    plt.close(g.fig)


def plot_per_tissue_bar(df_corr: pd.DataFrame, organ_vocab: Dict[int, str], out_path: Path) -> None:
    if df_corr.empty:
        return
    df_corr = df_corr.sort_values("rho", ascending=False)
    labels = [organ_vocab.get(int(r.organ_id), str(int(r.organ_id))) for r in df_corr.itertuples()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels[::-1], df_corr["rho"].to_list()[::-1], color="#59a14f")
    ax.set_xlabel("Pearson ρ")
    ax.set_title("Per-tissue correlation")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fig2A/2B: pred-vs-obs and per-tissue correlations")
    parser.add_argument("--preds", type=Path, default=Path("outputs/preds/test_preds.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/seq_cnn_v1_rbp_trna/manifest.json"))
    parser.add_argument("--outdir", type=Path, default=Path("reports/figures/fig2_fig3"))
    parser.add_argument("--excel", type=Path, default=Path("outputs/fig2_fig3_data/fig2A2B.xlsx"))
    parser.add_argument("--organs", nargs="*", default=DEFAULT_ORGANS, help="Preferred organ names/ids to show (<=8)")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    organ_vocab = manifest["organ_vocab"]
    df = standardise_preds(pd.read_parquet(args.preds))
    metrics = compute_reg_metrics(df)
    corr = per_tissue_corr(df)

    # select 8 organs
    sel = select_organs(organ_vocab, preferred=args.organs, top_n=8, per_org_scores={r.organ_id: r.rho for r in corr.itertuples()})
    sel_ids = {oid for oid, _ in sel}
    corr_sel = corr[corr["organ_id"].isin(sel_ids)]

    plot_pred_obs(df, metrics, args.outdir / "fig2A_pred_vs_obs.png")
    plot_per_tissue_bar(corr_sel, organ_vocab, args.outdir / "fig2B_per_tissue_corr.png")

    # Excel export
    df["organ_name"] = df["organ_id"].map(lambda x: organ_vocab.get(int(x), str(int(x))))
    corr_full = corr.copy()
    corr_full["organ_name"] = corr_full["organ_id"].map(lambda x: organ_vocab.get(int(x), str(int(x))))
    save_excel_sheets(
        {
            "preds": df[["y_true", "y_pred", "organ_id", "organ_name"]],
            "per_tissue": corr_full[["organ_id", "organ_name", "rho", "rmse", "count"]],
        },
        args.excel,
    )
    with open(args.excel.with_suffix(".metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
