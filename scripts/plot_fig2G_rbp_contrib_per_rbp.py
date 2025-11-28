#!/usr/bin/env python3
"""
Plot Fig2G (per-RBP Top-10 contribution) using per-RBP features.

Required input: eval_with_rbp_hits_per_rbp.*.parquet produced by
merge_rbp_per_rbp_features_into_eval.py (contains y_true, y_pred, organ_id/name, *_hits_per_kb).

Example:
  export PATH="/root/miniconda3/bin:$PATH"
  cd /root/autodl-tmp/Sample_mRNA_011_5090-main_regpatched/Sample_mRNA_011_5090-main
  python -u scripts/plot_fig2G_rbp_contrib_per_rbp.py \
    --eval Phase2Data/eval/eval_with_rbp_hits_per_rbp.test.parquet \
    --tissue Liver \
    --output reports/figures/fig2_fig3/fig2G_rbp_contrib_per_rbp.liver.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor


LOGGER = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def select_rbp_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.endswith("_hits_per_kb")]


def compute_spearman(df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in feat_cols:
        rho, _ = spearmanr(df[c], df["y_pred"], nan_policy="omit")
        rows.append({"rbp": c.replace("_hits_per_kb", ""), "spearman_r": rho})
    res = pd.DataFrame(rows)
    res["abs_rho"] = res["spearman_r"].abs()
    return res.sort_values("abs_rho", ascending=False)


def compute_shap_importance(df: pd.DataFrame, feat_cols: List[str], sample_size: int = 5000) -> pd.DataFrame:
    X = df[feat_cols].fillna(0.0)
    y = df["y_pred"].to_numpy()
    if len(X) > sample_size:
        X = X.sample(n=sample_size, random_state=42)
        y = y[X.index]
    model = RandomForestRegressor(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1, min_samples_leaf=2
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame({"rbp": [c.replace("_hits_per_kb", "") for c in feat_cols], "shap_importance": mean_abs})


def plot_top10(importances: pd.DataFrame, metric_col: str, title: str, output: Path) -> None:
    top = importances.nlargest(10, metric_col).iloc[::-1]  # reverse for barh order
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["rbp"], top[metric_col], color="#1f77b4")
    for i, val in enumerate(top[metric_col]):
        ax.text(val, i, f"{val:.3f}", va="center", ha="left", fontsize=8)
    ax.set_xlabel(metric_col.replace("_", " "))
    ax.set_ylabel("RBP")
    ax.set_title(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fig2G per-RBP contribution plot (Top-10).")
    parser.add_argument("--eval", type=Path, default=Path("Phase2Data/eval/eval_with_rbp_hits_per_rbp.test.parquet"))
    parser.add_argument("--tissue", default="Liver", help="Organ/tissue name string to filter on.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/fig2_fig3/fig2G_rbp_contrib_per_rbp.liver.png"),
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)

    LOGGER.info("Loading eval table from %s", args.eval)
    df = pd.read_parquet(args.eval)
    tissue_mask = df["organ_name"].str.lower() == args.tissue.lower()
    df_t = df[tissue_mask].copy()
    LOGGER.info("Filtered tissue='%s': %d rows", args.tissue, len(df_t))
    if len(df_t) == 0:
        raise ValueError(f"No rows found for tissue={args.tissue}")

    feat_cols = select_rbp_cols(df_t)
    if not feat_cols:
        raise ValueError("No per-RBP feature columns (*_hits_per_kb) found.")

    LOGGER.info("Computing Spearman correlations for %d features", len(feat_cols))
    spearman_df = compute_spearman(df_t, feat_cols)

    LOGGER.info("Computing SHAP importances (RandomForest)")
    shap_df = compute_shap_importance(df_t, feat_cols)

    merged = shap_df.merge(spearman_df, on="rbp", how="left")
    LOGGER.info("Top-10 RBPs by SHAP:\n%s", merged.nlargest(10, "shap_importance")[["rbp", "shap_importance"]])

    title = f"Top 10 RNA-binding proteins associated with {args.tissue}"
    plot_top10(merged, "shap_importance", title, args.output)

    # Save metrics alongside?
    metrics_out = args.output.with_suffix(".metrics.csv")
    merged.sort_values("shap_importance", ascending=False).to_csv(metrics_out, index=False)
    LOGGER.info("Saved plot to %s and metrics to %s", args.output, metrics_out)


if __name__ == "__main__":
    main()
