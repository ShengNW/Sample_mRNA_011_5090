#!/usr/bin/env python3
"""Fig 3D: distribution shift placeholder (pred vs ground truth)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent))
from common import save_excel_sheets, standardise_preds  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Fig3D distribution shift (placeholder real vs real)")
    parser.add_argument("--preds", type=Path, default=Path("outputs/preds/test_preds.parquet"))
    parser.add_argument("--outdir", type=Path, default=Path("reports/figures/fig2_fig3"))
    parser.add_argument("--excel", type=Path, default=Path("outputs/fig2_fig3_data/fig3D_shift.xlsx"))
    args = parser.parse_args()

    df = standardise_preds(pd.read_parquet(args.preds))
    plot_df = pd.concat(
        [
            pd.DataFrame({"value": df["y_true"], "type": "Observed"}),
            pd.DataFrame({"value": df["y_pred"], "type": "Predicted"}),
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(data=plot_df, x="value", hue="type", fill=True, common_norm=False, alpha=0.5, ax=ax)
    ax.set_xlabel("Expression score")
    ax.set_title("Distribution shift (placeholder)")
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.outdir / "fig3D_distribution_shift.png", dpi=300)
    plt.close(fig)

    save_excel_sheets({"shift": plot_df}, args.excel)


if __name__ == "__main__":
    main()
