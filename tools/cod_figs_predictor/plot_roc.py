#!/usr/bin/env python3
"""Fig 2C: organ-specific ROC curves using regression score as surrogate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent))
from common import load_manifest, save_excel_sheets, select_organs, standardise_preds  # type: ignore

DEFAULT_ORGANS = ["Liver", "Muscle - Skeletal", "Heart - Left Ventricle", "Lung"]


def main():
    parser = argparse.ArgumentParser(description="Fig2C ROC (per selected organs)")
    parser.add_argument("--preds", type=Path, default=Path("outputs/preds/test_preds.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/seq_cnn_v1_rbp_trna/manifest.json"))
    parser.add_argument("--outdir", type=Path, default=Path("reports/figures/fig2_fig3"))
    parser.add_argument("--excel", type=Path, default=Path("outputs/fig2_fig3_data/fig2C_roc.xlsx"))
    parser.add_argument("--organs", nargs="*", default=DEFAULT_ORGANS, help="Organ names/ids to plot (3-4)")
    args = parser.parse_args()

    vocab = load_manifest(args.manifest)["organ_vocab"]
    df = standardise_preds(pd.read_parquet(args.preds))
    sel = select_organs(vocab, preferred=args.organs, top_n=4)

    fig, ax = plt.subplots(figsize=(6, 5))
    records = {}
    for oid, name in sel:
        y_true = (df["organ_id"] == oid).astype(int).to_numpy()
        scores = df["y_pred"].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
        records[name] = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Organ specificity ROC (surrogate)")
    ax.legend()
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.outdir / "fig2C_roc.png", dpi=300)
    plt.close(fig)

    save_excel_sheets(records, args.excel)


if __name__ == "__main__":
    main()
