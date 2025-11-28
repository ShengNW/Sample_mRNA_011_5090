#!/usr/bin/env python3
"""Fig 3B: error raincloud grouped by GC and length tertiles."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt
import torch
from torch.utils.data import DataLoader

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent))
from common import load_manifest, save_excel_sheets, standardise_preds  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
from src.side.dataset import UTRFeatureShardDataset  # type: ignore


def one_hot_to_seq(x: torch.Tensor) -> str:
    # x: (7, L); first 5 channels are A,C,G,T,N
    bases = "ACGTN"
    arr = x[:5].argmax(dim=0).cpu().numpy()
    return "".join(bases[i] for i in arr)


def gc_and_len(seq: str) -> tuple[float, int]:
    seq = seq.replace("N", "")
    if not seq:
        return 0.0, 0
    gc = sum(1 for c in seq if c in ("G", "C")) / len(seq)
    return float(gc), len(seq)


def collect_gc_len(dataset_dir: Path, split: str, max_samples: int | None = None, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    ds = UTRFeatureShardDataset(str(dataset_dir), split=split, max_cache_shards=1)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    gcs, lens = [], []
    seen = 0
    for batch in loader:
        for u5, u3 in zip(batch["utr5"], batch["utr3"]):
            seq = one_hot_to_seq(u5) + one_hot_to_seq(u3)
            gc, ln = gc_and_len(seq)
            gcs.append(gc)
            lens.append(ln)
            seen += 1
            if max_samples and seen >= max_samples:
                return np.array(gcs), np.array(lens)
    return np.array(gcs), np.array(lens)


def tertile_bins(arr: np.ndarray) -> np.ndarray:
    qs = np.quantile(arr, [0.33, 0.66])
    return np.digitize(arr, qs, right=False)


def make_raincloud(df: pd.DataFrame, group_col: str, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    pt.RainCloud(x=group_col, y="abs_error", data=df, ax=ax, palette="Set2", width_viol=.6, width_box=.2, orient="h")
    ax.set_xlabel("Absolute error")
    ax.set_ylabel(group_col.replace("_", " ").title())
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fig3B raincloud plot")
    parser.add_argument("--preds", type=Path, default=Path("outputs/preds/test_preds.parquet"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/seq_cnn_v1_rbp_trna"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--outdir", type=Path, default=Path("reports/figures/fig2_fig3"))
    parser.add_argument("--excel", type=Path, default=Path("outputs/fig2_fig3_data/fig3B_raincloud.xlsx"))
    parser.add_argument("--max-samples", type=int, default=30000, help="limit samples for GC/len computation")
    args = parser.parse_args()

    _ = load_manifest(args.dataset / "manifest.json")  # ensures manifest exists
    preds = standardise_preds(pd.read_parquet(args.preds))
    gcs, lens = collect_gc_len(args.dataset, args.split, max_samples=args.max_samples)
    if len(gcs) < len(preds):
        preds = preds.iloc[: len(gcs)].reset_index(drop=True)
    preds["gc"] = gcs[: len(preds)]
    preds["length"] = lens[: len(preds)]
    preds["abs_error"] = (preds["y_pred"] - preds["y_true"]).abs()
    preds["gc_bin"] = tertile_bins(preds["gc"])
    preds["len_bin"] = tertile_bins(preds["length"])

    make_raincloud(preds, "gc_bin", args.outdir / "fig3B_raincloud_gc.png", "Error by GC tertile")
    make_raincloud(preds, "len_bin", args.outdir / "fig3B_raincloud_len.png", "Error by length tertile")

    save_excel_sheets(
        {"raincloud_gc": preds[["y_true", "y_pred", "organ_id", "gc", "gc_bin", "abs_error"]],
         "raincloud_len": preds[["y_true", "y_pred", "organ_id", "length", "len_bin", "abs_error"]]},
        args.excel,
    )


if __name__ == "__main__":
    main()
