#!/usr/bin/env python3
"""
Merge per-RBP features into prediction table for Fig2 evaluations.

Inputs:
  --preds: predictions parquet (y_true, y_pred, organ_id) from inference
  --index: dataset split index containing global_idx (sample_id) for alignment
  --features: per-RBP hits_per_kb file produced by compute_rbp_hits_per_rbp.py
  --manifest: dataset manifest (for organ_vocab lookup)

Output:
  Parquet with predictions + organ_name + per-RBP columns.

Example:
  export PATH="/root/miniconda3/bin:$PATH"
  cd /root/autodl-tmp/Sample_mRNA_011_5090-main_regpatched/Sample_mRNA_011_5090-main
  python -u scripts/merge_rbp_per_rbp_features_into_eval.py \
    --preds outputs/preds/test_preds.parquet \
    --index data/processed/seq_cnn_v1_rbp_trna/index/test/index.parquet \
    --features Phase2Data/features/features_rbp_hits_per_rbp.test.parquet \
    --manifest data/processed/seq_cnn_v1_rbp_trna/manifest.json \
    --output Phase2Data/eval/eval_with_rbp_hits_per_rbp.test.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_manifest(path: Path) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def main():
    parser = argparse.ArgumentParser(description="Merge per-RBP features into prediction table.")
    parser.add_argument("--preds", type=Path, default=Path("outputs/preds/test_preds.parquet"))
    parser.add_argument(
        "--index", type=Path, default=Path("data/processed/seq_cnn_v1_rbp_trna/index/test/index.parquet")
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("Phase2Data/features/features_rbp_hits_per_rbp.test.parquet"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/seq_cnn_v1_rbp_trna/manifest.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("Phase2Data/eval/eval_with_rbp_hits_per_rbp.test.parquet")
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)

    LOGGER.info("Loading predictions from %s", args.preds)
    preds = pd.read_parquet(args.preds)

    LOGGER.info("Loading split index from %s", args.index)
    idx_df = pd.read_parquet(args.index)
    if "global_idx" not in idx_df.columns:
        raise ValueError(f"Index file {args.index} lacks 'global_idx' column.")
    sample_ids = idx_df["global_idx"].to_numpy()
    if len(sample_ids) != len(preds):
        LOGGER.warning(
            "Preds rows (%d) != index rows (%d); will align by position and truncate to min length.",
            len(preds),
            len(sample_ids),
        )
        n = min(len(preds), len(sample_ids))
        preds = preds.iloc[:n].copy()
        sample_ids = sample_ids[:n]
    preds.insert(0, "sample_id", sample_ids)
    if "organ_id" not in preds.columns and "organ_id" in idx_df.columns:
        preds["organ_id"] = idx_df["organ_id"].iloc[: len(preds)].to_numpy()

    LOGGER.info("Loading per-RBP features from %s", args.features)
    feats = pd.read_parquet(args.features)

    LOGGER.info("Merging predictions with features on sample_id")
    merged = preds.merge(feats, on="sample_id", how="left", suffixes=("", "_feat"))
    matched = merged["sample_id"].isin(feats["sample_id"]).sum()
    LOGGER.info("Matched %d / %d rows with RBP features", matched, len(merged))

    manifest = load_manifest(args.manifest)
    organ_vocab = {int(k): v for k, v in manifest.get("organ_vocab", {}).items()}
    if "organ_name" not in merged.columns and "organ_id" in merged.columns:
        merged["organ_name"] = merged["organ_id"].map(organ_vocab)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    LOGGER.info("Saved merged eval to %s (shape=%s)", args.output, merged.shape)
    rbp_cols = [c for c in merged.columns if c.endswith("_hits_per_kb")]
    LOGGER.info("RBP feature columns: %d; example: %s", len(rbp_cols), rbp_cols[:5])


if __name__ == "__main__":
    main()
