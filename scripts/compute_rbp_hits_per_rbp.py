#!/usr/bin/env python3
"""
Compute per-RBP eCLIP peak hits (per kb) for UTR samples.

Input:
  - Raw sample table with columns: gene_id, utr5_len, utr3_len, split (default: data/rawData/train_pair.v1.parquet)
  - Ensembl GTF for UTR coordinates (default: data/external/ref/ensembl/Homo_sapiens.GRCh38.115.gtf.gz)
  - eCLIP peak directory containing *.bed.gz (default: data/external/rbp/encode_eclip/HepG2)

Output:
  - Parquet with sample_id (global_idx), gene_id, split, organ_id, and one column per RBP: <RBP>_hits_per_kb

Example (CPU-heavy; run with nohup):
  export PATH="/root/miniconda3/bin:$PATH"
  cd /root/autodl-tmp/Sample_mRNA_011_5090-main_regpatched/Sample_mRNA_011_5090-main
  nohup python -u scripts/compute_rbp_hits_per_rbp.py \
    --split test \
    --num-workers 48 \
    --output Phase2Data/features/features_rbp_hits_per_rbp.test.parquet \
    > /tmp/log_compute_rbp_hits_per_rbp_$(date +%F).log 2>&1 &
"""

from __future__ import annotations

import argparse
import gzip
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.side.features import load_utr_coords


LOGGER = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@dataclass
class GeneIntervals:
    starts: np.ndarray
    ends: np.ndarray
    idxs: np.ndarray


def parse_rbp_name(name_field: str, path: Path) -> str:
    """
    Extract an RBP label from the BED name field.
    Example: 'GTF2F1_HepG2_IDR' -> 'GTF2F1'.
    Fallback to filename stem if parsing fails.
    """
    if name_field:
        token = name_field.split("_")[0].strip()
        if token:
            return token
    return path.stem


def list_peak_files(root: Path) -> List[Path]:
    files = sorted(root.rglob("*.bed.gz"))
    if not files:
        raise FileNotFoundError(f"No *.bed.gz files under {root}")
    return files


def build_gene_index(
    utr_coords: Dict[str, Dict[str, object]], genes: Iterable[str]
) -> Tuple[Dict[str, int], Dict[str, GeneIntervals]]:
    """Return gene->idx and per-chrom interval arrays restricted to the given gene set."""
    gene_set = set(genes)
    gene_to_idx: Dict[str, int] = {}
    chrom_data: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
    for gene in gene_set:
        info = utr_coords.get(gene)
        if not info:
            continue
        chrom = str(info.get("chr"))
        if chrom.lower().startswith("chr"):
            chrom = chrom[3:]
        start = info.get("min_start")
        end = info.get("max_end")
        if chrom is None or start is None or end is None:
            continue
        idx = len(gene_to_idx)
        gene_to_idx[gene] = idx
        chrom_data[str(chrom)].append((int(start), int(end), idx))

    chrom_intervals: Dict[str, GeneIntervals] = {}
    for chrom, rows in chrom_data.items():
        rows.sort(key=lambda x: x[0])
        starts = np.fromiter((r[0] for r in rows), dtype=np.int32)
        ends = np.fromiter((r[1] for r in rows), dtype=np.int32)
        idxs = np.fromiter((r[2] for r in rows), dtype=np.int32)
        chrom_intervals[chrom] = GeneIntervals(starts=starts, ends=ends, idxs=idxs)
    return gene_to_idx, chrom_intervals


def process_peak_file(
    path: Path, chrom_intervals: Dict[str, GeneIntervals], gene_dim: int
) -> Tuple[str, np.ndarray]:
    """
    Count overlaps between peaks in `path` and gene intervals.
    Returns (rbp_name, counts_per_gene_idx).
    """
    counts = np.zeros(gene_dim, dtype=np.int32)
    rbp_name = None
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split()
            if len(parts) < 4:
                continue
            chrom, start_s, end_s, name_field = parts[0], parts[1], parts[2], parts[3]
            if chrom.lower().startswith("chr"):
                chrom = chrom[3:]
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            if rbp_name is None:
                rbp_name = parse_rbp_name(name_field, path)
            intervals = chrom_intervals.get(chrom)
            if intervals is None:
                continue
            # genes whose start <= peak_end
            idx_end = np.searchsorted(intervals.starts, end, side="right")
            if idx_end == 0:
                continue
            candidate_ends = intervals.ends[:idx_end]
            overlap_mask = candidate_ends >= start
            if not overlap_mask.any():
                continue
            hit_idxs = intervals.idxs[:idx_end][overlap_mask]
            np.add.at(counts, hit_idxs, 1)
    if rbp_name is None:
        rbp_name = parse_rbp_name("", path)
    return rbp_name, counts


def compute_counts(
    peak_files: List[Path], chrom_intervals: Dict[str, GeneIntervals], num_workers: int, gene_dim: int
) -> Tuple[List[str], np.ndarray]:
    """
    Compute counts matrix: shape (num_rbp, num_gene).
    """
    from multiprocessing import Pool
    from functools import partial

    results: List[Tuple[str, np.ndarray]] = []
    worker = partial(process_peak_file, chrom_intervals=chrom_intervals, gene_dim=gene_dim)
    with Pool(processes=num_workers) as pool:
        for rbp_name, counts in pool.imap_unordered(worker, peak_files):
            results.append((rbp_name, counts))

    # Align to common gene dimension (max idx + 1)
    gene_dim = max(gene_dim, max((c.shape[0] for _, c in results), default=0))
    rbp_names = []
    counts_matrix = []
    for name, cnt in results:
        rbp_names.append(name)
        if cnt.shape[0] < gene_dim:
            padded = np.zeros(gene_dim, dtype=cnt.dtype)
            padded[: cnt.shape[0]] = cnt
            counts_matrix.append(padded)
        else:
            counts_matrix.append(cnt)
    if counts_matrix:
        counts_arr = np.vstack(counts_matrix)
    else:
        counts_arr = np.zeros((0, gene_dim), dtype=np.int32)
    return rbp_names, counts_arr


def build_feature_table(
    sample_df: pd.DataFrame,
    gene_to_idx: Dict[str, int],
    rbp_names: List[str],
    counts: np.ndarray,
) -> pd.DataFrame:
    """
    Build per-sample feature table with hits_per_kb columns.
    """
    sample_gene_idx = sample_df["gene_id"].map(gene_to_idx).fillna(-1).to_numpy(dtype=np.int64)
    lengths_kb = (sample_df["utr5_len"].to_numpy() + sample_df["utr3_len"].to_numpy()) / 1000.0
    lengths_kb[lengths_kb == 0] = np.nan  # avoid divide-by-zero

    data = {}
    for i, rbp in enumerate(rbp_names):
        hits = np.zeros_like(lengths_kb, dtype=np.float32)
        valid_mask = sample_gene_idx >= 0
        hits[valid_mask] = counts[i, sample_gene_idx[valid_mask]]
        data[f"{rbp}_hits_per_kb"] = hits / lengths_kb
    features = pd.DataFrame(data)
    features.insert(0, "sample_id", sample_df["sample_id"].values)
    features.insert(1, "gene_id", sample_df["gene_id"].values)
    features.insert(2, "split", sample_df["split"].values)
    features.insert(3, "organ_id", sample_df["organ_id"].values)
    return features


def main():
    parser = argparse.ArgumentParser(description="Compute per-RBP eCLIP hit counts per kb.")
    parser.add_argument("--raw", type=Path, default=Path("data/rawData/train_pair.v1.parquet"))
    parser.add_argument(
        "--gtf", type=Path, default=Path("data/external/ref/ensembl/Homo_sapiens.GRCh38.115.gtf.gz")
    )
    parser.add_argument("--peak-dir", type=Path, default=Path("data/external/rbp/encode_eclip/HepG2"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=Path("Phase2Data/features/features_rbp_hits_per_rbp.parquet"))
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)

    LOGGER.info("Loading raw samples from %s", args.raw)
    cols = ["gene_id", "utr5_len", "utr3_len", "split", "organ_id"]
    raw_df = pd.read_parquet(args.raw, columns=cols)
    raw_df = raw_df.reset_index().rename(columns={"index": "sample_id"})
    sample_df = raw_df[raw_df["split"].str.lower() == args.split.lower()].copy()
    LOGGER.info("Split '%s' samples: %d", args.split, len(sample_df))

    genes = sample_df["gene_id"].unique().tolist()
    LOGGER.info("Unique genes in split: %d", len(genes))

    LOGGER.info("Loading UTR coordinates from %s", args.gtf)
    utr_coords = load_utr_coords(args.gtf)
    gene_to_idx, chrom_intervals = build_gene_index(utr_coords, genes)
    if not gene_to_idx:
        raise RuntimeError("No gene intervals matched; check GTF and gene IDs.")
    gene_dim = len(gene_to_idx)
    LOGGER.info("Prepared intervals for %d genes across %d chromosomes", gene_dim, len(chrom_intervals))

    peak_files = list_peak_files(args.peak_dir)

    LOGGER.info("Processing %d peak files with %d workers", len(peak_files), args.num_workers)
    rbp_names, counts = compute_counts(peak_files, chrom_intervals, args.num_workers, gene_dim)
    LOGGER.info("Computed counts for %d RBPs; counts shape=%s", len(rbp_names), counts.shape)

    LOGGER.info("Building per-sample features")
    features = build_feature_table(sample_df, gene_to_idx, rbp_names, counts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    LOGGER.info("Saved features to %s (shape=%s, RBPs=%d)", args.output, features.shape, len(rbp_names))
    LOGGER.info("Example feature columns: %s", list(features.columns[: min(8, len(features.columns))]))


if __name__ == "__main__":
    main()
