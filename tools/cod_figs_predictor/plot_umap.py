#!/usr/bin/env python3
"""Fig 2D: UMAP of encoder embeddings (FiLM CNN)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import umap

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent))
from common import load_manifest, save_excel_sheets, select_organs  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from src.side.dataset import UTRFeatureShardDataset, load_manifest as load_ds_manifest  # type: ignore
from src.side.model import DualBranchCNNFiLM  # type: ignore


def load_model(manifest: Dict, checkpoint: Path, device: torch.device, cfg_path: Path | None = None) -> DualBranchCNNFiLM:
    in_ch = int(manifest["shapes"]["utr5"][0])
    num_organs = len(manifest.get("organ_vocab", {}))
    conv_channels = [64, 128, 256]
    stem_channels = 32
    film_dim = 32
    hidden_dim = 256
    if cfg_path and cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
        conv_channels = cfg.get("conv_channels", conv_channels)
        stem_channels = cfg.get("stem_channels", stem_channels)
        film_dim = cfg.get("film_dim", film_dim)
        hidden_dim = cfg.get("hidden_dim", hidden_dim)
    model = DualBranchCNNFiLM(
        in_channels=in_ch,
        num_organs=num_organs,
        conv_channels=conv_channels,
        stem_channels=stem_channels,
        film_dim=film_dim,
        hidden_dim=hidden_dim,
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def extract_embeddings(
    dataset_dir: Path, checkpoint: Path, split: str, max_samples: int, batch_size: int, num_workers: int
) -> Tuple[np.ndarray, np.ndarray]:
    ds_manifest = load_ds_manifest(str(dataset_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = UTRFeatureShardDataset(str(dataset_dir), split=split, max_cache_shards=2)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = load_model(ds_manifest, checkpoint, device, cfg_path=REPO_ROOT / "configs" / "train_rbp_trna.yaml")
    feats = []
    organs = []
    with torch.no_grad():
        for batch in loader:
            if max_samples and len(organs) >= max_samples:
                break
            utr5 = batch["utr5"].to(device)
            utr3 = batch["utr3"].to(device)
            organ = batch["organ_id"].to(device)
            cond = model.tissue_embedding(organ)
            feat5 = model.branch5(utr5, cond)
            feat3 = model.branch3(utr3, cond)
            fused = torch.cat([feat5, feat3], dim=-1)
            feats.append(fused.detach().cpu())
            organs.append(organ.detach().cpu())
    feats_arr = torch.cat(feats).numpy()
    organs_arr = torch.cat(organs).numpy()
    if max_samples:
        feats_arr = feats_arr[:max_samples]
        organs_arr = organs_arr[:max_samples]
    return feats_arr, organs_arr


def plot_umap(emb: np.ndarray, organs: np.ndarray, vocab: Dict[int, str], out_path: Path) -> None:
    reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=30, min_dist=0.1)
    emb2d = reducer.fit_transform(StandardScaler().fit_transform(emb))
    df = pd.DataFrame({"x": emb2d[:, 0], "y": emb2d[:, 1], "organ_id": organs})
    df["organ_name"] = df["organ_id"].map(lambda x: vocab.get(int(x), str(int(x))))

    fig, ax = plt.subplots(figsize=(8, 6))
    sel = select_organs(vocab, top_n=8)
    sel_ids = {oid for oid, _ in sel}
    for oid, name in sel:
        sub = df[df["organ_id"] == oid]
        ax.scatter(sub["x"], sub["y"], s=6, alpha=0.6, label=name)
    ax.set_title("UMAP of encoder embeddings")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return df


def main():
    parser = argparse.ArgumentParser(description="Fig2D UMAP embedding plot")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/seq_cnn_v1_rbp_trna"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/cnn_film_rbp_trna_best.pt"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--outdir", type=Path, default=Path("reports/figures/fig2_fig3"))
    parser.add_argument("--excel", type=Path, default=Path("outputs/fig2_fig3_data/fig2D_umap.xlsx"))
    args = parser.parse_args()

    manifest = load_manifest(args.dataset / "manifest.json")
    vocab = manifest["organ_vocab"]
    emb, organs = extract_embeddings(args.dataset, args.checkpoint, args.split, args.max_samples, args.batch_size, args.num_workers)
    df = plot_umap(emb, organs, vocab, args.outdir / "fig2D_umap.png")
    save_excel_sheets({"umap": df}, args.excel)


if __name__ == "__main__":
    main()
