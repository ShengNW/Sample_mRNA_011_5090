#!/usr/bin/env python3
"""Run inference with FiLM CNN and dump predictions for plotting.

This helper materialises y_true/y_pred/organ_id parquet files so downstream
plotting (Pred vs Obs, per-organ R2, calibration) can be done offline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.side.dataset import UTRFeatureShardDataset, load_manifest
from src.side.model import DualBranchCNNFiLM


def _load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) if path.endswith(".json") else __import__("yaml").safe_load(f)


def build_model(cfg: Dict, manifest: Dict, checkpoint: str, device: torch.device) -> DualBranchCNNFiLM:
    in_channels = int(manifest["shapes"]["utr5"][0])
    organ_vocab = manifest.get("organ_vocab", {})
    num_organs = int(cfg.get("num_organs", 0)) or len(organ_vocab)
    if num_organs == 0:
        raise ValueError("num_organs not found in cfg or manifest")
    model = DualBranchCNNFiLM(
        in_channels=in_channels,
        num_organs=num_organs,
        conv_channels=cfg.get("conv_channels", [64, 128, 256]),
        stem_channels=cfg.get("stem_channels", 32),
        film_dim=cfg.get("film_dim", 32),
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.2),
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def run_inference(cfg: Dict, checkpoint: str, split: str, out_path: str, batch_size: int, num_workers: int) -> str:
    manifest = load_manifest(cfg["dataset_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = UTRFeatureShardDataset(cfg["dataset_dir"], split=split, max_cache_shards=cfg.get("max_cache_shards", 2))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = build_model(cfg, manifest, checkpoint, device)

    preds = []
    trues = []
    organs = []
    with torch.no_grad():
        for batch in loader:
            utr5 = batch["utr5"].to(device)
            utr3 = batch["utr3"].to(device)
            organ = batch["organ_id"].to(device)
            target = batch["label"].to(device)
            out = model(utr5, utr3, organ)
            preds.append(out.detach().cpu())
            trues.append(target.detach().cpu())
            organs.append(organ.detach().cpu())
    y_pred = torch.cat(preds).numpy().astype(np.float32)
    y_true = torch.cat(trues).numpy().astype(np.float32)
    organ_ids = torch.cat(organs).numpy().astype(np.int64)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_pred": y_pred, "y_true": y_true, "organ_id": organ_ids}).to_parquet(out_path, index=False)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Run FiLM CNN inference and save predictions")
    parser.add_argument("--config", required=True, help="Training config (yaml/json) with dataset_dir + model dims")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--split", default="val", help="Dataset split to run (val/test/train)")
    parser.add_argument("--out", required=True, help="Output parquet path for predictions")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    out_path = run_inference(cfg, args.checkpoint, args.split, args.out, args.batch_size, args.num_workers)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
