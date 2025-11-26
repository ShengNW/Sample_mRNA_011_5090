#!/usr/bin/env python3
"""Plot hard metrics (Pred vs Obs, per-organ R2, calibration, learning curves)."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAIN_RE = re.compile(r"Epoch (\d+)/\d+ completed .*?train_loss=([0-9.]+)")
VAL_RE = re.compile(r"Validation completed .*? R2=([\-0-9.]+)")


def parse_log(log_path: Path) -> Dict[str, List[Tuple[int, float]]]:
    train, val = [], []
    current_epoch = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = TRAIN_RE.search(line)
            if m:
                current_epoch = len(train) + 1
                train.append((current_epoch, float(m.group(2))))
            mv = VAL_RE.search(line)
            if mv:
                ep = current_epoch if current_epoch is not None else len(val) + 1
                val.append((ep, float(mv.group(1))))
    return {"train_loss": train, "val_r2": val}


def load_manifest(manifest_path: Path) -> Dict[int, str]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    organ_vocab = data.get("organ_vocab", {})
    return {int(k): v for k, v in organ_vocab.items()}


def standardize_preds(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for key in ("y_true", "target", "label"):
        if key in cols:
            rename[cols[key]] = "y_true"
            break
    for key in ("y_pred", "pred", "prediction"):
        if key in cols:
            rename[cols[key]] = "y_pred"
            break
    if "organ_id" in df.columns:
        rename["organ_id"] = "organ_id"
    df = df.rename(columns=rename)
    missing = [c for c in ("y_true", "y_pred") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "organ_id" not in df.columns:
        df["organ_id"] = -1
    return df


def compute_metrics(df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    resid = y_pred - y_true
    mse = float(np.mean(resid ** 2))
    var = float(np.var(y_true)) + 1e-12
    r2 = float(1.0 - mse / var)
    rho = float(np.corrcoef(y_true, y_pred)[0, 1])
    rmse = float(math.sqrt(mse))
    per_org = df.groupby("organ_id").apply(
        lambda g: pd.Series({
            "r2": 1.0 - float(np.mean((g.y_pred - g.y_true) ** 2)) / (float(np.var(g.y_true)) + 1e-12),
            "count": len(g),
        })
    ).reset_index()
    return {"r2": r2, "rho": rho, "rmse": rmse}, per_org


def plot_learning_curves(curves: Dict[str, List[Tuple[int, float]]], out_path: Path) -> None:
    if not curves["train_loss"] and not curves["val_r2"]:
        return
    fig, ax1 = plt.subplots(figsize=(7, 4))
    if curves["train_loss"]:
        xs, ys = zip(*curves["train_loss"])
        ax1.plot(xs, ys, label="train_loss", color="#4e79a7")
        ax1.set_ylabel("Train loss")
    if curves["val_r2"]:
        ax2 = ax1.twinx()
        xs, ys = zip(*curves["val_r2"])
        ax2.plot(xs, ys, label="val R2", color="#f28e2b")
        ax2.set_ylabel("Val R2")
    ax1.set_xlabel("Epoch")
    ax1.set_title("Learning curves")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pred_obs(df: pd.DataFrame, metrics: Dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df["y_true"], df["y_pred"], s=4, alpha=0.4, color="#4e79a7")
    min_v = float(min(df["y_true"].min(), df["y_pred"].min()))
    max_v = float(max(df["y_true"].max(), df["y_pred"].max()))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title("Pred vs Obs")
    ax.text(
        0.05,
        0.95,
        f"R2={metrics['r2']:.3f}\nρ={metrics['rho']:.3f}\nRMSE={metrics['rmse']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_residual_hist(df: pd.DataFrame, out_path: Path) -> None:
    resid = df["y_pred"] - df["y_true"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(resid, bins=40, color="#59a14f", alpha=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Residuals")
    ax.set_xlabel("Pred - Obs")
    ax.set_ylabel("Count")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_calibration(df: pd.DataFrame, out_path: Path, n_bins: int = 20) -> None:
    preds = df["y_pred"].to_numpy()
    trues = df["y_true"].to_numpy()
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(preds, quantiles)
    bin_ids = np.digitize(preds, bins[1:-1], right=True)
    bin_pred = []
    bin_true = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        bin_pred.append(preds[mask].mean())
        bin_true.append(trues[mask].mean())
    if not bin_pred:
        return
    lo = min(bin_pred + bin_true)
    hi = max(bin_pred + bin_true)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--")
    ax.plot(bin_pred, bin_true, marker="o", color="#e15759")
    ax.set_xlabel("Mean predicted (bin)")
    ax.set_ylabel("Mean observed")
    ax.set_title("Calibration")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_r2_by_organ(per_org: pd.DataFrame, organ_names: Dict[int, str], out_path: Path) -> None:
    if per_org.empty:
        return
    per_org = per_org.sort_values("r2", ascending=False)
    labels = [organ_names.get(int(r.organ_id), str(int(r.organ_id))) for r in per_org.itertuples()]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels[::-1], per_org["r2"].to_list()[::-1], color="#76b7b2")
    ax.set_xlabel("R2")
    ax.set_title("Per-organ R2")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_metrics_json(metrics: Dict, per_org: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(metrics)
    data["per_organ"] = per_org.to_dict(orient="records") if not per_org.empty else []
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Plot hard metrics panels")
    parser.add_argument("--log", type=Path, required=True, help="Training log path (side_train_rbp_trna.log)")
    parser.add_argument("--preds", type=Path, help="Parquet/CSV with y_true,y_pred,organ_id")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/seq_cnn_v1_rbp_trna/manifest.json"))
    parser.add_argument("--outdir", type=Path, default=Path("reports/figures/hard_metrics"))
    parser.add_argument("--split", default="val", help="Label for output files (val/test)")
    args = parser.parse_args()

    curves = parse_log(args.log)
    os.makedirs(args.outdir, exist_ok=True)
    plot_learning_curves(curves, args.outdir / "learning_curves.png")

    if args.preds:
        df = standardize_preds(pd.read_parquet(args.preds) if args.preds.suffix == ".parquet" else pd.read_csv(args.preds))
        metrics, per_org = compute_metrics(df)
        organ_names = load_manifest(args.manifest)
        plot_pred_obs(df, metrics, args.outdir / f"pred_obs_{args.split}.png")
        plot_residual_hist(df, args.outdir / f"residuals_{args.split}.png")
        plot_calibration(df, args.outdir / f"calibration_{args.split}.png")
        plot_r2_by_organ(per_org, organ_names, args.outdir / f"r2_by_organ_{args.split}.png")
        save_metrics_json(metrics, per_org, args.outdir / f"metrics_{args.split}.json")


if __name__ == "__main__":
    main()
