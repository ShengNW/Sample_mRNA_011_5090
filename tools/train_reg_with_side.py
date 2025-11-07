
# -*- coding: utf-8 -*-
"""
用侧表特征训练多任务回归。
约定：
  - data/processed/seq_cnn_v1_reg 为回归分片目录（含 manifest.json & shards/）
  - 若存在 index/{split}.json（含 organ_id/transcript_id），将用于侧表查表
  - 侧表文件路径在 CLI 指定
"""
from __future__ import annotations
import os, json, argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from src.cnn_v1.dataset import ShardedUTRDataset
from src.cnn_v1.dataset_side import AugmentedUTRDataset, collate_with_side
from src.cnn_v1.model import DualBranchCNN
from src.cnn_v1.model_side import DualBranchCNNWithSide
from src.features.store import FeatureStore

def load_manifest(dataset_dir: str):
    man_path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.exists(man_path):
        raise FileNotFoundError(f"manifest not found: {man_path}")
    with open(man_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="data/processed/seq_cnn_v1_reg")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--huber_beta", type=float, default=1.0)
    ap.add_argument("--logdir", default="outputs/cnn_v1_reg_side")
    # side feature files
    ap.add_argument("--tissue", default=None, help="organ_id keyed vector file (.npy/.json/.csv)")
    ap.add_argument("--rbp", default=None, help="organ_id keyed vector file")
    ap.add_argument("--struct", default=None, help="transcript_id keyed vector file")
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    # feature store
    fs = FeatureStore(tissue_path=args.tissue, rbp_path=args.rbp, struct_path=args.struct)

    # datasets
    man = load_manifest(args.dataset_dir)
    out_dim = int(man.get("y_dim", man.get("num_outputs", 54)))

    def build_loader(split):
        base = ShardedUTRDataset(args.dataset_dir, split=split)
        index_json = os.path.join(args.dataset_dir, "index", f"{split}.json")
        aux = AugmentedUTRDataset(base, fs, index_json=index_json if os.path.exists(index_json) else None)
        return DataLoader(aux, batch_size=args.batch_size, shuffle=(split=="train"),
                          num_workers=4, pin_memory=True, collate_fn=collate_with_side)

    loaders = {s: build_loader(s) for s in args.splits.split(",")}

    # model: base + side head
    base = DualBranchCNN(in_ch=5, emb_dim=8, channels=[64,128,256], num_classes=out_dim)
    dims = fs.dims
    model = DualBranchCNNWithSide(base, side_dims=dims, out_dim=out_dim, hidden=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # loss/optim
    crit = nn.HuberLoss(delta=args.huber_beta)
    opt  = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_val = float("inf")
    for epoch in range(1, args.epochs+1):
        model.train()
        tot = 0.0; n=0
        for batch in loaders["train"]:
            x5 = batch["utr5"].to(device)
            x3 = batch["utr3"].to(device)
            y  = batch["label"].to(device).float()
            side = {k:v.to(device) for k,v in batch.get("side", {}).items()}
            opt.zero_grad()
            yhat = model(x5, x3, side)
            loss = crit(yhat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()) * y.size(0); n += y.size(0)
        train_loss = tot / max(n,1)

        # val
        model.eval()
        with torch.no_grad():
            tot = 0.0; n=0
            for batch in loaders.get("val", []):
                x5 = batch["utr5"].to(device)
                x3 = batch["utr3"].to(device)
                y  = batch["label"].to(device).float()
                side = {k:v.to(device) for k,v in batch.get("side", {}).items()}
                yhat = model(x5, x3, side)
                loss = crit(yhat, y)
                tot += float(loss.item()) * y.size(0); n += y.size(0)
            val_loss = tot / max(n,1) if n>0 else float('nan')
        print(f"[{epoch}] train={train_loss:.5f} val={val_loss:.5f}")
        with open(os.path.join(args.logdir, "loss.txt"), "a", encoding="utf-8") as f:
            f.write(f"{epoch}\t{train_loss}\t{val_loss}\n")
        if n>0 and val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, 
                       os.path.join(args.logdir, "best.pt"))

    # 简单测试评估
    if "test" in loaders:
        from src.eval_suite.metrics import r2_score_np
        import numpy as np
        model.eval(); yh_list=[]; y_list=[]
        with torch.no_grad():
            for batch in loaders["test"]:
                x5 = batch["utr5"].to(device)
                x3 = batch["utr3"].to(device)
                y  = batch["label"].to(device).float()
                side = {k:v.to(device) for k,v in batch.get("side", {}).items()}
                yhat = model(x5, x3, side)
                yh_list.append(yhat.cpu().numpy()); y_list.append(y.cpu().numpy())
        Y = np.concatenate(y_list, 0); P = np.concatenate(yh_list, 0)
        r2 = r2_score_np(Y, P, average="macro")
        print("TEST macro R2:", float(r2))

if __name__ == "__main__":
    main()
