# Fig2G per-RBP contribution (CPU pipeline)

Data flow:
- `data/rawData/train_pair.v1.parquet` (contains gene_id, utr5/3 sequences, split)  
  → `scripts/compute_rbp_hits_per_rbp.py` (uses GTF + eCLIP peaks)  
  → `Phase2Data/features/features_rbp_hits_per_rbp.<split>.parquet`
- Predictions `outputs/preds/test_preds.parquet` + dataset index  
  → `scripts/merge_rbp_per_rbp_features_into_eval.py`  
  → `Phase2Data/eval/eval_with_rbp_hits_per_rbp.<split>.parquet`
- Evaluation + per-RBP features  
  → `scripts/plot_fig2G_rbp_contrib_per_rbp.py`  
  → `reports/figures/fig2_fig3/fig2G_rbp_contrib_per_rbp.<tissue>.png`

Environment (CPU node):
```bash
export PATH="/root/miniconda3/bin:$PATH"
# Use pip only; conda CLI is broken.
```

End-to-end commands (HepG2 eCLIP, liver target):
```bash
export PATH="/root/miniconda3/bin:$PATH"
cd /root/autodl-tmp/Sample_mRNA_011_5090-main_regpatched/Sample_mRNA_011_5090-main

# 1) Compute per-RBP hits (test split, HepG2 peaks)
nohup python -u scripts/compute_rbp_hits_per_rbp.py \
  --split test \
  --peak-dir data/external/rbp/encode_eclip/HepG2 \
  --num-workers 48 \
  --output Phase2Data/features/features_rbp_hits_per_rbp.test.parquet \
  > /tmp/log_compute_rbp_hits_per_rbp_$(date +%F).log 2>&1 &
# Monitor: tail -f /tmp/log_compute_rbp_hits_per_rbp_$(date +%F).log

# 2) Merge features into eval table
python -u scripts/merge_rbp_per_rbp_features_into_eval.py \
  --preds outputs/preds/test_preds.parquet \
  --index data/processed/seq_cnn_v1_rbp_trna/index/test/index.parquet \
  --features Phase2Data/features/features_rbp_hits_per_rbp.test.parquet \
  --manifest data/processed/seq_cnn_v1_rbp_trna/manifest.json \
  --output Phase2Data/eval/eval_with_rbp_hits_per_rbp.test.parquet

# 3) Plot Fig2G per-RBP (liver)
python -u scripts/plot_fig2G_rbp_contrib_per_rbp.py \
  --eval Phase2Data/eval/eval_with_rbp_hits_per_rbp.test.parquet \
  --tissue Liver \
  --output reports/figures/fig2_fig3/fig2G_rbp_contrib_per_rbp.liver.png
```

Logging / sanity:
- `compute_rbp_hits_per_rbp.py` logs number of RBPs, feature shape, example columns.
- `merge_rbp_per_rbp_features_into_eval.py` logs matched rows and RBP column count.
- `plot_fig2G_rbp_contrib_per_rbp.py` logs Top-10 SHAP RBPs and writes a CSV with metrics next to the figure.
