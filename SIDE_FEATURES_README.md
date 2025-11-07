
# Side-Feature Wiring (1b/1c minimal)

This patch adds:
- `src/features/store.py`: load side tables:
    - organ_id → tissue_vec / rbp_vec
    - transcript_id → x_struct
  Supported formats: `.npy` ({"keys": [...], "data": [[...]]}), `.json`, `.csv` (first col key).
- `src/cnn_v1/dataset_side.py`: `AugmentedUTRDataset` wrapper + `collate_with_side` that injects side features into batches.
- `src/cnn_v1/heads/concat_side.py`: `ConcatSideHead` MLP head that concatenates trunk features with side features.
- `src/cnn_v1/model_side.py`: `DualBranchCNNWithSide` that reuses the two sequence branches and swaps in the concat head.
- `tools/train_reg_with_side.py`: training entrypoint wiring everything together (no changes to your existing training loop).
- `src/eval_suite/report_per_organ.py`: per-dimension R² / Spearman utilities.

## Run
1) Build your regression dataset (N,54) at `data/processed/seq_cnn_v1_reg` (as you already do).
2) Prepare side tables (any subset is fine):
   - `--tissue path/to/organ_tissue_vec.csv`
   - `--rbp path/to/organ_rbp_vec.csv`
   - `--struct path/to/transcript_struct_vec.csv`
3) Train:
```bash
python -m tools.train_reg_with_side \
  --dataset_dir data/processed/seq_cnn_v1_reg \
  --tissue data_intermediate/features/organ_tissue_vec.csv \
  --rbp    data_intermediate/features/organ_rbp_vec.csv \
  --struct data_intermediate/features/transcript_struct_vec.csv
```

Notes:
- If `dataset_dir/index/{split}.json` contains `organ_id` / `transcript_id` for each `global_idx`, they will be used. Otherwise the corresponding side features are silently zeroed (you can still benefit from transcript-only features).
- Nothing in the original files is modified; you can continue to use your existing configs and loops side-by-side.
