import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

class UTRDataset(IterableDataset):
    """
    Dataset for UTR sequences and associated features.
    Supports sharded data loading and optional shuffle buffer for improved I/O performance.
    Each data sample is yielded as a tuple: (seq5, seq3, tissue_vec, extra_features, label).
      - seq5: 5' UTR sequence (as a tensor or array, e.g. one-hot encoded) 
      - seq3: 3' UTR sequence 
      - tissue_vec: tissue embedding vector (float tensor)
      - extra_features: additional numeric features (tensor), e.g. RBP and tRNA features
      - label: target value (e.g. expression level)
    """
    def __init__(self, shards, tissue_embeddings, rbp_features=None, trna_features=None, 
                 include_rbp=True, include_trna=True, include_film=True, shuffle_buffer=0, rank=None, world_size=None):
        """
        shards: either a directory path containing data shard files, or a list of file paths.
        tissue_embeddings: dict mapping tissue_id -> embedding vector.
        rbp_features: dict as returned by load_rbp_features (or None if not used).
        trna_features: dict as returned by load_trna_features (or None if not used).
        include_rbp, include_trna, include_film: flags to include respective features (in case we want to disable them).
        shuffle_buffer: size of buffer for shuffling. If >0, will shuffle within this buffer size to avoid global random reads.
        rank, world_size: for distributed training, specify the process rank and total number of processes to split shards.
        """
        super(UTRDataset, self).__init__()
        # Determine list of shard files
        if isinstance(shards, str):
            # Treat as directory
            self.shard_files = sorted([os.path.join(shards, f) for f in os.listdir(shards)])
        else:
            # Assume list of file paths
            self.shard_files = list(shards)
        # If using DDP, split shards among ranks to avoid duplication
        if world_size is not None and world_size > 1 and rank is not None:
            self.shard_files = [f for i, f in enumerate(self.shard_files) if i % world_size == rank]
        self.tissue_embeddings = tissue_embeddings
        self.rbp_features = rbp_features or {}  # default to empty dict if None
        self.trna_features = trna_features or {}
        self.include_rbp = include_rbp
        self.include_trna = include_trna
        self.include_film = include_film
        self.shuffle_buffer = shuffle_buffer

    def parse_shard_file(self, file_path):
        """
        Load data from a single shard file.
        Assumes the shard contains multiple samples. 
        This function should return an iterable (list or generator) of samples.
        Each sample could be a tuple (gene_id, tissue_id, seq5, seq3, label) or a dict with those fields.
        """
        # Example implementation assuming numpy npz or pickle for simplicity.
        data_samples = []
        if file_path.endswith(".npz"):
            # If shards are stored as NPZ with arrays
            npz = np.load(file_path, allow_pickle=True)
            # Assuming npz contains arrays: gene_ids, tissue_ids, seq5, seq3, labels
            gene_ids = npz["gene_ids"]
            tissue_ids = npz["tissue_ids"]
            seq5_arr = npz["seq5"]  # could be 2D array [N, L5] or 3D one-hot [N, L5, 4]
            seq3_arr = npz["seq3"]
            labels = npz["labels"]
            N = len(labels)
            for i in range(N):
                gene = gene_ids[i]
                tissue = tissue_ids[i]
                seq5 = seq5_arr[i]
                seq3 = seq3_arr[i]
                label = labels[i]
                data_samples.append((gene, tissue, seq5, seq3, label))
        else:
            # Assume pickle file with list of samples (each sample could be a dict or tuple)
            import pickle
            with open(file_path, "rb") as pf:
                content = pickle.load(pf)
            # content is expected to be list of samples
            for sample in content:
                # Normalize sample format to tuple (gene, tissue, seq5, seq3, label)
                if isinstance(sample, dict):
                    gene = sample.get("gene_id") or sample.get("gene")
                    tissue = sample.get("tissue_id") or sample.get("tissue")
                    seq5 = sample.get("seq5")
                    seq3 = sample.get("seq3")
                    label = sample.get("label")
                else:
                    # assume tuple format already
                    gene, tissue, seq5, seq3, label = sample
                data_samples.append((gene, tissue, seq5, seq3, label))
        return data_samples

    def __iter__(self):
        # If multiple workers, split shard files among workers to avoid duplicates
        worker_info = get_worker_info()
        if worker_info is None:
            # Single worker (or main process)
            shard_files = self.shard_files
        else:
            # Partition the shard list among workers
            total_shards = len(self.shard_files)
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # Split into contiguous chunks
            shards_per_worker = math.ceil(total_shards / num_workers)
            start = worker_id * shards_per_worker
            end = min(total_shards, start + shards_per_worker)
            shard_files = self.shard_files[start:end]
        # Iterate through assigned shard files in order
        for shard_path in shard_files:
            # Load all samples from this shard
            samples = self.parse_shard_file(shard_path)
            # If shuffle_buffer is used, implement a buffered shuffle of samples
            if self.shuffle_buffer and self.shuffle_buffer > 0:
                buffer = []
                it = iter(samples)
                # Fill initial buffer
                for _ in range(self.shuffle_buffer):
                    try:
                        buffer.append(next(it))
                    except StopIteration:
                        break
                # Yield items randomly from buffer and refill from iterator
                import random
                while len(buffer) > 0:
                    try:
                        # Randomly select an index from buffer to yield
                        idx = random.randrange(len(buffer))
                        sample = buffer[idx]
                        # Replace that item with a new one from iterator
                        buffer[idx] = next(it)
                    except StopIteration:
                        # If iterator is exhausted, pop remaining buffer sequentially
                        sample = buffer.pop(idx)
                    else:
                        # Yield the chosen sample
                        yield self._process_sample(sample)
                        continue
                    # If StopIteration reached: yield the item we popped, then break loop to flush rest
                    yield self._process_sample(sample)
                    # Now yield the rest of buffer (since no new items to fill)
                    for remaining in buffer:
                        yield self._process_sample(remaining)
                    buffer = []
            else:
                # If no shuffle buffer, we can optionally shuffle the list locally, or yield sequentially for deterministic order.
                # We'll shuffle within the shard for randomness, to avoid strictly sorted order.
                import random
                random.shuffle(samples)
                for sample in samples:
                    yield self._process_sample(sample)

    def _process_sample(self, sample):
        """
        Convert raw sample tuple into final output tuple (seq5_tensor, seq3_tensor, tissue_vec_tensor, extra_feat_tensor, label_tensor).
        Also attach RBP/tRNA features if available.
        """
        gene_id, tissue_id, seq5, seq3, label = sample
        # Ensure seq5, seq3 are torch tensors (or numpy arrays) of appropriate shape.
        # If they are one-hot encoded as numpy arrays, we might want to convert to torch tensors.
        seq5_tensor = torch.tensor(seq5, dtype=torch.float32) if not isinstance(seq5, torch.Tensor) else seq5.clone().detach()
        seq3_tensor = torch.tensor(seq3, dtype=torch.float32) if not isinstance(seq3, torch.Tensor) else seq3.clone().detach()
        # Retrieve tissue embedding vector
        tissue_vec = self.tissue_embeddings.get(str(tissue_id), None)
        if tissue_vec is None:
            # If no embedding found (should not happen if all tissues covered), use zeros
            # We assume all embeddings have same dimension as first entry
            embed_dim = len(next(iter(self.tissue_embeddings.values())))
            tissue_vec = [0.0] * embed_dim
        tissue_tensor = torch.tensor(tissue_vec, dtype=torch.float32)
        # Gather extra features
        extra_feats = []
        if self.include_rbp:
            # Get RBP features for this gene and this tissue (cell line)
            rbp_vals = [0, 0.0, 0, 0.0]  # default [count5, avg5, count3, avg3]
            if tissue_id in self.rbp_features:
                gene_rec = self.rbp_features[tissue_id].get(gene_id)
                if gene_rec:
                    rbp_vals = [
                        gene_rec.get("count_5utr", 0),
                        gene_rec.get("avg_signal_5utr", 0.0),
                        gene_rec.get("count_3utr", 0),
                        gene_rec.get("avg_signal_3utr", 0.0)
                    ]
            # Normalize or scale if needed (e.g., log transform signals) - here we just use raw values
            extra_feats.extend([float(rbp_vals[0]), float(rbp_vals[1]), float(rbp_vals[2]), float(rbp_vals[3])])
        if self.include_trna:
            trna_rec = self.trna_features.get(gene_id)
            if trna_rec:
                overlap = trna_rec.get("tRNA_overlap", 0)
                dist = trna_rec.get("tRNA_distance", 1e6)
            else:
                overlap = 0
                dist = 1e6
            extra_feats.extend([float(overlap), float(dist)])
        # Convert extra_feats to tensor
        if extra_feats:
            extra_tensor = torch.tensor(extra_feats, dtype=torch.float32)
        else:
            # If no extra features, use an empty tensor of shape (0,)
            extra_tensor = torch.tensor([], dtype=torch.float32)
        # Label to tensor (assuming regression target)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return seq5_tensor, seq3_tensor, tissue_tensor, extra_tensor, label_tensor
