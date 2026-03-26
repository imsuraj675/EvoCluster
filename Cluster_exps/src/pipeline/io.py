import os
import csv
import sys
import logging
import pickle
from datetime import datetime

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import esm

EPS = 1e-12

def setup_logging(organism: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{organism}_{timestamp}.log")
    logger = logging.getLogger("multiscale")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_file}")
    return logger

def get_device(use_gpu: bool = True) -> str:
    if use_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def log_device_info(logger, use_gpu: bool = True):
    device = get_device(use_gpu)
    logger.info(f"  Device: {device}")
    if device == "cuda":
        logger.info(f"    GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info(f"    VRAM: {vram:.1f} GB")
    elif use_gpu:
        logger.info(f"    (GPU requested but CUDA not available — using CPU)")
    return device

def l2_normalize(X: np.ndarray, eps: float = EPS) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (nrm + eps)

def load_embeddings(fasta_path, emb_path, layers, logger):
    label_to_index = {}
    csv_path = os.path.join(emb_path, "index_to_label.csv")
    assert os.path.isfile(csv_path), f"index_to_label.csv not found at {csv_path}"
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        _ = next(reader)
        for idx, label in reader:
            label_to_index[label.strip()] = idx.strip()
    X_by_layer = {layer: [] for layer in layers}
    prot_meta = []
    for fasta_header, seq in esm.data.read_fasta(fasta_path):
        short_header = fasta_header.split(" ")[0]
        if short_header not in label_to_index:
            continue
        idx = label_to_index[short_header]
        emb_file = os.path.join(emb_path, f"{idx}.pt")
        if not os.path.isfile(emb_file):
            continue
        embs = torch.load(emb_file, map_location="cpu", weights_only=False)
        for layer in layers:
            X_by_layer[layer].append(embs[layer, :])
        prot_meta.append(fasta_header.split("|"))
    for layer in layers:
        X_by_layer[layer] = np.stack(X_by_layer[layer], axis=0)
    logger.info(f"Loaded {len(prot_meta)} proteins")
    return X_by_layer, prot_meta

def prepare_embeddings(X_raw, *, pca_dim=400, scale=True, normalize=True, logger=None):
    log = logger or logging.getLogger("multiscale")
    log.info(f"Step 1: Preparing embeddings — PCA({pca_dim}), scale={scale}, normalize={normalize}")
    log.debug(f"  Input shape: {X_raw.shape}")
    if pca_dim > 0 and pca_dim < X_raw.shape[1]:
        pca_model = PCA(n_components=pca_dim, svd_solver="full")
        X = pca_model.fit_transform(X_raw)
    else:
        pca_model = None
        X = X_raw.copy()
    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    if normalize:
        X = l2_normalize(X)
    log.info(f"  Output shape: {X.shape}")
    return {"X": X, "pca_model": pca_model, "scaler": scaler, "N": X.shape[0]}

def save_results(results, results_path, organism, logger=None):
    log = logger or logging.getLogger("multiscale")
    os.makedirs(results_path, exist_ok=True)
    pkl_path = os.path.join(results_path, f"{organism}_multiscale_results.pkl")
    save_data = {
        "metrics": results["metrics"],
        "config": results["config"],
        "graph_stats": results["graph_stats"],
        "stability": results["stability"],
        "n_merges": results["n_merges"],
        "merge_log": results["merge_log"],
        "labels": results["labels"],
        "labels_all": {k: v.tolist() for k, v in results["labels_all"].items()},
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(save_data, f)
    log.info(f"Saved results to: {pkl_path}")
    tsv_path = os.path.join(results_path, f"{organism}_multiscale_summary.tsv")
    headers = [
        "level", "n_clusters", "n_singletons",
        "naive_P", "naive_R", "naive_F1",
        "dist_P", "dist_R", "dist_F1",
        "121_P", "121_R", "121_F1",
        "AMI", "combined",
    ]
    with open(tsv_path, "w") as f:
        f.write("\t".join(headers) + "\n")
        for level_name, m in results["metrics"].items():
            row = [
                level_name, str(m["n_clusters"]), str(m["n_singletons"]),
                f"{m['naive'][0]:.4f}", f"{m['naive'][1]:.4f}", f"{m['naive'][2]:.4f}",
                f"{m['distance'][0]:.4f}", f"{m['distance'][1]:.4f}", f"{m['distance'][2]:.4f}",
                f"{m['121'][0]:.4f}", f"{m['121'][1]:.4f}", f"{m['121'][2]:.4f}",
                f"{m['AMI']:.4f}", f"{m['combined_score']:.4f}",
            ]
            f.write("\t".join(row) + "\n")
    log.info(f"Saved summary to: {tsv_path}")
