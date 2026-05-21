import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
from math import log2
from tqdm import tqdm
from utils_get_embeddings import get_plm_representation

# =========================================================
# CONFIGURATION
# =========================================================
MODEL_TYPES = ["esmc", "esm2"]
PFAM_LIST_FILE = "all_pfams.txt"
DATA_DIR = "../data"
OUT_ROOT = "results_kmeans_layerwise"

N_CLUSTERS = 114
PCA_DIM = 400
LAYER_START = 34      # run for layers >= 27
RANDOM_STATE = 42

# =========================================================
# REQUIRED EXTERNAL FUNCTION
# =========================================================
# get_plm_representation must exist in your environment


def load_plm_dataset(pfam, model_type):
    fasta_file_wo_gap = f"{DATA_DIR}/{pfam}/{pfam}_nogap.aln"
    fasta_file_std_gap = f"{DATA_DIR}/{pfam}/{pfam}_stdgap.aln"
    layers = "all"
    return get_plm_representation(
        model_type,
        fasta_file_wo_gap,
        fasta_file_std_gap,
        layers
    )


# =========================================================
# MERGE DATASETS (KEEP LAYER DIM)
# =========================================================
def merge_plm_datasets(plm_list, pfam_ids):
    """
    Returns:
        X_all: shape (N, L, D)
        pfam_labels: shape (N,)
    """
    embeddings = []
    pfam_labels = []

    for plm, pfam in zip(plm_list, pfam_ids):
        emb = plm.seq_embeddding   # (n_i, L, D)
        embeddings.append(emb)
        pfam_labels.extend([pfam] * emb.shape[0])

    X_all = np.concatenate(embeddings, axis=0)
    pfam_labels = np.array(pfam_labels)

    return X_all, pfam_labels


# =========================================================
# PAIRWISE METRICS (NO O(N^2))
# =========================================================
def compute_pairwise_metrics(pfam_labels, cluster_labels):

    pfam_counts = Counter(pfam_labels)
    cluster_counts = Counter(cluster_labels)

    total_same_pfam = sum(c * (c - 1) // 2 for c in pfam_counts.values())
    total_same_cluster = sum(c * (c - 1) // 2 for c in cluster_counts.values())

    cluster_pfam = defaultdict(Counter)
    for p, c in zip(pfam_labels, cluster_labels):
        cluster_pfam[c][p] += 1

    tp = sum(
        cnt * (cnt - 1) // 2
        for cdict in cluster_pfam.values()
        for cnt in cdict.values()
    )

    fp = total_same_cluster - tp
    fn = total_same_pfam - tp
    total_pairs = len(pfam_labels) * (len(pfam_labels) - 1) // 2
    tn = total_pairs - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    return tp, fp, fn, tn, precision, recall, f1


# =========================================================
# PURITY & ENTROPY
# =========================================================
def compute_cluster_purity_entropy(pfam_labels, cluster_labels):

    clusters = defaultdict(list)
    for p, c in zip(pfam_labels, cluster_labels):
        clusters[c].append(p)

    purities = []
    entropies = []

    for pfams in clusters.values():
        counts = Counter(pfams)
        total = len(pfams)

        purities.append(max(counts.values()) / total)

        entropy = 0.0
        for cnt in counts.values():
            p = cnt / total
            entropy -= p * log2(p)
        entropies.append(entropy)

    return np.mean(purities), np.mean(entropies)


# =========================================================
# MAIN EXPERIMENT
# =========================================================
def run_experiment(model_type, pfam_ids):

    print(f"\n========== {model_type.upper()} ==========")

    out_dir = os.path.join(OUT_ROOT, model_type)
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    plm_datasets = []
    for pfam in tqdm(pfam_ids, desc="Loading Pfams"):
        plm_datasets.append(load_plm_dataset(pfam, model_type))

    X_all, true_pfams = merge_plm_datasets(plm_datasets, pfam_ids)

    num_layers = X_all.shape[1]
    print(num_layers)
    #return
    print(f"Total layers detected: {num_layers}")

    metrics_file = os.path.join(out_dir, "metrics_layerwise.txt")

    with open(metrics_file, "w") as f:

        f.write(f"Model: {model_type}\n")
        f.write(f"Pfams: {len(pfam_ids)}\n")
        f.write(f"PCA dim: {PCA_DIM}\n")
        f.write(f"Clusters: {N_CLUSTERS}\n")
        f.write(f"Layers evaluated: {LAYER_START}–{num_layers-1}\n")
        f.write("=" * 60 + "\n\n")

        for layer in [num_layers-2]:

            print(f"Layer {layer}")

            X_layer = X_all[:, layer, :]   # (N, D)

            # PCA
            pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
            X_pca = pca.fit_transform(X_layer)

            # KMeans
            kmeans = KMeans(
                n_clusters=N_CLUSTERS,
                n_init=20,
                random_state=RANDOM_STATE
            )
            cluster_labels = kmeans.fit_predict(X_pca)

            # Metrics
            tp, fp, fn, tn, p, r, f1 = compute_pairwise_metrics(
                true_pfams, cluster_labels
            )

            purity, entropy = compute_cluster_purity_entropy(
                true_pfams, cluster_labels
            )

            # Save
            f.write(f"Layer {layer}\n")
            f.write("-" * 40 + "\n")
#            f.write(f"TP: {tp}\nFP: {fp}\nFN: {fn}\nTN: {tn}\n")
            f.write(f"Precision: {p:.4f}\n")
            f.write(f"Recall:    {r:.4f}\n")
            f.write(f"F1-score:  {f1:.4f}\n")
            f.write(f"Purity:    {purity:.4f}\n")
            f.write(f"Entropy:   {entropy:.4f}\n\n")

            print(
                f"  P={p:.3f} R={r:.3f} F1={f1:.3f} "
                f"Purity={purity:.3f} Entropy={entropy:.3f}"
            )


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":

    with open(PFAM_LIST_FILE) as f:
        pfam_ids = [l.strip() for l in f if l.strip()]

    assert len(pfam_ids) == N_CLUSTERS, \
        f"Expected {N_CLUSTERS} Pfams, got {len(pfam_ids)}"

    os.makedirs(OUT_ROOT, exist_ok=True)

    for model in MODEL_TYPES:
        run_experiment(model, pfam_ids)