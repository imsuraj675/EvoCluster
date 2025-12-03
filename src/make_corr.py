import os
import numpy as np
import pickle
from tqdm import tqdm

# === Import required functions ===
from utils_get_embeddings import get_plm_representation
from homology_corr import layer_wise_lg_corr


# === Define helper function to load PLM embeddings ===
def load_plm_dataset(pfam, model_type):
    fasta_file_wo_gap  = f"../data/{pfam}/{pfam}_nogap.aln"
    fasta_file_std_gap = f"../data/{pfam}/{pfam}_stdgap.aln"
    layers = "all"

    # Call get_plm_representation -> returns PLMEmb_Dataset
    plmdata = get_plm_representation(model_type, fasta_file_wo_gap, fasta_file_std_gap, layers)
    return plmdata


# === Merge two PLMEmb_Dataset objects ===
def merge_plm_datasets(plm1, plm2):
    seq_labels = plm1.seq_labels + plm2.seq_labels
    seq_embedding = np.concatenate([plm1.seq_embeddding, plm2.seq_embeddding], axis=0)
    return type(plm1)(seq_labels, seq_embedding)


# === Merge sequence reference dictionaries ===
def merge_seq_ref(seq_ref1, seq_ref2):
    merged_ref = {}
    x = len(seq_ref1)
    merged_ref.update({k: v for k, v in seq_ref1.items()})
    merged_ref.update({k: v + x for k, v in seq_ref2.items()})
    return merged_ref


# === Main pipeline ===
def main():
    pair_file = "all_pfam_pairs.txt"
    results_dir = "results_interfamily/results"
    os.makedirs(results_dir, exist_ok=True)

    with open(pair_file) as f:
        pairs = [line.strip().split() for line in f if line.strip()]

    for pf1, pf2 in tqdm(pairs, desc="Processing Pfam pairs"):
        # Load reference dicts
        #if os.path.exists(f"./results_interfamily/results/esmc_{pf1}_{pf2}_spear.npy"):
        #    print(f"Skipping {pf1}_{pf2}")
        #    continue

        seq_ref1_path = f"../data/{pf1}/sequence_ref_dict.pkl"
        seq_ref2_path = f"../data/{pf2}/sequence_ref_dict.pkl"

        with open(seq_ref1_path, "rb") as f1, open(seq_ref2_path, "rb") as f2:
            seq_ref1 = pickle.load(f1)
            seq_ref2 = pickle.load(f2)

        # Merge sequence refs
        seq_ref_merged = merge_seq_ref(seq_ref1, seq_ref2)

        # Load merged LG matrix
        lg_mat_path = f"results_interfamily/data/{pf1}_{pf2}_merged.npy"
        lg_mat = np.load(lg_mat_path)

        #un for both models
        for model_type in ["esm2", "esmc", "pt", "esm1"]:
            # Load PLM representations
            if os.path.exists(f"{results_dir}/{model_type}_{pf1}_{pf2}_spear.npy"):
                print(f"Skipping {pf1}-{pf2}-----{model_type}")
                continue
            plm1 = load_plm_dataset(pf1, model_type)
            plm2 = load_plm_dataset(pf2, model_type)

            # Merge embeddings
            plm_merged = merge_plm_datasets(plm1, plm2)

            # Compute correlations
            spear, pearson = layer_wise_lg_corr(plm_merged, lg_mat, seq_ref_merged)

            # Save results
            np.save(os.path.join(results_dir, f"{model_type}_{pf1}_{pf2}_pearson.npy"), pearson)
            np.save(os.path.join(results_dir, f"{model_type}_{pf1}_{pf2}_spear.npy"), spear)


if __name__ == "__main__":
    main()

