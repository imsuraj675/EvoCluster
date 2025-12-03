import numpy as np
import pickle
from ete3 import Tree
from tqdm import tqdm
import os

pairs_file = "all_pfam_pairs.txt"
master_tree_path = "../data/all_consensus/all_pfams_merged.tree"

master = Tree(master_tree_path, format=1)

with open(pairs_file) as f:
    pairs = [line.strip().split() for line in f if line.strip()]

for pf1, pf2 in pairs:
    print(f"Processing {pf1} — {pf2}")
    if os.path.exists(f'results_interfamily/{pf1}_{pf2}_merged.npy'):
        print(f"Skipping pfam - {pf1} - {pf2}")
        continue
    # Load references and matrices
    with open(f"../data/{pf1}/sequence_ref_dict.pkl", "rb") as f:
        seq_ref1 = pickle.load(f)
    with open(f"../data/{pf2}/sequence_ref_dict.pkl", "rb") as f:
        seq_ref2 = pickle.load(f)

    lg1 = np.load(f"../data/{pf1}/lg_mat.npy")
    lg2 = np.load(f"../data/{pf2}/lg_mat.npy")
    
    # Trees
    tree1 = Tree(f"../data/{pf1}/{pf1}.tree", format=1)
    tree2 = Tree(f"../data/{pf2}/{pf2}.tree", format=1)

    node1 = master.search_nodes(name=pf1)[0]
    node2 = master.search_nodes(name=pf2)[0]

    # Distances
    dist1 = [node1.get_distance(x) for x in tqdm(seq_ref1.keys(), desc=f"{pf1} dist", leave=False)]
    dist2 = [node2.get_distance(x) for x in tqdm(seq_ref2.keys(), desc=f"{pf2} dist", leave=False)]
    dst = node1.get_distance(node2)

    n1, n2 = len(seq_ref1), len(seq_ref2)
    res = np.zeros((n1 + n2, n1 + n2))

    # Fill self distances
    res[:n1, :n1] = lg1
    res[n1:, n1:] = lg2

    # Fill cross-Pfam part
    for i, x in enumerate(tqdm(list(seq_ref1.keys()), leave=False, desc=f'{pf1}-{pf2}')):
        sm1 = dist1[i]
        for j, y in enumerate(seq_ref2.keys()):
            sm2 = dist2[j]
            val = sm1 + dst + sm2
            res[i, n1 + j] = val
            res[n1 + j, i] = val

    np.save(f"results_interfamily/data/{pf1}_{pf2}_merged.npy", res)

