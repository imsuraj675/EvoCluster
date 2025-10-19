import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from homology_corr import hm_correlation_analysis

# ===========================
# Input handling
# ===========================
if len(sys.argv) < 2:
    print("Usage: python run_hm_analysis.py <PfamID>")
    sys.exit(1)

pfam_id = sys.argv[1]
shuffle = 'N'
colattn = 'N'

# ===========================
# File paths
# ===========================
seed_file = f"{pfam_id}/{pfam_id}.aln"
full_file = f"{pfam_id}_f/{pfam_id}_f.aln"

if not os.path.exists(seed_file):
    raise FileNotFoundError(f"Seed alignment not found at {seed_file}")
if not os.path.exists(full_file):
    raise FileNotFoundError(f"Full alignment not found at {full_file}")

# ===========================
# Models and plotting setup
# ===========================
models = ["esm2", "esmc", "pt", "msa"]
colors = {
    "esm2": "#4fa3ff",     # blue
    "esmc": "#ffb347",     # orange
    "pt": "#8cff9f",       # green
    "msa": "#ff66c4"       # pink
}

# ===========================
# Alignment types
# ===========================
alignments = {
    "Seed": seed_file,
    "Full": full_file
}

results = {align_type: {} for align_type in alignments}

# ===========================
# Run correlation analysis
# ===========================
for align_type, fasta_aln_file in alignments.items():
    for model in models:
        print(f"Running correlation analysis for {model.upper()} using {align_type} alignment ({fasta_aln_file}) ...")

        rho_layer_corr, pearson_layer_corr = hm_correlation_analysis(
            fasta_aln_file, model, shuffle, colattn
        )
        results[align_type][model] = {
            "order": np.array(rho_layer_corr),
            "mag": np.array(pearson_layer_corr)
        }

print("\nAll models and alignments processed successfully.\n")

# ===========================
# Plot results (4 graphs)
# ===========================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_config = [
    ("Seed", "order", "RSS (Order) - Seed Alignment"),
    ("Seed", "mag", "RSS (Magnitude) - Seed Alignment"),
    ("Full", "order", "RSS (Order) - Full Alignment"),
    ("Full", "mag", "RSS (Magnitude) - Full Alignment")
]

for ax, (align_type, metric, title) in zip(axes.flatten(), plot_config):
    for model, vals in results[align_type].items():
        depth = np.arange(1, len(vals[metric]) + 1) / len(vals[metric]) * 100
        ax.plot(depth, vals[metric], label=model.upper(), color=colors[model], linewidth=2)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Depth (% of total layers)")
    ax.set_ylabel(metric.upper())
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.show()