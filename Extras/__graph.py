#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

results_dir = "./data/results"
models = ["esmc", "esm2", "esm1", "pt", "msa"]
metrics = [("pear", "ESSr (Pearson)"), ("spear", "ESSp (Spearman)")]

def load_model_data_arrays(model, metric):
    """
    Return a list of 1D numpy arrays (one array per Pfam file) for given model and metric.
    """
    pattern = os.path.join(results_dir, f"{model}_*_{metric}.npy")
    files = sorted(glob(pattern))
    arrays = []
    for f in files:
        try:
            arr = np.load(f)
            arr = np.asarray(arr).ravel()
            arrays.append(arr)
            print(f"📦 Loaded {f} (shape = {arr.shape})")
        except Exception as e:
            print(f"⚠️ Failed to load {f}: {e}")
    return arrays  # list of 1D arrays, lengths may vary

# ------------------------------
# 1) Boxplots per model × metric: one box per layer
# ------------------------------
for model in models:
    for metric_key, metric_label in metrics:
        arrays = load_model_data_arrays(model, metric_key)
        if not arrays:
            print(f"⚠️ No files for {model} - {metric_key}, skipping.")
            continue

        # Align to max length among Pfams for this model
        max_len = max(len(a) for a in arrays)
        padded = [np.pad(a, (0, max_len - len(a)), constant_values=np.nan) for a in arrays]
        # For each layer index, collect values across Pfams (ignore NaNs)
        per_layer_vals = []
        for layer_idx in range(max_len):
            vals = [arr[layer_idx] for arr in padded if not np.isnan(arr[layer_idx])]
            # If no values for this layer (unlikely), keep an empty list to avoid crash
            per_layer_vals.append(np.array(vals) if vals else np.array([]))

        # Remove trailing layers that have zero counts (all empty) if any
        valid_layer_count = sum(1 for v in per_layer_vals if v.size > 0)
        if valid_layer_count < len(per_layer_vals):
            per_layer_vals = [v for v in per_layer_vals if v.size > 0]
            max_len = len(per_layer_vals)

        # Create boxplot: one box per layer
        plt.figure(figsize=(max(8, max_len*0.3), 5))
        # matplotlib.boxplot expects non-empty sequences; handle possible empty slots by replacing with tiny arrays
        safe_data = [v if v.size>0 else np.array([np.nan]) for v in per_layer_vals]
        bxp = plt.boxplot(safe_data, patch_artist=True, positions=range(max_len), widths=0.6, manage_ticks=False)

        plt.title(f"{metric_label} distribution across layers for {model.upper()}")
        plt.xlabel("Layer index")
        plt.ylabel(metric_label)
        plt.xticks(range(max_len), [str(i) for i in range(max_len)], rotation=45)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        out_fname = f"{model}_{metric_key}_layers_boxplot.png"
        plt.savefig(out_fname, dpi=300)
        plt.close()
        print(f"📦 Saved {out_fname} (boxes = {max_len})")

# ------------------------------
# 2) Mean layer-wise line plots across models (as percent depth)
# ------------------------------
for metric_key, metric_label in metrics:
    plt.figure(figsize=(10, 6))
    legend_models = []
    for model in models:
        arrays = load_model_data_arrays(model, metric_key)
        if not arrays:
            continue
        max_len = max(len(a) for a in arrays)
        padded = [np.pad(a, (0, max_len - len(a)), constant_values=np.nan) for a in arrays]
        mean_vals = np.nanmean(np.vstack(padded), axis=0)  # shape (max_len,)
        # convert layer indices to % depth
        x_percent = np.linspace(0, 100, len(mean_vals))
        plt.plot(x_percent, mean_vals, label=model.upper(), linewidth=2)
        legend_models.append(model.upper())

    if not legend_models:
        print(f"⚠️ No data for metric {metric_key} across any model, skipping mean plot.")
        continue

    plt.title(f"Mean {metric_label} vs. Layer Depth (%)")
    plt.xlabel("Layer Depth (%)")
    plt.ylabel(metric_label)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_fname = f"{metric_key}_mean_layerwise.png"
    plt.savefig(out_fname, dpi=300)
    plt.close()
    print(f"📈 Saved {out_fname}")

print("\n✅ Done — created boxplots (one box per layer) and mean-layer plots.")
