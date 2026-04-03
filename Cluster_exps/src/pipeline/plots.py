import logging
import os
from collections import Counter

import numpy as np

PLOT_LEVEL_ORDER = ["coarse", "fine", "adaptive", "mid"]
PLOT_COLORS = {
    "coarse": "#4e79a7",
    "fine": "#59a14f",
    "adaptive": "#e15759",
    "mid": "#f28e2b",
}
MAX_RANK_POINTS = 1500


def _cluster_sizes_from_labels(labels):
    counts = Counter(int(label) for label in labels if int(label) >= 0)
    return np.array(sorted(counts.values(), reverse=True), dtype=np.int32)


def save_diagnostic_plots(results, results_path, organism, logger=None):
    log = logger or logging.getLogger("multiscale")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.debug(f"Matplotlib unavailable; skipping diagnostic plots: {exc}")
        return []

    plots_dir = os.path.join(results_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    saved_paths = []

    stability = results.get("stability", {})
    metrics = results.get("metrics", {})
    labels_all = results.get("labels_all", {})
    stage_summaries = results.get("stage_summaries", [])

    per_level_scores = stability.get("per_level_scores", [])
    selected_levels = set(stability.get("selected_levels", []))

    if per_level_scores:
        resolutions = np.array([row["resolution"] for row in per_level_scores], dtype=float)
        ks = np.array([row["n_clusters"] for row in per_level_scores], dtype=float)
        composites = np.array([row["composite"] for row in per_level_scores], dtype=float)
        feasible = np.array([row["feasible"] for row in per_level_scores], dtype=bool)
        selected_mask = np.array([idx in selected_levels for idx in range(len(per_level_scores))], dtype=bool)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

        axes[0].plot(resolutions, ks, color="#4e79a7", linewidth=1.5, alpha=0.8)
        axes[0].scatter(resolutions[feasible], ks[feasible], s=30, color="#59a14f", label="Feasible")
        axes[0].scatter(resolutions[~feasible], ks[~feasible], s=22, color="#bab0ab", label="Infeasible")
        axes[0].scatter(
            resolutions[selected_mask],
            ks[selected_mask],
            s=90,
            facecolor="none",
            edgecolor="#e15759",
            linewidth=2,
            label="Selected",
        )
        axes[0].set_title("Scale Discovery and Selection")
        axes[0].set_ylabel("Clusters (K)")
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        feasible_comp = np.where(feasible, composites, np.nan)
        axes[1].plot(resolutions, feasible_comp, color="#f28e2b", linewidth=1.5)
        axes[1].scatter(resolutions[feasible], composites[feasible], s=30, color="#f28e2b")
        axes[1].scatter(
            resolutions[selected_mask],
            composites[selected_mask],
            s=90,
            facecolor="none",
            edgecolor="#e15759",
            linewidth=2,
        )
        axes[1].set_xlabel("Resolution / Scale")
        axes[1].set_ylabel("Composite score")
        axes[1].grid(alpha=0.25)

        path = os.path.join(plots_dir, f"{organism}_scale_selection.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

    if metrics:
        ordered_levels = [level for level in PLOT_LEVEL_ORDER if level in metrics]
        if ordered_levels:
            pairwise_f1 = [metrics[level]["pairwise"]["f1"] for level in ordered_levels]
            ami = [metrics[level]["AMI"] for level in ordered_levels]

            fig, ax1 = plt.subplots(figsize=(9, 5), constrained_layout=True)
            x = np.arange(len(ordered_levels))
            bars = ax1.bar(x, pairwise_f1, color=[PLOT_COLORS.get(level, "#4e79a7") for level in ordered_levels])
            ax1.set_xticks(x, ordered_levels)
            ax1.set_ylim(0, 1)
            ax1.set_ylabel("Pairwise F1")
            ax1.set_title("Output-Level Quality")
            ax1.grid(axis="y", alpha=0.25)
            for bar, score in zip(bars, pairwise_f1):
                ax1.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.3f}", ha="center", va="bottom", fontsize=9)

            ax2 = ax1.twinx()
            ax2.plot(x, ami, color="#e15759", marker="o", linewidth=1.8)
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("AMI", color="#e15759")
            ax2.tick_params(axis="y", colors="#e15759")

            path = os.path.join(plots_dir, f"{organism}_output_quality.png")
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(path)

    if labels_all:
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        any_series = False
        levels_to_plot = [level for level in PLOT_LEVEL_ORDER if level in labels_all][:3]
        for level_name in levels_to_plot:
            labels = labels_all[level_name]
            sizes = _cluster_sizes_from_labels(labels)
            if sizes.size == 0:
                continue
            shown = min(sizes.size, MAX_RANK_POINTS)
            ranks = np.arange(1, shown + 1)
            ax.plot(
                ranks,
                sizes[:shown],
                linewidth=2.0,
                label=f"{level_name} (K={sizes.size})",
                color=PLOT_COLORS.get(level_name, "#4e79a7"),
            )
            any_series = True

        if any_series:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"Cluster rank (top {MAX_RANK_POINTS:,} max)")
            ax.set_ylabel("Cluster size")
            ax.set_title("Cluster Size Rank Plot")
            ax.grid(alpha=0.25, which="both")
            ax.legend()

            path = os.path.join(plots_dir, f"{organism}_cluster_size_rank.png")
            fig.savefig(path, dpi=180, bbox_inches="tight")
            saved_paths.append(path)
        plt.close(fig)

    if labels_all:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        any_series = False
        levels_to_plot = [level for level in PLOT_LEVEL_ORDER if level in labels_all][:3]
        bins = np.logspace(0, 3, 25)

        for level_name in levels_to_plot:
            labels = labels_all[level_name]
            sizes = _cluster_sizes_from_labels(labels)
            if sizes.size == 0:
                continue
            clipped = np.clip(sizes, bins[0], bins[-1])
            ax.hist(
                clipped,
                bins=bins,
                histtype="step",
                linewidth=2.0,
                label=f"{level_name} (median={np.median(sizes):.0f})",
                color=PLOT_COLORS.get(level_name, "#4e79a7"),
            )
            any_series = True

        if any_series:
            ax.set_xscale("log")
            ax.set_xlabel("Cluster size")
            ax.set_ylabel("Number of clusters")
            ax.set_title("Cluster Size Distribution")
            ax.grid(alpha=0.25, which="both")
            ax.legend()

            path = os.path.join(plots_dir, f"{organism}_cluster_size_distribution.png")
            fig.savefig(path, dpi=180, bbox_inches="tight")
            saved_paths.append(path)
        plt.close(fig)

    if stage_summaries:
        stages = [row["stage"] for row in stage_summaries]
        merges = [row["n_merges"] for row in stage_summaries]
        rejects = [row["n_rejected"] for row in stage_summaries]
        k_before = [row["fine_k_before"] for row in stage_summaries]
        k_after = [row["fine_k_after"] for row in stage_summaries]

        fig, ax1 = plt.subplots(figsize=(9, 5), constrained_layout=True)
        x = np.arange(len(stages))
        width = 0.35
        ax1.bar(x - width / 2, merges, width=width, color="#59a14f", label="Merged")
        ax1.bar(x + width / 2, rejects, width=width, color="#bab0ab", label="Rejected")
        ax1.set_xticks(x, [f"Stage {stage}" for stage in stages])
        ax1.set_ylabel("Pair count")
        ax1.set_title("Merge Stage Summary")
        ax1.grid(axis="y", alpha=0.25)
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(x, k_before, color="#4e79a7", marker="o", linewidth=1.5, label="K before")
        ax2.plot(x, k_after, color="#e15759", marker="o", linewidth=1.5, label="K after")
        ax2.set_ylabel("Cluster count (K)")

        path = os.path.join(plots_dir, f"{organism}_merge_stages.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

    # ── Extended evaluation plots ──
    # Size-binned F1 bar chart (using "fine" level if available)
    fine_metrics = metrics.get("fine", {})
    ext = fine_metrics.get("extended", {})
    size_bins = ext.get("size_binned", {})

    if size_bins:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            bin_names = list(size_bins.keys())
            precisions = [size_bins[b].get("precision", 0) for b in bin_names]
            recalls = [size_bins[b].get("recall", 0) for b in bin_names]
            f1s = [size_bins[b].get("f1", 0) for b in bin_names]
            n_prots = [size_bins[b].get("n_proteins", 0) for b in bin_names]

            fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
            x = np.arange(len(bin_names))
            w = 0.25
            ax.bar(x - w, precisions, width=w, color="#4e79a7", label="Precision")
            ax.bar(x,     recalls,    width=w, color="#59a14f", label="Recall")
            ax.bar(x + w, f1s,        width=w, color="#e15759", label="F1")
            ax.set_xticks(x, [f"{b}\n({n})" for b, n in zip(bin_names, n_prots)], fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score")
            ax.set_title("Size-Binned Evaluation (fine level)")
            ax.grid(axis="y", alpha=0.25)
            ax.legend()

            for i, f1_val in enumerate(f1s):
                ax.text(x[i] + w, f1_val + 0.02, f"{f1_val:.2f}", ha="center", va="bottom", fontsize=7)

            path = os.path.join(plots_dir, f"{organism}_size_binned_f1.png")
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(path)
        except Exception:
            pass

    # Split/merge error summary
    sm = ext.get("split_merge", {})
    if sm and sm.get("n_true_groups", 0) > 0:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

            # Pie: splits
            n_split = sm.get("n_split_groups", 0)
            n_total = sm.get("n_true_groups", 1)
            axes[0].pie(
                [n_split, n_total - n_split],
                labels=[f"Split ({n_split})", f"Intact ({n_total - n_split})"],
                colors=["#e15759", "#59a14f"],
                autopct="%1.0f%%", startangle=90,
            )
            axes[0].set_title("True Groups: Split vs Intact")

            # Pie: merges
            n_merge = sm.get("n_merge_clusters", 0)
            n_pred = sm.get("n_pred_clusters", 1)
            axes[1].pie(
                [n_merge, max(n_pred - n_merge, 0)],
                labels=[f"Impure ({n_merge})", f"Pure ({max(n_pred - n_merge, 0)})"],
                colors=["#f28e2b", "#4e79a7"],
                autopct="%1.0f%%", startangle=90,
            )
            axes[1].set_title("Pred Clusters: Impure vs Pure")

            path = os.path.join(plots_dir, f"{organism}_split_merge.png")
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(path)
        except Exception:
            pass

    for path in saved_paths:
        log.info(f"Saved plot to: {path}")

    return saved_paths

