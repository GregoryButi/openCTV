import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D

# === Ablation Variants ===
variants = [
    "train", "valid",
    "train_margin_only", "valid_margin_only",
    "train_no_barriers", "valid_no_barriers"
]

# === Paths ===
def get_result_dirs(prefix):
    return {
        v: f"/home/gregory/Documents/Projects/CTV_RO1/Results/PeerReview/results_folder_edits_{prefix}_{v}"
        for v in variants
    }

input_variants = {
    "HAS": get_result_dirs("HAS"),
    "KO": get_result_dirs("KO"),
    "AEM": get_result_dirs("AEM"),
}

labels = ['MD 1', 'MD 2', 'MD 3']  # inter-MD labels

# === Load Metrics ===
def load_results(base_dir):
    dice = np.load(os.path.join(base_dir, "dice_PIDs.npy"))
    hd95 = np.load(os.path.join(base_dir, "HD95_PIDs.npy"))
    sds = np.load(os.path.join(base_dir, "SDS_PIDs.npy"))[:, 1]
    return dice, hd95, sds


dice_results_train = []
hd95_results_train = []
sds_results_train = []

dice_results_test = []
hd95_results_test = []
sds_results_test = []

for prefix in ["HAS", "KO", "AEM"]:
    dirs = input_variants[prefix]

    # Train
    dice_full, hd95_full, sds_full = load_results(dirs["train"])
    dice_margin, hd95_margin, sds_margin = load_results(dirs["train_margin_only"])
    dice_no_barriers, hd95_no_barriers, sds_no_barriers = load_results(dirs["train_no_barriers"])

    dice_results_train.append([dice_full, dice_margin, dice_no_barriers])
    hd95_results_train.append([hd95_full, hd95_margin, hd95_no_barriers])
    sds_results_train.append([sds_full, sds_margin, sds_no_barriers])

    # Test
    dice_full, hd95_full, sds_full = load_results(dirs["valid"])
    dice_margin, hd95_margin, sds_margin = load_results(dirs["valid_margin_only"])
    dice_no_barriers, hd95_no_barriers, sds_no_barriers = load_results(dirs["valid_no_barriers"])

    dice_results_test.append([dice_full, dice_margin, dice_no_barriers])
    hd95_results_test.append([hd95_full, hd95_margin, hd95_no_barriers])
    sds_results_test.append([sds_full, sds_margin, sds_no_barriers])


# === Plotting Function with Δ Annotations ===
def add_multi_boxplots(ax, data_all, labels_inter, labels_intra, ylabel, width=0.25, annotate_diff=True):
    n = len(data_all)  # number of MDs
    k = len(labels_intra)  # number of ablation variants

    # Position groups
    positions = [np.arange(n) * (k + 1) + j for j in range(k)]

    group_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Full, Margin-only, No barriers

    for i in range(n):  # loop over MDs
        y_max_case = []
        y_min_case = []

        for j, group in enumerate(data_all[i]):
            pos = positions[j][i]

            # Boxplot
            ax.boxplot(group, positions=[pos], widths=width,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=group_colors[j], alpha=0.4, color='black'),
                       capprops=dict(color='black'),
                       whiskerprops=dict(color='black'),
                       medianprops=dict(color='black'))

            # Scatter
            x = np.random.normal(loc=pos, scale=0.05, size=len(group))
            ax.scatter(x, group, alpha=0.6, s=30, color=group_colors[j])

            y_max_case.append(np.max(group))
            y_min_case.append(np.min(group))

        # Δ annotations (compare each variant vs Full model)
        if annotate_diff:
            y_max = max(y_max_case)
            y_min = min(y_min_case)

            ref_median = np.median(data_all[i][0])  # Full model reference

            for j in range(1, k):
                comp_median = np.median(data_all[i][j])
                delta = comp_median - ref_median

                pos1 = positions[0][i]
                pos2 = positions[j][i]

                y_pos = y_max + (j * 0.08) * (y_max - y_min)
                ax.plot([pos1, pos2], [y_pos, y_pos], color='black', lw=1.5)
                ax.text((pos1 + pos2) / 2, y_pos + 0.01 * (y_max - y_min),
                        f"Δ = {delta:.2f}", ha='center', va='bottom', fontsize=16, color='black')

    # X-axis labels
    mid_positions = [np.mean([positions[j][i] for j in range(k)]) for i in range(n)]
    ax.set_xticks(mid_positions)
    ax.set_xticklabels(labels_inter, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='y', labelsize=14)

    # return handles for global legend
    legend_elements = [
        Line2D([0], [0], marker='s', linestyle='None', label=lab,
               markerfacecolor='white', markeredgecolor=col, color=col, markersize=10)
        for lab, col in zip(labels_intra, group_colors)
    ]
    return legend_elements


# === Plot Ablation Train ===

Labels_intra = ['Full parametrization (m+d+$\\rho$)', 'Constrained (only m)', 'Unconstrained (only m)']

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

handles = add_multi_boxplots(axes[0], dice_results_train, labels,
                             Labels_intra, 'DSC')
#_ = add_multi_boxplots(axes[1], hd95_results_train, labels,
#                       Labels_intra, 'HD95 (mm)')
_ = add_multi_boxplots(axes[1], sds_results_train, labels,
                       Labels_intra, 'SDSC')

# Shared legend above
fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=18, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("similarity_ablation_train.pdf", bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

handles = add_multi_boxplots(axes[0], dice_results_test, labels,
                             Labels_intra, 'DSC')

_ = add_multi_boxplots(axes[1], sds_results_test, labels,
                       Labels_intra, 'SDSC')

# Shared legend above the figure
fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=18, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.92])  # leave space at top
fig.savefig("similarity_ablation_test.pdf", bbox_inches='tight')
plt.show()