import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

input_dir_HAS_edits_train = "/home/gregory/Documents/Projects/CTV_RO1/Results/PeerReview/results_folder_edits_HAS_train"
input_dir_KO_edits_train = "/home/gregory/Documents/Projects/CTV_RO1/Results/PeerReview/results_folder_edits_KO_train"
input_dir_AEM_edits_train = "/home/gregory/Documents/Projects/CTV_RO1/Results/PeerReview/results_folder_edits_AEM_train"

input_dir_HAS_edits_test = "/home/gregory/Documents/Projects/CTV_RO1/Results/PeerReview/results_folder_edits_HAS_valid"
input_dir_KO_edits_test = "/home/gregory/Documents/Projects/CTV_RO1/Results/PeerReview/results_folder_edits_KO_valid"
input_dir_AEM_edits_test = "/home/gregory/Documents/Projects/CTV_RO1/Results/PeerReview/results_folder_edits_AEM_valid"

moving_barrier = ['Midline', 'Ventricles_connected']
soft_barrier = ['Brainstem', 'Optic_structure']

dice_all_results_train = []
hd95_all_results_train = []
sds_all_results_train = []

dice_all_results_test = []
hd95_all_results_test = []
sds_all_results_test = []

labels = ['MD 1', 'MD 2', 'MD 3']
results_dirs_train = [input_dir_HAS_edits_train, input_dir_KO_edits_train, input_dir_AEM_edits_train]
results_dirs_test = [input_dir_HAS_edits_test, input_dir_KO_edits_test, input_dir_AEM_edits_test]

for results_dir_train, results_dir_test in zip(results_dirs_train, results_dirs_test):

    # From results_folder_*
    dice_all_results_train.append(np.load(os.path.join(results_dir_train, "dice_PIDs.npy")))
    hd95_all_results_train.append(np.load(os.path.join(results_dir_train, "HD95_PIDs.npy")))
    sds_all_results_train.append(np.load(os.path.join(results_dir_train, "SDS_PIDs.npy"))[:, 1])

    dice_all_results_test.append(np.load(os.path.join(results_dir_test, "dice_PIDs.npy")))
    hd95_all_results_test.append(np.load(os.path.join(results_dir_test, "HD95_PIDs.npy")))
    sds_all_results_test.append(np.load(os.path.join(results_dir_test, "SDS_PIDs.npy"))[:, 1])

print(np.median(np.array(list(dice_all_results_train[0])+list(dice_all_results_train[1])+list(dice_all_results_train[2]))))
print(np.median(np.array(list(hd95_all_results_train[0])+list(hd95_all_results_train[1])+list(hd95_all_results_train[2]))))
print(np.median(np.array(list(sds_all_results_train[0])+list(sds_all_results_train[1])+list(sds_all_results_train[2]))))
print(np.median(np.array(list(dice_all_results_test[0])+list(dice_all_results_test[1])+list(dice_all_results_test[2]))))
print(np.median(np.array(list(hd95_all_results_test[0])+list(hd95_all_results_test[1])+list(hd95_all_results_test[2]))))
print(np.median(np.array(list(sds_all_results_test[0])+list(sds_all_results_test[1])+list(sds_all_results_test[2]))))

def add_paired_boxplots(ax, data1_all, data2_all, labels_inter, labels_intra, ylabel, width=0.35, annotate_diff=True):
    n = len(data1_all)
    positions1 = np.arange(n) * 2 + 1
    positions2 = positions1 + 0.8

    base_colors = ['#1f77b4', '#2ca02c', '#d62728']
    transparent_colors = [to_rgba(c, alpha=0.4) for c in base_colors]

    marker1 = 'o'
    marker2 = 's'

    for i in range(n):
        # Boxplot group 1
        ax.boxplot(data1_all[i:i+1], positions=[positions1[i]], widths=width,
                   patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=transparent_colors[i], color='black'),
                   capprops=dict(color='black'),
                   whiskerprops=dict(color='black'),
                   medianprops=dict(color='black'))

        # Boxplot group 2
        ax.boxplot(data2_all[i:i+1], positions=[positions2[i]], widths=width,
                   patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=transparent_colors[i], color='black'),
                   capprops=dict(color='black'),
                   whiskerprops=dict(color='black'),
                   medianprops=dict(color='black'))

        # Jittered scatter points (larger and clearer)
        x1 = np.random.normal(loc=positions1[i], scale=0.05, size=len(data1_all[i]))
        x2 = np.random.normal(loc=positions2[i], scale=0.05, size=len(data2_all[i]))
        ax.scatter(x1, data1_all[i], alpha=0.6, s=40, color=base_colors[i], marker=marker1,
                   label=labels_intra[0] if i == 0 else "")
        ax.scatter(x2, data2_all[i], alpha=0.6, s=40, color=base_colors[i], marker=marker2,
                   label=labels_intra[1] if i == 0 else "")

        # Δ annotation
        if annotate_diff:
            y1 = np.median(data1_all[i])
            y2 = np.median(data2_all[i])
            y_max = max(np.max(data1_all[i]), np.max(data2_all[i]))
            y_min = min(np.min(data2_all[i]), np.min(data1_all[i]))
            y_pos = y_max + 0.05 * (y_max - y_min)
            delta = y2 - y1
            ax.plot([positions1[i], positions2[i]], [y_pos, y_pos], color='black', lw=1.5)
            ax.text((positions1[i] + positions2[i]) / 2, y_pos + 0.01 * (y_max - y_min),
                    f"Δ = {delta:.2f}", ha='center', va='bottom', fontsize=14, color='black')

    # X-axis labels
    mid_positions = (positions1 + positions2) / 2
    ax.set_xticks(mid_positions)
    ax.set_xticklabels(labels_inter, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='y', labelsize=14)

    # Marker legend
    legend_elements = [
        Line2D([0], [0], marker='o', linestyle='None', label=labels_intra[0],
               markerfacecolor='white', markeredgecolor='black', color='black', markersize=8),
        Line2D([0], [0], marker='s', linestyle='None', label=labels_intra[1],
               markerfacecolor='white', markeredgecolor='black', color='black', markersize=8)
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=13)

# === Plotting ===

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

add_paired_boxplots(axes2[0], dice_all_results_train, dice_all_results_test, labels,
                    ['Train', 'Test'], 'DSC')

add_paired_boxplots(axes2[1], hd95_all_results_train, hd95_all_results_test, labels,
                    ['Train', 'Test'], 'HD95 (mm)')

add_paired_boxplots(axes2[2], sds_all_results_train, sds_all_results_test, labels,
                    ['Train', 'Test'], 'SDSC')

plt.tight_layout()
fig2.savefig("similarity_train_test.pdf", bbox_inches='tight')
plt.show()