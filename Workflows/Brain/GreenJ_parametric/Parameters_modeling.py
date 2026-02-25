import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import mannwhitneyu
import itertools

input_dir_HAS = "/home/gregory/Documents/Projects/CTV_RO1/Results/Parametric_editing/results_folder_edits_HAS_train"
input_dir_KO = "/home/gregory/Documents/Projects/CTV_RO1/Results/Parametric_editing/results_folder_edits_KO_train"
input_dir_AEM = "/home/gregory/Documents/Projects/CTV_RO1/Results/Parametric_editing/results_folder_edits_AEM_train"

moving_barrier = ['Midline', 'Ventricles']
soft_barrier = ['Brainstem', 'Optic structure']

bm0_all = []
bm1_all = []
res0_all = []
res1_all = []
red0_all = []
red1_all = []
vol_diff_all = []
isodistance_all = []
labels = []
for i, input_dir in enumerate([input_dir_HAS, input_dir_KO, input_dir_AEM]):

    labels.append(f'MD {i+1}')

    # save numpy arrays
    isodistance_PIDs = np.load(os.path.join(input_dir, "isodistance_PIDs.npy"))

    barrier_movement_PIDs = np.load(os.path.join(input_dir, "barrier_movement_PIDs.npy"))
    resistance_PIDs = np.load(os.path.join(input_dir, "resistance_PIDs.npy"))

    distance_barrier_movement = np.load(os.path.join(input_dir, "distance_barrier_movement.npy"))
    distance_barrier_soft = np.load(os.path.join(input_dir, "distance_barrier_soft.npy"))

    margin_reduction_soft = np.load(os.path.join(input_dir, "margin_reduction_soft.npy"))

    volume_diff = np.load(os.path.join(input_dir, "volume_diff_PIDs.npy"))

    # j = 0
    # plt.figure(j)
    # plt.scatter(distance_barrier_soft[distance_barrier_soft[:, 0] <= isodistance_PIDs, 0], resistance_PIDs[distance_barrier_soft[:, 0] <= isodistance_PIDs, 0])
    # plt.xlabel(f'Distance {soft_barrier[0]}')
    # plt.ylabel(f'Resistance {soft_barrier[0]}')
    #
    # j += 1
    # plt.figure(j)
    # plt.scatter(distance_barrier_soft[distance_barrier_soft[:, 1] <= isodistance_PIDs, 1], resistance_PIDs[distance_barrier_soft[:, 1] <= isodistance_PIDs, 1])
    # plt.xlabel(f'Distance {soft_barrier[1]}')
    # plt.ylabel(f'Resistance {soft_barrier[1]}')
    #
    # j += 1
    # plt.figure(j)
    # plt.scatter(distance_barrier_movement[distance_barrier_movement[:, 0] <= isodistance_PIDs, 0],
    #             barrier_movement_PIDs[distance_barrier_movement[:, 0] <= isodistance_PIDs, 0])
    # plt.xlabel(f'Distance {moving_barrier[0]}')
    # plt.ylabel(f'Barrier movement {moving_barrier[0]}')
    #
    # j += 1
    # plt.figure(j)
    # plt.scatter(distance_barrier_movement[distance_barrier_movement[:, 1] <= isodistance_PIDs, 1],
    #             barrier_movement_PIDs[distance_barrier_movement[:, 1] <= isodistance_PIDs, 1])
    # plt.xlabel(f'Distance {moving_barrier[1]}')
    # plt.ylabel(f'Barrier movement {moving_barrier[1]}')

    # Collect data for boxplots (masked by isodistance)
    bm0_all.append(barrier_movement_PIDs[distance_barrier_movement[:, 0] <= isodistance_PIDs, 0])
    bm1_all.append(barrier_movement_PIDs[distance_barrier_movement[:, 1] <= isodistance_PIDs, 1])
    res0_all.append(resistance_PIDs[distance_barrier_soft[:, 0] <= isodistance_PIDs, 0])
    res1_all.append(resistance_PIDs[distance_barrier_soft[:, 1] <= isodistance_PIDs, 1])
    red0_all.append(margin_reduction_soft[distance_barrier_soft[:, 0] <= isodistance_PIDs, 0])
    red1_all.append(margin_reduction_soft[distance_barrier_soft[:, 1] <= isodistance_PIDs, 1])
    isodistance_all.append(isodistance_PIDs)
    vol_diff_all.append(volume_diff)

# j = 0
# plt.figure(j)
# plt.legend(labels)
#
# j += 1
# plt.figure(j)
# plt.legend(labels)
#
# j += 1
# plt.figure(j)
# plt.legend(labels)
#
# j += 1
# plt.figure(j)
# plt.legend(labels)

red_all = np.concatenate((red0_all[0], red0_all[1], red1_all[0], red1_all[1]))

plt.hist(red_all)
plt.xlabel('margin reduction (mm)')
plt.ylabel('frequency')

plt.figure()
plt.plot(res0_all[0], red0_all[0], 'ko')
plt.plot(res0_all[1], red0_all[1], 'ko')
#plt.plot(res0_all[2], red0_all[2], 'ko')
plt.plot(res1_all[0], red1_all[0], 'ko')
plt.plot(res1_all[1], red1_all[1], 'ko')
#plt.plot(res1_all[2], red1_all[2], 'ko')
plt.xlabel('resistance coefficient')
plt.ylabel('margin reduction (mm)')
plt.show()

def p_to_stars(p):
    if p <= 0.001:
        return '***'
    elif p <= 0.01:
        return '**'
    elif p <= 0.05:
        return '*'
    else:
        return 'ns'

colors = ['#1f77b4', '#2ca02c', '#d62728']

def add_boxplot_with_points(ax, data_all, labels, ylabel, alpha=0.4):
    positions = np.arange(1, len(data_all) + 1)
    base_colors = ['#1f77b4', '#2ca02c', '#d62728']

    # Create boxplot with patch_artist=True to allow colored boxes
    bp = ax.boxplot(data_all, positions=positions, labels=labels, patch_artist=True, showfliers=False)

    # Apply transparent colors to boxes
    for i, patch in enumerate(bp['boxes']):
        color = base_colors[i % len(base_colors)]
        patch.set_facecolor(color)
        patch.set_alpha(alpha)  # Set transparency
        patch.set_edgecolor('black')

    # Style whiskers, caps, medians
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='black', linewidth=1.5)

    # Add jittered scatter points
    for i, data in enumerate(data_all):
        x = np.random.normal(loc=positions[i], scale=0.05, size=len(data))  # jitter
        ax.scatter(x, data, alpha=0.6, s=30, color=base_colors[i % len(base_colors)])

    x_offset = 0.18  # horizontal shift to the right of the box

    for i, data in enumerate(data_all):
        median = np.median(data)
        ax.text(
            positions[i] + x_offset,
            median,
            f'{median:.2f}',
            ha='left',
            va='center',
            fontsize=13,
            fontweight='bold',
            color='black'
        )

    # Add significance bars
    y_max = max([max(data) for data in data_all]) * 1.1
    step = (y_max * 0.05)

    for idx, (i, j) in enumerate(itertools.combinations(range(len(data_all)), 2)):
        stat, p = mannwhitneyu(data_all[i], data_all[j], alternative='two-sided')
        stars = p_to_stars(p)
        y = y_max + idx * step
        x1, x2 = positions[i], positions[j]
        ax.plot([x1, x1, x2, x2], [y, y + step / 4, y + step / 4, y], lw=1.5, c='k')
        ax.text((x1 + x2) / 2, y + step / 4 + 0.01 * y_max, stars, ha='center', va='bottom', fontsize=16)

    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

# Plot setup
fig, axes = plt.subplots(1, 5, figsize=(24, 6))  # Adjusted height to 6 for spacing
axes = axes.flatten()

add_boxplot_with_points(axes[0], bm0_all, labels, f'Positional uncertainty {moving_barrier[0]}')
add_boxplot_with_points(axes[1], bm1_all, labels, f'Positional uncertainty {moving_barrier[1]}')
add_boxplot_with_points(axes[2], res0_all, labels, f'Resistance in {soft_barrier[0]}')
add_boxplot_with_points(axes[3], res1_all, labels, f'Resistance in {soft_barrier[1]}')
add_boxplot_with_points(axes[4], isodistance_all, labels, 'Iso-distance')
# add_boxplot_with_points(axes[5], vol_diff_all, labels, 'Volume difference')

plt.tight_layout()
fig.savefig("parameters.pdf", bbox_inches='tight')
plt.show()




