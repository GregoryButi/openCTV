import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon
import numpy as np
import os

# results directory
results_folder = "/home/gregory/Documents/Projects/CTV_RO1/Results/New"

Barriers_DL_dice = np.load(os.path.join(results_folder, 'dice_barriers_DL.npy'))
Barriers_DL_surface_dice = np.load(os.path.join(results_folder, 'SDS_barriers_DL.npy'))
Barriers_DL_HD95 = np.load(os.path.join(results_folder, 'HD95_barriers_DL.npy'))

CTV_DL_dice = np.load(os.path.join(results_folder, 'dice_ctv_DL.npy'))
CTV_DL_dice = [i[0] for i in CTV_DL_dice]

CTV_DL_surface_dice = np.load(os.path.join(results_folder, 'SDS_ctv_DL.npy'))[..., 1]
CTV_DL_surface_dice = [i[0] for i in CTV_DL_surface_dice]

CTV_DL_HD95 = np.load(os.path.join(results_folder, 'HD95_ctv_DL.npy'))
CTV_DL_HD95 = [i[0] for i in CTV_DL_HD95]

CTV_TS_dice = np.load(os.path.join(results_folder, 'dice_ctv_TS.npy'))
CTV_TS_dice = [i[0] for i in CTV_TS_dice]

CTV_TS_surface_dice = np.load(os.path.join(results_folder, 'SDS_ctv_TS.npy'))[..., 1]
CTV_TS_surface_dice = [i[0] for i in CTV_TS_surface_dice]

CTV_TS_HD95 = np.load(os.path.join(results_folder, 'HD95_ctv_TS.npy'))
CTV_TS_HD95 = [i[0] for i in CTV_TS_HD95]


# Wilcoxon tests
dice_stat, dice_p = wilcoxon(CTV_DL_dice, CTV_TS_dice)
surface_dice_stat, surface_dice_p = wilcoxon(CTV_DL_surface_dice, CTV_TS_surface_dice)
hd95_stat, hd95_p = wilcoxon(CTV_DL_HD95, CTV_TS_HD95)

# Dataframe creation helper
def create_df(dl, ts, metric_name):
    return pd.DataFrame({
        "Value": dl + ts,
        "Model": ["Auto CTV"] * len(dl) + ["TotalSegmentator CTV"] * len(ts),
        "Metric": [metric_name] * (len(dl) + len(ts))
    })

# Create combined dataframe
df_dice = create_df(CTV_DL_dice, CTV_TS_dice, "Dice")
df_surface_dice = create_df(CTV_DL_surface_dice, CTV_TS_surface_dice, "Surface Dice")
df_hd95 = create_df(CTV_DL_HD95, CTV_TS_HD95, "HD95 [mm]")
df_all = pd.concat([df_dice, df_surface_dice, df_hd95])

# Plot
sns.set(style="white")
metrics = ["Dice", "Surface Dice", "HD95 [mm]"]
p_values = [dice_p, surface_dice_p, hd95_p]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, metric, pval in zip(axes, metrics, p_values):
    df_metric = df_all[df_all["Metric"] == metric]
    sns.violinplot(x="Model", y="Value", data=df_metric, inner=None, ax=ax, alpha=0.5)
    sns.boxplot(x="Model", y="Value", data=df_metric, width=0.2, ax=ax)

    ax.set_xlabel("", fontsize=16)
    ax.set_ylabel(metric, fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    # Annotate p-value
    #y_max = df_metric["Value"].max()
    #ax.text(0.5, y_max + (0.05 * y_max), f"p = {pval:.6f}", fontsize=14, ha='center')

plt.tight_layout()
#plt.savefig(os.path.join(results_folder, "CTV_comparison_plots.png"), format="png", dpi=300, bbox_inches="tight")
plt.show()