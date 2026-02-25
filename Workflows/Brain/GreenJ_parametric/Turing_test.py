
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.stats import mannwhitneyu
from statsmodels.stats.power import TTestIndPower

Ratings_clinical_MD1 = np.array([2, 2, 1, 1, 2, 2, 2, 3, 2, 2])
Ratings_auto_MD1 =   np.array([2, 2, 2, 2, 2, 2, 3, 3, 3, 2])

Ratings_clinical_MD3 = np.array([2, 1, 3, 3, 3, 3, 3, 3, 3, 2])
Ratings_auto_MD3 = np.array([2, 2, 2, 3, 3, 2, 3, 2, 3, 1])

# Rating categories
all_ratings = np.array([0, 1, 2, 3])

# Count occurrences
def count_ratings(ratings, categories):
    return np.array([(rating == ratings).sum() for rating in categories])

counts_clinical_MD1 = count_ratings(Ratings_clinical_MD1, all_ratings)
counts_auto_MD1     = count_ratings(Ratings_auto_MD1, all_ratings)
counts_clinical_MD3 = count_ratings(Ratings_clinical_MD3, all_ratings)
counts_auto_MD3     = count_ratings(Ratings_auto_MD3, all_ratings)

# Plot setup
bar_width = 0.35
rating_descriptions = ['0\nUnusable', '1\nMajor edits', '2\nMinor edits', '3\nAcceptable']

fig = plt.figure(figsize=(10, 6))

# Colors (base + lighter shades for MD3)
clinical_color = '#1f77b4'   # darker blue (MD1)
clinical_shade = '#6baed6'   # lighter blue (MD3)
auto_color     = '#ff7f0e'   # darker orange (MD1)
auto_shade     = '#fdae6b'   # lighter orange (MD3)

# Clinical (stacked MD1 + MD3)
plt.bar(all_ratings - bar_width/2, counts_clinical_MD1,
        width=bar_width, label='Clinical CTV rated by MD A', color=clinical_color, edgecolor='black')
plt.bar(all_ratings - bar_width/2, counts_clinical_MD3,
        width=bar_width, bottom=counts_clinical_MD1, label='Clinical CTV rated by MD B',
        color=clinical_shade, edgecolor='black', hatch='///')

# Auto (stacked MD1 + MD3)
plt.bar(all_ratings + bar_width/2, counts_auto_MD1,
        width=bar_width, label='Model CTV rated by MD A', color=auto_color, edgecolor='black')
plt.bar(all_ratings + bar_width/2, counts_auto_MD3,
        width=bar_width, bottom=counts_auto_MD1, label='Model CTV rated by MD B',
        color=auto_shade, edgecolor='black', hatch='///')

# Customize axes
plt.xticks(all_ratings, rating_descriptions, fontsize=14)
plt.xlabel('Rating', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
fig.savefig("Ratings.pdf", bbox_inches='tight')
plt.show()

Ratings_clinical_all = np.concatenate((Ratings_clinical_MD1, Ratings_clinical_MD3))
Ratings_model_all = np.concatenate((Ratings_auto_MD1, Ratings_auto_MD3))

np.mean(Ratings_clinical_all)
np.mean(Ratings_model_all)

# ---- Stats ----
stat, p = mannwhitneyu(Ratings_clinical_all, Ratings_model_all, alternative='two-sided')
print(f"MD1 - two-sided test: Statistic={stat}, p-value={p}")

stat, p = mannwhitneyu(Ratings_clinical_all, Ratings_model_all, alternative='greater')
print(f"MD1 - greater test: Statistic={stat}, p-value={p}")

power_analysis = TTestIndPower()
sample_size = power_analysis.solve_power(effect_size=0.5, power=0.8, alpha=0.05)
print(f"Samples per group (power analysis): {round(sample_size)}")
