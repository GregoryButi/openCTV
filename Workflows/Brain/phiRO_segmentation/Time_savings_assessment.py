import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Your rating lists (as strings)
#Ratings_manual_without_edits = [2, 1.5, 1.5, 2.5, 3, 1.5, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 3, 3, 2]
Ratings_auto_without_edits = [1.5, 1.5, 1.5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
Ratings_auto_with_edits = [2, 2, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 3, 2]

# Convert to string for categorical processing
#manual_str = [str(r) for r in Ratings_manual_without_edits]
auto_no_edit_str = [str(r) for r in Ratings_auto_without_edits]
auto_with_edit_str = [str(r) for r in Ratings_auto_with_edits]

# Get all unique rating categories in sorted order
#all_ratings = sorted(set(manual_str + auto_no_edit_str + auto_with_edit_str), key=lambda x: float(x))
all_ratings = sorted(set(auto_no_edit_str + auto_with_edit_str), key=lambda x: float(x))

# Count occurrences
#manual_counts = Counter(manual_str)
auto_no_edit_counts = Counter(auto_no_edit_str)
auto_with_edit_counts = Counter(auto_with_edit_str)

# Get counts in consistent order
#manual_vals = [manual_counts.get(r, 0) for r in all_ratings]
auto_no_edit_vals = [auto_no_edit_counts.get(r, 0) for r in all_ratings]
auto_with_edit_vals = [auto_with_edit_counts.get(r, 0) for r in all_ratings]

# Set bar positions
x = np.arange(len(all_ratings))
bar_width = 0.25

# Plot
plt.figure(figsize=(10, 6))
#plt.bar(x - bar_width, manual_vals, width=bar_width, label='Manual (no edits)', edgecolor='black')
plt.bar(x, auto_no_edit_vals, width=bar_width, label='Auto (no edits)', edgecolor='black')
plt.bar(x + bar_width, auto_with_edit_vals, width=bar_width, label='Auto (with edits)', edgecolor='black')

# Customize x-axis
plt.xticks(x, all_ratings)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Counts (Categorical)')
plt.legend()
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()