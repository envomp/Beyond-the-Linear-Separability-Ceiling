import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# --- 1. Define the Data based on the Bongard task structure ---
np.random.seed(42)

# 6 positive example embeddings, clustered around a point
X_pos_support = np.random.randn(6, 2) * 0.4 + [1.2, 2.8]
# 6 negative example embeddings, clustered around another point
X_neg_support = np.random.randn(6, 2) * 0.4 + [2.8, 1.2]

# --- 2. Calculate the LSC (Nearest Centroid) ---
c_pos = X_pos_support.mean(axis=0)  # Positive Centroid
c_neg = X_neg_support.mean(axis=0)  # Negative Centroid

# The LSC "probe" is the perpendicular bisector of the centroids
mid_point = (c_pos + c_neg) / 2
slope = (c_neg[1] - c_pos[1]) / (c_neg[0] - c_pos[0])
slope_perp = -1 / slope
intercept_perp = mid_point[1] - slope_perp * mid_point[0]

x_vals = np.array([0, 4])
y_vals_boundary = slope_perp * x_vals + intercept_perp

# Define two query points. Both *truly* belong to the "Positive" class
# (a) A query where the LSC probe succeeds
query_lsc_correct = np.array([1.5, 2.5])
# (b) A query where the LSC probe fails (it's on the wrong side)
query_lsc_fails = np.array([2.5, 1.8])

# --- 3. Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), sharey=True, sharex=True)
plot_lims = (0, 4)


def plot_base(ax):
    """Plots the common elements: support sets, centroids, and LSC boundary."""
    ax.set_xlim(plot_lims)
    ax.set_ylim(plot_lims)
    ax.set_xlabel('Reduced dimension 1', fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.6)

    # Plot support sets (lightly)
    ax.scatter(X_pos_support[:, 0], X_pos_support[:, 1], c='#0072B2', marker='o', s=50, alpha=0.3, label='Positive examples')
    ax.scatter(X_neg_support[:, 0], X_neg_support[:, 1], c='#D95319', marker='o', s=50, alpha=0.3, label='Negative examples')

    # Plot Centroids (strongly)
    ax.scatter(c_pos[0], c_pos[1], c='#0072B2', marker='X', s=250, edgecolors='k', zorder=10, label='Positive centroid ($\mathbf{c}_P$)')
    ax.scatter(c_neg[0], c_neg[1], c='#D95319', marker='X', s=250, edgecolors='k', zorder=10, label='Negative centroid ($\mathbf{c}_N$)')

    # Plot the LSC Boundary and fill
    ax.plot(x_vals, y_vals_boundary, 'k--', label='LSC probe (boundary)', zorder=5)
    ax.fill_between(x_vals, y_vals_boundary, 4, color='#0072B2', alpha=0.1, label='LSC "positive" guess area')
    ax.fill_between(x_vals, y_vals_boundary, 0, color='#D95319', alpha=0.1, label='LSC "negative" guess area')


# --- Panel (a): LSC Probe (Setup) ---
ax = axes[0]
plot_base(ax)
ax.set_ylabel('Reduced dimension 2', fontsize=12)
ax.set_title('(a) LSC probe', fontsize=12)

# Plot a query where LSC is correct
ax.scatter(query_lsc_correct[0], query_lsc_correct[1], c='green', marker='P', s=350, edgecolors='k', zorder=12, label='Query ($\mathbf{v}_Q$)')
ax.text(3.9, 3.8, "LSC guess: correct", ha='right', va='top', fontsize=11, color='k', weight='bold')

# --- Panel (b): Alignment Gap ---
ax = axes[1]
plot_base(ax)
ax.set_title('(b) Alignment gap ($acc_{gen} < LSC$)', fontsize=12)

# Plot the same query where LSC is correct
ax.scatter(query_lsc_correct[0], query_lsc_correct[1], c='green', marker='P', s=350, edgecolors='k', zorder=12, label='Query ($\mathbf{v}_Q$)')
ax.text(3.9, 3.8, "LSC guess: correct", ha='right', va='top', fontsize=11, color='k', weight='bold')
# ...but the generative model fails!
ax.text(3.9, 3.5, "Generative output: wrong", ha='right', va='top', fontsize=11, color='red', weight='bold')

# --- Panel (c): Surpassing the Ceiling ---
ax = axes[2]
plot_base(ax)
ax.set_title('(c) Surpassing the ceiling ($acc_{gen} > LSC$)', fontsize=12)

# Plot a query where LSC fails
ax.scatter(query_lsc_fails[0], query_lsc_fails[1], c='green', marker='P', s=350, edgecolors='k', zorder=12, label='Query ($\mathbf{v}_Q$)')
ax.text(3.9, 3.8, "LSC guess: wrong", ha='right', va='top', fontsize=11, color='red', weight='bold')
# ...but the generative model succeeds!
ax.text(3.9, 3.5, "Generative output: correct", ha='right', va='top', fontsize=11, color='green', weight='bold')

# Collect handles and labels from all axes, ensuring uniqueness
handles, labels = [], []
for ax in axes.flat: # Use .flat to iterate through all axes easily
    h, l = ax.get_legend_handles_labels()
    for handle, label in zip(h, l):
        if label not in labels:
            labels.append(label)
            handles.append(handle)

# Add the figure legend at the bottom center
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=11)

plt.tight_layout(rect=[0, 0.24, 1, 0.95])
plt.savefig("concept_illustration.png", dpi=300)
plt.show()
