import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ca = np.loadtxt('cluster_assignments.dat').astype(int)

counts = np.bincount(ca)[1:]
labels = list(range(1, len(counts)+1))
ymax = np.max(counts)+300


# plt.rcParams["font.size"] = 10
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(4,2), tight_layout=True)
b = ax.bar(labels, counts, color='seagreen', edgecolor='black')

ax.bar_label(b, label_type='edge', fontsize=10)

# ax.set_xlabel('Cluster')
# ax.set_ylabel('Cluster Size')
ax.set_xlim(0.5, len(counts)+0.5)
ax.set_ylim(0, ymax+ymax/10)
# ax.yaxis.set_major_locator(MultipleLocator(1000))
# ax.yaxis.set_minor_locator(MultipleLocator(200))
ax.set_xticks(labels)
# ax.tick_params(which='both', width=2)
# ax.grid(axis='y', linestyle='dotted')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
plt.savefig('cluster_sizes.png', dpi=300)
plt.show()
