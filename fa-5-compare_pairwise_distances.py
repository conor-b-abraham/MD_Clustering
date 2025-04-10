import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
from scipy.spatial.distance import squareform
from mpl_toolkits.axes_grid1 import make_axes_locatable
# ------------------------------------------------------------------------------ #
#                              F U N C T I O N S                                 #
# ------------------------------------------------------------------------------ #
def parse_input():
    helptext = ["To use this program you must specify two pairwise distance files that\n",
            "were generated using 5-compare_pairwise_distances.py.\n",
            "    i.e.     > python 5-compare_pairwise_distances.py [path1]/pairwise_distances.dat [path2]/pairwise_distances.dat\n"]
    stop = False
    if len(sys.argv) < 2:
        print(f'INPUT ERROR: files not specified')
        stop = True
    elif len(sys.argv) == 2 and sys.argv[1] =='-h' or sys.argv[1] == '--help':
        stop = True
    elif len(sys.argv) == 3:
        files = [os.path.abspath(file) for file in [sys.argv[1], sys.argv[2]]]
        if not os.path.isfile(files[0]):
            print(f'INPUT ERROR: {files[0]} not found')
            stop = True
        if not os.path.isfile(files[1]):
            print(f'INPUT ERROR: {files[1]} not found')
            stop = True
    if stop:
        sys.exit(f'{"".join(helptext)}')
    else:
        print(f'Comparing the following pairwise distances:\n    {files[0]}\n    {files[1]}\n')
    return files

def get_names(s1, s2):
    n1, n2 = [],[]
    toggle = False
    for i, v1 in enumerate(s1):
        if len(s2) > i and v1 != s2[i]:
            toggle = True
            n1.append(v1)
            n2.append(s2[i])
        elif toggle == True:
            break
    if '/' in n1:
        n1 = n1[:n1.index('/')]
    if '/' in n2:
        n2 = n2[:n2.index('/')]
    n1 = ''.join(n1)
    n2 = ''.join(n2)
    ask = input(f"The name related to {s1} is {n1}\nThe name related to {s2} is {n2}\n\nWould you like to change the names [y/N]? ")
    if ask.upper() == 'Y' or ask.upper() == 'YES':
        n1 = input(f"What should the first name be? ")
        n2 = input(f"What should the second name be? ")
    return n1, n2

# ------------------------------------------------------------------------------ #
#                                   M A I N                                      #
# ------------------------------------------------------------------------------ #
print("------------------------------------------------------------------------------")
print("           C O M P A R E    P A I R W I S E    D I S T A N C E S              ")
print(" ")
print(" Compare pairwise distances between atom selection for clustering for two     ")
print(" different trajectories.")
print("------------------------------------------------------------------------------\n")
files = parse_input()
name1, name2 = get_names(files[0], files[1])

print("\nLoading Pairwise Distances")
pd1 = np.loadtxt(files[0])
pd2 = np.loadtxt(files[1])

print(f"\nComputing D_i-D_j where i is {name1} and j is {name2}")
pd1av = np.mean(pd1, axis=0)
pd2av = np.mean(pd2, axis=0)
avdiff = pd1av-pd2av
avdiff = squareform(avdiff)

print(f"\nPlotting Results")

bound = np.ceil(max([np.abs(np.max(avdiff)),np.abs(np.min(avdiff))]))

plt.rcParams["font.size"] = 12
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 2

fig, ax = plt.subplots(figsize=(14,14))

im = ax.imshow(avdiff, cmap="RdBu", vmin=-bound, vmax=bound)

ax.set_xlabel('Residue Number',fontsize=18)
ax.set_ylabel('Residue Number',fontsize=18)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.set_ylabel(f'$D_i-D_j$ where $D_i$ '+f'is for {name1} and '+r'$D_j$ '+f'is for {name2}',fontsize=18)

ax.set_xticks(np.arange(avdiff.shape[1]))
ax.set_yticks(np.arange(avdiff.shape[0]))
ax.set_xticklabels(list(range(1,avdiff.shape[1]+1)), rotation=90)
ax.set_yticklabels(list(range(1,avdiff.shape[0]+1)))
#ax.spines[:].set_color('white')
ax.set_xticks(np.arange(avdiff.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(avdiff.shape[0]+1)-.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)
ax.tick_params(which='both', width=2)

plt.tight_layout()
plt.savefig('compare_pairwise_distances.png')
plt.show()
