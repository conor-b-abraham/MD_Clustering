import numpy as np
import MDAnalysis as mda
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

# ------------------------------------------------------------------------------ #
#                              F U N C T I O N S                                 #
# ------------------------------------------------------------------------------ #
def parse_input():
    helptext = ["To use this program you must specify the paths to where you performed the\n",
            "clusterings using 1-pairwise_distances, 2-cluster.jl, and 3-results.py.\n",
            "    i.e.     > python 4-match_clusterings.py [path1] [path2]\n",
            "This is the directory where you can find the cluster1, cluster2, ..., clusterN\n",
            "directories and pairwise_distances.dat, frame_IDs.dat, dRMSDs.jld, etc."]
    stop = False
    if len(sys.argv) < 2:
        print(f'INPUT ERROR: paths not specified')
        stop = True
    elif len(sys.argv) == 2 and sys.argv[1] =='-h' or sys.argv[1] == '--help':
        stop = True
    elif len(sys.argv) == 3:
        paths = [os.path.abspath(path) for path in [sys.argv[1], sys.argv[2]]]
        if not os.path.isdir(paths[0]):
            print(f'INPUT ERROR: {paths[0]} directory not found')
            stop = True
        if not os.path.isdir(paths[1]):
            print(f'INPUT ERROR: {paths[1]} directory not found')
            stop = True
    if stop:
        sys.exit(f'{"".join(helptext)}')
    return paths

def how_many_clusters(paths):
    nclusters = [len(np.unique(np.loadtxt(f'{path}/cluster_assignments.dat'))) for path in paths]
    if nclusters[1] > nclusters[0]: # organize them from greatest numbers of clusters to smallest number of clusters
        nclusters = nclusters[::-1]
        paths = paths[::-1]
    return paths, nclusters

def get_positions(paths, nclusters):
    positions, names = {}, []
    for i, path in enumerate(paths):
        positions[path] = []
        for c in range(1, nclusters[i]+1):
            for f in os.listdir(f'{path}/cluster{c}'):
                if os.path.isfile(f'{path}/cluster{c}/{f}') and f[-4:] == '.gro':
                    file = f
            if c == 1:
                names.append(file.replace('_medoid.gro',''))
            u = mda.Universe(f'{path}/cluster{c}/{file}')
            ag = u.select_atoms("(protein and backbone) or (name BB)")
            coordinates = []
            for residue in ag.residues:
                if residue.atoms.n_atoms > 1:
                    coordinates.append(residue.atoms.center_of_mass())
                else:
                    coordinates.append(residue.atoms.positions)
            positions[path].append(np.vstack(coordinates))
    if positions[paths[0]][0].shape != positions[paths[1]][0].shape:
        sys.exit('ERROR: The number of residues in the specified trajectories do not match.')
    return positions, names

def kabsch(a, b):
    '''
    find the rotation matrix that minimizes the RMSD between position arrays a and b.
    return the a array centered on the origin and the b array centered on the origin
    and rotated to best fit a.
    '''
    a -= np.average(a, axis=0)
    b -= np.average(b, axis=0)

    cov = np.ndarray((3, 3))
    for i in range(3):
        for j in range(3):
            cov[i,j] = np.dot(a[:,i], b[:,j])

    (u, v, vh) = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(np.dot(vh.T, u.T)))

    mat = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,d]])
    rot = np.dot(vh.T, np.dot(mat, u.T))
    b = np.dot(b, rot)
    return a, b

def rmsd(a, b):
    '''
    calculate the rmsd between position arrays a and b
    '''
    return np.std(a-b)

# ------------------------------------------------------------------------------ #
#                                   M A I N                                      #
# ------------------------------------------------------------------------------ #
print("------------------------------------------------------------------------------")
print("                     M A T C H    C L U S T E R I N G S                       ")
print(" ")
print("                Match clusters from two different clusterings.                ")
print("------------------------------------------------------------------------------\n")
paths = parse_input()
paths, nclusters = how_many_clusters(paths)

print(f"Matching the {nclusters[0]} clusters in {paths[0]} \nto the {nclusters[1]} clusters in {paths[1]}\n")

positions, names = get_positions(paths, nclusters)

rmsds = np.zeros(nclusters)
for i, a in enumerate(tqdm(positions[paths[0]])):
    for j, b in enumerate(positions[paths[1]]):
        fittedA, fittedB = kabsch(a, b)
        rmsds[i,j] = rmsd(fittedA, fittedB)

# Plot the heatmap
plt.rcParams["font.size"] = 12
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 2
fig, ax = plt.subplots()
im = ax.imshow(rmsds, cmap="cool")

ax.set_xlabel(names[1])
ax.set_ylabel(names[0])
ax.xaxis.set_label_position('top')

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('RMSD', rotation=-90, va="bottom")
cbar.ax.spines[:].set_color('white')
cbar.ax.tick_params(which='both',width=2)
ax.set_xticks(np.arange(rmsds.shape[1]))
ax.set_yticks(np.arange(rmsds.shape[0]))
ax.set_xticklabels(list(range(1,nclusters[1]+1)))
ax.set_yticklabels(list(range(1,nclusters[0]+1)))

ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

ax.spines[:].set_color('white')

ax.set_xticks(np.arange(rmsds.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(rmsds.shape[0]+1)-.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=5)
ax.tick_params(which="minor", bottom=False, left=False)
ax.tick_params(which="both", width=2)
kw = dict(horizontalalignment="center", verticalalignment="center")
#threshold = im.norm(rmsds.max())/2.
valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")
textcolors = ["black"]
for i in range(rmsds.shape[0]):
    for j in range(rmsds.shape[1]):
        #kw.update(color=textcolors[int(im.norm(rmsds[i, j]) > threshold)])
        kw.update(color="black")
        text = im.axes.text(j, i, valfmt(rmsds[i, j], None), **kw)

plt.tight_layout()
plt.savefig('medoid_rmsds.png')
plt.show()
