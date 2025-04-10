import numpy as np
import MDAnalysis as mda
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# ------------------------------------------------------------------------------ #
#                              F U N C T I O N S                                 #
# ------------------------------------------------------------------------------ #
def process_input(inputfilename):
    KILL = False
    if not os.path.isfile(f'cluster_assignments.dat'):
        print(f'The clustering result file {os.getcwd()}/cluster_assignments.dat does not exist')
        KILL = True
    if not os.path.isfile(f'frame_IDs.dat'):
        print(f'The IDs file {os.getcwd()}/frame_IDs.dat does not exist')
        KILL = True
    if KILL:
        sys.exit(f'ERROR: Necessary input files were not found:\n'
                  'Make sure you ran 1-pairwise_distances.py AND 2-cluster.jl before running this program\n'
                  'and that your current working directory is where their output files are stored.\n')
    elif not os.path.isfile(inputfilename):
        sys.exit(f'ERROR: The input file, {inputfilename}, was not found')
    else:
        clusters = np.loadtxt(f'cluster_assignments.dat')
        IDs = np.loadtxt(f'frame_IDs.dat',dtype='str')
        inputfile = np.loadtxt(inputfilename, dtype='str')
    return clusters, IDs, inputfile

# ------------------------------------------------------------------------------ #
#                                   M A I N                                      #
# ------------------------------------------------------------------------------ #
print("------------------------------------------------------------------------------")
print("                   S T E P  3 :    G E T    R E S U L T S                     ")
print(" ")
print("  Breaking up trajectories into their individual clusters, extracting medoids,")
print("  and plotting the size of each cluster for each type of trajectory.")
print("------------------------------------------------------------------------------\n")

parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputfile',required=True, help='input file (same as used for 1-pairwise_distances.py)')
args = parser.parse_args()
inputfilename = args.inputfile

# Collect input
clusters, IDs, inputfile = process_input(inputfilename)
cluster_assignment_dict = {f"{ID}":[c for i, c in enumerate(clusters) if IDs[i] == ID] for ID in np.unique(IDs)}
if inputfile.ndim == 1:
    types = np.array([inputfile[0]])
    tops = {inputfile[0]:[inputfile[1]]}
    trajs = {inputfile[0]:[inputfile[2]]}
else:
    types = np.unique(inputfile[:,0])
    tops = {ID:[t for i, t in enumerate(inputfile[:,1]) if inputfile[i,0] == ID] for ID in np.unique(inputfile[:,0])}
    trajs = {ID:[t for i, t in enumerate(inputfile[:,2]) if inputfile[i,0] == ID] for ID in np.unique(inputfile[:,0])}

# Create Directories for Each Cluster
dirs = {}
for cluster in np.unique(clusters):
    dirs[cluster] = f'cluster{int(cluster)}'
    if not os.path.exists(dirs[cluster]):
        os.makedirs(dirs[cluster])

# Write Trajectories for each Cluster
cluster_sizes = {}
for ID, cluster_assignments in cluster_assignment_dict.items():
    cluster_sizes[ID] = [0]*len(np.unique(clusters))
    u = mda.Universe(tops[ID][0], trajs[ID])
    for cluster in tqdm(np.unique(cluster_assignments), desc=f"Creating {ID} Cluster Trajectories:"):
        cluster_frames = np.where(cluster_assignments==cluster)[0]
        cluster_sizes[ID][int(cluster)-1]=len(cluster_frames)
        traj_file = f'{dirs[cluster]}/{ID}_trajectory.xtc'
        if not os.path.isfile(traj_file):
            with mda.Writer(traj_file, u.atoms.n_atoms) as W:
                for ts in u.trajectory:
                    if ts.frame in cluster_frames:
                        W.write(u.atoms)

# Extract medoid structures
dRMSDs = np.load('np_dRMSDs.npy')
for ID, cluster_assignments in cluster_assignment_dict.items():
    IDdRMSDs = dRMSDs[:,IDs==ID] # make smaller array of all of the columns corresponding to the current traj type
    for cluster in tqdm(np.unique(cluster_assignments), desc=f'Extracting {ID} Cluster Medoids'):
        IDCdRMSDs = IDdRMSDs[:,np.array(cluster_assignments)==cluster] # strip columns outside of current cluster
        clusterdRMSDs = IDCdRMSDs[clusters==cluster, :] # strip rows outside the current cluster
        dRMSDmeans = np.mean(clusterdRMSDs, axis=0) # find mean rmsd of each frame in trajectory
        medoidframe = np.argmin(dRMSDmeans) # find frame with min rmsd
        u = mda.Universe(tops[ID][0], f'{dirs[cluster]}/{ID}_trajectory.xtc')
        u.trajectory[medoidframe]
        u.atoms.write(f'{dirs[cluster]}/{ID}_medoid.gro')

# Plotting the cluster sizes
plt.rcParams["font.size"] = 12
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 2

labels = list(map(str, list(range(1,len(np.unique(clusters))+1))))
fig, ax = plt.subplots(figsize=(12,8))
for i, (k, v) in enumerate(cluster_sizes.items()):
    if i == 0:
        b = ax.bar(labels, v, label=k)
    else:
        b = ax.bar(labels, v, label=k, bottom=store_last)
    ax.bar_label(b, label_type='center', fmt='%.2f')
    store_last = v
ax.legend()
ax.set_xlabel('Cluster Number')
ax.set_ylabel('Cluster Size')
ax.tick_params(which='both', width=2)
ax.tick_params(which='both', width=2)
plt.tight_layout()
plt.savefig(f'cluster_sizes.png')
