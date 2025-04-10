import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.distances as distances
from scipy.spatial.distance import pdist
from tqdm import tqdm
import sys
import os
import argparse

# ------------------------------------------------------------------------------ #
#                              F U N C T I O N S                                 #
# ------------------------------------------------------------------------------ #
def pairwise_distances(u, ag, usecom):
    '''
    Calculate pairwise distances of atoms in atomgroup
    *** Modified from save_pair_distances.py by George A. Pantelopulos
    INPUT
    -----
    u : MDAnalysis.Universe
        -- Universe to which ag belongs
    ag : MDAnalysis.AtomGroup
        -- AtomGroup for which to calculate pairwise distances
    RETURNS
    -------
    pairwise_distances : nd.array (nF, (N*N-1)/2)
        -- Array with pairwise distances for the N atoms in ag for a nF length trajectory.
    '''
    print(f'Calculating Pairwise Distances for {ag.atoms.n_atoms} atoms in {ag.residues.n_residues}',
          f' residues over a {u.trajectory.n_frames} frame trajectory:')
    if usecom:
        print(f'Using center of mass of atoms in each residue if there are multiple atoms per residue.')
        pairwise_distances = np.zeros((u.trajectory.n_frames, int((ag.residues.n_residues*(ag.residues.n_residues-1))/2)))
        for ts in tqdm(u.trajectory):
            coordinates = []
            for residue in ag.residues:
                if residue.atoms.n_atoms > 1:
                    coordinates.append(residue.atoms.center_of_mass())
                else:
                    coordinates.append(residue.atoms.positions)
            coordinates = np.vstack(coordinates)
            pairwise_distances[ts.frame,:] = pdist(coordinates)
    else:
        pairwise_distances = np.zeros((u.trajectory.n_frames, int((ag.atoms.n_atoms*(ag.atoms.n_atoms-1))/2)))
        for ts in tqdm(u.trajectory):
            pairwise_distances[ts.frame,:] = distances.self_distance_array(ag.positions)
    return pairwise_distances

def check_atom_ordering(ags, names):
    '''
    Prints the ordering of the residues in two ags so you can make sure that they are consistent across different trajectories
    INPUT
    -----
    ags  : list
         -- List containing MDAnalysis.AtomGroups for which pairwise distances are to be calculated
    names : numpy Array
         -- numpy array containing names for each trajectory
    '''
    # Check to make sure all atomgroups have the same number of atoms
    narray = np.array([ag.residues.n_residues for ag in ags])
    if not np.all(narray==narray[0]):
        for i, v in enumerate(names):
            print(f'{i}:{v} {narray[i]} residues')
        sys.exit('Number of Residues in each AtomGroup do not match. Cannot calculate pairwise distances.')

    # Print the order of each atom group
    print('Residue Ordering (Please confirm they match before proceeding with clustering)')
    for i, ag in enumerate(ags):
        if ag.n_atoms == 0:
            sys.exit("ERROR: Bad Selection Command created empty atom groups.")
        residues = " ".join(ag.residues.resnames.tolist())
        print(f"{names[i]} (AG{i}): {residues}")
    print(" ")
    ask = input("Please hit [ENTER] to continue (Entering any character before hitting [ENTER] will abort program): ")
    if ask != "":
        sys.exit("User Abort. Pairwise Distances were not calculated.")

def process_input(inputfilename):
    '''
    Process input file and confirm that everything is in order before proceeding
    '''
    if os.path.isfile(inputfilename):
        inputfile = np.loadtxt(inputfilename, dtype='str')
        if inputfile.ndim == 1 and len(inputfile) == 3:
            print(f"Pairwise distances will be calculated for {inputfile.shape[0]} trajectories.")
            names = np.array([inputfile[0]])
            tops = np.array([inputfile[1]])
            trajs = np.array([inputfile[2]])
            EXIT = False
        elif inputfile.ndim > 1 and inputfile.shape[1] == 3:
            print(f"Pairwise distances will be calculated for {inputfile.shape[0]} trajectories.")
            names = inputfile[:,0]
            tops = inputfile[:,1]
            trajs = inputfile[:,2]
            EXIT = False
        else:
            EXIT = True
            print("Improper formatting for input file.")
    else:
        EXIT = True
        print("Provided input file does not exist.")

    if not EXIT:
        for top in tops:
            if not os.path.isfile(top):
                EXIT = True
                print(f"Input topology file {top} does not exist")
        for traj in trajs:
            if not os.path.isfile(traj):
                EXIT = True
                print(f"Input trajectory file {traj} does not exist")

    if EXIT:
        sys.exit("Input file must be formatted with names, topology files, and trajectory files in columns 1, 2, and 3, respectively.\n"
                 "EXAMPLE:   MARTINI    rep1.tpr      rep1.xtc\n"
                 "           MARTINI    rep2.tpr      rep2.xtc\n"
                 "           MARTINI    rep3.tpr      rep3.xtc\n"
                 "           MARTINI    rep4.tpr      rep4.xtc\n"
                 "           GREST      system.pdb    system.dcd\n")

    return names, tops, trajs

# ------------------------------------------------------------------------------ #
#                                   M A I N                                      #
# ------------------------------------------------------------------------------ #
print("------------------------------------------------------------------------------")
print("            S T E P  1 :    P A I R W I S E    D I S T A N C E S              ")
print(" ")
print("  Calculating Pairwise distances between the backbone atoms of a protein in  ")
print("  a single or multiple trajectories.")
print("------------------------------------------------------------------------------\n")

parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputfile',required=True, help='input file (use -i showformat to see proper formatting of input file).')
parser.add_argument('-s','--selection',default="(protein and name CA) or (name BB)",help='MDAnalysis selection command to form atomgroup for pairwise distance calculation (put it in quotes). (default : "(protein and name CA) or (name BB)")')
parser.add_argument('-c','--usecom', action='store_true', help='take the center of mass of atoms in each residue as the reference position.')
args = parser.parse_args()
inputfilename = args.inputfile
selection_command = args.selection
usecom = args.usecom

names, tops, trajs = process_input(inputfilename)

print(f'The Pairwise Distances will be calculated for the AtomGroups defined by selection command: "{selection_command}"')

multiverse = [mda.Universe(top, trajs[i]) for i, top in enumerate(tops)]
ags=[u.select_atoms(selection_command) for u in multiverse]
check_atom_ordering(ags, names)

pair_dists, IDs = [], []
for i, u in enumerate(multiverse):
    print(f'\n{names[i]} ({i+1}/{len(multiverse)})')
    pair_dists.append(pairwise_distances(u, ags[i], usecom))
    IDs += [names[i]]*u.trajectory.n_frames

print("\nFinishing Up")
pair_dists = np.vstack(pair_dists)

np.savetxt(f'pairwise_distances.dat', pair_dists)
with open(f'frame_IDs.dat', 'w+') as wf:
    for i in IDs:
        wf.write(f"{i}\n")

print(f"\nPairwise Distances saved to {f'pairwise_distances.dat'}")
print(f"Frame IDs saved to frame_IDs.dat")
print("\nCOMPLETE: Next, run 2-cluster.jl")
