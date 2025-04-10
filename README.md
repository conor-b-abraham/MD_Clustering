# MD Agglomerative Clustering

Perform Hierarchical Clustering with the Ward Metric In Julia & Python

## Overview
Main Programs: 1-pairwise_distances.py, 2-cluster.jl, 3-results.py

Further Analysis Programs: 4-compare_clusterings.py, 5-compare_pairwise_distances.py

## Preparation
Because this clustering scheme is based on interatomic distances, RMSD alignment
of the structures is not necessary before you begin. You should, however, make
sure that the molecules are made whole (i.e. PBC unwrapping) prior to using these
programs. Alternatively, you could modify 1-pairwise_distances.py so that the
PBC box dimensions are considered (i.e. add 'box=u.trajectory.dimensions' to
the self_distance_array command); however, this will be more prone to error than if
you just unwrap the trajectories first and the output from 3-results.py will be
more useful.

## Main Programs
all programs will write the output files and trajectories to your current working directory. Therefore, you should run all three programs from the same directory.

### STEP 1 : 1-pairwise_distances.py
The cartesian coordinates of the atoms are read from all trajectories you want to be
clustered. This is done in python because MDAnalysis is significantly more flexible than
any Julia package for reading and writing trajectories. This program will use these
coordinates to calculate the "self distance array" for the backbone atoms of a protein
and write a single file with these arrays for each frame in each trajectory. For Martini
trajectories, the backbone atoms are represented by a single bead, but for all atom trajectories
they are represented by individual beads. So that all atom and martini trajectories can be
clustered together, the center of mass of the all atom backbone atoms is taken to represent
them. From this point on all trajectories are treated as a single trajectory. A second file
is written containing the IDs of each frame so that they can be mapped back to their
original trajectory.

**Dependencies** : Python3, Numpy, MDAnalysis

**Usage** : > python 1-pairwise_distances.py -i INPUT_FILE [-s SELECTION]
    where,
    -i, --inputfile INPUT_FILE : is the name of your properly formatted input file.

    -s, --selection SELECTION : is the selection command to establish atom group for pairwise distance calculations. (Default: "(protein and name CA) or (name BB)")

    -c, --usecom : if there are multiple atoms per residue, calculate the pairwise distances between the centers of mass of each residue instead of between each atom. This is useful for clustering all atom trajectories together with martini trajectories.

    Your input file needs to have 3 columns and at least 1 row. Each trajectory
    you want to have included in the clustering needs to have its own row.
    Column 1 contains the name/ID of the trajectory, Column 2 contains the file
    name (and path) of the topology file, and column 3 contains the file name
    (and path) of the trajectory file. If you have multiple replicates of a
    trajectory type, you should give all replicates the same name/ID. The topology
    and trajectory file types can be any that MDAnalysis accepts. Obviously, if
    you are comparing two different trajectories they don't need to have the same
    total number of atoms, however, they do need to have the same number of alpha
    carbons.

    EXAMPLES OF PROPERLY FORMATTED INPUT FILES:
    Example 1 : A single gREST trajectory
    gREST [path]/topology.pdb [path]/trajectory.dcd

    Example 2: 5 martini trajectories (replicates)
    MARTINI [path]/topology.gro [path]/rep1.xtc
    MARTINI [path]/topology.gro [path]/rep2.xtc
    MARTINI [path]/topology.gro [path]/rep3.xtc
    MARTINI [path]/topology.gro [path]/rep4.xtc
    MARTINI [path]/topology.gro [path]/rep5.xtc

    Example 3: 10 martini trajectories (5 replicates for two different membranes)
    MEMBRANE1 [membrane1 path]/topology.gro [membrane1 path]/rep1.xtc
    MEMBRANE1 [membrane1 path]/topology.gro [membrane1 path]/rep2.xtc
    MEMBRANE1 [membrane1 path]/topology.gro [membrane1 path]/rep3.xtc
    MEMBRANE1 [membrane1 path]/topology.gro [membrane1 path]/rep4.xtc
    MEMBRANE1 [membrane1 path]/topology.gro [membrane1 path]/rep5.xtc
    MEMBRANE2 [membrane2 path]/topology.gro [membrane2 path]/rep1.xtc
    MEMBRANE2 [membrane2 path]/topology.gro [membrane2 path]/rep2.xtc
    MEMBRANE2 [membrane2 path]/topology.gro [membrane2 path]/rep3.xtc
    MEMBRANE2 [membrane2 path]/topology.gro [membrane2 path]/rep4.xtc
    MEMBRANE2 [membrane2 path]/topology.gro [membrane2 path]/rep5.xtc

### STEP 2 : 2-cluster.jl
The self distance array file is read in and the pairframe dRMSDs are calculated
between all pairs of frames. For this, Julia is significantly faster than python.
The speedup is very obvious for large trajectories. Clustering is then computed and the
silhouette scores are calculated for various numbers of clusters. The number of clusters
with the highest silhouette score is automatically selected to be used to write the output.
The user can optionally rerun this program with a defined number of clusters if they
would like to override the automatically selected number of clusters. This program will
write a large number of files (most of them are checkpoint files so that long calculations
don't have to be recalculated). The outputs are as follows:
    - dRMSDs.jld : The dRMSD value array
    - np_dRMSDs.jld : The dRMSD value array in a numpy readable format (can be turned off)
    - clustering_object.jld : the clustering result (dtype::Hclust)
    - dendrogram.png : the dendrogram from the clustering
    - silhouette_scores.dat : the silhouette scores
    - silhouette_scores.png : the silhouette scores plot
    - cluster_assignments.dat : the final results for the chosen number of clusters. A single
                                column of the cluster number to which each frame belongs (will
                                match the ID file written from 1-pairwise_distances.py)

**Dependencies** : Julia - Julia 1.6.2+, StatsPlots, Clustering, DelimitedFiles, LinearAlgebra, Distances, ProgressMeter, Statistics, JLD, Argparse, PyPlot, NPZ
                       ** These can easily be installed from the Julia Pkg REPL
               
		   Python - Python3, Matplotlib
                       ** These are required for the PyPlot Julia Package to work

**Usage** : > julia 2-cluster.jl [-k NCLUSTERS] [-l MINCHECK] [-u MAXCHECK] [-d] [-h]
          where,

          -k, --nclusters NCLUSTERS : number of clusters (0 will cause silhouette score to be calculated and determine the optimal number of clusters)(type: Int64, default: 0)

          -o, --barjoseph : if used, the fast optimal leaf ordering algorithm will be used while performing clustering. if not used, leaf ordering will be based on the input order of frames.

          -r, --recluster : if used, clustering will be recomputed even if past clustering result is found.

          -l, --mincheck MINCHECK : minimum number of clusters to compute silhouette score for (type: Int64, default: 2)

          -u, --maxcheck MAXCHECK : maximum number of clusters to compute silhouette score for (type: Int64, default:100)

          -d, --dendon : force recreation of dendrogram if clustering has already been completed (default: false)

          -t, --silthreshold : threshold to which to start checking silhouette scores for only every 10 nclusters. If 0, there will be no threshold and the silhouette score for every n clusters between --mincheck and --maxcheck will be checked. This is useful for preliminarily checking the silhouette scores for a large range of nclusters because checking every n clusters will become very expensive. (type: Int64, default: 0)

          -n, --nonpy : prevent program from writing dRMSDs to a numpy file. Writing this file takes a long time so this option can be used to make the program faster. Eventually, you will need to rerun this program without this option so that 3-results can extract the clusters' medoids.
          -h, --help : show this help message and exit

### STEP 3 : 3-results.py
Individual trajectories will be written for each cluster and each trajectory type. If you clustered 5 Martini Trajectories with 1 All atom trajectory, each cluster would have two trajectories written (1 for the martini trajectories and 1 for the all atom trajectories). This program will also produce a plot of the cluster sizes for each trajectory type.

**Dependencies** : Python3, Numpy, MDAnalysis, Matplotlib

**Usage** : > python 3-results.py -i INPUT_FILE

        where,

        -i, --inputfile INPUT_FILE : is the same inputfile you specified with 1-pairwise_distances.py

## Further Analysis Programs
### 1. 4-match_clusterings.py
Use this program to match clusterings of two different trajectories (i.e. a MARTINI trajectory and an Atomistic trajectory of the same program). This program will calculate the RMSDs between all of the medoids for each trajectory to find matching clusters. So that you do not have to coarse-grain the atomistic trajectory, the kabsch algorithm is implemented so that RMSDs can be calculated based on supplied position arrays rather than calculating them through MDAnalysis (which requires matching system sizes).

 **Dependencies** : Python3, numpy, sys, os, matplotlib, tqdm

### 2. 5-compare_pairwise_distances.py
Compare pairwise distances between atom selection for clustering for two different trajectories. 

**Dependencies** : Python3, Numpy, Matplotlib, Scipy
