println("-------------------------------------------------------------------")
println("                S T E P  2 :    C L U S T E R I N G                ")
println()
println("  Performing Hierarchical Clustering with the Ward linkage Metric  ")
println("-------------------------------------------------------------------\n")
# Packages
using StatsPlots
using Clustering
using DelimitedFiles
using LinearAlgebra
using Distances
using ProgressMeter
using Statistics
using JLD
using ArgParse
using PyPlot
using NPZ
using Plots.PlotMeasures
# ------------------------------------------------------------------------------ #
#                              F U N C T I O N S                                 #
# ------------------------------------------------------------------------------ #
function parse_commandline()
    arguments = ArgParseSettings()

    @add_arg_table arguments begin
        "--nclusters", "-k"
        help = "number of clusters (0 will cause silhouette score to be calculated and determine the optimal number of clusters)"
        default = 0
        arg_type = Int64
        "--barjoseph", "-o"
        help = "if used, the fast optimal leaf ordering algorithm will be used while performing clustering. if not used, leaf ordering will be based on the input order of frames."
        action = :store_true
        "--recluster", "-r"
        help = "if used, clustering will be recomputed even if past clustering result is found."
        action = :store_true
        "--mincheck", "-l"
        help = "minimum number of clusters to compute silhouette score for"
        default = 2
        arg_type = Int64
        "--maxcheck", "-u"
        help = "maximum number of clusters to compute silhouette score for"
        default = 100
        arg_type = Int64
        "--silthreshold", "-t"
        help = "threshold to which to start checking silhouette scores for only every 10 nclusters. If 0, there will be no threshold. This is useful for preliminarily checking the silhouette scores for a large range of nclusters."
        default = 0
        arg_type = Int64
        "--dendon", "-d"
        help = "force recreation of dendrogram if clustering has already been completed. This is automatic if clustering is performed."
        action = :store_true
        "--nonpy", "-n"
        help = "prevent program from writing dRMSDs to a numpy file. Writing this file takes a long time so this option can be used to make the program faster. Eventually, you will need to rerun this program so that 3-results can extract the clusters' medoids."
        action = :store_true
    end
    return parse_args(arguments)
end

function printover(out::String)
    # Replaces previous line printed to terminal
    print("\u1b[1F")
    print(out)
    print("\u1b[0K")
    println()
end

function pairframe_dRMSDs(pd::Matrix{Float64}, rmsdfile::String)
    nframes::Int64 = size(pd)[1]
    ndists::Int64 = size(pd)[2]
    println("Calculating the pairwise dRMSDs between the $ndists pairwise distances of $nframes frames. This could take a while.")
    @time RMSD = pairwise(rmsd, pd, dims=1) # computes pairwise root mean squared displacement with BLAS (significantly faster than iterating)
    println("Writing dRMSDs to $rmsdfile")
    save(rmsdfile, "RMSD", RMSD)
    printover("Wrote dRMSDs to $rmsdfile")
    return RMSD
end

function plot_dendrogram(cluster::Hclust, dendfile::String)
    println("\nPlotting dendrogram and saving to $dendfile")
    StatsPlots.plot(cluster)
    StatsPlots.plot!(xticks=false)
    StatsPlots.plot!(size=(600, 1000))
    StatsPlots.plot!(ylabel="Ward Distance", left_margin=10mm)
    StatsPlots.savefig(dendfile)
    printover("Plotted dendrogram and saved to $dendfile")
end

function calculate_silhouettes(clustering::Hclust, RMSD::Matrix{Float64}, silfile::String, plotfile::String, mincheck::Int64, maxcheck::Int64, sil_thresh::Int64)
    # compute silhouettes
    if sil_thresh == 0
        check_range = range(mincheck, maxcheck, step=1)
    else
        check_range = sort(unique(hcat(collect(range(mincheck, sil_thresh, step=1))', collect(range(sil_thresh, maxcheck, step=10))')))
    end
    silhouette_scores = zeros(Float64, (length(check_range), 2))
    p = Progress(length(check_range))
    for (ndx, nclust) in enumerate(check_range)
        cutcluster = cutree(clustering, k=nclust) # cut the dendrogram
        tempsils = silhouettes(cutcluster, RMSD)
        silhouette_scores[ndx, 1] = nclust
        silhouette_scores[ndx, 2] = mean(tempsils) # compute the mean of the silhouettes
        next!(p; showvalues=[(:nclust, nclust)])
    end

    # write silhouette scores for each nclusters
    println("Writing Silhouette Scores to $silfile")
    open(silfile, "w") do io
        writedlm(io, silhouette_scores, '\t')
    end
    printover("Wrote Silhouette Scores to $silfile")

    # find optimal number of clusters
    bestsil, bestndx = findmax(silhouette_scores[:, 2])
    best = round(Int64, silhouette_scores[bestndx, 1])

    # Plotting Silhouettes
    println("Plotting silhouette scores and saving as $plotfile")

    PyPlot.plot(silhouette_scores[:, 1], silhouette_scores[:, 2], color="tab:gray")
    #axvline(best, color="tab:red")
    xlabel("Number of Clusters")
    ylabel("Silhouette Score")
    title("Best Number of Clusters is $best")
    xlim(0, maxcheck)

    PyPlot.savefig(plotfile)
    printover("Plotted silhouette scores and saved as $plotfile")

    return best
end

function final_cut(cluster::Hclust, nclusters::Int64, outfile::String)
    cutcluster = cutree(cluster, k=nclusters)
    open(outfile, "w") do io
        writedlm(io, cutcluster, '\t')
    end
end

# ------------------------------------------------------------------------------ #
#                                   M A I N                                      #
# ------------------------------------------------------------------------------ #
function main()
    # Arguments from commandline
    parsed_args = parse_commandline()
    nclusters = parsed_args["nclusters"]
    mincheck = parsed_args["mincheck"]
    maxcheck = parsed_args["maxcheck"]
    dend_on = parsed_args["dendon"]
    sil_thresh = parsed_args["silthreshold"]
    barjoseph = parsed_args["barjoseph"]
    recluster = parsed_args["recluster"]
    nonpy = parsed_args["nonpy"]

    # File Names
    rmsdfile::String = "dRMSDs.jld"
    numpyrmsdfile::String = "np_dRMSDs.npy"
    pdfile::String = "pairwise_distances.dat"
    clusfile::String = "clustering_object.jld"
    dendfile::String = "dendrogram.png"
    silfile::String = "silhouette_scores.dat"
    plotfile::String = "silhouette_scores.png"
    outfile::String = "cluster_assignments.dat"

    # Check to make sure pairdists were already computed
    if isfile(pdfile) == false
        println("The pairwise distance file $pdfile was not found. Please run 1-pairwise_distances.py")
        println("before running this program.")
        exit(86)
    end

    # Calculate RMSDs or read saved result
    if isfile(rmsdfile)
        println("Reading saved dRMSDs from $rmsdfile")
        RMSD = load(rmsdfile)["RMSD"]
        printover("Read saved dRMSDs from $rmsdfile")
        if isfile(numpyrmsdfile)
            nonpy = true
        end
    else
        println("Reading pairwise distances from $pdfile")
        pd = readdlm(pdfile, ' ', Float64, '\n')
        printover("Read pairwise distances from $pdfile")
        RMSD = pairframe_dRMSDs(pd, rmsdfile)
    end

    if nonpy == false
        println("Writing dRMSDs in numpy readable format to $numpyrmsdfile")
        @time npzwrite(numpyrmsdfile, RMSD)
    end

    # Perform Clustering or read saved result
    if isfile(clusfile) && recluster == false
        println("Reading clustering result from $clusfile")
        cluster = load(clusfile)["cluster"]
        printover("Read clustering result from $clusfile")
    else
        if barjoseph == true
            println("\nPerforming Clustering with fast optimal leaf ordering")
            @time cluster = hclust(RMSD, linkage=:ward, branchorder=:barjoseph)
        else
            println("\nPerforming Clustering with regular leaf ordering")
            @time cluster = hclust(RMSD, linkage=:ward)

        end
        println("Writing clustering result to $clusfile")
        save(clusfile, "cluster", cluster)
        printover("Wrote clustering result to $clusfile")
        dend_on = true
    end

    # make dendrogram (if '-d' is given or if clustering is computed for the first time)
    if dend_on
        plot_dendrogram(cluster, dendfile)
    end

    # calculate silhouettes if nclust == 0 or if the silhouette file does not already exist
    if nclusters == 0 || isfile(silfile) == false
        # Calculate Silhouettes
        println("\nCalculating Silhouette Scores")
        bestnclusters = calculate_silhouettes(cluster, RMSD, silfile, plotfile, mincheck, maxcheck, sil_thresh)
        println("\nSilhouette Score is the highest for $bestnclusters clusters")
    end

    # perform final clustering
    if nclusters == 0
        println("\nCutting Tree at nclusters based on max silhouette score and saving to $outfile")
        final_clust = final_cut(cluster, bestnclusters, outfile)
        printover("Cut Tree at nclusters based on max silhouette score and saved to $outfile")
    else
        println("\nCutting Tree at user defined nclusters and saving to $outfile")
        final_clust = final_cut(cluster, nclusters, outfile)
        printover("Cut Tree at userdefined nclusters and saved to $outfile")
    end

    println("\nCOMPLETE: Next, run 3-results.py\n")
end

main()
