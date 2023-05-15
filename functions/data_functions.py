import numpy as np
import matplotlib.pyplot as plt
import pickle

#Compares to datasets, bin-bin in histograms
def make_comparison_plots( data_1, data_2, var_name, label1 = "ROOT", label2 = "HDF5", verbose=2,yscale='log' ) :
    
    fig,ax = plt.subplots( 1, 2, figsize=(20, 9) ,constrained_layout=True)
    
    data1_hist = ax[0].hist( data_1.flatten(), bins=200 )
    data2_hist   = ax[1].hist( data_2.flatten(), bins=200 )

    ax[0].set_yscale( yscale )
    ax[1].set_yscale( yscale )

    ax[0].set_title(label1, fontsize=21 )
    ax[1].set_title(label2, fontsize=21)

    ax[0].set_xlabel( var_name, fontsize=16 )
    ax[1].set_xlabel( var_name, fontsize=16 )

    plt.show()

    n_diffs = 0
    for bi in range( len(data1_hist[0]) ):
        if data1_hist[0][bi] == 0 and data2_hist[0][bi] == 0 : continue

        if data1_hist[0][bi] != data2_hist[0][bi] :
            n_diffs = n_diffs + 1
            if (verbose>=2):
                print( '%4d  :  %9.0f  %9.0f  %s' % (bi, data1_hist[0][bi], data2_hist[0][bi],  data1_hist[2][bi]), end='' )
                print(" *** %.0f" % (data1_hist[0][bi] - data2_hist[0][bi]))

    if (verbose >= 1):
        print("  array length:  %d" % len(data1_hist[0]) )
        print("\n\n Number of differences:  %d\n\n" % n_diffs )

    if label1 and label2:
        plt.savefig(f"{label1}_vs_{label2}_{var_name}_comparison.pdf")


# Get Resolution, scale, and distributions of Pred/X in bins of truth
def get_res_scale(truth,pred,binning=np.linspace(0,100,21),label=""):

    if (len(truth) != len(pred)):
        print("truth and prediction arrays must be same length")
        return

    indecies = np.digitize(truth,binning)-1 #Get the bin number each element belongs to.
    N_Bins = len(binning)
    indecies[indecies==-1.] = 0.

    max_count = np.bincount(indecies).max()
    slices = np.empty((N_Bins,max_count))
    slices.fill(np.nan)
    scale_array = np.empty((N_Bins,max_count+1))
    scale_array.fill(np.nan)

    counter = np.zeros(N_Bins,int) #for getting mean from sum, and incrementing element number in bin
    avg_truth = np.zeros(N_Bins,float)


    for i in range(len(pred)):
        bin = indecies[i]
        slices[bin][counter[bin]] = pred[i] #slice_array[bin number][element number inside bin] = pred[i]
        counter[bin]+=1
        avg_truth[bin]+=truth[i]
        scale_array[bin][counter[bin]] = pred[i]/truth[i]

    #Resoluton = stdev(pred)/avg_truth 
    avg_truth = avg_truth/counter
    pred_stdev = np.nanstd(slices,axis=1)
    resolution = pred_stdev/avg_truth

    #Scale = <pred/truth>
    avg_scale  =   np.nanmean(scale_array,axis=-1)
    median_scale = np.nanmedian(scale_array,axis=-1)

    dict = {}
    dict["avg_truth"]    = avg_truth
    dict["resolution"]   = resolution
    dict["median_scale"] = median_scale
    dict["avg_scale"]    = avg_scale
    dict["slices"]       = slices
    dict["scale_array"]  = scale_array

    if (label != ""):
        with open(f'{label}/res_scale.pickle', 'wb') as pickle_file:
            pickle.dump(dict, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    return dict


def get_res_scale_in_reco_bins(truth,pred,reco,binning=np.linspace(0,100,21),label=""):
    #FIXME: can do: if not(reco) -> us truth for determining slices-> or reco=truth?
    if (len(truth) != len(pred)):
        print("truth and pred arrays must be same length")
        return

    N_Bins = len(binning)
    indecies = np.digitize(reco,binning)-1 #get bin number of reco bins for each element
    indecies[indecies==-1.] = 0.
    max_count = np.bincount(indecies).max()

    #distributions of pred. in bins of truth
    slices = np.empty((N_Bins,max_count)) 
    slices.fill(np.nan)

    #distributions of pred/truth in bins of truth
    scale_array = np.empty((N_Bins,max_count+1))
    scale_array.fill(np.nan)

    #Scale and Averages
    counter = np.zeros(N_Bins,int) #for getting mean from sum, and incrementing element number in bin
    avg_reco = np.zeros(N_Bins,float)
    avg_truth = np.zeros(N_Bins,float)


    for i in range(len(pred)):
        bin = indecies[i]
        if (bin >= N_Bins): continue
        slices[bin][counter[bin]] = pred[i] #slice_array[bin number][element number inside that bin] = pred[i]
        counter[bin]+=1 #increment the element number inside [bin]
        avg_reco[bin]+=reco[i]
        avg_truth[bin]+=truth[i]
        scale_array[bin][counter[bin]] = pred[i]/truth[i]

        #above is a faster way of doing:
        # if prediction inside bin:
        #     array_for_that_bin.append(prediction)

    #Resoluton = stdev(pred)/avg_truth 
    avg_reco = avg_reco/counter
    avg_truth = avg_truth/counter
    pred_stdev = np.nanstd(slices,axis=1)
    resolution = pred_stdev/avg_truth

    #Scale = <pred/truth>
    median_scale = np.nanmedian(scale_array,axis=-1)
    avg_scale  =   np.nanmean(scale_array,axis=-1)


    dict = {}
    dict["avg_reco"]     = avg_reco
    dict["avg_truth"]    = avg_truth
    dict["resolution"]   = resolution
    dict["median_scale"] = median_scale
    dict["avg_scale"]    = avg_scale
    dict["slices"]       = slices
    dict["scale_array"]  = scale_array

    if (label != ""):
        with open(f'{label}/res_scale_RecoBins.pickle', 'wb') as pickle_file:
            pickle.dump(dict, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    return dict
    # return resolution, median_scale, avg_reco, avg_truth, slices, scale_array

