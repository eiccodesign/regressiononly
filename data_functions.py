import numpy as np
import matplotlib.pyplot as plt

#Compare ROOT and HDF5
def make_comparison_plots( root_array, h5_array, var_name, yscale='log' ) :
    
    fig,ax = plt.subplots( 1, 2, figsize=(20, 9) ,constrained_layout=True)
    
    root_hist = ax[0].hist( root_array.flatten(), bins=200 )
    h5_hist   = ax[1].hist( h5_array.flatten(), bins=200 )

    ax[0].set_yscale( yscale )
    ax[1].set_yscale( yscale )

    ax[0].set_title('Root', fontsize=21 )
    ax[1].set_title('H5', fontsize=21)

    ax[0].set_xlabel( var_name )
    ax[1].set_xlabel( var_name )

    plt.show()

    print("  array length:  %d" % len(root_hist[0]) )

    n_diffs = 0
    for bi in range( len(root_hist[0]) ):
        if root_hist[0][bi] == 0 and h5_hist[0][bi] == 0 : continue

        if root_hist[0][bi] != h5_hist[0][bi] :
            print( '%4d  :  %9.0f  %9.0f  %s' % (bi, root_hist[0][bi], h5_hist[0][bi],  root_hist[2][bi]), end='' )
            print(" *** %.0f" % (root_hist[0][bi] - h5_hist[0][bi]))
            n_diffs = n_diffs + 1

    print("\n\n Number of differences:  %d\n\n" % n_diffs )


# Get Resolution, scale, and distributions of Pred/X in bins of truth
def get_res_scale(truth,pred,N_Bins=20,min=0,max=100):
    if (len(truth) != len(pred)):
        print("truth and prediction arrays must be same length")
        return

    binning = np.linspace(min,max,N_Bins+1)
    indecies = np.digitize(truth,binning)-1 #Get the bin number each element belongs to.
    max_count = np.bincount(indecies).max()
    slices = np.empty((N_Bins,max_count))
    slices.fill(np.nan)

    counter = np.zeros(N_Bins,int) #for getting mean from sum, and incrementing element number in bin
    avg_truth = np.zeros(N_Bins,float)
    pred_over_truth = np.zeros(N_Bins,float)


    for i in range(len(pred)):
        bin = indecies[i]
        slices[bin][counter[bin]] = pred[i] #slice_array[bin number][element number inside bin] = pred[i]
        counter[bin]+=1
        avg_truth[bin]+=truth[i]
        pred_over_truth[bin] += pred[i]/truth[i]

    #Resoluton = stdev(pred)/avg_truth 
    avg_truth = avg_truth/counter
    pred_stdev = np.nanstd(slices,axis=1)
    resolution = pred_stdev/avg_truth
    #Scale = <pred/truth>

    scale = pred_over_truth/counter

    return resolution,scale,avg_truth,slices

def get_res_scale_in_reco_bins(truth,pred,reco,N_Bins=20,min=0,max=100):
    if (len(truth) != len(pred)):
        print("truth and pred arrays must be same length")
        return

    binning = np.linspace(min,max,N_Bins+1)
    indecies = np.digitize(reco,binning)-1 #get bin number of reco bins for each element
    max_count = np.bincount(indecies).max()
    slices = np.empty((N_Bins,max_count))
    slices.fill(np.nan)

    counter = np.zeros(N_Bins,int) #for getting mean from sum, and incrementing element number in bin
    avg_reco = np.zeros(N_Bins,float)
    avg_truth = np.zeros(N_Bins,float)
    pred_over_truth = np.zeros(N_Bins,float)


    for i in range(len(pred)):
        bin = indecies[i]
        if (bin >= N_Bins): continue
        slices[bin][counter[bin]] = pred[i] #slice_array[bin number][element number inside bin] = pred[i]
        counter[bin]+=1 #increment the element number inside [bin]
        avg_reco[bin]+=reco[i]
        avg_truth[bin]+=truth[i]
        pred_over_truth[bin] += pred[i]/truth[i]


    #Resoluton = stdev(pred)/avg_truth 
    avg_reco = avg_reco/counter
    avg_truth = avg_truth/counter
    pred_stdev = np.nanstd(slices,axis=1)
    resolution = pred_stdev/avg_truth

    #Scale = <pred/truth>
    scale = pred_over_truth/counter

    return resolution,scale,avg_reco,avg_truth,slices
