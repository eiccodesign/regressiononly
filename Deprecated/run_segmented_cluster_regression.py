import sys  
sys.path.insert(0, 'functions')
sys.path.insert(0, 'training')

from Clusterer import *
from plotting import *
from data_functions import *
from NN_Regression import *

root_file = "Log10_Continuous_17deg.root"
label ="FiftyFour_Segmentation_17deg"
detector_name = "HcalEndcapPHitsReco" #or "HcalEndcapPInsertHitsReco"
sampling_fraction = 0.02 #or 0.0098
NEvents_Max = 1000000 #OK if tree has less events than this
Energy_Bins = binning=np.linspace(0.1,110,21) #Plotting
n_calo_layers = 54

take_log10 = False
if (take_log10):
    Energy_Bins = np.logspace()

Do_Processing = True
Do_Training = True
#saves to path/numpy_file. Only needs to be run frist time. 
# Change to False for messing with plots

if Do_Processing:
    
    Clusterer = Strawman_Clusterer(root_file, 
                                   label, 
                                   detector_name, 
                                   sampling_fraction, 
                                   NEvents_Max, 
                                   n_calo_layers,
                                   take_log=take_log10)

    Clusterer.run_segmentation_clusterer()

    del Clusterer

ClusterSum = load_ClusterSum(label)
segmented_ClusterSum = load_segmented_ClusterSum(label)
GenP = load_GenP(label)

flat_hits_e = load_flat_hits_e(label)
energy_QA_plots(flat_hits_e, GenP, segmented_ClusterSum, label)

ClusterSum_vs_GenP(ClusterSum, GenP, label)


print("Clusterer and Plotting Done. Starting Regression")
NN_Regression = NN_Regressor(label,n_calo_layers)

if (Do_Training):
    NN_Regression.run_NN_regression()

x_test = np.load(f"./{label}/x_test.npy")
y_test = np.load(f"./{label}/y_test.npy")
preds = np.load(f"./{label}/predictions.npy")
loss = np.load(f"./{label}/loss.npy")
val_loss = np.load(f"./{label}/val_loss.npy")

ClusterSum_vs_GenP(preds[:,0],y_test,label)
Plot_Loss_Curve(loss,val_loss,label,loss_string="MAE") #label loss yourself here

simple_sum = x_test
if len(np.shape(x_test))>1:
    simple_sum = np.sum(x_test,axis=-1)

NN = get_res_scale(y_test,preds,Energy_Bins,label) #Label here will save. Make sure not to save strawman!
strawman = get_res_scale(y_test,simple_sum,Energy_Bins)

NN_in_RecoBins = get_res_scale_in_reco_bins(y_test,preds,simple_sum,Energy_Bins,label)
strawman_in_RecoBins = get_res_scale_in_reco_bins(y_test,simple_sum,simple_sum,Energy_Bins)

print(NN.keys())
print(NN_in_RecoBins.keys())

Plot_Resolutions(NN,strawman,label)
Plot_Energy_Scale(NN,label,sampling_fraction,strawman)
Plot_Energy_Scale(NN_in_RecoBins,label,sampling_fraction,None,"reco")

plot_slices(NN["slices"],NN['avg_truth'],label,Energy_Bins,"Truth",)
plot_slices(NN["scale_array"],NN['avg_truth'],label, Energy_Bins,"Truth",scale=True)

