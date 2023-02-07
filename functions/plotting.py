import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from copy import copy
from matplotlib import cm
import pickle

import sys  
sys.path.insert(0, 'functions')
from Clusterer import E_binning

star_energies = [12,16,20,25,30,50,60,70]
star_res = [0.18, 0.16, 0.15, 0.14, 0.13, 0.098, 0.092, 0.090]

ECCE_res = [0.15,0.127,0.117,0.121,0.106,0.102,0.092,0.098]
ECCE_energies = [10,20,30,40,50,60,80,100]

def ClusterSum_vs_GenP(clusterSum, genP, label, take_log = False, ylabel="Cluster Sum", plot_offset = 5.0):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10), constrained_layout=True)
    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))

    #Bins and Range
    sumE_maxPlot = 125.0 #max cluster energy to plot. Has high tails that mess up plotting
    sumE_maxPlot = min(sumE_maxPlot,np.max(clusterSum))
    cluster_bins = E_binning(np.min(clusterSum),sumE_maxPlot+plot_offset, take_log)
    truth_bins   = E_binning(np.min(genP),np.max(genP)+plot_offset, take_log)

    h, xedges, yedges = np.histogram2d(genP, clusterSum, bins=[truth_bins, cluster_bins])
    pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=LogNorm(vmin=1.0e-2,vmax=1.1e4), rasterized=True)
    cb = fig.colorbar(pcm, ax=ax, pad=0)
    cb.ax.tick_params(labelsize=20)
    ax.set_xlabel("Generated Energy",fontsize=22)
    ax.set_ylabel("Cluster Sum Energy",fontsize=25)
    ax.set_title(f"Cluster Sum vs. Generated Energy",fontsize=30)

    draw_identity_line(ax, color='cyan', linewidth=2, alpha=0.5, label="Ideal")
    ax.legend(loc="upper left")
    fig.text(0.95,-0.05,label,transform=ax.transAxes,fontsize=10)


    path = "./"+label
    plt.savefig(f"{path}/ClusterSum_vs_GenP_Colormap.pdf") 

def energy_QA_plots(flat_hits_e, genP, cluster_sum, label):

    print("Plotting QA Distributions...")

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(36, 10), constrained_layout=True)
    axes = np.ravel(ax)

    max_hits_e = np.mean(flat_hits_e) + np.std(flat_hits_e)

    bins_hits_e = np.linspace(np.min(flat_hits_e),max_hits_e,100)
    axes[0].hist(flat_hits_e, bins=bins_hits_e, color="gold", alpha=0.8)
    axes[0].set_ylabel("Counts",fontsize=22) 
    axes[0].set_xlabel("Cell Hit Energy [GeV]",fontsize=22) 
    axes[0].set_title("Cell Energy Distribution",fontsize=22) 

    axes[1].hist(np.ravel(genP),color="red",alpha=0.8,bins=100)
    axes[1].set_ylabel("Counts",fontsize=22) 
    axes[1].set_xlabel("Generated Momentum [GeV]",fontsize=22) 
    axes[1].set_title("Gen. Momentum Distribution",fontsize=22) 

    if len(np.shape(cluster_sum)) > 1:
        n_zbins = np.shape(cluster_sum)[1]
        print(f"N Z bins = {n_zbins}")
        cm_subsection = np.linspace(0.0, 1.0, n_zbins) 
        colors = [ cm.winter(x) for x in cm_subsection ]

        for zbin in range(n_zbins):
            axes[2].hist(cluster_sum[:,zbin],color=colors[zbin],
                         label="Layer %i"%(zbin),alpha=0.8,bins=20)

        axes[2].set_ylabel("Counts",fontsize=22) 
        axes[2].set_xlabel("Cluster Energy [GeV]",fontsize=22) 
        axes[2].set_title("Cluster Sum Distribution (Raw)",fontsize=22) 
        axes[2].legend(fontsize=22) 
    else:
        axes[2].hist(cluster_sum,color="blue",alpha=0.8)
        axes[2].set_ylabel("Counts",fontsize=22) 
        axes[2].set_xlabel("Cluster Energy [GeV]",fontsize=22) 
        axes[2].set_title("Cluster Sum Distribution (Raw)",fontsize=22) 

    path = "./"+label
    plt.savefig(f"{path}/energy_QA_plots.pdf")

def Plot_Loss_Curve(loss,val_loss,label,loss_string):

    fig,axes = plt.subplots(1,1,figsize=(14,10))
    axes = [axes,axes] #easier to add axes later, if need be
    axes[0].plot(loss,label="loss")
    axes[0].plot(val_loss,label="val_loss")
    axes[0].set_title('Model Loss vs. Epoch',fontsize=26)

    # fig.text(1.05,1.1,label,transform=axes[0].transAxes,fontsize=10)
    plt.text(0.8,-0.08,label,transform=axes[0].transAxes,fontsize=10)
    axes[0].set_ylabel(f'Loss ({loss_string})',fontsize=22)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('epoch',fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(direction='in',right=True,top=True,length=10)
    plt.tick_params(direction='in',right=True,top=True,which='minor')
    axes[0].set_xlim([-1,101])
    axes[0].legend(['train', 'validation'], loc='upper right',fontsize=22)
    plt.savefig(f"./{label}/ROOT_Correlation.png")


def Plot_Resolutions(NN, strawman,label):
    mask = ~np.isnan(NN["resolution"])
    fig=plt.figure(figsize=(14,10))
    plt.title("AI Codesign Resolution",fontsize=25)
    plt.ylabel("$(\sigma_{E,\mathrm{Pred}}/E_\mathrm{Truth})$",fontsize=24)
    plt.xlabel("$E_\mathrm{Truth}$ [GeV]",fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(direction='in',right=True,top=True,length=10)
#plt.ylim(-0.02,0.4)
    plt.ylim(0,2)
    plt.ylim(0,.32)
    plt.xlim(-1,100.01)
    plt.xlim(0.0,100)
#errors = 1.0/(np.sqrt(2*counter-2))*stdev_pred
    ax = plt.subplot(1,1,1)
    first_bin = 0
    last_bin = len(NN["avg_truth"])

    plt.text(0.8,-0.08,label,transform=ax.transAxes,fontsize=10)
    plt.errorbar(NN["avg_truth"][mask][first_bin:last_bin],NN["resolution"][mask][first_bin:last_bin],#yerr=errors[first_bin:last_bin],
                 linestyle="-",linewidth=2.0,capsize=4,capthick=1.2,elinewidth=1.2,ecolor='black',marker="o",color='dodgerblue',alpha=0.7,label="Simple NN")

    plt.plot(ECCE_energies,ECCE_res,"-o",label = "EIC Ref",color="limegreen")
    plt.plot(star_energies,star_res,"-o",label = "STAR",color="deeppink")
    plt.legend(fontsize=15,loc="upper left")

    path = "./"+label
    plt.savefig("%s/resolution_plot.pdf"%(path))


def Plot_Energy_Scale(NN, label, sampling_fraction, strawman=None, bin_label="truth", ymin=0.95,ymax=1.05):
    mask = ~np.isnan(NN["median_scale"])
    fig=plt.figure(figsize=(14,10))
    plt.title("AI Codesign Scale $E_\mathrm{%s}$ Bins"%(bin_label),fontsize=25)
    plt.ylabel("$(E_\mathrm{Pred}/(E_\mathrm{%s})$"%(bin_label),fontsize=24)
    plt.xlabel("$E_\mathrm{%s}$ [GeV]"%(bin_label) ,fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(direction='in',right=True,top=True,length=10)
    plt.axhline(y=1.0, color='k', linestyle='--',alpha=0.5)#plt.ylim(-0.02,0.4)
    plt.ylim(ymin,ymax)

    ax = plt.subplot(1,1,1)
    first_bin = 0
    last_bin = len(NN[f"avg_{bin_label}"][mask])

    color1 = 'blue'
    color2 = 'dodgerblue'

    #NN   
    plt.errorbar(NN[f"avg_{bin_label}"][mask][first_bin:last_bin],NN["median_scale"][mask][first_bin:last_bin],#yerr=errors[first_bin:last_bin],
                 linestyle="--",linewidth=2.0,capsize=4,capthick=1.2,elinewidth=1.2,
                 ecolor='black',marker="o",color=color1,alpha=0.7,label="Neural Network")

    #Strawman
    if (strawman):
        plt.errorbar(strawman[f"avg_{bin_label}"][mask][first_bin:last_bin],strawman["median_scale"][mask][first_bin:last_bin],
                 linestyle="-",linewidth=2.0,capsize=4,capthick=1.2,elinewidth=1.2,ecolor='black',
                 marker="o",color=color2,alpha=0.7,label="Strawman $\sum_\mathrm{Cluster\ E} /\ %1.2f$"%(sampling_fraction))


#plt.text(0.7,0.7,"ROOT",transform=ax.transAxes,fontsize=25)

    plt.legend(fontsize=20)
    plt.text(0.8,-0.08,label,transform=ax.transAxes,fontsize=10)

    path = "./"+label
    plt.savefig("%s/scale_plot.pdf"%(path))


def plot_slices(input_slices,truth,label,E_Bins,bin_label="Truth",scale=False):

    # mask = ~(np.all(np.isnan(input_slices)))
    mask = []
    for i in range(len(input_slices)):  
        mask.append(~(np.all(np.isnan(input_slices[i])))) 
    N_Bins = len(truth[mask])
    input_slices = input_slices[mask]
    truth = truth[mask]
    nrows = int(N_Bins/10)
    if (nrows < 1): 
        nrows =1

    fig,axs = plt.subplots(nrows,int(N_Bins/nrows), figsize=(32, 10),sharex=False,sharey=True,constrained_layout=True)
    axs = np.asarray(axs)

    for i in range(N_Bins):
        row = int(i/10)
        col = i%10
        if (nrows==1): ax = axs[col]
        else: ax = axs[row,col]
        
        if (col==0):
            ax.set_ylabel("Counts",fontsize=15)
        if (np.all(np.isnan(input_slices[i]))): continue

        ax.set_title("%1.1f $ < E_\mathrm{%s} < $%1.1f [GeV]"%(E_Bins[mask][i],bin_label,E_Bins[mask][i]+E_Bins[1]),fontsize=10)
        #^^^assums linspace

        ax.set_xlabel("Predicted Eenergy")
        ax.hist(input_slices[i],label="Predicted Energies",bins=30)

        if (scale):
            ax.axvline(x=truth[i]/truth[i],color='red',alpha=0.3,linestyle="--",
                       label="Median $E_\mathrm{Truth}/E_\mathrm{Truth} = %1.2f$"%(truth[i]/truth[i]))
            ax.axvline(x=np.nanmedian(input_slices,axis=-1)[i],color='cyan',alpha=0.3,linestyle="--",
                       label="Avg. $E_\mathrm{Pred} = %1.2f$"%(np.nanmedian(input_slices,axis=-1)[i]))
        else:
            ax.axvline(x=truth[i],color='red',alpha=0.3,linestyle="--",
                       label="Median $E_\mathrm{Truth} = %1.2f$"%(truth[i]))
            ax.axvline(x=np.nanmedian(input_slices,axis=-1)[i],color='cyan',alpha=0.3,linestyle="--",
                       label="Median $E_\mathrm{Pred} = %1.2f$"%(np.nanmedian(input_slices,axis=-1)[i]))

        if (nrows==1):
            ax.legend(fontsize=15)
        else: 
            ax.legend(fontsize=7.5)

        ax.tick_params(direction='in',right=True,top=True,length=5)

    if (scale):
        plt.suptitle("Distributions of $E_\mathrm{Pred} / E_\mathrm{Truth}$",fontsize=25)
    else:
        plt.suptitle("Distributions of $E_\mathrm{Pred}$",fontsize=25)

    plt.savefig("./%s/resolutions_%sslices.pdf"%(label,bin_label))


def draw_identity_line(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes




def Compare_Resolutions(dirs,labels,logscale=False):

    cm_subsection = np.linspace(0.2, 0.8, len(labels))
    colors = [ cm.plasma(x) for x in cm_subsection ]

    fig=plt.figure(figsize=(14,10))
    plt.title("AI Codesign Resolution",fontsize=25)
    plt.ylabel("$(\sigma_{E,\mathrm{Pred}}/E_\mathrm{Truth})$",fontsize=24)
    plt.xlabel("$E_\mathrm{Truth}$ [GeV]",fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(direction='in',right=True,top=True,length=10)
    # plt.ylim(0.075,.2)
    # plt.xlim(-1,110.01)

    ax = plt.subplot(1,1,1)
    plt.text(0.8,-0.08,dirs[0],transform=ax.transAxes,fontsize=10)

    if (logscale):
        plt.xscale('log')

    for i, (dir,label) in enumerate (zip(dirs,labels)):
        with open(f'./{dir}/res_scale.pickle', 'rb') as handle:
            dict = pickle.load(handle)

            mask = ~np.isnan(dict["resolution"])
            first_bin = 0
            last_bin = len(dict["avg_truth"])

            plt.errorbar(dict["avg_truth"][mask][first_bin:last_bin],
                         dict["resolution"][mask][first_bin:last_bin],
                         linestyle="-", linewidth=2.0, capsize=4, 
                         capthick=1.2,elinewidth=1.2,ecolor='black',
                         marker="o",color=colors[i],alpha=0.99,label=label)

    plt.legend(fontsize=15,loc="upper right")
    plt.savefig("comparing_resolution_plot.pdf")


def Compare_Scales(dirs,labels,bin_label="Truth",ymin=0.95,ymax=1.05,logscale=False):

    cm_subsection = np.linspace(0.2, 0.8, len(labels))
    colors = [ cm.plasma(x) for x in cm_subsection ]
    bin_label = "Truth"

    fig=plt.figure(figsize=(14,10))

    plt.title("AI Codesign Scale $E_\mathrm{%s}$ Bins"%(bin_label),fontsize=25)
    plt.ylabel("$(E_\mathrm{Pred}/(E_\mathrm{%s})$"%(bin_label),fontsize=24)
    plt.xlabel("$E_\mathrm{%s}$ [GeV]"%(bin_label) ,fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(direction='in',right=True,top=True,length=10)
    plt.axhline(y=1.0, color='k', linestyle='--',alpha=0.5)#plt.ylim(-0.02,0.4)
    plt.ylim(ymin,ymax)
    if (logscale):
        plt.xscale('log')

    ax = plt.subplot(1,1,1)
    plt.text(0.8,-0.08,dirs[0],transform=ax.transAxes,fontsize=10)

    for i, (dir,label) in enumerate (zip(dirs,labels)):
        with open(f'./{dir}/res_scale.pickle', 'rb') as handle:
            dict = pickle.load(handle)

            mask = ~np.isnan(dict["median_scale"])
            first_bin = 0
            last_bin = len(dict["avg_truth"])

            plt.errorbar(dict["avg_truth"][mask][first_bin:last_bin],
                         dict["median_scale"][mask][first_bin:last_bin],
                         linestyle="-", linewidth=2.0, capsize=4, 
                         capthick=1.2,elinewidth=1.2,ecolor='black',
                         marker="o",color=colors[i],alpha=0.99,label=label)

    plt.legend(fontsize=15,loc="upper right")
    plt.savefig("comparing_median_scale_plot.pdf")


