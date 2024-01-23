import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm
from copy import copy
from matplotlib import cm
import awkward as ak
import glob
import math
import json
import os
import random
import compress_pickle as pickle
from scipy.optimize import curve_fit
import uproot3 as ur
def gaussian(x, amp, mean, sigma):
    return amp * np.exp( -0.5*((x - mean)/sigma)**2) /sigma
import sys  
glob_Nbin=64
#sys.path.insert(0, 'functions')
from Clusterer import E_binning
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
star_energies = [12,16,20,25,30,50,60,70]
star_res = [0.18, 0.16, 0.15, 0.14, 0.13, 0.098, 0.092, 0.090]

ECCE_res = [0.15,0.127,0.117,0.121,0.106,0.102,0.092,0.098]
ECCE_energies = [10,20,30,40,50,60,80,100]

MIP=0.0006 ## GeV    
MIP_ECAL=0.13
time_TH=150  ## ns                                                                                                               
MIP_TH_HCAL=0.5*MIP
MIP_TH_ECAL=0.5*MIP_ECAL
def ClusterSum_vs_GenP(clusterSum, genP, label, take_log = False, ylabel="Cluster Sum", plot_offset = 5.0):
    '''
    if ylabel=="Cluster Sum"
    title_x="Energy [GeV]"
    title_x="Theta [Deg]"
    title_y="Cluster Sum Energy [GeV]"
    title_y='Predicted Theta [Deg]'
    '''
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


    #path =label
    #plt.savefig(f"{path}/ClusterSum_vs_GenP_Colormap.pdf") 

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

    path = label
    plt.savefig(f"{path}/energy_QA_plots.pdf")

def Plot_Loss_Curve(loss,val_loss,label,loss_string):

    fig,axes = plt.subplots(1,1,figsize=(14,10))
    axes = [axes,axes] #easier to add axes later, if need be
    axes[0].plot(loss,label="loss")
    axes[0].plot(val_loss,label="val_loss")
    axes[0].set_title('Model Loss vs. Epoch',fontsize=26)

    # fig.text(1.05,1.1,label,transform=axes[0].transAxes,fontsize=10)
    #plt.text(0.8,-0.08,label,transform=axes[0].transAxes,fontsize=10)
    axes[0].set_ylabel(f'Loss ({loss_string})',fontsize=22)
    #axes[0].set_yscale('log')
    axes[0].set_xlabel('Epoch',fontsize=22)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    #plt.tick_params(direction='in',right=True,top=True,length=10)
    #plt.tick_params(direction='in',right=True,top=True,which='minor')
    axes[0].set_xlim([-1,101])
    axes[0].set_ylim(0.0,0.15)
    
    axes[0].legend(['train', 'validation'], loc='upper right',fontsize=22)
    plt.savefig(f"{label}/ROOT_Correlation.png")


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
                 linestyle="-",linewidth=2.0,capsize=4,capthick=1.2,elinewidth=1.2,ecolor='black',marker="o",color='dodgerblue',alpha=0.7,label="Deepsets")

    plt.plot(ECCE_energies,ECCE_res,"-o",label = "EIC Ref",color="limegreen")
    plt.plot(star_energies,star_res,"-o",label = "STAR",color="deeppink")
    plt.legend(fontsize=15,loc="upper left")

    path =label
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

    plt.legend(fontsize=30)
    plt.text(0.8,-0.08,label,transform=ax.transAxes,fontsize=10)

    path = label
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
    
     
## Gives number of energy generated and puts them in bin    
def get_binning_Nbins(log_base,particle):
    binning=[]
    if log_base==2:
        if particle=='pi+' or particle=='pp':
            N_Bins=9
        elif particle=='e-' or particle =='ele':
            N_Bins=8
        else:
            print("You are analyzing log2 base generated data but have not provided the particle like pion/electron") 
            
        for n in range(0,N_Bins):
            egen=pow(2,n)
            binning.append(egen)   
    
        
    elif log_base==10:
        if particle=='pi+' or particle=='pp':
            N_Bins=11
        elif particle=='e-' or particle =='ele':
            N_Bins=11
    
        else:
            print("You are analyzing log2 base generated data but have not provided the particle like pion/electron")
        for n in range(0,N_Bins):
            ran_power=(n+1)*0.20
            pevent=int(pow(10,ran_power))
            binning.append(pevent)
         
    elif log_base=='continuous':
        N_Bins=glob_Nbin
        if particle=='pi+' or particle=='pp' :
            bin = np.linspace(0.,110,N_Bins)
            
            
        elif particle=='e-' or particle =='ele':
            bin = np.linspace(0.,92,N_Bins)
            
        print("I am uniform")
        binning=bin    
    else:
        print('Chose the data if generated in log10 or log2 base')
        return
    #nn=int(np.log2(upto)) +1
    print(N_Bins)
    #print('binning range. ',binning)
    return N_Bins, binning    
    
    
    
    
   
    
def print_parameter(variable, name_tag, detector):
    name=name_tag+"_"+detector+ "=np."
    true=np.array(variable)
    np.set_printoptions(precision=6)
    print(name,repr(true))

def get_greek_particle(particle):
    if particle=='pi-':
        greek_particle='$\pi^{-}$'
    elif particle=='e-' or particle=='ele':
        greek_particle='$e^-$'
    elif particle=='pi+' or particle=='pp':
        greek_particle='$\pi^{+}$'    
    elif particle=='mu-' or particle=='muon':
        greek_particle='$\mu^{-}$'
    elif particle=='neutron' or particle=='n':
        greek_particle='neutron'
        
    else:
        print("You forgot to pick the particle")
    return greek_particle
        
 
def draw_plot_res_scale(var_X, var_Y,labels, title, xlimits,ylimits, particle, detector, legend_position):
    fig=plt.figure(figsize=(8,6))
    greek_particle=get_greek_particle(particle)
    #plt.title("AI Codesign Scale",fontsize=25
    if title=='scale':
        plt.ylabel("$E_\mathrm{Pred}/E_\mathrm{Truth}$",fontsize=24)
        ylim_min=0.9
        title_head='Energy Scale for '+ greek_particle + ' in ' +' '+detector
    elif title=='resolution':
        #plt.ylabel("$(\sigma_{E,\mathrm{Pred}}/E_\mathrm{Truth})$",fontsize=24)
        plt.ylabel("Resolution",fontsize=24)
        ylim_min=0
        title_head='Energy Resolution for '+ greek_particle + ' in' +' '+ detector
    elif title=='angular-resolution-theta':
        #plt.ylabel("$(\sigma_{E,\mathrm{Pred}}/E_\mathrm{Truth})$",fontsize=24)
        plt.ylabel(r"$\sigma_{\theta} \quad [mrad]$",fontsize=24)
        ylim_min=0
        title_head='Angular Resolution  for '+ greek_particle + ' in' +' '+ detector
        
    elif title=='angular-resolution-phi':
        #plt.ylabel("$(\sigma_{E,\mathrm{Pred}}/E_\mathrm{Truth})$",fontsize=24)
        plt.ylabel(r"$\sigma_{\phi} \quad [mrad]$",fontsize=24)
        ylim_min=0
        title_head='Angular Resolution  for '+ greek_particle + ' in' +' '+ detector    
    plt.xlabel("$E_\mathrm{Truth}$ [GeV]",fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(direction='in',right=True,top=True,length=10)
    plt.axhline(y=1.0, color='k', linestyle='--',alpha=0.5)#plt.ylim(-0.02,0.4)
    plt.ylim(ylimits)
    plt.xlim(xlimits)
    #plt.grid()
    #errors = 1.0/(np.sqrt(2*counter-2))*stdev_pred
    ax = plt.subplot(1,1,1)
    for x,y,label in zip(var_X,var_Y,labels):
        print(label)
        plt.errorbar(x, y,linewidth=2.0,marker="o",alpha=0.7,  label=label)

    #plt.errorbar(straw_X,straw_Y,linewidth=2.0,marker="o",alpha=0.7,\
    #             label="Strawman")
    plt.legend(fontsize=15,loc=legend_position)
    #plt.title(title_head, fontsize=20)

### compare the loss and val loss for given directories    

        #label_train.append("train_" +ilabel)
        #label_val.append("val_"+ilabel)
        
def compare_loss_plots(result_paths, labels, title, particle, xlimits,ylimits): 
    num_plots = len(labels)
    col = min(num_plots, int(np.sqrt(num_plots)))  # Determine number of columns, minimum of sqrt(num_plots) and num_plots
    row = int(np.ceil(num_plots / col)) 
    if col == 1:
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
    else:    
        fig,axes = plt.subplots(row,col, figsize=(22, 10),  constrained_layout=True) #sharex=True, sharey=True,
    #plt.subplots_adjust(wspace=0, hspace=0)      
    
    for index, (result_path, ilabel) in  enumerate(zip(result_paths, labels)):
        npz_unpacked_loss = np.load(result_path+"/losses.npz")
        loss = npz_unpacked_loss['training']
        val_loss = npz_unpacked_loss['validation']
            
        
        if col == 1:
            axes[index].plot(loss[:, -1], label="train_" + ilabel)
            axes[index].plot(val_loss[:,-1],  label="val_"+ilabel)
            axes[row-1].set_xlabel('Epoch',fontsize=30)
            axes[row-2].set_ylabel('Loss',fontsize=30)
            axes[index].legend(loc='upper right', fontsize=20)   
            axes[index].xaxis.set_tick_params(labelsize=30, length=10)
            axes[index].yaxis.set_tick_params(labelsize=30, length=10)
            axes[index].set_ylim(ylimits)
            axes[index].set_xlim(xlimits)
            axes[index].minorticks_on()
            axes[index].tick_params(axis='both', which='minor', length=4, width=4, labelsize=20)
           
            
        else:    
            irow=index//col
            icol=index%col
            print(irow,icol)    
            axes[irow,icol].plot(loss[:,-1],   label="train_" +ilabel)
            axes[irow,icol].plot(val_loss[:,-1],  label="val_"+ilabel)
            
            
            axes[irow, icol].set_xlim(xlimits)
            axes[irow, icol].set_ylim(ylimits)
            axes[irow, icol].minorticks_on()
            #axes[0,0].set_title(title,fontsize=30)
            #axes.title.set_position([.5, 1.05])
            #axes[irow,icol].set_yscale('log')
            axes[1,0].set_xlabel('Epoch',fontsize=30)
            axes[1,0].set_ylabel('Loss ',fontsize=30)
            
            axes[irow,icol].xaxis.set_tick_params(labelsize=30, length=20)
            axes[irow,icol].yaxis.set_tick_params(labelsize=30, length=20)
            
            axes[irow,icol].tick_params(axis='both', which='minor', length=4, width=4, labelsize=20)
        
            for xlabel in axes[irow, icol].get_xticklabels():#, ax[irow, icol].get_yticklabels()):
                #ylabel.set_weight('bold')
                xlabel.set_weight('bold')

            for ylabel in axes[irow, icol].get_yticklabels():#, ax[irow, icol].get_yticklabels()):
                ylabel.set_weight('bold')
            
            axes[irow,icol].legend(loc='upper right', fontsize=30)    
        fig.suptitle(title, fontsize=30,ha="center")       
        
## Read ROOT FILE WITH FEW EVENTS BETWEEN START AND STOP EVENTS      
## RETURNS HIT E
def read_start_stop(file_path, detector, entry_start, entry_stop):
    ur_file = ur.open(file_path)
    ur_tree = ur_file['events']
    num_entries = ur_tree.numentries
    #num_entries=int(train_frac*num_entriesss)

    #print(means.shape,'      ',stds.shape)
    #print("PRINT  DETECTOR ", detector)    
    if detector=="hcal":
        detector_name = "HcalEndcapPHitsReco"
    elif detector=='ecal':
        detector_name = "EcalEndcapPHitsReco"
        
    elif detector=="hcal_insert":
        detector_name= "HcalEndcapPInsertHitsReco"
        
    elif detector=="zdc":
        detector_name="ZDCHcalHitsReco"
    else:
        print("Please make sure you have picked right detector name")     
        print("Pick: hcal or hcal_insert for endcap calo/ hcal_insert for insert")
            
    if(entry_stop<entry_start):
        return
        
    genPx = ur_tree.array('MCParticles.momentum.x',entrystart=entry_start, entrystop=entry_stop)[:,2]
    genPy = ur_tree.array('MCParticles.momentum.y',entrystart=entry_start, entrystop=entry_stop)[:,2]
    genPz = ur_tree.array('MCParticles.momentum.z',entrystart=entry_start, entrystop=entry_stop)[:,2]
    mass = ur_tree.array("MCParticles.mass", entrystart=entry_start      , entrystop=entry_stop)[:,2]
    
    root_gen_P = np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
    gen_energy=np.sqrt(root_gen_P**2 + mass**2)

    hit_e =ur_tree.array(f'{detector_name}.energy',entrystart=entry_start, entrystop=entry_stop)
    time =ur_tree.array(f'{detector_name}.time',entrystart=entry_start, entrystop=entry_stop)
    
    return hit_e, time, gen_energy    
    

    
    
    
    


## Fits the prediction distrubution 
## get the resolution and energy scale 
## using fit parameter and media

def get_res_scale_fit_log10_log2(truth,pred, binning, nbins, data_type, particle, label='energy', fit='True', plot_range=3):   
    N_Bins=len(binning)
    
    FIT_SIGMA=3 ## fit within +- 3 sigma                                                                                                                                                                                                             
    row=math.ceil(np.sqrt(N_Bins-1))
    if (row**2-N_Bins)>1:
        col=row-1
    else:
        col=row

    resolution_arr=[]
    mean_arr=[]
    resolution_cor_arr=[]
    scale_arr=[]
    avg_truth_arr=[]
    slices_arr=[]
    slices_pred_truth_arr=[]
    slices_truth_arr=[]
    scale_median_arr=[]
    
    res_stdev_pred_median_arr=[]
    res_sigma_median_arr=[]
    
    
    
    if label=='energy':
        xtitle='$E_{Pred}/E_{Truth}$'
        unit='GeV'
        y_ticks_size=14
        x_ticks_size=14
        major_x_locator=0.25
           
    elif label=='theta-energy':
        xtitle='$\\theta_{pred} - \\theta_{true}$[mrad]' 
        unit='GeV'
        y_ticks_size=14
        x_ticks_size=15
        major_x_locator=0.5
        
        
    elif label=='phi-energy':
        xtitle='$\phi_{pred} - \phi_{true}$ [mrad]' 
        unit='GeV'  
        y_ticks_size=14
        x_ticks_size=14
        major_x_locator=200   
       
        
    else:
        print('PLEASE PROVIDE THE VARIABLE YOU WANT TO REGRESS')
        return
    
    #fig,axs = plt.subplots(row,col, figsize=(22, 10),sharex=True,constrained_layout=True)
    fig,axs = plt.subplots(row,col, figsize=(18, 15),sharex=False)#,constrained_layout=True)
    plt.subplots_adjust(wspace=0, hspace=0.3)
    if (len(truth) != len(pred)):
        print("truth and prediction arrays must be same length")
        #return
    if data_type=='discrete':
        truth=np.rint(truth)
    
    indecies = np.digitize(truth,binning)-1 #Get the bin number each element belongs to.
    indecies=np.where(indecies < 0, 0, indecies)
    #if any(indecies<0): print(indicies)
    max_count = np.bincount(indecies).max()
    slices = np.empty((N_Bins,max_count))
    
    slices_truth=np.empty((N_Bins,max_count))
    slices_truth.fill(np.nan)
    
    #slices_pred_truth=np.empty((N_Bins,max_count))
    #slices_pred_truth.fill(np.nan)
    
    slices.fill(np.nan)
    
    scale_array = np.empty((N_Bins,max_count+1))
    scale_array.fill(np.nan)

    counter = np.zeros(N_Bins,int) #for getting mean from sum, and incrementing element number in bin                                                                
    avg_truth = np.zeros(N_Bins,float)
    pred_over_truth = np.zeros(N_Bins,float)
    xticks=np.linspace(0,100,6)
    one_ytics=round(max_count,-3)## round to thousand                                                                                                                
    yticks=np.linspace(0,one_ytics,6)
    
    #print(binning)
    
    for i in range(len(pred)):
        bin = indecies[i]
        if (bin>=N_Bins): continue
        slices[bin][counter[bin]] = pred[i] #slice_array[bin number][element number inside bin] = pred[i]                                                            
        slices_truth[bin][counter[bin]] = truth[i]
        
        avg_truth[bin]+=truth[i]
        #if truth[i]<=0:
        #    truth[i]=999
        pred_over_truth[bin] += pred[i]/truth[i]
        if label=='energy':
            scale_array[bin][counter[bin]] = pred[i]/truth[i]
        elif (label=='theta-energy') or (label=='phi-energy'):
            scale_array[bin][counter[bin]] = pred[i]
        
        counter[bin]+=1
    counter[counter == 0] = 1
    
    avg_truth = avg_truth/counter

    #stdev_pred = np.nanstd(slices,axis=1)
    #avg_pred   = np.nanmean(slices,axis=1)
    
    stdev_pred = np.nanstd(scale_array,axis=1)
    avg_pred   = np.nanmean(scale_array,axis=1)
    
    stdev_truth = np.nanstd(slices_truth,axis=1)
    scale_median_comp=np.nanmedian(scale_array,axis=-1)
    median_pred=np.nanmedian(slices,axis=-1)
    #median_scale = np.nanmedian(scale_array,axis=-1)
    #print(avg_pred)
    
    for ii in range(0,N_Bins-1):
        ## guess parameters for fitting      
                                   
        mean_guess=avg_pred[ii]
        sigma_guess=stdev_pred[ii]
        #print(ii,'   mean guess.  ', mean_guess,'  binning ',binning[ii], ' - ', binning[ii+1])
        
        ## min and max range for histogram to be fitted to extract resolution/scale                                                                                  
        min_range=mean_guess - plot_range*sigma_guess
        max_range=mean_guess + plot_range*sigma_guess
        
        if label=='energy':
            if min_range<0:
                min_range=0.2
    

        irow=int(ii/col)
        icol=int(ii%col)
        
        if irow<row:
            
            ax = axs[irow,icol]
            #count, bins,_=ax.hist(slices[ii].ravel(),bins=nbins,alpha=0.5,range=(min_range,max_range),label='HCAL',color='b',linewidth=8)
            count, bins,_=ax.hist(scale_array[ii].ravel(),bins=nbins,alpha=0.5,range=(min_range,max_range),label='HCAL',color='b',linewidth=8)
            
            count=count[~np.isnan(count)]
            bins=bins[~np.isnan(bins)]
            
            binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
            
          
            # PARAMETER BOUNDS ARE NOT USED FOR NOW
            if fit:
                mean, std=gaussian_fit_on_distribution(FIT_SIGMA, sigma_guess, mean_guess, binscenters,  count,ax, min_range, max_range)
                #, binning[i], binning[i+1])
            else:
                mean=mean_guess
                std=sigma_guess
        
            for_plot=round(avg_truth[ii])
            y_text_val=int(np.max(count))*0.7
            
            if data_type=='continuous':
                ax.set_title("{0:.1f} - {1:.1f} {2} ".format(binning[ii],  binning[ii+1], unit ) , fontsize=15)
            else: 
                ax.set_title("{0} {1}".format(binning[ii], unit), fontsize=15)
        
            
            if ((irow==row-1) & (icol==int(col/2))):
                ax.set_xlabel('{0}'.format(xtitle),fontsize=25) 
    
            if icol==0:
                ax.set_ylabel("Entries",fontsize=15)


            #ax.xaxis.set_major_locator(MultipleLocator(major_x_locator))
            ax.tick_params(axis='x', labelsize=x_ticks_size)
            ax.tick_params(axis='y', labelsize=y_ticks_size)
            #scale_median=scale_median_comp[ii]
            #scale=mean/avg_truth[ii]
            scale=mean
            
            mean_arr.append(mean)
            avg_truth_arr.append(avg_truth[ii])
            #resolution=std
            
            if label=='energy':
                resolution=(std/mean)  #/scale    #Scale corrected Strawman
                
                #slices_pred_truth=  (slices[ii] - slices_truth[ii])/slices_truth[ii]
                slices_pred_truth=  scale_array[ii] #slices[ii] /slices_truth[ii]

                
                
                #resolution_cor_arr.append(resolution_scale_corrected)
                scale_arr.append(scale)
                slices_arr.append(slices[ii])
                
                #scale_median_arr.append(scale_median)


                slices_pred_truth_arr.append(slices_pred_truth)

                #res_stdev_pred_median_arr.append(res_stdev_pred_median)
                #res_sigma_median_arr.append(res_sigma_median)
            elif label=='theta' or label=='phi' or label =='theta-energy' or label =='phi-energy':
                 resolution=std
                
            else:
                print("Please provide the right label like energy or theta")
            resolution_arr.append(resolution)
            
        else:
            continue
            
    if label=='energy' or label=='theta-energy' or label=='phi-energy':      
        return resolution_arr, scale_arr, avg_truth_arr, slices_arr, slices_pred_truth_arr
    else:
        return avg_truth_arr, resolution_arr, scale_arr, mean_arr
    
    #return avg_truth, mean_arr

def generate_file_name_dict(input_dims, latent_sizes,num_layers, learning_rates, folders_used, data_types,labels, particles):
    file_name_dict = {}
    path_to_result_dir='/media/miguel/Elements/Data_hcali/Data1/log10_Uniform_03-23/DeepSets_output'
    
    for dim, size, layer, lr, folder, data_type, label, part in zip(input_dims, latent_sizes, num_layers, learning_rates, folders_used, data_types, labels, particles):
        result_path = f"{path_to_result_dir}/results_{dim}_{data_type}_{folder}Fol_{size}_{lr}_{layer}Lay_{part}/{label}"
        key = f"{dim}_{size}_{layer}_{lr}_{folder}_{data_type}_{part}"
        file_name_dict[key] = result_path
    with open('file_name_dict.json', 'w') as f:
        json.dump(file_name_dict, f)
        
        
        
def get_fit_parameters_strawman(root_file, detector, binning, particle, total_events, data_type, output_path, nbins, time_TH, MIP_TH_HCAL):
    if detector=='hcal': 
        sampling_fraction=0.0139 #0.0224 (ATHENA CONFIG 0.0224 
    elif detector=='hcal_insert':
        sampling_fraction=0.0089 ## time 150 ns and MIP 0.5*MIP
       
    
    ur_file = ur.open(root_file) 
    ur_tree = ur_file['events']
    num_entries = ur_tree.numentries
    val_fraction=0.3
    train_fraction=0.5
    test_fraction=1-val_fraction-train_fraction
    '''
    train_start=0
    train_end=int(num_entries*train_fraction)

    val_start=train_end
    val_end=train_end + int(num_entries*val_fraction)

    test_start=val_end
    test_end=num_entries
    '''
    #begin_from=0
    entry_begin=0  ##test_start + 
    stop=entry_begin+total_events
    print("Test start  ",entry_begin, '  test end  ', stop, ' total events between test start and end   ', stop - entry_begin)
    
    
    
    hit_e_raw, time, gen_energy=read_start_stop(root_file, detector, entry_begin, stop)
    
   
    mask = (hit_e_raw > MIP_TH_HCAL) & (time < time_TH) & (hit_e_raw < 1e10)
    hit_e=hit_e_raw[mask]
    
    root_cluster_hcali_raw = ak.sum(hit_e, axis=-1)
    root_cluster_sum_hcali_temp = ak.to_numpy(root_cluster_hcali_raw)
    cluster_sum=np.divide(root_cluster_sum_hcali_temp,sampling_fraction)
    
    #NN = get_res_scale(gen_energy, cluster_sum,Energy_Bins,path)
    resolution_fit, pred_over_truth_fit, true_fit,slices_fit,resolution_scale_corr_median, median_scale_fit,slices_pred_truth,\
    res_std_median, res_sigma_median =get_res_scale_fit_log10_log2(gen_energy,cluster_sum, binning, nbins, data_type, particle)
    #get_res_scale_fit_log10_log2(target_ene,pred_ene, binning,
    #            nbins, data_type, particle, 'energy', fit=True, plot_range=0.1)
    #truth, prediction= get_res_scale_fit_log10_log2(gen_energy,cluster_sum, nbins, data_type, particle)
    #return truth, prediction 
   
    
    name_tag=f'straw{glob_Nbin}_'   
    particle_detector=f'{particle}_{detector}_{data_type}'
    
    #print_parameter(truth, name_tag +'truth', particle_detector )
    #print_parameter(prediction, name_tag +'prediction', particle_detector )
    
    
    print_parameter(true_fit, name_tag +'energy',particle_detector ) 
    print_parameter(resolution_scale_corr_median, name_tag+'res_cor',particle_detector  )
    #print_parameter(pred_over_truth_fit, name_tag +'scale_fit', particle_detector ) 
    print_parameter(resolution_fit, name_tag +'res_sigma', particle_detector ) 
    print_parameter(res_std_median, name_tag +'res_std_median', particle_detector ) 
    print_parameter(res_sigma_median, name_tag +'res_sigma_median', particle_detector )
    print_parameter(median_scale_fit, name_tag +'scale_median', particle_detector )
    
    
 
    
    '''
    df = pd.DataFrame(slices_fit)
    df.to_csv(f'{output_path}/pred_strawman_{particle_detector}.csv', index=False)         
    
    
    df_straw_truth = pd.DataFrame(slices_pred_truth)
    df_straw_truth.to_csv(f'{output_path}/pred_true_strawman_{particle_detector}.csv', index=False)
    '''
    
    
## PLOT (E_PRED - E_TRUTH )/ E_TRUTH    
    
def compare_energy_response_E_over_pred(files_pred_truth, Gen_Energy, data_type, particle, detector, labels, 
                                    ratio_E_pred=True, ylogscale=True):

    #N_Bins, Gen_Energy =get_binning_Nbins(data_type,particle)
    nbins=40
    times=0.9
    N_Bins=len(Gen_Energy)
    if data_type=='continuous':
        mean_values = (Gen_Energy[:-1] + Gen_Energy[1:]) / 2

    val_alpha=0.7
    row=math.ceil(np.sqrt(N_Bins-1))
    if (row**2-N_Bins)>1:
        col=row-1
    else:
        col=row

    params = {'axes.titlesize':'20',
              'xtick.labelsize':'20',
              'ytick.labelsize':'20'}
    plt.rcParams.update(params)
    if ratio_E_pred:
        fig,axs = plt.subplots(row,col, figsize=(30, 18), sharex=True, sharey=False)
    else:
        fig,axs = plt.subplots(row,col, figsize=(30, 18))
    plt.subplots_adjust(wspace=0.05, hspace=0.0)
    #width_plot=2
   
    
    for file_pred_truth,   label in zip(files_pred_truth,   labels):

        df = pd.read_csv(file_pred_truth)

        #df=df_pred.subtract(df_truth)
        for ii in range(0,N_Bins-1):
            irow=ii//col
            icol=ii%col
            if irow<row:

                ax = axs[irow,icol]
                
                counts=np.zeros(N_Bins,int)
                bins=np.zeros(N_Bins,int)

                
                if(ratio_E_pred):
                    min_range=-1
                    max_range=3
                    
                    ytitle="$E_{Truth}$"
                    xlabel_title=r'$\frac{E_{Pred} - E_{Truth}}{E_{Truth}}$'


                ## For the ratio    
                else:
                    min_range=Gen_Energy[ii] - Gen_Energy[ii] * times
                    max_range=Gen_Energy[ii] + Gen_Energy[ii] * times
                    
                    if min_range<0:
                        min_range=0
                    if max_range<3:
                        max_range=3
                   
                    xlabel_title="$E_{Truth}$ [GeV]"
                df.iloc[ii:ii+1].T.hist(bins=nbins,ax=axs[irow,icol],label=label,
                                        alpha=val_alpha, range=(min_range,max_range),histtype='step',linewidth=4,
                                        density=True)   

                #axs[irow,icol].set_title("E =[{0} - {1}] GeV".format(E_min,E_max))
                if data_type=='continuous':
                    axs[irow,icol].set_title("{:.1f} - {:.1f}".format(Gen_Energy[ii], (Gen_Energy[ii]+Gen_Energy[1])), 
                                            fontsize=15)
                else:    
                    axs[irow,icol].set_title("{0} GeV".format(Gen_Energy[ii]))
                if (ylogscale):
                    axs[irow,icol].set_yscale('log')
                    axs[0,0].legend(loc='upper right', fontsize=20)
                else:
                    axs[0,0].legend(loc='upper right', fontsize=20)
                axs[0,0].legend(loc='upper left', fontsize=15)
                if ratio_E_pred:
                    axs[irow,icol].axvline(0,linestyle='dashed',linewidth=3,color='b')
                else:
                    axs[irow,icol].axvline(Gen_Energy[ii],linestyle='dashed',linewidth=3,color='b')
                    
                
                mid_col=int(col/2)
                mid_row=int(row/2)
                axs[mid_row,0].set_ylabel("Count", fontsize=20)
               
                
                axs[row-1,mid_col].set_xlabel(xlabel_title, fontsize=30, labelpad=0.05)
                
            else:
                continue 
                
                
def draw_plot_res_scale_withInset(var_X, var_Y,labels, title, xlimits,ylimits, particle, detector, legend_position):
    fig, ax=plt.subplots(figsize=(8,6))
    greek_particle=get_greek_particle(particle)
    #plt.title("AI Codesign Scale",fontsize=25
    if title=='scale':
        plt.ylabel("$(E_\mathrm{Pred}/E_\mathrm{Truth})$",fontsize=24)
        ylim_min=0.9
        xmin=0
        xmax=10
        ymin=0.80
        ymax=1.10
        title_head='Energy Scale for '+ greek_particle + ' in ' +' '+detector
    elif title=='resolution':
        plt.ylabel("Resolution",fontsize=24)
        ylim_min=0
        xmin=1
        xmax=10
        ymin=0.1
        ymax=0.5
        title_head='Energy Resolution for '+ greek_particle + ' in' +' '+ detector

    plt.xlabel("$E_\mathrm{Truth}$ [GeV]",fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(direction='in',right=True,top=True,length=10)
    plt.axhline(y=1.0, color='k', linestyle='--',alpha=0.5)#plt.ylim(-0.02,0.4)
    plt.ylim(ylimits)
    plt.xlim(xlimits)
    #plt.grid()
    #errors = 1.0/(np.sqrt(2*counter-2))*stdev_pred
    ax = plt.subplot(1,1,1)
    for x,y,label in zip(var_X,var_Y,labels):
        print(label)
        plt.errorbar(x, y,linewidth=2.0,marker="o",alpha=0.7,  label=label)

    plt.legend(fontsize=15,loc=legend_position)
    plt.title(title_head, fontsize=20)
    
    
    axins = ax.inset_axes([0.5, 0.55, 0.4, 0.4])
    axins.set_xlim(xmin, xmax)
    axins.set_ylim(ymin, ymax)
    #axins.set_xlabel("$E_\mathrm{Truth}$ [GeV]",fontsize=24)
    
    #plt.title(title_head, fontsize=20)
    for x,y,label in zip(var_X,var_Y,labels):
        axins.errorbar(x[1:], y[1:],linewidth=2.0,marker="o",alpha=0.7,  label=label) 
        
    
    
def get_cluster_sum_from_hits(detector, ur_tree):
    gen_energy_arr=[]
    hit_e_arr=[]
    if detector=="hcal":
        detector_name = "HcalEndcapPHitsReco"
        sampling_fraction= 0.0224 #0.0139 #0.0224 (ATHENA CONFIG 0.0224
        MIP_TH=MIP_TH_HCAL
        
    elif detector=="zdc":
        detector_name= "ZDCHcalHitsReco"
        sampling_fraction=0.0224
        MIP_TH=MIP_TH_HCAL    

    elif detector=="insert":
        detector_name= "HcalEndcapPInsertHitsReco"
        sampling_fraction=0.0203 # 0.0089
        MIP_TH=MIP_TH_HCAL

    elif detector =='ecal':
        detector_name= "EcalEndcapPHitsReco"
        sampling_fraction=1.
        MIP_TH=MIP_TH_ECAL
    else:
        print("Please make sure you have picked right detector name")     
        print("Pick: hcal or hcal_insert for endcap calo/ hcal_insert for insert")
       
    hit_raw =ur_tree.array(f'{detector_name}.energy')
    time =ur_tree.array(f'{detector_name}.time')
    condition1 = hit_raw > MIP_TH
    condition2= time < time_TH
    condition3= hit_raw < 1e10
    combined_mask = condition1 & condition2 & condition3

    hit_e = ak.mask(hit_raw, combined_mask)
    #mask = (ak.num(hit_raw) > MIP_TH) & (ak.num(time) < time_TH) & (ak.num(hit_raw) < 1e10)
    #hit_e=hit_raw[mask]
        
    #PosRecoX_hcal = ur_tree.array(f'{detector_name}.position.x')/10.0
    #PosRecoY_hcal = ur_tree.array(f'{detector_name}.position.y')/10.0
    #PosRecoZ_hcal = ur_tree.array(f'{detector_name}.position.z')/10.0

    #hit_e_ecal =ur_tree.array(f'{detector_ecal}.energy')

    cluster_raw = ak.sum(hit_e, axis=-1)
    cluster_sum_temp = ak.to_numpy(cluster_raw)
    cluster_sum=np.divide(cluster_sum_temp,sampling_fraction)
    #cluster_sums.append(cluster_sum)
    return cluster_sum       

## READS ROOT FILES FROM LARGE SET OF DATA
## FOR GIVEN DATA DIRECTORY AND HADRONIC DETECTOR IT GIVES TOTAL CLUSTER SUM AND HCAL SUM
## IF FLAG det_Ecal is True, then it also gives sum of HCAL + ECAL
        
def read_root_files_chain(data_dir, hadronic_detector, start,total_files, ecal_hcal_both=True):
   
    root_files_total = np.sort(glob.glob(data_dir+'*root'))
    file_list=root_files_total[start:total_files]
        
    genP=[]
    cluster_sums_hcal=[]
    cluster_sums_ecal=[]
    tot_energy=[]
    thetas=[]
    for file_num in file_list:
        #print(file_num)
        ur_tree=ur.open(file_num)['events']
        num_entries=ur_tree.numentries
        print("Total Entries. == ", num_entries)
        
        genPx = ur_tree.array('MCParticles.momentum.x')[:,2]
        genPy = ur_tree.array('MCParticles.momentum.y')[:,2]
        genPz = ur_tree.array('MCParticles.momentum.z')[:,2]
        mass = ur_tree.array("MCParticles.mass")[:,2]
        root_gen_P = np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
        gen_energy=np.sqrt(root_gen_P**2 + mass**2)
        theta = np.arccos(genPz/root_gen_P)*180/np.pi
        
        cluster_sum_hcal= get_cluster_sum_from_hits(hadronic_detector, ur_tree)
        if ecal_hcal_both:
            cluster_sum_ecal= get_cluster_sum_from_hits('ecal', ur_tree)
            total_clust_energy = cluster_sum_hcal +  cluster_sum_ecal
            cluster_sums_ecal.append(cluster_sum_ecal)
                   
        else:
            total_clust_energy = cluster_sum_hcal
            
        cluster_sums_hcal.append(cluster_sum_hcal)  
    
        genP.append(gen_energy)
        thetas.append(theta)

        tot_energy.append(total_clust_energy)
    
    #print(cluster_sums_hcal.shape)
    combined_cluster_sums_hcal= np.concatenate(cluster_sums_hcal)   
    combined_genP = np.concatenate(genP)
    combined_thetas=np.concatenate(thetas)
    
    combined_total_energy=np.concatenate(tot_energy)
    
    if ecal_hcal_both:
        combined_cluster_sums_ecal= np.concatenate(cluster_sums_ecal)
        return combined_genP, combined_thetas, combined_cluster_sums_hcal,  combined_cluster_sums_ecal, combined_total_energy 
    else:
        combined_cluster_sums_ecal=combined_cluster_sums_hcal ## if there is not ecal then hcal = ecal for easy ness
        return combined_genP, combined_thetas, combined_cluster_sums_hcal,  None, combined_total_energy
    

      
        
def gaussian_fit_on_distribution(FIT_SIGMA, sigma_guess, mean_guess, binscenters,  count, ax, min_range, max_range):
            mask=(binscenters>(mean_guess-FIT_SIGMA*sigma_guess)) & (binscenters<(mean_guess+FIT_SIGMA*sigma_guess))
            error_counts=np.sqrt(count)
            error_counts=np.where(error_counts==0,1,error_counts)
        
            param_bounds=([-np.inf,-np.inf,-np.inf], [np.inf,np.inf,np.inf])
            
            try:
                popt,pcov=curve_fit(gaussian,binscenters[mask],count[mask],p0=[np.max(count),mean_guess,sigma_guess],bounds=param_bounds)
            except RuntimeError:
            # Fitting failed, set mean and std to 9999
                mean = 9999
                std = 9999
            else:    
                ax.plot(binscenters[mask], gaussian(binscenters[mask], *popt), color='red', linewidth=2.5, label=r'F')
            
                #ax.set_xlim(math.floor(min_range),math.ceil(max_range))
                mean=popt[1]
                std=popt[2]
            return mean, std        
        
        
        
####FOR MIP ANALYSIS SELECTS THE RIGHT FILE FOR GIVEN DETECTOR AND PARTICLE
## IT TAKES NUMBER OF FILES FROM START TO STOP YOU WANT TO ANALYSIS
## RETURNS HITS AND GENERATED ENERGY
def get_hitE_genE_fromChain(detector, particle, start, stop):
    hit_e_arr=[]
    MIP_TH=0
    gen_energy_arr=[]
    if detector=="hcal":
        detector_name = "HcalEndcapPHitsReco"
        sampling_fraction=0.0139 #0.0224 (ATHENA CONFIG 0.0224 
        Mev_to_GeV=1

    elif detector=="insert":
        detector_name= "HcalEndcapPInsertHitsReco"
        sampling_fraction=0.0089
        Mev_to_GeV=1

    elif detector =='ecal':
        detector_name= "EcalEndcapPHitsReco"
        sampling_fraction=1.
        Mev_to_GeV=1 #33.33
    else:
        print("Please make sure you have picked right detector name")     
        print("Pick: hcal or hcal_insert for endcap calo/ hcal_insert for insert")
        
        
    conditions = {
    ('ecal', 'mu-'): '/media/miguel/Elements/Data_hcali/Data1/log10_Uniform_03-23/log10_20GeV_mu-_Ecal_Only_10_5k_17deg/',
    ('ecal', 'e-'): '/media/miguel/Elements/Data_hcali/Data1/log10_Uniform_03-23/log10_20GeV_e-_Ecal_Only_20_5k_17deg/',
    ('hcal', 'mu-'): '/media/miguel/Elements/Data_hcali/Data1/log10_Uniform_03-23/log10_20GeV_mu-_Hcal_Only_10_5k_17deg/',
    ('hcal', 'e-'): '/media/miguel/Elements/Data_hcali/Data1/log10_Uniform_03-23/log10_20GeV_e-_Hcal_Only_20_5k_17deg/',
    # Add more conditions and corresponding file paths as needed
    }

    default_file = "default.txt"
    rootfile_dir = conditions.get((detector, particle), default_file)
    if rootfile_dir==default_file:
        print('File not found')    
        
    root_files_total = np.sort(glob.glob(rootfile_dir+'*root'))
    file_lists=root_files_total[start:stop]   
    
    
    #trees = [ur.open(file_list)["events"] for file_list in file_lists]  # Replace "tree_name" with the actual tree name
    #chain = ur.tree.concatenate(trees)
    #num_entries=chain.numentries
    #print(num_entries)
    
    for file_num in file_lists:
        #print(file_num)
        ur_tree=ur.open(file_num)['events']
        num_entries=ur_tree.numentries
        #print('Hello hello hello. ', num_entries)
        
        genPx = ur_tree.array('MCParticles.momentum.x')[:,2]
        
        genPy = ur_tree.array('MCParticles.momentum.y')[:,2]
        genPz = ur_tree.array('MCParticles.momentum.z')[:,2]
        mass = ur_tree.array("MCParticles.mass")[:,2]
        root_gen_P = np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
        gen_energy=np.sqrt(root_gen_P**2 + mass**2)
        
        #theta = np.arccos(genPz/root_gen_P)*180/np.pi    
        
        hit_raw =ur_tree.array(f'{detector_name}.energy')/Mev_to_GeV
        
        time =ur_tree.array(f'{detector_name}.time')
        condition1 = hit_raw > MIP_TH
        condition2= time < time_TH
        condition3= hit_raw < 1e10
        combined_mask = condition1 & condition2 & condition3

    
        hit_e = ak.mask(hit_raw, combined_mask)
        #mask = (hit_raw > MIP_TH_HCAL) & (time < time_TH) & (hit_raw < 1e10)
        #hit_e=hit_raw[mask]
        
        
        hit_e=hit_e.astype(float)
        hit_e=ak.flatten(hit_e)
        gen_energy_arr.append(gen_energy)
        hit_e_arr.append(hit_e)
        
        #PosRecoX_hcal = ur_tree.array(f'{detector_name}.position.x')/10.0
        #PosRecoY_hcal = ur_tree.array(f'{detector_name}.position.y')/10.0
        #PosRecoZ_hcal = ur_tree.array(f'{detector_name}.position.z')/10.0
    #hit_e_arr=np.(hit_e_arr)
    #hit_e_arr=np.array(hit_e_arr)
    
    gen_energy_arr=np.concatenate(gen_energy_arr)
    hit_e_arr=np.concatenate(hit_e_arr)
    gen_energy_arr = gen_energy_arr.astype(int)
    return hit_e_arr, gen_energy_arr


def Ecal_hcal_generate_file_name_dict(input_dir,output_file, input_dims, latent_sizes,num_layers, learning_rates,labels, particles,error_types):
    file_name_dict = {}
    path_to_result_dir=f'/media/miguel/Elements/Data_hcali/Data1/log10_Uniform_03-23/DeepSets_output/Deepset_Models/{input_dir}'
    
    for dim, size, layer, lr, label, part, err in zip(input_dims, latent_sizes, num_layers, learning_rates, labels, particles,error_types):
        result_path = f"{path_to_result_dir}/results_{dim}_size{size}_lr{lr}_{layer}Lay_{part}_{err}/{label}"
        key = f"{dim}_{size}_{layer}_{lr}_{part}_{err}"
        file_name_dict[key] = result_path
        print(file_name_dict[key])
        #print('__________________________', key)
    with open(f'{output_file}', 'w') as f:
        json.dump(file_name_dict, f)
        
### THIS PART OF THE CODE FITS THE RESOLUTION CURVES AND GET THE SCHOSTASTIC, NOISE AND CONSTANT TERM  
## WITHOUT CONSTANT TERM 

def fit_resolution_curve(xx, stochastic, const):
    return np.sqrt(stochastic**2/xx + const**2)

def get_resolution_fit_terms(energies, resolutions,  xpos, ypos, text_size, title, xlabel, ylabel):
    #energy_all=np.array(energies)
    #resolution_all=np.array(resolutions)
    #mask=np.logical_and(energy_all>energy_lim[0] , energy_all<energy_lim[1])
    #energy=energy_all[mask]
    #resolution=resolution_all[mask]
    p0=[0.5,0.5,0.1]
    plt.errorbar(energies, resolutions , marker='o', linestyle='None')
    plt.xlabel(xlabel,  fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    
    popt, pcov = curve_fit(fit_resolution_curve, energies, resolutions)
    plt.plot(energies, fit_resolution_curve(energies, *popt), color='red', linewidth=2.5, label=r'Fit to Insert data')
    stochastic_term, const_term=popt
    plt.text(xpos, ypos, f'$\sigma/E $ =  {stochastic_term:.2f}/$\sqrt{{E}} \quad \oplus$ {const_term:.2e} ', fontsize=text_size)     
    
    #goodness_of_fit=chi_sq #/dof
    plt.title(title) 
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    
def get_3D_inference_from_discrete_data(path_to_result, path_to_stat, granularity, file="test_predictions.npz"):

    npz_unpacked = np.load(f'{path_to_result}/{file}')

    predictions_arr = npz_unpacked['outputs']
    targets_arr = npz_unpacked['targets']

    #means = pickle.load(f"preprocessed_data/means.p")

    means = pickle.load(open(f"{path_to_stat}/means.p", 'rb'), compression='gzip')
    stdvs = pickle.load(open(f"{path_to_stat}/stdvs.p", 'rb'), compression='gzip')
    
    ### True ENERGY
    targets_ene = targets_arr[:,0]*stdvs['genP'] + means['genP']
    ### PREDICTED ENERGY
    prediction_ene=predictions_arr[:,0]*stdvs['genP'] + means['genP']
    
    if granularity=='full_cell_hits':
        targets_ene_plt = 10**targets_ene
        prediction_ene_plt=10**prediction_ene
        
    elif granularity=='z_sections':
        targets_ene_plt = targets_ene
        prediction_ene_plt=prediction_ene

    ### True Theta
    targets_theta= targets_arr[:,1]*stdvs['theta'] + means['theta']


    ### PREDICTED THETA
    predictions_theta=predictions_arr[:,1]*stdvs['theta'] + means['theta']
    #prediction_theta_plt=predictions_theta


    ### True phi
    targets_phi= targets_arr[:,2]*stdvs['phi'] + means['phi']

    ### PREDICTED PHI
    predictions_phi=predictions_arr[:,2]*stdvs['phi'] + means['phi']

    return targets_ene_plt, prediction_ene_plt, targets_theta, predictions_theta, targets_phi, predictions_phi


def get_1D_inference_from_discrete_data(granularity, path_to_result, path_to_stat, file="test_predictions.npz"):

    npz_unpacked = np.load(f'{path_to_result}/{file}')
    print("path_to_result", path_to_result,'     c ', file)

    predictions_arr = npz_unpacked['outputs']
    targets_arr = npz_unpacked['targets']

    #means = pickle.load(f"preprocessed_data/means.p")

    means = pickle.load(open(f"{path_to_stat}/means.p", 'rb'), compression='gzip')
    stdvs = pickle.load(open(f"{path_to_stat}/stdvs.p", 'rb'), compression='gzip')
    targets_ene = targets_arr*stdvs['genP'] + means['genP']
    

    ### PREDICTED ENERGY
    prediction_ene=predictions_arr*stdvs['genP'] + means['genP']
    
    if granularity=='full_cell_hits':
        targets_ene = 10**targets_ene
        prediction_ene=10**prediction_ene
        
    return targets_ene, prediction_ene
   
        
### Get the model for loss curve validation and traning loss function  from continuous        
def get_loss_curve_from_training_hcal(path_to_deepset_models, granularity, output_dim, input_features, \
                             incident_angle, hadronic_detector, num_z_layers, include_ecal):
    if include_ecal:
        ecal_status='ecal'
    else:
        ecal_status='Noecal'
    model_dir=f'Output{output_dim}D_{incident_angle}_{ecal_status}_{hadronic_detector}_{granularity}'
    
    if granularity=='full_cell_hits':
        result_dir=f'results_{input_features}D_size64_lre3_4Lay_pp_mse'
        
    elif granularity=='z_sections':
        result_dir=f'results_{num_z_layers}Z_size64_lre3_4Lay_pp_mse'
        input_features=num_z_layers
        
        
    print(model_dir)
    print(result_dir)
    conditions = {
    ('full_cell_hits', 3, 4, 'hcal', True): 'Block_20230714_1113_concatTrue',
        
    ('full_cell_hits', 1, 4, 'hcal', True): 'Block_20230714_1536_concatTrue',
        
    ('full_cell_hits', 1, 3, 'hcal', True): 'Block_20230714_2002_concatTrue',
        
    ('full_cell_hits', 1, 2, 'hcal', True): 'Block_20230714_2313_concatTrue',
        
    ('full_cell_hits', 1, 1, 'hcal', True): 'Block_20230715_0707_concatTrue',  
        
    ('z_sections',     1, 1, 'hcal', False): 'Block_20230711_1822_concatTrue', 
        
    ('z_sections',     1, 5, 'hcal', False): 'Block_20230629_1535_concatTrue',
        
    ('z_sections',     1, 15, 'hcal', False): 'Block_20230630_0125_concatTrue',
        
    ('z_sections',     1, 25, 'hcal', False): 'Block_20230630_0758_concatTrue',    
        
    }
    print(granularity, '   ', output_dim, '   ',  input_features,  '   ', hadronic_detector, '   ', include_ecal)
    default_file = "default"
    block_name = conditions.get((granularity, output_dim, input_features, hadronic_detector, include_ecal), default_file)
    if block_name==default_file:
        print('File not found') 
        return None
    else:
        final_model_path= f'{path_to_deepset_models}/{model_dir}/{result_dir}/{block_name}'
        
    return final_model_path   


def read_start_stop_local(file_path, detector, NumEvents, include_ecal=True):
    import uproot as ur
    MIP_TH_ecal=0.5 * 0.13
    Time_TH_ecal=150
    root_files = os.listdir(file_path)
    selected_file = random.choice(root_files)
    file = os.path.join(file_path, selected_file)
    
    ur_tree = ur.open(file)['events']
    #ur_tree = ur_file['events']
    num_entries = ur_tree.num_entries
    if (NumEvents==-1) or (NumEvents>num_entries):
        NumEvents=num_entries
    event_data=ur_tree.arrays(entry_stop=NumEvents)
    #num_entries=int(train_frac*num_entriesss)
    print("Total Entries ", num_entries)
    
    #print(means.shape,'      ',stds.shape)
    #print("PRINT  DETECTOR ", detector)    
    if detector=="hcal":
        detector_name = "HcalEndcapPHitsReco"
        MIP_TH=0.5 * 0.0006
        Time_TH=150
        theta_max=600.0
        sampling_fraction=0.0224
    elif detector=='ecal':
        detector_name = "EcalEndcapPHitsReco"
        MIP_TH=0.5 * 0.13
        Time_TH=150
        theta_max=600.0
        sampling_fraction=1.0
        
    elif detector=="insert":
        detector_name= "HcalEndcapPInsertHitsReco"
        MIP_TH=0.5 * 0.0006
        Time_TH=150.0
        theta_max=600.0
        sampling_fraction=0.0224
        
    elif detector=="zdc":
        detector_name="ZDCHcalHitsReco"
        MIP_TH=0.5 * 0.000393
        Time_TH=275.0
        theta_max=10.0
        sampling_fraction=0.0216
        
    else:
        print("Please make sure you have picked right detector name")     
        print("Pick: hcal or hcal_insert for endcap calo/ hcal_insert for insert")
            
    
    genPx = event_data['MCParticles.momentum.x'][:,2]
    genPy = event_data['MCParticles.momentum.y'][:,2]
    genPz = event_data['MCParticles.momentum.z'][:,2]
    mass = event_data["MCParticles.mass"][:,2]
    if detector != 'zdc':
        genPx, genPz = rotateY(genPx, genPz, .025)
    
    root_gen_P = np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
    mom=np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
    theta=np.degrees(np.arccos(genPz/mom))  ## in mili radians
    gen_energy=np.sqrt(root_gen_P**2 + mass**2)
    
    hit_e =event_data[f'{detector_name}.energy']
    time =event_data[f'{detector_name}.time']
    posx =event_data[f'{detector_name}.position.x']/10.
    posy =event_data[f'{detector_name}.position.y']/10.
    posz =event_data[f'{detector_name}.position.z']/10.
    
    mask=np.logical_and(hit_e>MIP_TH , time<Time_TH)
    hit_e = hit_e[mask]
    posx=posx[mask]
    posy=posy[mask]
    posz=posz[mask]
    cluster_sum_hcal=(np.sum(hit_e, axis=-1))/sampling_fraction

    
    if include_ecal==True:
        detector_name='EcalEndcapPHitsReco'
        hit_e_ecal =event_data[f'{detector_name}.energy']
        time_ecal =event_data[f'{detector_name}.time']
        mask_ecal=np.logical_and(hit_e_ecal>MIP_TH_ecal , time_ecal<Time_TH_ecal)
        hit_e_ecal = hit_e_ecal[mask_ecal]
        cluster_sum_ecal=np.sum(hit_e_ecal, axis=-1)
        cluster_sum_total=cluster_sum_hcal + cluster_sum_ecal
        
        
    else:
        cluster_sum_total=cluster_sum_hcal
        cluster_sum_ecal=None
        
    
   
    
    return hit_e, posx, posy, posz, genPx, genPy, gen_energy, theta, cluster_sum_total , cluster_sum_hcal, cluster_sum_ecal 

def rotateY(xdata, zdata, angle):
    s = np.sin(angle)
    c = np.cos(angle)
    rotatedz = c*zdata - s*xdata
    rotatedx = s*zdata + c*xdata
    return rotatedx, rotatedz
     