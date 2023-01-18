import numpy as np
import matplotlib.pyplot as plt

star_energies = [12,16,20,25,30,50,60,70]
star_res = [0.18, 0.16, 0.15, 0.14, 0.13, 0.098, 0.092, 0.090]

ECCE_res = [0.15,0.127,0.117,0.121,0.106,0.102,0.092,0.098]
ECCE_energies = [10,20,30,40,50,60,80,100]

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



def Plot_Energy_Scale(NN, strawman,label,sampling_fraction,ymin=0.95,ymax=1.05):
    mask = ~np.isnan(NN["resolution"])
    fig=plt.figure(figsize=(14,10))
    plt.title("AI Codesign Scale",fontsize=25)
    plt.ylabel("$(E_\mathrm{Pred}/E_\mathrm{Truth})$",fontsize=24)
    plt.xlabel("$E_\mathrm{Truth}$ [GeV]",fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(direction='in',right=True,top=True,length=10)
    plt.axhline(y=1.0, color='k', linestyle='--',alpha=0.5)#plt.ylim(-0.02,0.4)
    plt.ylim(ymin,ymax)

    ax = plt.subplot(1,1,1)
    first_bin = 0
    last_bin = len(NN["avg_truth"])

    color1 = 'blue'
    color2 = 'dodgerblue'

    #NN   
    plt.errorbar(NN["avg_truth"][mask][first_bin:last_bin],NN["median_scale"][mask][first_bin:last_bin],#yerr=errors[first_bin:last_bin],
                 linestyle="--",linewidth=2.0,capsize=4,capthick=1.2,elinewidth=1.2,
                 ecolor='black',marker="o",color=color1,alpha=0.7,label="Neural Network")

    #Strawman
    plt.errorbar(strawman["avg_truth"][mask][first_bin:last_bin],strawman["median_scale"][mask][first_bin:last_bin],
                 linestyle="-",linewidth=2.0,capsize=4,capthick=1.2,elinewidth=1.2,ecolor='black',
                 marker="o",color=color2,alpha=0.7,label="Strawman $\sum_\mathrm{Cluster\ E} /\ %1.2f$"%(sampling_fraction))


#plt.text(0.7,0.7,"ROOT",transform=ax.transAxes,fontsize=25)

    plt.legend(fontsize=20)
    plt.text(0.8,-0.08,label,transform=ax.transAxes,fontsize=10)

    path = "./"+label
    plt.savefig("%s/scale_plot.pdf"%(path))

