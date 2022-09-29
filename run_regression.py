#Python/Data
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
import numpy as np
import h5py as h5

#ML
import energyflow as ef
from energyflow.archs import PFN
from energyflow.utils import data_split
import tensorflow as tf
import os
import shutil

def step_decay(epoch, lr):
    min_rate = 1.01e-6
    N_start = 50
    N_epochs = 30
    
    if epoch > N_start and lr >= min_rate:
        if (epoch%N_epochs==0):
            return lr * 0.1
    return lr
     
label = "var_theta_phi"
path = "./"+label

if not os.path.exists(path):
    os.makedirs(path)
else:
    shutil.rmtree(path)
    os.makedirs(path)

filename = 'vartheta_layered.hdf5'
h5_file = h5.File(filename,'r')
print(list(h5_file.keys()))
images = h5_file['calo_images']
truth = h5_file['truth']

N_Events = 1_000_000
#N_Events = 100_000
# N_Events = 1_000
X = images[0:N_Events]
Y = truth[0:N_Events,0,0] #rm last 0 for E+phi

(X_train, X_val, X_test,
Y_train, Y_val, Y_test) = data_split(X, Y, val=0.2, test=0.3,shuffle=True)


fig = plt.figure(figsize=(16,9))

cm = plt.cm.get_cmap('plasma')
cell_vars = ["Energy","Cell X","Cell Y","Cell Depth","Layer 1 Position", "Layer 2 Position"]
bins = [np.linspace(-50,200,101),np.linspace(-500,751,100),
        np.linspace(-500,751,100),np.linspace(-1.5,2.5,100), #,np.linspace(-25,25,100),
        np.linspace(-275,350,100),np.linspace(-275,350,100)]
data=[]

#plt.subplots_adjust(left=None, bottom=1, right=None, top=1.5, wspace=None, hspace=None)
for i in range(images.shape[1]):
    ax = plt.subplot(2, 3, i+1)
    #ax = axsLeft[int(i/3)][int(i/2)]
    data.insert(i,np.ravel(images[0:10000,i,:]))
    data[i] = data[i][data[i]!=0]
    plt.hist(data[i],bins=bins[i],color=cm(i/images.shape[1]))
    plt.title("%s"%(cell_vars[i]),fontsize=20)
    plt.suptitle("Normalized Cell Input Data",fontsize=25)
    
plt.savefig("%s/Normalized_Cell_Data.pdf"%(path))
plt.clf()

plt.hist(Y_val,alpha=0.5,label="Validation Energy")
plt.hist(Y_test,alpha=0.5,label="Test Energy")
plt.legend()
plt.savefig("%s/Val_Test_Energy.pdf"%(path))
plt.clf()

callbacks = tf.keras.callbacks.LearningRateScheduler(step_decay,verbose=0)

learning_rate = 1e-4
Phi_sizes, F_sizes = (100, 100, 128), (100, 100, 100)
output_act, output_dim = 'linear', 1
loss = 'mse' #mean-squared error
pfn = PFN(input_dim=X.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, 
          output_act=output_act, output_dim=output_dim, loss=loss, 
          optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

the_fit = pfn.fit(X_train, Y_train,
                  epochs=300,
                  batch_size=400,
                  callbacks=[callbacks],
                  validation_data=(X_val, Y_val),
                  verbose=1)

pfn.layers
pfn.save("%s/energy_regression.h5"%(label))
mypreds = pfn.predict(X_test,batch_size=400)

np.save("%s/predictions.npy"%(label),mypreds)
np.save("%s/y_test.npy"%(label),Y_test)
np.save("%s/x_test.npy"%(label),X_test)

fig = plt.figure(figsize=(28,10))
ax = plt.subplot(1, 2, 1)
plt.scatter(Y_test,mypreds,alpha=0.3)
plt.xlabel("Y Test [GeV]",fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tick_params(direction='in',right=True,top=True,length=10)
#plt.ylim(-0.01,100.01)
plt.ylim(-10,110)
plt.ylabel("Y Pred [GeV]",fontsize=22)
_ = plt.title("Prediction vs. Test",fontsize=26)
#FIXME: find indecies in mypreds where mypreds 50

ax = plt.subplot(1, 2, 2)
plt.plot(the_fit.history['loss'])
plt.plot(the_fit.history['val_loss'])
plt.title('Model Loss vs. Epoch',fontsize=26)
plt.text(0.73,0.73,"Step Decayed \n Learning Rate \n {:.1e} to {:.1e}".format(learning_rate,1e-6),
         transform=ax.transAxes,fontsize=20)
plt.ylabel('Mean-Squared Error',fontsize=22)
plt.yscale('log')
plt.xlabel('epoch',fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tick_params(direction='in',right=True,top=True,length=10)
plt.tick_params(direction='in',right=True,top=True,which='minor')
plt.xlim([-1,301])
plt.legend(['train', 'validation'], loc='upper right',fontsize=22)
plt.savefig("%s/StepDecay_Prediction_Test.pdf"%(label))
plt.clf()

#Binning
N = 51
E_Max = 100
E_Bins = np.linspace(0,E_Max,N)

#Goal: slices defined by bin of truthE, filled with prediction distributions
indecies = np.digitize(Y_test,E_Bins)
max_count = ((np.bincount(indecies).max()))
slices = np.empty((N,max_count))
slices.fill(np.nan)

counter = np.zeros(N,int)
avg_truth = np.zeros(N,float)

for pred,bin,truth in zip(mypreds,indecies,Y_test):
    slices[bin-1][counter[bin-1]] = pred
    counter[bin-1]+=1
    avg_truth[bin-1]+=truth

#Resoluton: stdev(pred)/avg_truth    
avg_truth = avg_truth/counter
stdev_pred = np.nanstd(slices,axis=1)
resolution = stdev_pred/avg_truth

fig=plt.figure(figsize=(14,10))
plt.title("AI Codesign Resolution",fontsize=25)
plt.ylabel("$(\sigma_{E,\mathrm{Pred}}/E_\mathrm{Truth})$",fontsize=24)
plt.xlabel("$E_\mathrm{Truth}$ [GeV]",fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tick_params(direction='in',right=True,top=True,length=10)
#plt.ylim(-0.02,0.4)
#plt.ylim(0,0.22)
plt.xlim(-0.01,100.01)
errors = 1.0/(np.sqrt(2*counter-2))*stdev_pred
ax = plt.subplot(1,1,1)
first_bin = 0
last_bin = N
plt.errorbar(avg_truth[first_bin:last_bin],resolution[first_bin:last_bin],yerr=errors[first_bin:last_bin],
             linestyle="-",linewidth=2.0,capsize=4,capthick=1.2,elinewidth=1.2,ecolor='black',marker="o",color='dodgerblue',alpha=0.7)
_ = plt.text(0.7,0.93,"Stat. Error: $\dfrac{\sigma}{\sqrt{2N-2} } $",transform=ax.transAxes,fontsize=20)
plt.savefig("%s/resolution_plot.pdf"%(label))



