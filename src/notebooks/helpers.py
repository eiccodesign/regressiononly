import numpy as np
import awkward as ak
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uproot
import os


# Gaussian functional form
def gaussian(x, amp, mean, sigma_squared):
    return amp * np.exp( -0.5*((x - mean)**2/sigma_squared))
# Function to fit with the Gaussian
# Returns two tuples:
# (mean of gaussian fit, error of mean of gaussian fit)
# (std. dev. of gaussian fit, error of std. dev. of gaussian fit)
def gaussian_fit_on_distribution(nsigma_fit_range,
                                 sigma_of_data,
                                 mean_of_data,
                                 bin_centers,
                                 bin_counts,
                                 ax):
    # Mask to get the bins within +- nsigma_fit_range*sigma_of_data of mean_of_data
    # e.g. fitting within +- 3 sigma of mean
    fit_range_mask = (bin_centers > (mean_of_data - nsigma_fit_range*sigma_of_data))\
    & (bin_centers < (mean_of_data + nsigma_fit_range*sigma_of_data))
    
    try:
        # Fitting the data with a Gaussian
        param_bounds=([0,-np.inf,0], [np.inf,np.inf,np.inf])
        popt, pcov = curve_fit(gaussian,
                              bin_centers[fit_range_mask],
                              bin_counts[fit_range_mask],
                              p0 = [np.max(bin_counts), mean_of_data, sigma_of_data*sigma_of_data],
                              bounds = param_bounds)
        # Errors of each fit parameter
        errors = np.sqrt(np.diag(pcov))
    except Exception as e:
        # Log the exception for debugging purposes
        print(f"Exception occurred during curve fitting: {e}")
        return None  # Explicitly return None in case of an exception
    else:
        # Drawing the fit
        ax.plot(bin_centers[fit_range_mask],
                gaussian(bin_centers[fit_range_mask], *popt),
                color='red',
                linewidth=2.5)

        fit_mean = popt[1]
        fit_std = np.sqrt(popt[2])
        fit_mean_error = errors[1]
        fit_std_error = 0.5*errors[2]/popt[2]*fit_std
        return (fit_mean, fit_mean_error), (fit_std, fit_std_error)
# Propogates the errors of dividing two quantities with errors
# Returns tuple with divided number and new error
def DivideWithErrors(numerator, numerator_error, denominator, denominator_error):
    divided = numerator/denominator
    error = divided * np.sqrt((numerator_error/numerator)**2
                              + (denominator_error/denominator)**2)
    return (divided, error)
# Propogates the errors of adding two quantities with errors
# Returns tuple with added number and new error
def AddWithErrors(num1, num1_error, num2, num2_error):
    summed = num1 + num2
    error = np.sqrt(num1_error * num1_error + num2_error * num2_error)
    return (summed, error)

def AverageWithErrors(value_list, error_list):
    num_entries = len(value_list)
    if num_entries != len(error_list):
        print("Averaging: List are not equal length!")
    average = np.sum(np.asarray(value_list))/(num_entries)
    error = np.sqrt(np.sum(np.square(np.asarray(error_list)))/(num_entries*num_entries))
    return (average, error)

# Function that plots the energy distributions and returns an array of the energy resolution and scales
def get_resolutions(data_to_fit,
                    genE,
                    binning,
                    nbins,
                    data_name = "energy", # Either "energy", "theta", or "phi"
                    divide_by_genE = False,
                    title=""):   
    N_Bins=len(binning)
    
    n_sigma_fit= 3 # fit within +- 3 sigma   
    plot_range = 3                                                                                                                                                                                                          
    row=math.ceil(np.sqrt(N_Bins))
    if (row**2-N_Bins)>row:
        col=row-1
    else:
        col=row

    resolution_list = []
    resolution_error_list = []
    energy_scale_list = []
    energy_scale_error_list = []
    events_below_3sigma_list = []

    y_ticks_size=14
    x_ticks_size=14
    major_x_locator=0.25
    unit = "GeV"
    
    fig,axs = plt.subplots(row,col, figsize=(18, 15),sharex=False)
    plt.subplots_adjust(wspace=0, hspace=0.3)

    # Rounding generated energies to nearest integers
    genE = np.rint(genE)
    mask = (genE >= binning[0]) & (genE <= binning[-1])
    data_to_fit = data_to_fit[mask]
    genE = genE[mask]
    # Putting each event in its associated energy bin
    # Will get a np array with each entry being the bin number of the event
    indecies = np.digitize(genE, binning)-1 

    # Making sure there are no negative indecies
    # These would be entries that are smaller than the lowest bin edge
    indecies = np.where(indecies < 0, 0, indecies) 
    if any(indecies<0): print(indecies)

    # Takes the number of entries from the bin with the most entries
    max_count = np.max(np.bincount(indecies)) 

    # Will store the event quantity in these arrays
    # 2D array: N_bins number of arrays, each initalized to allow for max number of bin entries
    binned_data = np.empty((N_Bins, max_count))
    binned_data.fill(np.nan)
    
    # Arrays used to count the number of events in each energy bin
    event_counter = np.zeros(N_Bins, int)
    # In case there are multiple energies within a bin,
    # this will give the mean energy of the entries in that bin
    avg_truth = np.zeros(N_Bins, float)

    # Storing the energies in binned_quantity arrays
    for i in range(len(genE)):
        bin = indecies[i]
        # Skipping events that are greater than the max bin edge
        if (bin>=N_Bins): continue
        if data_name == "energy" and divide_by_genE:
            binned_data[bin][event_counter[bin]] = data_to_fit[i]/genE[i]
        else:
            binned_data[bin][event_counter[bin]] = data_to_fit[i] 
        avg_truth[bin] += genE[i]                                                      
        event_counter[bin]+=1
    
    avg_truth = avg_truth/event_counter
    
    # Removing any nan entries and taking mean/std. deviation
    data_stdev = np.nanstd(binned_data, axis=1)
    data_mean = np.nanmean(binned_data, axis=1)


    for i_bin in range(N_Bins):
        
        # Using the means and std. dev of bin data
        # as the initial values for the Gaussian fit
        bin_mean = data_mean[i_bin]
        bin_stddev = data_stdev[i_bin]
        
        # Min and max range for histogram                                                                                  
        min_range = bin_mean - plot_range * bin_stddev
        max_range = bin_mean + plot_range * bin_stddev
        
        irow=int(i_bin/col)
        icol=int(i_bin%col)

        if irow < row:
            
            ax = axs[irow,icol]
            i_data = binned_data[i_bin][~np.isnan(binned_data[i_bin])]
            if len(i_data)<5:
                continue
            bin_counts, bin_edges, _ = ax.hist(i_data,
                                     bins = nbins,
                                     alpha=0.5,
                                     range=(min_range, max_range),
                                     color='b',
                                     linewidth=8)
            
            bin_counts = bin_counts[~np.isnan(bin_counts)]
            bin_edges = bin_edges[~np.isnan(bin_edges)]
            
            bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(bin_edges)-1)])
            
            fit_result = gaussian_fit_on_distribution(n_sigma_fit,
                                                      bin_stddev,
                                                      bin_mean,
                                                      bin_centers,
                                                      bin_counts,
                                                      ax)
            if fit_result is None:
                print(f"gaussian_fit_on_distribution returned None for bin {i_bin}")
                continue
            fit_mean = fit_result[0]
            fit_std = fit_result[1]
            if data_name == "energy":
                # Fraction of events with less than mean-3sigma energy 
                leakage_mask = i_data < (fit_mean[0] - 3*fit_std[0])
                leakage_fraction = len(i_data[leakage_mask])/len(i_data)
                events_below_3sigma_list.append(leakage_fraction)
                

            ax.set_title("{0} {1}".format(binning[i_bin], unit), fontsize=15)
    
            if icol==0:
                ax.set_ylabel("Entries",fontsize=15)

            ax.tick_params(axis='x', labelsize=x_ticks_size)
            ax.tick_params(axis='y', labelsize=y_ticks_size)
            
            if data_name == "energy":
                resolution = DivideWithErrors(fit_std[0],
                                              fit_std[1],
                                              fit_mean[0],
                                              fit_mean[1])
                # Energy scale is the fit mean divided by true energy
                if divide_by_genE:
                    energy_scale = fit_mean
                else:
                    energy_scale = DivideWithErrors(fit_mean[0],
                                                    fit_mean[1],
                                                    binning[i_bin],
                                                    0)
                energy_scale_list.append(energy_scale[0])
                energy_scale_error_list.append(energy_scale[1])
            else:
                resolution = (fit_std[0], fit_std[1])
            
            resolution_list.append(resolution[0])
            resolution_error_list.append(resolution[1])
        else:
            continue
    if data_name == "energy":
        average_scale = AverageWithErrors(energy_scale_list, energy_scale_error_list)
        # Divide the resolutions by the average energy scale
        for i in range(len(resolution_list)):
            res_over_scale = DivideWithErrors(
                resolution_list[i],
                resolution_error_list[i],
                average_scale[0],
                average_scale[1])
            resolution_list[i] = res_over_scale[0]
            resolution_error_list[i] = res_over_scale[1]
            
            
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    if data_name == "energy":
        if divide_by_genE:
            plt.xlabel("$E_{dep}/E_{truth}$", fontsize=24, labelpad=20)
        else:
            plt.xlabel("$E_{dep} (GeV)$", fontsize=24, labelpad=20)
    elif data_name == "theta":
        plt.xlabel("$\\theta_{pred} - \\theta_{true} (mrad)$", fontsize=24, labelpad=20)
    elif data_name == "phi":
        plt.xlabel("$\phi_{pred} - \phi_{true} (rad)$", fontsize=24, labelpad=20)
    plt.suptitle(title)
    return avg_truth, (resolution_list, resolution_error_list), (energy_scale_list, energy_scale_error_list), events_below_3sigma_list

def get_theta_resolutions(
    data_to_fit,
    gen_theta,
    binning,
    nbins,
    data_name = "theta", # Either "energy", "theta", or "phi"
    title="",
    units="",
):   
    N_Bins=len(binning)-1
    
    n_sigma_fit= 3 # fit within +- 3 sigma   
    plot_range = 3                                                                                                                                                                                                          
    row=math.ceil(np.sqrt(N_Bins))
    if (row**2-N_Bins)>row:
        col=row-1
    else:
        col=row

    sigma_list = []
    sigma_error_list = []
    mean_list = []
    mean_error_list = []

    y_ticks_size=14
    x_ticks_size=14
    major_x_locator=0.25
    unit = units
    
    fig,axs = plt.subplots(row,col, figsize=(18, 15),sharex=False)
    plt.subplots_adjust(wspace=0, hspace=0.3)

    # Rounding generated energies to nearest integers
    mask = (gen_theta >= binning[0]) & (gen_theta <= binning[-1])
    data_to_fit = data_to_fit[mask]
    gen_theta = gen_theta[mask]
    # Putting each event in its associated bin
    # Will get a np array with each entry being the bin number of the event
    indecies = np.digitize(gen_theta, binning)-1 

    # Making sure there are no negative indecies
    # These would be entries that are smaller than the lowest bin edge
    indecies = np.where(indecies < 0, 0, indecies) 
    if any(indecies<0): print(indecies)

    # Takes the number of entries from the bin with the most entries
    max_count = np.max(np.bincount(indecies)) 

    # Will store the event quantity in these arrays
    # 2D array: N_bins number of arrays, each initalized to allow for max number of bin entries
    binned_data = np.empty((N_Bins, max_count))
    binned_data.fill(np.nan)
    event_counter = np.zeros(N_Bins, int)
    # Storing the energies in binned_quantity arrays
    for i in range(len(gen_theta)):
        bin = indecies[i]
        # Skipping events that are greater than the max bin edge
        if (bin>=N_Bins): continue
        binned_data[bin][event_counter[bin]] = data_to_fit[i] 
        event_counter[bin]+=1
    
    # Removing any nan entries and taking mean/std. deviation
    data_stdev = np.nanstd(binned_data, axis=1)
    data_mean = np.nanmean(binned_data, axis=1)
    valid_bin_centers = []

    for i_bin in range(N_Bins):
        
        # Using the means and std. dev of bin data
        # as the initial values for the Gaussian fit
        bin_mean = data_mean[i_bin]
        bin_stddev = data_stdev[i_bin]
        
        # Min and max range for histogram                                                                                  
        min_range = bin_mean - plot_range * bin_stddev
        max_range = bin_mean + plot_range * bin_stddev
        
        irow=int(i_bin/col)
        icol=int(i_bin%col)

        if irow < row:
            ax = axs[irow,icol]
            i_data = binned_data[i_bin][~np.isnan(binned_data[i_bin])]
            ax.set_title(f"{binning[i_bin]:.2f} - {binning[i_bin+1]:.2f} {unit}", fontsize=15)
            if len(i_data)<5:
                continue
            theta_bin_center = (binning[i_bin] + binning[i_bin+1])/2
            valid_bin_centers.append(theta_bin_center)
            bin_counts, bin_edges, _ = ax.hist(i_data,
                                     bins = nbins,
                                     alpha=0.5,
                                     range=(min_range, max_range),
                                     color='b',
                                     linewidth=8)

            bin_counts = bin_counts[~np.isnan(bin_counts)]
            bin_edges = bin_edges[~np.isnan(bin_edges)]

            bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(bin_edges)-1)])

            fit_result = gaussian_fit_on_distribution(n_sigma_fit,
                                                      bin_stddev,
                                                      bin_mean,
                                                      bin_centers,
                                                      bin_counts,
                                                      ax)
            if fit_result is None:
                print(f"gaussian_fit_on_distribution returned None for bin {i_bin}")
                continue
            fit_mean = fit_result[0]
            fit_std = fit_result[1]

            

            if icol==0:
                ax.set_ylabel("Entries",fontsize=15)

            ax.tick_params(axis='x', labelsize=x_ticks_size)
            ax.tick_params(axis='y', labelsize=y_ticks_size)

            sigma = (fit_std[0], fit_std[1])
            mean = (fit_mean[0], fit_mean[1])

            sigma_list.append(sigma[0])
            sigma_error_list.append(sigma[1])

            mean_list.append(mean[0])
            mean_error_list.append(mean[1])
        else:
            continue
            
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    if data_name == "theta":
        plt.xlabel(f"$\\theta_{{pred}} - \\theta_{{true}} ({unit})$", fontsize=24, labelpad=20)
    elif data_name == "phi":
        plt.xlabel(f"$\phi_{{pred}} - \phi_{{true}} ({unit})$", fontsize=24, labelpad=20)
    plt.suptitle(title)
    return valid_bin_centers, (sigma_list, sigma_error_list), (mean_list, mean_error_list)

def rotateY(x, z, angle_rad):
    s, c = np.sin(angle_rad), np.cos(angle_rad)
    x_new =  s * z + c * x
    z_new =  c * z - s * x
    return x_new, z_new

def rotated_eta_from_theta_phi(theta_mrad, phi_rad, angle_rad=0.025):
    """
    theta_mrad: array of true polar angles in milliradians (your targets_theta)
    phi_rad:   array of true azimuth in radians (your targets_phi)
    angle_rad: rotation about +Y in radians
    returns:   rotated pseudorapidity array (same shape as inputs)
    """
    theta = theta_mrad * 1e-3  # mrad -> rad

    # unit vector in original frame
    sinth = np.sin(theta)
    costh = np.cos(theta)
    cosph = np.cos(phi_rad)
    sinph = np.sin(phi_rad)

    x = sinth * cosph
    y = sinth * sinph
    z = costh

    # rotate around Y
    x_r, z_r = rotateY(x, z, angle_rad)

    # new theta, then eta*
    # guard numerical edges
    z_r = np.clip(z_r, -1.0, 1.0)
    theta_r = np.arccos(z_r)
    eta_r = -np.log(np.tan(theta_r / 2.0))
    return eta_r
def rotate_theta(theta_mrad, phi_rad, angle_rad=0.025):
    """
    theta_mrad: array of true polar angles in milliradians (your targets_theta)
    phi_rad:   array of true azimuth in radians (your targets_phi)
    angle_rad: rotation about +Y in radians
    returns:   rotated pseudorapidity array (same shape as inputs)
    """
    theta = theta_mrad * 1e-3  # mrad -> rad

    # unit vector in original frame
    sinth = np.sin(theta)
    costh = np.cos(theta)
    cosph = np.cos(phi_rad)
    sinph = np.sin(phi_rad)

    x = sinth * cosph
    y = sinth * sinph
    z = costh

    # rotate around Y
    x_r, z_r = rotateY(x, z, angle_rad)

    # new theta, then eta*
    # guard numerical edges
    # z_r = np.clip(z_r, -1.0, 1.0)
    theta_r = np.arccos(z_r)
    return theta_r*180/np.pi