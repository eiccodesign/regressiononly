import h5py
import numpy as np
from tqdm import tqdm

var_str = ["E","X","Y","Z"]
print(f"\nUsing variable strings {var_str} from binning_utils.py\n")

def get_bin_width(centers):

    # Assumes fixed-width binning, 
    # Can be changed to centers[1:] - centers[:-1]
    width = np.round(centers[2] - centers[1], 2)  # avoids [0] edgecase
    
    return width

def get_bin_edges(g4_cell_data):

    print("BINNING UTILS LEN G4 data",np.shape(g4_cell_data))
    centers = np.unique(g4_cell_data)

    width = get_bin_width(centers) 

    edges = centers - width/2
    max_edge = centers[-1] + width/2
    edges = (np.append(edges,max_edge))
    edges = np.round(edges, 2)

    return centers, edges, width



def get_bin_dict(geant4_name, nevts = 100_000):

    bin_dict = {}

    with h5py.File(geant4_name, 'r') as g4:

        for var in range(1,4): #Skips E, should be cont.

            g4_data = g4['hcal_cells'][:nevts,:,var]

            centers, edges, width = get_bin_edges(g4_data)

            # The MASK of 0 should not be included for Z
            if var_str[var] == "Z":
                if centers[0] == 0:
                    centers = np.delete(centers,0)
                    edges = np.delete(edges,0)
                
        
            bin_dict[f"centers{var_str[var]}"] = centers
            bin_dict[f"edges{var_str[var]}"] = edges 
            bin_dict[f"width{var_str[var]}"] = width 

        bin_dict[f"widthE"] = 2e-5 #20keV fake width, for smear.py

        # print("\nbinning_utils.py L48: Dictionary Keys = ",bin_dict.keys())

    return bin_dict


def get_digits_dict(continuous_file, dset_name, bin_dict):

    digit_dict = {}

    for var in range(1, 4):
        continuous_data = continuous_file[dset_name][:,:,var]
        digits = np.digitize(continuous_data,bin_dict[f"edges{var_str[var]}"])
        print("Sample Bin # ",var_str[var],": ",digits[100,:10])
        digit_dict[f"digits{var_str[var]}"] = digits - 1  # -1 for 0th index

    return digit_dict
