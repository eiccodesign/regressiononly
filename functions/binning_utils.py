import h5py
import numpy as np
from tqdm import tqdm
import sys

var_str = ["E","X","Y","Z"]
print(f"\nUsing variable strings {var_str} from binning_utils.py\n")

def get_bin_width(centers):

    # Assumes fixed-width binning, 
    # Can be changed to centers[1:] - centers[:-1]
    # print("\n\n get_bin_width: centers = ", centers)
    width = np.round(centers[2] - centers[1], 2)  # avoids [0] edgecase
    
    return width

def get_bin_edges(g4_cell_data):

    centers = np.unique(g4_cell_data)

    width = get_bin_width(centers) 

    edges = centers - width/2
    max_edge = centers[-1] + width/2
    edges = (np.append(edges,max_edge))
    edges = np.round(edges, 2)

    return centers, edges, width


def get_equidistant_layers(full_z_edges, n_segments):
    nZ = len(full_z_edges)-1
    print(f"\nIdentified {nZ} Longitudinal Layers\n")
    assert(nZ > n_segments)
    z_layers = []

    if nZ % n_segments != 0:
        sys.exit(f"ERROR: Please choose an integer factor for number of z-sections. \nThere are {nZ} sections total")

    #Add Front of Calorimeter

    n_skip = int(nZ/n_segments)
    z_layers = full_z_edges[::n_skip]

    #Make sure to include front of calorimeter
    if not(full_z_edges[-1] in z_layers):
        z_layers.insert(0,full_z_edges[0])

    #Make sure to include back of calorimeter
    if not(full_z_edges[-1] in z_layers):
        z_layers.append(full_z_edges[-1])

    return z_layers


def get_bin_dict(geant4_name, nevts = 100_000, dataset = 'hcal_cells'):

    bin_dict = {}

    with h5py.File(geant4_name, 'r') as g4:

        for var in range(1,4): #Skips E, should be cont.

            g4_data = g4[dataset][:nevts,:,var]

            centers, edges, width = get_bin_edges(g4_data)

            # The MASK of 0 should not be included for Z
            # XY however, naturally go through 0, and can have real 0 values
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
    #function to get the bin number each float would belong to
    #very useful for masking and assigning data to bins w/o loops

    digit_dict = {}

    for var in range(1, 4):
        continuous_data = continuous_file[dset_name][:,:,var]
        digits = np.digitize(continuous_data,bin_dict[f"edges{var_str[var]}"])
        print("Sample Bin # ",var_str[var],": ",digits[100,:10])
        digit_dict[f"digits{var_str[var]}"] = digits - 1  # -1 for 0th index

    return digit_dict


def get_random_z_pos(full_z_edges,n_seg):
    nZ = len(full_z_edges)
    rand_Ls = []

    rand_Ls.append(full_z_edges[0])  #Beginning of Calo

    assert(nZ > n_seg)  # avoid infinite loops

    while len(rand_Ls) < (n_seg - 1):  #Fill to second-to-last
        zi = np.random.randint(0,nZ-2)  #-1 fencepost, -1 again avoid end of calo
        randZ = full_z_edges[zi]
        
        if randZ not in rand_Ls:
            rand_Ls.append(randZ)

    rand_Ls.append(full_z_edges[-1])  #Add end of calo
            
    return np.round(np.sort(rand_Ls),2)


def get_nrand_z_pos(full_z_edges, n_seg, nrand=1):
    # sets n random Zs, and sets rest to back of calorimeter

    nZ = len(full_z_edges)-1
    assert nrand < n_seg
    assert (nZ > n_seg)  # avoid infinite loops

    rand_Ls = []

    while (len(rand_Ls) < nrand):
        rand_int = np.random.randint(0, nZ-(n_seg-nrand))
        randZ = full_z_edges[rand_int]
        if randZ not in rand_Ls:
            rand_Ls.append(full_z_edges[rand_int])

    for zedge in full_z_edges[-(n_seg-nrand):]:
        rand_Ls.append(zedge)

    assert (np.max(rand_Ls) == np.max(full_z_edges))
    return np.round(np.sort(rand_Ls), 2)


def Sum_EinZbins(cellE, cellZ, zbins):

    # this sums cells in the same Z range
    if (len(cellE) != len(cellZ)):
        exit("ERROR: cellE and cellZ must be the same lengeth")

    counts, bins = np.histogram(np.ravel(cellZ), bins=zbins,
                                weights = np.ravel(cellE))

    count_mask = counts != 0

    return(counts[count_mask], count_mask) 
    #count mask is used later to get appropriate z-centers


def get_newZbinned_cells(cellE, cellZ, cellX, cellY,
                         edgesX, edgesY, zbins):
    #currently assumes only varrying Z.
    #generalize by passing in bins for XY

    # centersX, edgesX, widthX = get_bin_edges(cellX)
    # centersY, edgesY, widthY = get_bin_edges(cellY)
    centersX = (edgesX[0:-1] + edgesX[1:])/2
    centersY = (edgesY[0:-1] + edgesY[1:])/2
    centersZ = (zbins[0:-1] + zbins[1:])/2

    nX = len(centersX)
    nY = len(centersY)
    nZ = len(zbins)-1

    assert(len(centersZ) == nZ)

    # this sums cells in the same XYZ range
    # XY are cell-width, and Z is nZ layers
    counts, bins = np.histogramdd(
        (cellX,cellY,cellZ),
        bins=(edgesX,edgesY,zbins),
        weights = cellE)

    # Now that we have cell-sums along Z,
    # Transorm back into coordinates
    newZ, newX, newY = xyz_coordinates_from_counts(counts, centersX, 
                                                   centersY, centersZ)

    count_mask = counts != 0
    cellE_log10 = np.log10(counts[count_mask])

    return(cellE_log10, newZ, newX, newY)


def xyz_masks_from_counts(counts):

    shape = np.shape(counts)
    assert(len(shape) == 3)
    nX = shape[0]
    nY = shape[1]
    nZ = shape[2]

    x_hits = []
    y_hits = []
    z_hits = []

    for zi in range(nZ):
        for yi in range(nY):
            x_hits.append(counts[:,yi,zi])
        for xi in range(nX):
            y_hits.append(counts[xi,:,zi])

    for yi in range(nY):
        for xi in range(nX):
            z_hits.append(counts[xi,yi,:])

    x_hits = np.ravel(x_hits)
    y_hits = np.ravel(y_hits)
    z_hits = np.ravel(z_hits)

    x_mask = x_hits != 0
    y_mask = y_hits != 0
    z_mask = z_hits != 0 

    return x_mask, y_mask, z_mask


def xyz_coordinates_from_counts(counts, centersX, centersY, centersZ):

    shape = np.shape(counts)
    assert(len(shape) == 3)

    x_mask, y_mask, z_mask = xyz_masks_from_counts(counts)

    nX = shape[0]
    nY = shape[1]
    nZ = shape[2]

    x_coords = np.tile(centersX,nY*nZ)
    y_coords = np.tile(centersY,nX*nZ)
    z_coords = np.tile(centersZ,nX*nY)

    newX = x_coords[x_mask]
    newY = y_coords[y_mask]
    newZ = z_coords[z_mask]

    return newZ, newX, newY  #generators expect this order
