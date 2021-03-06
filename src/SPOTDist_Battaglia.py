from numba import jit, prange
import numpy as np
import hdf5storage
import itertools
import seaborn as sns
import hdbscan
import os
from pattern_discovery.tools.misc import get_continous_time_periods


@jit(nopython=True)
def xcorr_list(in1, in2):
    """List of all time delays from a full cross correlation of the two inputs

    Parameters
    ----------
    in1 : numpy.ndarray
        Occurence times / indices
    in2 : numpy.ndarray
        Occurence times / indices
    """

    n1 = len(in1)
    n2 = len(in2)

    C = [0.0]*(n1*n2)
    for i in range(n1):
        for j in range(n2):
            C[i*n2+j] = in2[j] - in1[i]

    return C

@jit(nopython=True)
def signature_emd_(x, y):
    """A fast implementation of the EMD on sparse 1D signatures like described in:
    Grossberger, L., Battaglia, FP. and Vinck, M. (2018). Unsupervised clustering
    of temporal patterns in high-dimensional neuronal ensembles using a novel
    dissimilarity measure.

    Parameters
    ----------
    x : numpy.ndarray
        List of occurrences / a histogram signature
        Note: Needs to be non-empty and longer or equally long as y
    y : numpy.ndarray
        List of occurrences / a histogram signature
        Notes: Needs to be non-empty and shorter or equally long as x

    Returns
    -------
    Earth Mover's Distances between the two signatures / occurrence lists

    """

    Q = len(x)
    R = len(y)

    if Q == 0 or R == 0:
        return np.nan

    if Q < R:
        raise AttributeError('First argument must be longer than or equally long as second.')
    # print(f"x {x}")
    x.sort()
    y.sort()

    # Use integers as weights since they are less prone to precision issues when subtracting
    w_x = R  # = Q*R/Q
    w_y = Q  # = Q*R/R

    emd = 0.
    q = 0
    r = 0

    while q < Q:
        if w_x <= w_y:
            cost = w_x * abs(x[q] - y[r])
            w_y -= w_x
            w_x = R
            q += 1
        else:
            cost = w_y * abs(x[q] - y[r])
            w_x -= w_y
            w_y = Q
            r += 1

        emd += cost

    # Correct for the initial scaling to integer weights
    return emd/(Q*R)


@jit(nopython=True, parallel=True)
def xcorr_spotdis_cpu_(spike_times, ii_spike_times, epoch_index_pairs):
    """Compute distances between channel cross correlation pairs using all available CPU cores.
    The specific type of distance is provided via the parameter 'metric'.

    Parameters
    ----------
    spike_times : numpy.ndarray
        1 dimensional matrix containing all spike times

    ii_spike_times : numpy.ndarray
        MxNx2 dimensional matrix containing the start and end index for the spike_times array
        for any given epoch and channel combination

    epoch_index_pairs : numpy.ndarray
        (M*(M-1)/2)x2 dimensional matrix containing all unique epoch index pairs

    Returns
    -------
    distances : numpy.ndarray
        MxM distance matrix with numpy.nan for unknown distances and on the diagonal
    """

    # Get data dimensions
    n_epochs = ii_spike_times.shape[0]
    n_channels = ii_spike_times.shape[1]
    n_epoch_index_pairs = epoch_index_pairs.shape[0]

    # Initialize distance matrix
    distances = np.full((n_epochs, n_epochs), np.nan)

    nan_count = 0.0

    # For each epoch pair
    for i in prange(n_epoch_index_pairs): #TODO: add prange
        e1 = epoch_index_pairs[i,0]
        e2 = epoch_index_pairs[i,1]

        # Compute distances for all xcorr pairs between the two epochs
        xcorr_distances = np.full(int(n_channels * (n_channels-1) / 2), np.nan)
        n_xcorr_distances = 0
        i_xcorr_distance = -1
        for c1 in range(n_channels):
            for c2 in range(c1):
                i_xcorr_distance += 1

                # Only compute the xcorrs and distance in case there is a spike in all relevant channels
                if ((ii_spike_times[e1,c1,1] - ii_spike_times[e1,c1,0]) > 0
                    and (ii_spike_times[e1,c2,1] - ii_spike_times[e1,c2,0]) > 0
                    and (ii_spike_times[e2,c1,1] - ii_spike_times[e2,c1,0]) > 0
                    and (ii_spike_times[e2,c2,1] - ii_spike_times[e2,c2,0]) > 0):

                    # Compute the xcorrs
                    xcorr1 = xcorr_list(
                            spike_times[ii_spike_times[e1,c1,0]:ii_spike_times[e1,c1,1]],
                            spike_times[ii_spike_times[e1,c2,0]:ii_spike_times[e1,c2,1]])
                    xcorr2 = xcorr_list(
                            spike_times[ii_spike_times[e2,c1,0]:ii_spike_times[e2,c1,1]],
                            spike_times[ii_spike_times[e2,c2,0]:ii_spike_times[e2,c2,1]])
                    # print(f"xcorr1 {xcorr1}")
                    # EMD
                    if len(xcorr1) >= len(xcorr2):
                        xcorr_distances[i_xcorr_distance] = signature_emd_(xcorr1, xcorr2)
                    else:
                        xcorr_distances[i_xcorr_distance] = signature_emd_(xcorr2, xcorr1)

                    n_xcorr_distances = n_xcorr_distances + 1
                else:
                    nan_count = nan_count + 1

        # Save average xcorr distance
        if n_xcorr_distances > 0:
            distances[e1, e2] = np.nanmean(xcorr_distances)
            distances[e2, e1] = distances[e1, e2]

    percent_nan = nan_count / ((n_channels*(n_channels-1)/2)*n_epoch_index_pairs)

    return distances, percent_nan


def SPOT_Dist_Battaglia(data, len_epoch=100):
    spike_times_matrix = data
    n_cells, n_frames = spike_times_matrix.shape
    # print(f"spike_times matrix shape is {spike_times_matrix.shape}")

    # then computing the number of epoch in our raster
    n_epochs = n_frames // len_epoch
    print(f"Epoch lenght is {len_epoch} frames, total number of epochs is {n_epochs}")
    # to make things easy for now, the number of frames should be divisible by the length of epochs
    if (n_frames % len_epoch) != 0:
        raise Exception("number of frames {n_frames} not divisible by {len_epoch}")

    n_tot_spikes = np.sum(spike_times_matrix[:, :], dtype="int64")

    print(f"N total spikes is {n_tot_spikes}")

    spike_times = np.zeros(n_tot_spikes)
    ii_spike_times = np.zeros((n_epochs, n_cells, 2), dtype="int16")

    print(f"Start reshaping the data")
    k = 0
    for i in range(n_cells):
        for j in range(n_epochs):
            spike_times_cell_i_epoch_j = np.where(spike_times_matrix[i, np.arange(j*len_epoch, (j+1)*len_epoch)])
            if len(spike_times_cell_i_epoch_j[0]) > 0:
                spike_times[np.arange(k, (k + len(spike_times_cell_i_epoch_j[0])))] = spike_times_cell_i_epoch_j
                ii_spike_times[j, i, 0] = k
                ii_spike_times[j, i, 1] = k + len(spike_times_cell_i_epoch_j[0])
                k = k + len(spike_times_cell_i_epoch_j[0])
            elif len(spike_times_cell_i_epoch_j[0]) == 0:
                # print(f" Cell {i} has no spikes during epoch {j}")
                ii_spike_times[j, i, 0] = k
                ii_spike_times[j, i, 1] = k
    print(f"Data are reshaped")
    # print(f"spike_times array is {spike_times}")
    # print(f" matrix of start time in spike_times array for epoch i vs neuron j is {ii_spike_times[:, :, 0]}")
    # print(f" matrix of end time in spike_times array for epoch i vs neuron j is {ii_spike_times[:, :, 1]}")

    n_unique_pairs = (n_epochs * (n_epochs - 1)) / 2
    n_unique_pairs = int(n_unique_pairs)
    epoch_index_pairs = np.zeros((n_unique_pairs, 2), dtype="int16")

    k = 0
    for i in range(n_epochs - 1):
        for j in np.arange(i + 1, n_epochs):
            epoch_index_pairs[k, 0] = i
            epoch_index_pairs[k, 1] = j
            k = k + 1

    print(f"Starting to compute xcorr and EMD values")
    distances = xcorr_spotdis_cpu_(spike_times, ii_spike_times, epoch_index_pairs)[0]

    np.fill_diagonal(distances, 0)

    percent_nan = xcorr_spotdis_cpu_(spike_times, ii_spike_times, epoch_index_pairs)[1]

    print(f"percent of NaN is {percent_nan}")

    return distances, percent_nan


# SPOT_Dist_Battaglia(ms, len_epoch=100, use_raster=False)

