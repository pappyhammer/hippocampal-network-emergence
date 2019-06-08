from numba import jit, prange
import numpy as np
import hdf5storage
import itertools
import seaborn as sns
import hdbscan


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

    # Use integers as weights since they are less prome to precision issues when subtracting
    w_x = R # = Q*R/Q
    w_y = Q # = Q*R/R

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

def load_data():
    file_name = 'D:/Robin/data_hne/data/p41/p41_19_04_30_a000/predictions/' \
                'P41_19_04_30_a000_predictions_meso_v1_epoch_9.mat'
    data = hdf5storage.loadmat(file_name)
    predictions = data["predictions"]
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    spike_nums_dur = predictions.astype("int8")
    # print(f"toto {spike_nums_dur}")
    # spike_nums_dur = loaded_data["spike_nums_dur"]
    # n_cells, n_frames = spike_nums_dur.shape
    # spike_nums_dur = spike_nums_dur[:200, :10000]
    return spike_nums_dur


def build_spike_nums_and_peak_nums(spike_nums_dur):
    n_cells, n_frames = spike_nums_dur.shape
    spike_nums = np.zeros((n_cells, n_frames), dtype="int8")
    peak_nums = np.zeros((n_cells, n_frames), dtype="int8")
    for cell in np.arange(n_cells):
        transient_periods = get_continous_time_periods(spike_nums_dur[cell])
        for transient_period in transient_periods:
            onset = transient_period[0]
            peak = transient_period[1]
            # if onset == peak:
            #     print("onset == peak")
            spike_nums[cell, onset] = 1
            peak_nums[cell, peak] = 1
    return spike_nums, peak_nums


def main():
    rasterdur = True
    spike_nums_dur = load_data()

    if rasterdur is True:
        spike_times_matrix = spike_nums_dur
    if rasterdur is False:
        spike_times_matrix = build_spike_nums_and_peak_nums(spike_nums_dur)[0]

    n_cells, n_frames = spike_times_matrix.shape

    # we fix the length of an epoch, knowing than 1 frame is equal to 100 ms approximately
    len_epoch = 250
    # then computing the number of epoch in our raster
    n_epochs = n_frames // len_epoch
    # to make things easy for now, the number of frames should be divisible by the length of epochs
    if (n_frames % len_epoch) != 0:
        raise Exception("number of frames {n_frames} not divisible by {len_epoch}")

    n_tot_spikes = np.sum(spike_times_matrix[:, :])
    print(f"N total spikes is {n_tot_spikes}")

    spike_times = np.zeros(n_tot_spikes)
    ii_spike_times = np.zeros((n_epochs, n_cells, 2), dtype="int16")

    k = 0
    for i in range(n_cells):
        for j in range(n_epochs):
            spike_times_cell_i_epoch_j = np.where(spike_times_matrix[i, np.arange(j*len_epoch, (j+1)*len_epoch)])
            # print(f" spike_times for cell {i} during epoch {j} are {spike_times_cell_i_epoch_j}")
            if len(spike_times_cell_i_epoch_j[0]) > 0:
                spike_times[np.arange(k, (k + len(spike_times_cell_i_epoch_j[0])))] = spike_times_cell_i_epoch_j
                ii_spike_times[j, i, 0] = k
                ii_spike_times[j, i, 1] = k + len(spike_times_cell_i_epoch_j[0]) - 1
                k = k + len(spike_times_cell_i_epoch_j[0])
            elif len(spike_times_cell_i_epoch_j[0]) == 0:
                ii_spike_times[j, i, 0] = k
                ii_spike_times[j, i, 1] = k

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

    distances = xcorr_spotdis_cpu_(spike_times, ii_spike_times, epoch_index_pairs)[0]

    np.fill_diagonal(distances, 0)

    percent_nan = xcorr_spotdis_cpu_(spike_times, ii_spike_times, epoch_index_pairs)[1]

    print(f"percent of NaN is {percent_nan}")


    ax = sns.heatmap(distances)
    fig = ax.get_figure()
    save_formats = ["pdf"]
    if isinstance(save_formats, str):
        save_formats = [save_formats]

    path_results = "D:/Robin/data_hne/data/p41/p41_19_04_30_a000/spot_dist_src"
    for save_format in save_formats:
        fig.savefig(f'{path_results}/spot_dist_matrix'
                    f'.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())



    # ## DO HDBSCAN CLUSTERING ON DISSIMILARITY MATRIX (SPOTDIS VALUES) ##
    #
    # clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    #                             gen_min_span_tree=False, leaf_size=40,
    #                             metric='precomputed', min_cluster_size=2, min_samples=None, p=None)
    # # metric='precomputed' euclidean
    # clusterer.fit(distances)
    #
    # labels = clusterer.labels_
    # # print(f"labels.shape: {labels.shape}")
    # print(f"N clusters hdbscan: {labels.max()+1}")
    # # print(f"labels: {labels}")
    # print(f"With no clusters hdbscan: {len(np.where(labels == -1)[0])}")
    # n_clusters = 0
    # if labels.max() + 1 > 0:
    #     n_clusters = labels.max() + 1
    #
    # if n_clusters > 0:
    #     n_epoch_by_cluster = [len(np.where(labels == x)[0]) for x in np.arange(n_clusters)]
    #     print(f"Number of epochs by clusters hdbscan: {' '.join(map(str, n_epoch_by_cluster))}")
    #
    # distances_order = np.copy(distances)
    # labels_indices_sorted = np.argsort(labels)
    # distances_order = distances_order[labels_indices_sorted, :]
    # distances_order = distances_order[:, labels_indices_sorted]
    #
    # # Generate figure: dissimilarity matrice ordered by cluster
    # svm = sns.heatmap(distances_order)
    # svm.set_yticklabels(labels_indices_sorted)
    # svm.set_xticklabels(labels_indices_sorted)
    # fig = svm.get_figure()
    # # plt.show()
    # save_formats = ["pdf"]
    # if isinstance(save_formats, str):
    #     save_formats = [save_formats]
    #
    # path_results = "D:/Robin/data_hne/data/p41/p41_19_04_30_a000/spot_dist_src"
    # for save_format in save_formats:
    #     fig.savefig(f'{path_results}/hdbscan_order_distances'
    #                 f'.{save_format}',
    #                 format=f"{save_format}",
    #                 facecolor=fig.get_facecolor())
    # # plt.close()


main()

