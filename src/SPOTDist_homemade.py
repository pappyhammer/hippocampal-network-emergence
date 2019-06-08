import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as scistats
import seaborn as sns
import os
# import hdbscan
# from sklearn.manifold import TSNE as t_sne
import pandas as pd
import hdf5storage
from numba import jit, prange


class Epoch:
    """
    Represent an epoch with n_neurons and n_frames.
    Will contain the cross_correlation value between each pair of neurons of this epoch.
    """
    def __init__(self, epoch, id_value):
        """
        Epoch is a 2D array (cell vs frames)
        :param epoch:
        :param id: int value, use to identify this epoch, should be a different number than other epochs
        """
        self.epoch = epoch
        self.n_neurons = epoch.shape[0]
        self.n_frames = epoch.shape[1]
        self.id = id_value
        # a dict with key a tuple of int representing two neurons numbers
        # and value is a list of 2 arrays representing the lags indices and the normalize cross_correlation
        # value (only the ones > 0)
        self.neurons_pairs_cross_correlation = dict()

    def pairs_of_neurons(self):
        """
        Compute all pairs of neurons
        :return: a list of tuple of 2 int representing the pairs of neurons
        """
        pairs = []
        for neuron_1 in np.arange(self.n_neurons-1):
            for neuron_2 in np.arange(neuron_1+1, self.n_neurons):
                pairs.append((neuron_1, neuron_2))
        return pairs

    def compute_all_normalized_cross_correlation(self):
        """
        compute for all pair of neurons the normalize cross_correlation with respect of lag
        Will need to be optimized using C through Cython
        :return:
        """
        # we loop to get all pairs of neurons
        for neuron_1 in np.arange(self.n_neurons-1):
            for neuron_2 in np.arange(neuron_1+1, self.n_neurons):
                self.compute_normalized_cross_correlation(neuron_1, neuron_2)

    def compute_normalized_cross_correlation(self, neuron_1, neuron_2):
        """
        compute for a pair of neurons the normalize cross_correlation with respect of lag
        Will need to be optimized using C through Cython
        :param neuron_1: array of length self.n_frames
        :param neuron_2: array of length self.n_frames
        :return:
        """
        # we compute the cross_correlation of the two neuron for each lag
        n_lags = (self.n_frames * 2) + 1
        cross_correlation_array = np.zeros(n_lags)
        zero_lag_index = self.n_frames + 1
        lags = np.arange(-self.n_frames, self.n_frames+1)
        neuron_1_spikes = self.epoch[neuron_1, :]
        neuron_2_spikes = self.epoch[neuron_2, :]

        for lag_time_index, lag_time in enumerate(lags):
            # we shift the first neuron of lag_time
            shifted_neuron_1_spikes = np.zeros(self.n_frames, dtype="int8")
            if lag_time == 0:
                shifted_neuron_1_spikes = neuron_1_spikes
            elif lag_time > 0:
                shifted_neuron_1_spikes[lag_time:] = neuron_1_spikes[:-lag_time]
            else:
                shifted_neuron_1_spikes[:lag_time] = neuron_1_spikes[-lag_time:]
            # doing the correlation between neuron_1 shifted an neuron_2
            corr_value = np.correlate(shifted_neuron_1_spikes, neuron_2_spikes)
            cross_correlation_array[lag_time_index] = corr_value

        show_corr_bar_chart = (np.sum(cross_correlation_array) > 0)
        # if len(np.where(cross_correlation_array < 0)[0]) > 0:
        #     print(f"len neg {len(np.where(cross_correlation_array < 0)[0])}")
        # comment next line to show the correlation plot when the sum is > 0
        show_corr_bar_chart = False
        # if show_corr_bar_chart:
        #     fig, ax = plt.subplots(nrows=1, ncols=1,
        #                            figsize=(8, 3))
        #     rects1 = plt.bar(lags, cross_correlation_array,
        #                      color='black')
        #     plt.title("non normalized cross-correlation")
        #     plt.show()
        # normalization so that the sum of the values is equal to 1
        if np.sum(cross_correlation_array) > 0:
            normalized_cross_correlation_array = cross_correlation_array / np.sum(cross_correlation_array)
        else:
            normalized_cross_correlation_array = cross_correlation_array

        if show_corr_bar_chart:
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   figsize=(8, 3))
            rects1 = plt.bar(lags, normalized_cross_correlation_array,
                             color='black')
            plt.title(f"Normalized cross-correlation between neuron {neuron_1} and neuron {neuron_2} on epoch {self.id}")
            plt.show()

        # we could save the results as a np array to len n_frames
        # but to save memory we will just save the positive values
        # in one array we save the indices of the lags with positive corr (we could also keep a binary
        # array with 1 in time lags where a corr value is > 0 (might use less memory)
        # in another one we save the corr values as float value
        lags_indices = np.where(normalized_cross_correlation_array > 0)[0]
        cross_correlation_values = normalized_cross_correlation_array[normalized_cross_correlation_array > 0]
        self.neurons_pairs_cross_correlation[(neuron_1, neuron_2)] = [lags_indices, cross_correlation_values]
        # self.neurons_pairs_cross_correlation[(neuron_1, neuron_2)] = normalized_cross_correlation_array

    def get_normalized_cross_correlation_array(self, cells_pair):
        lags_indices, cross_correlation_values = self.neurons_pairs_cross_correlation[cells_pair]
        n_lags = (self.n_frames * 2) + 1
        normalized_cross_correlation_array = np.zeros(n_lags)
        normalized_cross_correlation_array[lags_indices] = cross_correlation_values
        return normalized_cross_correlation_array

    def __hash__(self):
        """
        compute a hash value, useful if we want to use Epoch as key in a dictionnary
        :return:
        """
        return self.id

    def __eq__(self, other):
        """
        Compare 2 epochs, useful if we want to use Epoch as key in a dictionnary
        :param other: Epoch object
        :return:
        """
        return self.id == other.id


def load_data_rasterdur(ms):
    """
    Used to load data. The code has to be manually change so far to change the data loaded.
    :return: return a 2D binary array representing a raster. Axis 0 (lines) represents the neurons (cells) and axis 1
    (columns) represent the frames (in our case sampling is approximatively 10Hz, so 100 ms by frame).
    """
    spike_nums_dur = ms.spike_struct.spike_nums_dur
    # spike_nums_dur = spike_nums_dur[:50, :10000] # TO TEST CODE
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


def median_normalization(traces):
    n_cells, n_frames = traces.shape
    for i in range(n_cells):
        traces[i, :] = traces[i, :] / np.median(traces[i, :])
    return traces


def load_data_traces(ms, use_median_norm=True):
    """
    Used to load data. The code has to be manually change so far to change the data loaded.
    :return: return a 2D binary array representing a raster. Axis 0 (lines) represents the neurons (cells) and axis 1
    (columns) represent the frames (in our case sampling is approximatively 10Hz, so 100 ms by frame).
    """
    if ms.raw_traces is None:
        raw_traces_loaded = ms.load_raw_traces_from_npy(path=f"p{ms.age}/{ms.description.lower()}/")
        if not raw_traces_loaded:
            ms.load_tiff_movie_in_memory()
            ms.raw_traces = ms.build_raw_traces_from_movie()
    raw_traces = ms.raw_traces
    traces = np.copy(raw_traces)
    if use_median_norm is True:
        median_normalization(traces)
    traces = traces[:100, :1000]  # TO TEST CODE
    return traces


@jit(nopython=True)
def signature_emd(x, y):
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


def compute_emd_for_a_pair_of_neurons_between_2_epochs(distance_metric, epoch_1, epoch_2, neurons_pair):
    """
    :param distance_metric: how to measure the distance between neuron-pair cross-correlation on 2 epochs
    :param epoch_1: first epoch (Epoch instance)
    :param epoch_2: second epoch (Epoch instance)
    :param neurons_pair: a tuple of in representing the two neurons pair
    :return: emd value is the generic name for the distance, not necessarily the emd value
    """

    epoch_1_cross_corr = epoch_1.get_normalized_cross_correlation_array(neurons_pair)
    epoch_2_cross_corr = epoch_2.get_normalized_cross_correlation_array(neurons_pair)

    if distance_metric == "Wasserstein distance":
        emd_value = scistats.wasserstein_distance(np.arange(len(epoch_1_cross_corr)),
                                             np.arange(len(epoch_2_cross_corr)),
                                             u_weights=epoch_1_cross_corr,
                                             v_weights=epoch_2_cross_corr)
        return emd_value

    if distance_metric == "EMD_Battaglia":
        emd_value = signature_emd(epoch_1_cross_corr, epoch_2_cross_corr)
        return emd_value


def compute_spotdis_between_2_epochs(distance_metric, epoch_1, epoch_2):
    """
    Will compute the spotdis value between 2 epochs, corresponding to the average EMD between 2 epochs
    :param distance_metric: how to measure the distance between neuron-pair cross-correlation on 2 epochs
    :param epoch_1: an Epoch instance
    :param epoch_2: an Epoch instance
    :return: a float number representing the spotdis value. The smaller the value the more similar
    are the pairs of neurons.

    """
    neurons_pairs = epoch_1.pairs_of_neurons()
    # count the number of emd values that has been sum, (equivalent to sum of Wij,km) in the paper equation
    total_count = 0
    emd_sum = 0
    for neurons_pair in neurons_pairs:
        # we get the lags and correlations values of both epoch to average the emd
        # lags_corr_values_e_1, c_1 = epoch_1.neurons_pairs_cross_correlation[neurons_pair]
        # lags_corr_values_e_2, c_2 = epoch_2.neurons_pairs_cross_correlation[neurons_pair]
        # # if one of the epoch pair of neurons has an empty correlation hisogram, then we don't count it
        # if (len(lags_corr_values_e_1[0]) == 0) or (len(lags_corr_values_e_2[0]) == 0):
        #     continue

        if (np.sum(epoch_1.neurons_pairs_cross_correlation[neurons_pair]) == 0) or \
                (np.sum(epoch_2.neurons_pairs_cross_correlation[neurons_pair]) == 0):
            continue

        emd_value = compute_emd_for_a_pair_of_neurons_between_2_epochs(distance_metric, epoch_1=epoch_1,
                                                                       epoch_2=epoch_2, neurons_pair=neurons_pair)
        total_count += 1
        emd_sum += emd_value
    # computing the average
    if total_count > 0:
        spotdis_value = emd_sum / total_count
    else:

        # TODO: take in consideration the case: no pair of neurons in which both neurons fired in both epochs k and m
        # TODO: in that case total_count will be equal to zeros, for now we put spotdis value to 1 (should it be zero ?)
        # In the paper, they assume, that for each pair of epochs k and m,
        # there is at least one pair of neurons in which both neurons fired in both epochs k and m
        spotdis_value = 1
    return spotdis_value


def SPOT_Dist_JD_RD(ms, data_to_use, len_epoch=100, use_raster=False, distance_metric="EMD_Battaglia"):
    """"
    :var: data_to_use: can be rasterdur, raster or traces
    :param: use_raster: if True work with onsets only, by default False and use raster_dur
    :param: distance_metric: this distance is the one used to measure the distance between distributions of
                            cross-correlations for a given neuron pair on 2 epochs
    :param: len_epoch: length of the epoch to compute cross-coreelation between all pair of neurons.
                       Must divide data _to_use length

    """

    print(f"Data used to run homemade SPOT_Dist is {data_to_use}")

    possible_distance_metrics = ["EMD_Battaglia", "Wasserstein_distance"]
    if distance_metric not in possible_distance_metrics:
        distance_metric = "EMD_Battaglia"
        raise Exception("This metric is not avalaible, keep going using Pearson correlation")
    if distance_metric is None:
        distance_metric = "EMD_Battaglia"
        raise Exception("Distance metric was not specified, keep going using EMD_Battaglia")

    if data_to_use == "rasterdur":
        spike_nums_dur = load_data_rasterdur(ms)
        if use_raster is True:
            print(f"Using raster with onsets only - Cleaning rasterdur")
            spike_nums = build_spike_nums_and_peak_nums(spike_nums_dur)[0]
            data = spike_nums

        elif use_raster is False:
            # to check the shape of the data
            print(f"spike_nums_dur shape is {spike_nums_dur.shape}")
            # number of frames in our raster
            n_frames = spike_nums_dur.shape[1]
            # number of neurons
            n_neurons = spike_nums_dur.shape[0]
            data = spike_nums_dur

    if data_to_use == "traces":
        use_raster = False
        traces = load_data_traces(ms)
        # to check the shape of the data
        print(f"traces shape is {traces.shape}")
        # number of frames in our raster
        n_frames = traces.shape[1]
        # number of neurons
        n_neurons = traces.shape[0]
        data = traces

    # then computing the number of epochs in our raster
    n_epochs = n_frames // len_epoch
    print(f"Epoch lenght is {len_epoch} frames, total number of epochs is {n_epochs}")
    # to make things easy for now, the number of frames should be divisible by the length of epochs
    if (n_frames % len_epoch) != 0:
        raise Exception("number of frames {n_frames} not divisible by {len_epoch}")

    # we creates the epochs
    # Two choices: produce a 3D array, with first dimension being the number of epochs, 2nd nb oc cells, and 3rd frames
    # or create an object Epoch to which we pass a 2D array (cells vs frames)
    # we do a sliding window to create the epochs over the spike_nums
    # epochs = np.zeros((n_epochs, n_neurons, len_epoch), dtype="int8")
    epochs = []
    for epoch_index, frame_index in enumerate(np.arange(0, n_frames, len_epoch)):
        epoch = data[:, frame_index:frame_index+len_epoch]
        epochs.append(Epoch(epoch=epoch, id_value=epoch_index))
        # epochs[epoch_index] = epoch

    # then we compute for each epoch the n_neurons(n_neurons − 1)/2 normalized cross-correlations between each pair of
    # neurons

    print(f"Computing all normalized cross correlation")
    for epoch in epochs:
        epoch.compute_all_normalized_cross_correlation()
    print(f"Computation of all normalized cross correlation done")

    # a 2d array that contains the spotdis value for each pair of epochs
    spotdis_values = np.zeros((n_epochs, n_epochs))

    print(f"Computing all spotdis value")
    for epoch_1_index in np.arange(n_epochs - 1):
        for epoch_2_index in np.arange(epoch_1_index+1, n_epochs):
            epoch_1 = epochs[epoch_1_index]
            epoch_2 = epochs[epoch_2_index]
            spotdis_value = compute_spotdis_between_2_epochs(distance_metric, epoch_1=epoch_1, epoch_2=epoch_2)
            spotdis_values[epoch_1_index, epoch_2_index] = spotdis_value
            spotdis_values[epoch_2_index, epoch_1_index] = spotdis_value
    # print(f"total_count is equal to zero for {len(spotdis_values==0)} spotdis values")
    print(f"Computation of all spotdis values is done")

    # Generate figure: dissimilarity matrice
    svm = sns.heatmap(spotdis_values)
    fig = svm.get_figure()
    save_formats = ["pdf", "png"]
    if isinstance(save_formats, str):
        save_formats = [save_formats]

    for save_format in save_formats:
        path_results = os.path.join(ms.param.path_results,
                                    f"{ms.description}_homemade_SPOTDist_heatmap_{data_to_use}_win_{len_epoch}.{save_format}")
        fig.savefig(f'{path_results}'
                    f'.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())

    np.save(os.path.join(ms.param.path_results, f"{ms.description}_homemade_SPOTDist_values_{data_to_use}_win_{len_epoch}.npy"),
            spotdis_values)

    return spotdis_value


# SPOT_Dist_JD_RD(ms, "traces", len_epoch=100, use_raster=False, distance_metric="EMD_Battaglia")
