import numpy as np
from matplotlib import pyplot as plt

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
        # comment next line to show the correlation plot when the sum is > 0
        show_corr_bar_chart = False
        if show_corr_bar_chart:
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   figsize=(8, 3))
            rects1 = plt.bar(lags, cross_correlation_array,
                             color='black')
            plt.title("non normalized")
            plt.show()
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
            plt.title("normalized")
            plt.show()
        # we could save the results as a np array to len n_frames
        # but to save memory we will just save the positive values
        # in one array we save the indices of the lags with positive corr (we could also keep a binary
        # array with 1 in time lags where a corr value is > 0 (might use less memory)
        # in another one we save the corr values as float value
        lags_indices = np.where(normalized_cross_correlation_array > 0)[0]
        cross_correlation_values = normalized_cross_correlation_array[normalized_cross_correlation_array > 0]
        self.neurons_pairs_cross_correlation[(neuron_1, neuron_2)] = [lags_indices, cross_correlation_values]

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

def load_data():
    """
    Used to load data. The code has to be manually change so far to change the data loaded.
    :return: return a 2D binary array representing a raster. Axis 0 (lines) represents the neurons (cells) and axis 1
    (columns) represent the frames (in our case sampling is approximatively 10Hz, so 100 ms by frame).
    """
    path_data = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/data/data_TIPE/"
    loaded_data = np.load(path_data + "P60_a529_2015_02_25_rasters_reduced.npz")
    # put to true to print the name of the variables available in this npz file
    print_variable_names = False
    if print_variable_names:
        for key in loaded_data.keys():
            print(f"Variable in npz file: {key}")
    # spike_nums is a binary 2D array containing the onsets (1 if onset)
    spike_nums = loaded_data["spike_nums"]
    # spike_nums is a binary 2D array containing the active period of neuron (a neuron is considered active from its
    # onset to its peak during a transient, thus when the value is 1 the neuron is active)
    spike_nums_dur = loaded_data["spike_nums_dur"]

    return spike_nums_dur


def compute_emd_between_2_pair_of_neurons(epoch_1, epoch_2, neurons_pair):
    """

    :param epoch_1: first epoch (Epoch instance)
    :param epoch_2: second epoch (Epoch instance)
    :param neurons_pair: a tuple of in representing the two neurons pair
    :return:
    """
    # TODO: write the code.

    return 0.


def compute_spotdis_between_2_epochs(epoch_1, epoch_2):
    """
    Will compute the spotdis value between 2 epochs, corresponding to the average EMD between 2 epochs
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
        emd_value = compute_emd_between_2_pair_of_neurons(epoch_1=epoch_1, epoch_2=epoch_2, neurons_pair=neurons_pair)
        # we get the lags and correlations values of both epoch to average the emd
        lags_corr_values_e_1 = epoch_1.neurons_pairs_cross_correlation[neurons_pair]
        lags_corr_values_e_2 = epoch_2.neurons_pairs_cross_correlation[neurons_pair]
        # if one of the epoch pair of neurons has an empty correlation hisogram, then we don't count it
        if (len(lags_corr_values_e_1[0]) == 0) or (len(lags_corr_values_e_2[0]) == 0):
            continue
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
        print("total_count is equal to zero")
        spotdis_value = 1

    return spotdis_value

def main():
    # loadind the raster
    spike_nums = load_data()
    # to check the shape of the data
    print(f"spike_nums.shape {spike_nums.shape}")
    # SPOTDis first step
    # Construct the pairwise epoch-to-epoch SPOTDis measure on the matrix of cross-correlations among all neuron pairs

    # number of frames in our raster
    n_frames = spike_nums.shape[1]
    # number of neurons
    n_neurons = spike_nums.shape[0]
    # we fix the length of an epoch, knowing than 1 frame is equal to 100 ms approximately
    len_epoch = 50
    # then computing the number of epoch in our raster
    n_epochs = n_frames // len_epoch
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
        epoch = spike_nums[:, frame_index:frame_index+len_epoch]
        epochs.append(Epoch(epoch=epoch, id_value=epoch_index))
        # epochs[epoch_index] = epoch

    # then we compute for each epoch the n_neurons(n_neurons − 1)/2 normalized cross-correlations between each pair of
    # neurons
    for epoch in epochs:
        epoch.compute_all_normalized_cross_correlation()

    # a 2d array that contains the spotdis value for each pair of epochs
    spotdis_values = np.zeros((n_epochs, n_epochs))

    for epoch_1_index in np.arange(n_epochs - 1):
        for epoch_2_index in np.arange(epoch_1_index+1, n_epochs):
            epoch_1 = epochs[epoch_1_index]
            epoch_2 = epochs[epoch_2_index]
            spotdis_value = compute_spotdis_between_2_epochs(epoch_1=epoch_1, epoch_2=epoch_2)
            spotdis_values[epoch_1_index, epoch_2_index] = spotdis_value
            spotdis_values[epoch_2_index, epoch_1_index] = spotdis_value


# periode1 = date1
# periode2 = date2
# duree =

def dissimilarite_neurones(n1,t1,n2,t2,duree):
    dis = 0
    for k in range(duree):
        dis += (A[n1][t1+k] - A[n2][t2+k])**2
    return(dis**(1/2))
    
    
# n = len A #nombre de neurones si les neurones sont en ligne
# m = len A[0]  #nombre d'enregistrements


def dissimilarite_periodes(periode1,periode2,duree):
    dissim = []
    for i in range(n):
        for j in range(i+1,n):
            dissim.append(dissimilarite(i,periode1,j,perode2,duree))
        
def normalisation(liste):
    n = len (liste)
    max = liste[0]
    for k in range(n):
        if liste[k] > max:
            max = liste[k]
    if max==0:     #ne peut arriver que si t1=t2 en théorie mais on sait jamais
        return liste
    else:
        return (1/max*liste)
        
#Pour une periode 1 fixée:

# lag =       #mettre l'écart entre deux periodes consécutives testées (1 par défaut mais peut etre pas utile, trop petit?

def dissimilarite_periode1(periode1,lag,duree):
    dis = []
    t = periode1
    while (periode + lag + duree) < m-1 :
        d = dissimilarite_periodes(periode1,periode+lag,duree)
        dis.append(normalisation(d))
    return (dis)
    
#Et on applique l'algo de clustering au résultat de cette fonction,
# puis on change la valeur de periode1 pour vérifier?? ou une seule periode1 suffit? Ou ce n'est pas ça du tout?

main()