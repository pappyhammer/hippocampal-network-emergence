"""
SPADE is the combination of a mining technique and multiple statistical tests
to detect and assess the statistical significance of repeated occurrences of
spike sequences (spatio-temporal patterns, STP).

Given a list of Neo Spiketrain objects, assumed to be recorded in parallel, the
SPADE analysis can be applied as demonstrated in this short toy example of 10
artificial spike trains of exhibiting fully synchronous events of order 10.

This modules relies on the implementation of the fp-growth algorithm contained
in the file fim.so which can be found here (http://www.borgelt.net/pyfim.html)
and should be available in the spade_src folder (elephant/spade_src/).
If the fim.so module is not present in the correct location or cannot be
imported (only available for linux OS) SPADE will make use of a python
implementation of the fast fca algorithm contained in
elephant/spade_src/fast_fca.py, which is about 10 times slower.

:copyright: Copyright 2017 by the Elephant team, see AUTHORS.txt.
:license: BSD, see LICENSE.txt for details.
"""
import neo as n
import elephant
import spade_copy_elephant
import elephant.spike_train_generation
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt
import quantities as pq
import warnings

warnings.simplefilter('once', UserWarning)
try:
    from mpi4py import MPI  # for parallelized routines

    HAVE_MPI = True
except ImportError:  # pragma: no cover
    HAVE_MPI = False

try:
    from elephant.spade_src import fim

    HAVE_FIM = True
except ImportError:  # pragma: no cover
    HAVE_FIM = False


def generate_poisson_pattern(n_cells, len_epoch, min_isi, max_isi, min_spikes_cell, max_spikes_cell):
    range_isi = np.random.randint(min_isi, max_isi, size=n_cells)
    range_num_spikes_per_cell = np.random.randint(min_spikes_cell, max_spikes_cell, size=n_cells)

    # create random pattern
    pattern = np.zeros((n_cells, len_epoch))

    for i in range(n_cells):
        isi = np.random.poisson(range_isi[i], (range_num_spikes_per_cell[i] - 1))
        spikes_times = np.zeros((range_num_spikes_per_cell[i]), dtype="int8")
        start = np.round(0.65 * len_epoch / range_num_spikes_per_cell[i])
        spikes_times[0] = np.random.randint(2, start)
        for j in np.arange(1, range_num_spikes_per_cell[i]):
            spikes_times[j] = spikes_times[j - 1] + isi[j - 1]
        pattern[i, spikes_times] = 1

    return pattern


def generate_artificial_data():
    random_pattern_order = True
    known_pattern_order = False  # This option is obsolete, do not use
    use_one_shuffle_per_pattern = True
    do_general_shuffling_on_full_raster = False
    fuse_raster_with_noise = True
    # DEFINE RASTER #
    n_cells = 50
    len_epoch = 100
    if random_pattern_order:
        n_frames = 3000
    if known_pattern_order:
        n_frames = 1200

    art_raster_dur = np.zeros((n_cells, n_frames), dtype="int8")
    art_raster_dur_pattern_shuffle = np.zeros((n_cells, n_frames), dtype="int8")
    noise_matrix = np.zeros((n_cells, n_frames), dtype="int8")

    n_epochs = n_frames // len_epoch
    print(f"Epoch length is {len_epoch} frames, total number of epochs is {n_epochs}")
    # to make things easy for now, the number of frames should be divisible by the length of epochs
    if (n_frames % len_epoch) != 0:
        raise Exception("number of frames {n_frames} not divisible by {len_epoch}")

    ############################################
    # CREATE PATTERNS ASSEMBLIES AND SEQUENCES #
    ############################################

    # create pattern#1 = sequence in order
    pattern1 = np.zeros((n_cells, len_epoch))
    pattern1[0, 0] = 1
    for i in np.arange(0, n_cells):
        pattern1[i, i] = 1
        pattern1[i, i + 50] = 1
    # create pattern#1 shuffle = sequence in a shuffle order
    pattern1_shuffle = np.copy(pattern1)
    np.random.shuffle(pattern1_shuffle)

    # create pattern#2 = assemblies in order
    pattern2 = np.zeros((n_cells, len_epoch))
    pattern2[3: 5, 2: 4] = 1
    pattern2[0: 3, 14:16] = 1
    pattern2[7:11, 26:28] = 1
    pattern2[5:7, 38:40] = 1
    pattern2[3: 5, 50:52] = 1
    pattern2[7: 11, 62:64] = 1
    pattern2[5:7, 74:76] = 1
    pattern2[7:11, 86:88] = 1
    # create pattern#2 shuffle = assemblies in shuffle order
    pattern2_shuffle = np.copy(pattern2)
    np.random.shuffle(pattern2_shuffle)

    # create pattern#3 = sequence together with noise
    pattern3 = np.zeros((n_cells, len_epoch))
    n_cells_in_sequence = 40
    noisy_cells = n_cells - n_cells_in_sequence
    for i in range(n_cells_in_sequence):
        pattern3[i, i:i + 2] = 1
        pattern3[i, 20 + i:i + 22] = 1
    pattern3[n_cells_in_sequence:n_cells, :] = generate_poisson_pattern(noisy_cells, len_epoch, 10, 50, 1, 2)
    # create pattern#3 shuffle
    pattern3_shuffle = np.copy(pattern3)
    np.random.shuffle(pattern3_shuffle)

    # create pattern#4 = assemblies together with noise
    pattern4 = np.zeros((n_cells, len_epoch))
    cells_in_assemblies = 41
    cells_with_noise = n_cells - cells_in_assemblies
    pattern4[11: 22, 2: 4] = 1
    pattern4[0: 11, 14:16] = 1
    pattern4[36:41, 26:28] = 1
    pattern4[22:36, 38:40] = 1
    pattern4[11: 22, 50:52] = 1
    pattern4[36: 41, 62:64] = 1
    pattern4[22:36, 74:76] = 1
    pattern4[0:11, 86:88] = 1
    pattern4[41:50, :] = generate_poisson_pattern(cells_with_noise, len_epoch, 10, 50, 1, 2)
    # create pattern#2 shuffle = assemblies in shuffle order
    pattern4_shuffle = np.copy(pattern4)
    np.random.shuffle(pattern4_shuffle)

    #########################################
    # USE PATTERNS ASSEMBLIES AND SEQUENCES #
    #########################################

    if known_pattern_order:
        # CREATE ARTIFICIAL RASTER FROM KNOWN COMBINATION OF PATTERN
        art_raster_dur[:, 0:100] = pattern1
        art_raster_dur[:, 100:200] = generate_poisson_pattern(n_cells, len_epoch, 20, 50, 1, 2)
        art_raster_dur[:, 200:300] = pattern2
        art_raster_dur[:, 300:400] = pattern1
        art_raster_dur[:, 400:500] = generate_poisson_pattern(n_cells, len_epoch, 10, 50, 1, 2)
        art_raster_dur[:, 500:600] = generate_poisson_pattern(n_cells, len_epoch, 10, 50, 1, 2)
        art_raster_dur[:, 600:700] = pattern1
        art_raster_dur[:, 700:800] = pattern2
        art_raster_dur[:, 800:900] = generate_poisson_pattern(n_cells, len_epoch, 10, 50, 1, 2)
        art_raster_dur[:, 900:1000] = generate_poisson_pattern(n_cells, len_epoch, 10, 50, 1, 2)
        art_raster_dur[:, 1000:1100] = pattern2
        art_raster_dur[:, 1100:1200] = generate_poisson_pattern(n_cells, len_epoch, 10, 50, 1, 2)

    if random_pattern_order:
        # CREATE ARTIFICIAL RASTER COMBINATION OF THESE ASSEMBLIES SEQUENCES PLUS NOISE
        # Half of the epochs are noise pattern, the other half if equally divided in patterns
        n_patterns = 2
        n_epochs_noise = n_epochs / 2
        n_epochs_noise = int(n_epochs_noise)
        n_epochs_pattern = n_epochs - n_epochs_noise
        n_epochs_pattern = int(n_epochs_pattern)
        n_epochs_pattern1 = n_epochs_pattern / n_patterns
        n_epochs_pattern1 = int(n_epochs_pattern1)
        n_epochs_pattern2 = n_epochs_pattern / n_patterns
        n_epochs_pattern2 = int(n_epochs_pattern2)
        pattern_id = np.zeros(n_epochs)
        pattern_id[0:n_epochs_noise] = 0
        pattern_id[n_epochs_noise:(n_epochs_noise + n_epochs_pattern1)] = 1
        pattern_id[(n_epochs_noise + n_epochs_pattern1):(n_epochs_noise + n_epochs_pattern1 + n_epochs_pattern2)] = 2
        np.random.shuffle(pattern_id)

        for i in range(n_epochs):
            if pattern_id[i] == 0:
                art_raster_dur[:, np.arange((i * len_epoch), (i * len_epoch) + len_epoch)] = generate_poisson_pattern(
                    n_cells, len_epoch, 10, 50, 1, 2)
            if pattern_id[i] == 1:
                art_raster_dur[:, np.arange((i * len_epoch), (i * len_epoch) + len_epoch)] = pattern3
            if pattern_id[i] == 2:
                art_raster_dur[:, np.arange((i * len_epoch), (i * len_epoch) + len_epoch)] = pattern4

        if use_one_shuffle_per_pattern is True:
            art_raster_dur_pattern_shuffle = np.zeros((n_cells, n_frames))
            for i in range(n_epochs):
                if pattern_id[i] == 0:
                    art_raster_dur_pattern_shuffle[:, np.arange((i * len_epoch),
                                                                (i * len_epoch) + len_epoch)] \
                        = generate_poisson_pattern(n_cells, len_epoch, 10, 50, 1, 2)
                if pattern_id[i] == 1:
                    art_raster_dur_pattern_shuffle[:, np.arange((i * len_epoch),
                                                                (i * len_epoch) + len_epoch)] = pattern3_shuffle
                if pattern_id[i] == 2:
                    art_raster_dur_pattern_shuffle[:, np.arange((i * len_epoch),
                                                                (i * len_epoch) + len_epoch)] = pattern4_shuffle

    if use_one_shuffle_per_pattern is False:
        rand_art_raster_dur = np.copy(art_raster_dur)
    elif use_one_shuffle_per_pattern is True:
        rand_art_raster_dur = np.copy(art_raster_dur_pattern_shuffle)

    if do_general_shuffling_on_full_raster is True:
        np.random.shuffle(rand_art_raster_dur)

    rand_art_raster_dur.astype(int)

    # CREATE ARTIFICIAL RASTER COMBINATION OF NOISE ONLY
    for i in np.arange(n_epochs):
        tmp_patt_noise = np.zeros((n_cells, len_epoch))
        patt_num_noise = np.random.randint(3)
        n_cells_to_clear = np.random.randint(np.round(n_cells / 5), n_cells)
        cell_to_clear_indices = np.random.randint(0, n_cells, size=n_cells_to_clear)
        # print(f"pattern number is {patt_num}")
        if patt_num_noise == 0:
            tmp_patt_noise = generate_poisson_pattern(n_cells, len_epoch, 10, 40, 1, 2)
            tmp_patt_noise[cell_to_clear_indices, :] = 0
        if patt_num_noise == 1:
            tmp_patt_noise = generate_poisson_pattern(n_cells, len_epoch, 25, 40, 1, 2)
            tmp_patt_noise[cell_to_clear_indices, :] = 0
        if patt_num_noise == 2:
            tmp_patt_noise = generate_poisson_pattern(n_cells, len_epoch, 30, 40, 1, 2)
            tmp_patt_noise[cell_to_clear_indices, :] = 0
        noise_matrix[:, np.arange((i * len_epoch), (i * len_epoch) + len_epoch)] = tmp_patt_noise

    if fuse_raster_with_noise is True:
        art_raster_dur_noise = np.copy(art_raster_dur)
        art_raster_dur_noise[np.where(noise_matrix == 1)] = 1
        rand_art_raster_dur_noise = np.copy(rand_art_raster_dur)
        rand_art_raster_dur_noise[np.where(noise_matrix == 1)] = 1
        rand_art_raster_dur = rand_art_raster_dur_noise

    return rand_art_raster_dur


def main():

    # Generate correlated data
    # sts = elephant.spike_train_generation.cpp(
    #     rate=5 * pq.Hz, A=[0] + [0.99] + [0] * 10 + [0.01], t_stop=10 * pq.s)
    # Load data from matlab
    data = hdf5storage.loadmat(
    "/media/julien/Not_today/hne_not_today/data/p60/a529_2015_02_25/a529_2015_02_25_RasterDur")['rasterdur']
    # data = data[25:45, 2000:4000]
    # data = generate_artificial_data()
    # data = data[20:30, 400:600]

    # print(data.shape)
    n_cells, n_times = data.shape
    spt = []
    for cell in np.arange(n_cells):
        cell_spike_nums = data[cell]
        spike_frames = np.where(cell_spike_nums)[0]
        # convert frames in s
        # spike_frames = spike_frames / ms.sampling_rate
        neo_spike_train = n.SpikeTrain(times=spike_frames, units='ms', t_stop=n_times)
        spt.append(neo_spike_train)

    # Mining patterns with SPADE using a binsize of 1 ms and a window length of 1
    # bin (i.e., detecting only synchronous patterns).
    size = 5 * pq.ms
    patterns = spade_copy_elephant.spade(
        data=spt, binsize=size, winlen=1, dither=15 * pq.ms, min_occ=5,
        min_spikes=5, n_surr=5, psr_param=[1, 2, 3], alpha=0.01, stat_corr='bonf',
        output_format='patterns')['patterns']

    ############
    # Plotting #
    ############

    # print(patterns)
    plt.figure()
    # Loop for each pattern found
    for i in patterns:
        # Initialize color for each pattern
        c = np.random.rand(3, )

        compteur = 0
        nearest_spike = []
        # Loop for each set of neuron forming a pattern
        for neu in i['neurons']:
            # Search for nearest spike in the bin instead of putting
            # a mark at the beginning of the bin
            for t in i['times']:
                cell_spike_nums = data[neu, int(t):int(t+size)]
                nearest_spike.append(t + max(np.where(cell_spike_nums)[0]) * pq.ms)
                nearest_spike_ms = nearest_spike * pq.ms
            compteur += 1
            if compteur == len(i['neurons']):
                plt.plot(
                    nearest_spike_ms, [neu] * len(nearest_spike_ms), 'ro', color=c,
                    label='pattern '+str(i['neurons']))
            else:
                plt.plot(
                    nearest_spike_ms, [neu] * len(nearest_spike_ms), 'ro', color=c)
    # Raster plot of the data
    for st_idx, st in enumerate(spt):
        if st_idx == 0:
            plt.plot(st.rescale(pq.ms), [st_idx] * len(st), 'k.', label='spikes')
        else:
            plt.plot(st.rescale(pq.ms), [st_idx] * len(st), 'k.')
    plt.ylim([-1, len(spt)])
    plt.xlabel('time (ms)')
    plt.ylabel('neurons ids')
    plt.legend()
    plt.show()


main()
