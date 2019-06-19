import numpy as np
import matplotlib.pyplot as plt
from SPOTDist_Battaglia import SPOT_Dist_Battaglia
import seaborn as sns
from SPOTDist_homemade import SPOT_Dist_JD_RD
import hdbscan
from sklearn.manifold import TSNE as t_sne
import pandas as pd
from pattern_discovery.display.raster import plot_spikes_raster
import os
import datetime
import time
import matplotlib.cm as cm
import hdf5storage


def load_data_rasterdur(ms):
    """
    Used to load data. The code has to be manually change so far to change the data loaded.
    :return: return a 2D binary array representing a raster. Axis 0 (lines) represents the neurons (cells) and axis 1
    (columns) represent the frames (in our case sampling is approximatively 10Hz, so 100 ms by frame).
    """
    spike_nums_dur = ms.spike_struct.spike_nums_dur
    # spike_nums_dur = spike_nums_dur[:50, :10000] # TO TEST CODE
    return spike_nums_dur


def load_data_raster(ms):
    """
    Used to load data. The code has to be manually change so far to change the data loaded.
    :return: return a 2D binary array representing a raster. Axis 0 (lines) represents the neurons (cells) and axis 1
    (columns) represent the frames (in our case sampling is approximatively 10Hz, so 100 ms by frame).
    """
    spike_nums = ms.spike_struct.spike_nums
    # spike_nums_dur = spike_nums_dur[:50, :10000] # TO TEST CODE
    return spike_nums


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
    # traces = traces[:100, :12500]  # TO TEST CODE
    return traces


def generate_poisson_pattern(n_cells, len_epoch, min_isi, max_isi, min_spikes_cell, max_spikes_cell):
    range_isi = np.random.randint(min_isi, max_isi, size=n_cells)
    range_num_spikes_per_cell = np.random.randint(min_spikes_cell, max_spikes_cell, size=n_cells)

    # create random pattern
    pattern = np.zeros((n_cells, len_epoch))

    for i in range(n_cells):
        isi = np.random.poisson(range_isi[i], (range_num_spikes_per_cell[i] - 1))
        spikes_times = np.zeros((range_num_spikes_per_cell[i]), dtype="int8")
        start = np.round(0.65*len_epoch/range_num_spikes_per_cell[i])
        spikes_times[0] = np.random.randint(2, start)
        for j in np.arange(1, range_num_spikes_per_cell[i]):
            spikes_times[j] = spikes_times[j - 1] + isi[j - 1]
        pattern[i, spikes_times] = 1

    return pattern


def spotdist_function(ms, param):

    ###################################################################################################################
    # CHOOSE METHOD TO USE #
    method_homemade = True  # The one to use if want to run on traces be careful with risk of memory error
    method_battaglia = False  # Run faster for raster_dur / raster but EMD is not normalized

    # DECIDE ON WHICH DATA TO WORK
    data_to_use = "traces"
    possible_data_to_use = ["raster_dur", "raster", "traces", "artificial_raster"]
    if data_to_use not in possible_data_to_use:
        data_to_use = "raster_dur"
        raise Exception("Can not run SpotDist on this data, by default use of raster_dur")

    # DECIDE EPOCH LENGTH  ONLY FOR NON ARTIFICIAL DATA
    len_epoch = 250

    # If you want to work on artificial data
    random_pattern_order = True
    known_pattern_order = False  # This option is obsolete, do not use
    use_one_shuffle_per_pattern = True
    do_general_shuffling_on_full_raster = False
    fuse_raster_with_noise = True

    # SET SAVING PATH
    path_results = param.path_results
    time_str = param.time_str

    ###################################################################################################################

    ################################
    # IF WORK ON ARTIFICIAL DATA   #
    ################################
    if data_to_use == "artificial_raster":
        # DEFINE RASTER #
        n_cells = 50
        len_pattern = 100
        if random_pattern_order:
            n_epochs = 100  # Put something that 4 can divide
            n_frames = len_pattern * n_epochs
        if known_pattern_order:
            n_epochs = 12  # Do not change, only 12 epochs are generated
            n_frames = len_pattern * n_epochs

        art_raster_dur = np.zeros((n_cells, n_frames), dtype="int8")
        art_raster_dur_noise = np.zeros((n_cells, n_frames), dtype="int8")
        rand_art_raster_dur = np.zeros((n_cells, n_frames), dtype="int8")
        art_raster_dur_pattern_shuffle = np.zeros((n_cells, n_frames), dtype="int8")
        noise_matrix = np.zeros((n_cells, n_frames), dtype="int8")
        rand_art_raster_dur_noise = np.zeros((n_cells, n_frames), dtype="int8")

        n_epochs = n_frames // len_pattern
        # to make things easy for now, the number of frames should be divisible by the length of epochs
        if (n_frames % len_pattern) != 0:
            raise Exception("number of frames {n_frames} not divisible by {len_epoch}")

        ############################################
        # CREATE PATTERNS ASSEMBLIES AND SEQUENCES #
        ############################################

        # create pattern#1 = sequence in order
        pattern1 = np.zeros((n_cells, len_pattern))
        for i in range(n_cells):
            pattern1[i, i] = 1
            pattern1[i, i+50] = 1
        # create pattern#1 shuffle = sequence in a shuffle order
        pattern1_shuffle = np.copy(pattern1)
        np.random.shuffle(pattern1_shuffle)

        # create pattern#2 = assemblies in order
        pattern2 = np.zeros((n_cells, len_pattern))
        pattern2[13: 26, 2: 4] = 1
        pattern2[0: 13, 14:16] = 1
        pattern2[39:50, 26:28] = 1
        pattern2[26:39, 38:40] = 1
        pattern2[13: 26, 50:52] = 1
        pattern2[39: 50, 62:64] = 1
        pattern2[26:39, 74:76] = 1
        pattern2[0:13, 86:88] = 1
        # create pattern#2 shuffle = assemblies in shuffle order
        pattern2_shuffle = np.copy(pattern2)
        np.random.shuffle(pattern2_shuffle)

        # create pattern#3 = sequence together with noise
        pattern3 = np.zeros((n_cells, len_pattern))
        n_cells_in_sequence = 40
        noisy_cells = n_cells - n_cells_in_sequence
        for i in range(n_cells_in_sequence):
            pattern3[i, i:i + 2] = 1
            pattern3[i, 20 + i:i + 22] = 1
        pattern3[n_cells_in_sequence:n_cells, :] = generate_poisson_pattern(noisy_cells, len_pattern, 10, 50, 1, 2)
        # create pattern#3 shuffle
        pattern3_shuffle = np.copy(pattern3)
        np.random.shuffle(pattern3_shuffle)

        # create pattern#4 = assemblies together with noise
        pattern4 = np.zeros((n_cells, len_pattern))
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
        pattern4[41:50, :] = generate_poisson_pattern(cells_with_noise, len_pattern, 10, 50, 1, 2)
        # create pattern#2 shuffle = assemblies in shuffle order
        pattern4_shuffle = np.copy(pattern4)
        np.random.shuffle(pattern4_shuffle)

        #########################################
        # USE PATTERNS ASSEMBLIES AND SEQUENCES #
        #########################################

        if known_pattern_order:
            # CREATE ARTIFICIAL RASTER FROM KNOWN COMBINATION OF PATTERN
            art_raster_dur[:, 0:100] = pattern1
            art_raster_dur[:, 100:200] = generate_poisson_pattern(n_cells, len_pattern, 20, 50, 1, 2)
            art_raster_dur[:, 200:300] = pattern2
            art_raster_dur[:, 300:400] = pattern1
            art_raster_dur[:, 400:500] = generate_poisson_pattern(n_cells, len_pattern, 10, 50, 1, 2)
            art_raster_dur[:, 500:600] = generate_poisson_pattern(n_cells, len_pattern, 10, 50, 1, 2)
            art_raster_dur[:, 600:700] = pattern1
            art_raster_dur[:, 700:800] = pattern2
            art_raster_dur[:, 800:900] = generate_poisson_pattern(n_cells, len_pattern, 10, 50, 1, 2)
            art_raster_dur[:, 900:1000] = generate_poisson_pattern(n_cells, len_pattern, 10, 50, 1, 2)
            art_raster_dur[:, 1000:1100] = pattern2
            art_raster_dur[:, 1100:1200] = generate_poisson_pattern(n_cells, len_pattern, 10, 50, 1, 2)

        if random_pattern_order:
            # CREATE ARTIFICIAL RASTER COMBINATION OF THESE ASSEMBLIES SEQUENCES PLUS NOISE
            # Half of the epochs are noise pattern, the other half if equally divided in patterns
            n_patterns = 2
            n_epochs_noise = n_epochs // 2
            n_epochs_pattern = n_epochs - n_epochs_noise
            # n_epochs_pattern = int(n_epochs_pattern)
            n_epochs_pattern1 = n_epochs_pattern // n_patterns
            n_epochs_pattern2 = n_epochs_pattern // n_patterns

            pattern_id = np.zeros(n_epochs)
            pattern_id[0:n_epochs_noise] = 0
            pattern_id[n_epochs_noise:(n_epochs_noise + n_epochs_pattern1)] = 1
            pattern_id[(n_epochs_noise + n_epochs_pattern1):(n_epochs_noise + n_epochs_pattern1 + n_epochs_pattern2)] = 2
            np.random.shuffle(pattern_id)

            for i in range(n_epochs):
                if pattern_id[i] == 0:
                    art_raster_dur[:, np.arange((i*len_pattern), (i*len_pattern)+len_pattern)] = generate_poisson_pattern(n_cells, len_pattern, 10, 50, 1, 2)
                    art_raster_dur_pattern_shuffle[:, np.arange((i * len_pattern), (i * len_pattern) + len_pattern)] = generate_poisson_pattern(n_cells, len_pattern, 10, 50, 1, 2)
                if pattern_id[i] == 1:
                    art_raster_dur[:, np.arange((i * len_pattern), (i * len_pattern) + len_pattern)] = pattern3
                    art_raster_dur_pattern_shuffle[:, np.arange((i * len_pattern), (i * len_pattern) + len_pattern)] = pattern3_shuffle
                if pattern_id[i] == 2:
                    art_raster_dur[:, np.arange((i * len_pattern), (i * len_pattern) + len_pattern)] = pattern4
                    art_raster_dur_pattern_shuffle[:, np.arange((i * len_pattern), (i * len_pattern) + len_pattern)] = pattern4_shuffle

        if use_one_shuffle_per_pattern is False:
            rand_art_raster_dur = np.copy(art_raster_dur)
        elif use_one_shuffle_per_pattern is True:
            rand_art_raster_dur = np.copy(art_raster_dur_pattern_shuffle)

        if do_general_shuffling_on_full_raster is True:
            np.random.shuffle(rand_art_raster_dur)

        rand_art_raster_dur.astype(int)

        # CREATE ARTIFICIAL RASTER COMBINATION OF NOISE ONLY
        for i in np.arange(n_epochs):
            tmp_patt_noise = np.zeros((n_cells, len_pattern))
            patt_num_noise = np.random.randint(3)
            n_cells_to_clear = np.random.randint(np.round(n_cells / 5), n_cells)
            cell_to_clear_indices = np.random.randint(0, n_cells, size=n_cells_to_clear)
            # print(f"pattern number is {patt_num}")
            if patt_num_noise == 0:
                tmp_patt_noise = generate_poisson_pattern(n_cells, len_pattern, 10, 40, 1, 2)
                tmp_patt_noise[cell_to_clear_indices, :] = 0
            if patt_num_noise == 1:
                tmp_patt_noise = generate_poisson_pattern(n_cells, len_pattern, 25, 40, 1, 2)
                tmp_patt_noise[cell_to_clear_indices, :] = 0
            if patt_num_noise == 2:
                tmp_patt_noise = generate_poisson_pattern(n_cells, len_pattern, 30, 40, 1, 2)
                tmp_patt_noise[cell_to_clear_indices, :] = 0
            noise_matrix[:, np.arange((i * len_pattern), (i * len_pattern) + len_pattern)] = tmp_patt_noise

        if fuse_raster_with_noise is True:
            art_raster_dur_noise = np.copy(art_raster_dur)
            art_raster_dur_noise[np.where(noise_matrix == 1)] = 1
            rand_art_raster_dur_noise = np.copy(rand_art_raster_dur)
            rand_art_raster_dur_noise[np.where(noise_matrix == 1)] = 1
            rand_art_raster_dur = rand_art_raster_dur_noise

        data = rand_art_raster_dur

        # PLOT ALL THESE RASTER #
        # noise-only pattern
        plot_spikes_raster(spike_nums=noise_matrix, param=None,
                           file_name=f"poisson_noise_raster",
                           # y_ticks_labels=np.arange(n_cells),
                           # y_ticks_labels_size=2,
                           save_raster=True,
                           show_raster=False,
                           without_activity_sum=True,
                           path_results=path_results,
                           save_formats=["pdf", "png"])

        # Artificial raster dur ordered
        plot_spikes_raster(spike_nums=art_raster_dur, param=None,
                           file_name=f"ordered_raster_with_patterns",
                           # y_ticks_labels=np.arange(n_cells),
                           # y_ticks_labels_size=2,
                           save_raster=True,
                           show_raster=False,
                           without_activity_sum=True,
                           path_results=path_results,
                           save_formats=["pdf", "png"])

        # Artificial raster dur with intra pattern shuffle of cell order
        plot_spikes_raster(spike_nums=art_raster_dur_pattern_shuffle, param=None,
                           file_name=f"raster_with_one_shuffle_per_pattern",
                           # y_ticks_labels=np.arange(n_cells),
                           # y_ticks_labels_size=2,
                           save_raster=True,
                           show_raster=False,
                           without_activity_sum=True,
                           path_results=path_results,
                           save_formats=["pdf", "png"])

        # Add an additional shuflle on the order of all cell
        plot_spikes_raster(spike_nums=rand_art_raster_dur, param=None,
                           file_name=f"raster_with_patterns_full_shuffle",
                           # y_ticks_labels=np.arange(n_cells),
                           # y_ticks_labels_size=2,
                           save_raster=True,
                           show_raster=False,
                           without_activity_sum=True,
                           path_results=path_results,
                           save_formats=["pdf", "png"])

        # Artificial raster dur ordered with random noise
        plot_spikes_raster(spike_nums=art_raster_dur_noise, param=None,
                           file_name=f"ordered_raster_with_patterns_and_noise",
                           # y_ticks_labels=np.arange(n_cells),
                           # y_ticks_labels_size=2,
                           save_raster=True,
                           show_raster=False,
                           without_activity_sum=True,
                           path_results=path_results,
                           save_formats=["pdf", "png"])

        # Artificial raster dur shuffled with random noise
        plot_spikes_raster(spike_nums=rand_art_raster_dur_noise, param=None,
                           file_name=f"raster_with_patterns_full_shuffle_and_noise",
                           # y_ticks_labels=np.arange(n_cells),
                           # y_ticks_labels_size=2,
                           save_raster=True,
                           show_raster=False,
                           without_activity_sum=True,
                           path_results=path_results,
                           save_formats=["pdf", "png"])

    #################################
    # WORK ON REAL DATA: RASTER_DUR #
    #################################
    if data_to_use == "raster_dur":
        print(f"Loading raster_dur")
        spike_nums_dur = load_data_rasterdur(ms)  # automatic way
        # spike_nums_dur = spike_nums_dur[:20, :2500]  # TO TEST THE CODE
        n_cells, n_frames = spike_nums_dur.shape
        print(f"spike_nums_dur has {n_cells} cells and {n_frames} frames")
        data = spike_nums_dur

    #############################
    # WORK ON REAL DATA: RASTER #
    #############################
    if data_to_use == "raster":
        print(f"Loading raster")
        spike_nums = load_data_raster(ms)  # automatic way
        # spike_nums = spike_nums[:20, :2500]  # TO TEST THE CODE
        n_cells, n_frames = spike_nums.shape
        print(f"spike_nums has {n_cells} cells and {n_frames} frames")
        data = spike_nums

    #############################
    # WORK ON REAL DATA: TRACES #
    #############################
    if data_to_use == "traces":
        print(f"Loading traces")
        traces = load_data_traces(ms)  # automatic way
        traces = traces[:100, :10000]  # TO TEST THE CODE
        n_cells, n_frames = traces.shape
        print(f"traces has {n_cells} cells and {n_frames} frames")
        data = traces

    #####################
    # COMPUTE DISTANCES #
    #####################
    n_epochs = n_frames // len_epoch
    # to make things easy for now, the number of frames should be divisible by the length of epochs
    if (n_frames % len_epoch) != 0:
        raise Exception("number of frames {n_frames} not divisible by {len_epoch}")
    if method_battaglia:
        method = "battaglia"
        distances = SPOT_Dist_Battaglia(data, len_epoch=len_epoch)[0]
    if method_homemade:
        method = "homemade"
        distances = SPOT_Dist_JD_RD(data, len_epoch=len_epoch, distance_metric="EMD_Battaglia")

    # Plot Distance matrix
    # ax = sns.heatmap(distances, annot=True)
    ax = sns.heatmap(distances)
    fig = ax.get_figure()

    save_formats = ["pdf", "png"]
    if isinstance(save_formats, str):
        save_formats = [save_formats]

    for save_format in save_formats:
        fig.savefig(f'{path_results}/{ms.description}_distances_matrix_{method}_SPOTDist_on_{data_to_use}_with_{len_epoch}_frame_epochs'
                    f'.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())
    plt.close()

    ##################
    ### CLUSTERING ###
    ##################

    # HDBSCAN is supposed to be be blind to Inf value, replace missing values by np.Inf for clustering
    distances[np.where(np.isnan(distances))] = np.Inf

    # DO HDBSCAN ON DISTANCES MATRIX - CONSIDER PRECOMPUTED DISTANCES
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, leaf_size=40,
                                metric='precomputed', min_cluster_size=2, min_samples=None, p=None)
    # metric='precomputed' euclidean

    clusterer.fit(distances)

    labels = clusterer.labels_
    # print(f"labels.shape: {labels.shape}")
    print(f"N clusters hdbscan: {labels.max()+1}")
    print(f"labels: {labels}")
    print(f"With no clusters hdbscan: {len(np.where(labels == -1)[0])}")
    n_clusters = 0
    if labels.max() + 1 > 0:
        n_clusters = labels.max() + 1

    if n_clusters > 0:
        n_epoch_by_cluster = [[len(np.where(labels == x)[0])] for x in np.arange(n_clusters)]
        print(f"Number of epochs by clusters hdbscan: {' '.join(map(str, n_epoch_by_cluster))}")

    distances_order = np.copy(distances)
    labels_indices_sorted = np.argsort(labels)
    distances_order = distances_order[labels_indices_sorted, :]
    distances_order = distances_order[:, labels_indices_sorted]

    # Generate figure: dissimilarity matrice ordered by cluster
    # Replace Inf values by NaN for better visualization
    distances_order[np.where(np.isinf(distances_order))] = np.nan
    # svm = sns.heatmap(distances_order, annot=True)  # if you want the value
    svm = sns.heatmap(distances_order)
    svm.set_yticklabels(labels_indices_sorted)
    svm.set_xticklabels(labels_indices_sorted)
    fig = svm.get_figure()
    # plt.show()
    save_formats = ["pdf", "png"]
    if isinstance(save_formats, str):
        save_formats = [save_formats]

    path_results = path_results
    for save_format in save_formats:
        fig.savefig(f'{path_results}/distances_matrix_hdbscan_ordered'
                    f'.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())
    plt.close()

    coords = []
    color = []
    for i in range(n_epochs):
        coords.append([[i*len_epoch, i*len_epoch + len_epoch]])
        color.append(cm.nipy_spectral(float(labels[i] + 2) / (len(np.unique(labels)) +2)))
    if data_to_use == "artificial_raster":
        plot_spikes_raster(spike_nums=art_raster_dur_noise, param=None,
                           file_name=f"raster_with_patterns_colored",
                           # y_ticks_labels=np.arange(n_cells),
                           # y_ticks_labels_size=2,
                           save_raster=True,
                           show_raster=False,
                           without_activity_sum=True,
                           span_area_coords=coords,
                           span_area_colors=color,
                           path_results=path_results,
                           save_formats=["pdf", "png"])
    if data_to_use == "raster_dur" or data_to_use == "raster":
        plot_spikes_raster(spike_nums=data, param=None,
                           file_name=f"raster_with_patterns_colored",
                           # y_ticks_labels=np.arange(n_cells),
                           # y_ticks_labels_size=2,
                           save_raster=True,
                           show_raster=False,
                           without_activity_sum=True,
                           span_area_coords=coords,
                           span_area_colors=color,
                           path_results=path_results,
                           save_formats=["pdf", "png"])


    # IF NO NaN DO T-SNE CLUSTERING ON DISTANCES VALUES - EUCLIDEAN DISTANCES # todo: find a way to do t-SNE anyway
    missing_values = np.isnan(distances)
    inf_values = np.isinf(distances)

    if (not np.any(missing_values)) and (not np.any(inf_values)):
        do_tsne_clustering = True
        print(f" do tsne clustering is {do_tsne_clustering}")
    elif bool(np.any(missing_values)) or bool(np.any(inf_values)):
        do_tsne_clustering = False
        print(f" do tsne clustering is {do_tsne_clustering}")

    if do_tsne_clustering is True:
        tsne = t_sne(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(distances)

        # first figure: plot t-sne without color
        df_subset = pd.DataFrame()
        df_subset['tsne-2d-one'] = tsne_results[:, 0]
        df_subset['tsne-2d-two'] = tsne_results[:, 1]
        df_subset['color'] = labels
        plt.figure(figsize=(16, 10))
        svm = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            data=df_subset,
            legend="full",
            alpha=1
        )
        fig = svm.get_figure()

        path_results = path_results
        for save_format in save_formats:
            fig.savefig(f'{path_results}/tsne_cluster'
                        f'.{save_format}',
                        format=f"{save_format}",
                        facecolor=fig.get_facecolor())
        plt.close()

        # second figure: plot t-sne with color from previous hdbscan result
        df_subset = pd.DataFrame()
        df_subset['tsne-2d-one'] = tsne_results[:, 0]
        df_subset['tsne-2d-two'] = tsne_results[:, 1]
        df_subset['color'] = labels

        plt.figure(figsize=(16, 10))
        svm = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="color",
            palette=sns.color_palette("hls", len(np.unique(labels))),
            data=df_subset,
            legend="full",
            alpha=1
        )
        fig = svm.get_figure()

        path_results = path_results
        for save_format in save_formats:
            fig.savefig(f'{path_results}/tsne_colors_from_previous_hdbscan_clustering'
                        f'.{save_format}',
                        format=f"{save_format}",
                        facecolor=fig.get_facecolor())
        plt.close()

        # DO CLUSTERING ON T-SNE RESULTS TO COLOR THE T-SNE FIGURE ##

        clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                    gen_min_span_tree=False, leaf_size=40,
                                    metric='euclidean', min_cluster_size=3, min_samples=None, p=None)
        clusterer.fit(tsne_results)
        labels_hdbscan_on_tsne = clusterer.labels_
        print(f"N clusters hdbscan on t-sne results: {labels_hdbscan_on_tsne.max()+1}")
        # print(f"labels: {labels_hdbscan_on_tsne}")

        df_subset = pd.DataFrame()
        df_subset['tsne-2d-one'] = tsne_results[:, 0]
        df_subset['tsne-2d-two'] = tsne_results[:, 1]
        df_subset['color'] = labels_hdbscan_on_tsne

        plt.figure(figsize=(16, 10))
        svm = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="color",
            palette=sns.color_palette("hls", len(np.unique(labels_hdbscan_on_tsne))),
            data=df_subset,
            legend="full",
            alpha=1
        )
        # plt.show()
        fig = svm.get_figure()

        path_results = path_results
        for save_format in save_formats:
            fig.savefig(f'{path_results}/tsne_colors_from_post_tsne_clustering'
                        f'.{save_format}',
                        format=f"{save_format}",
                        facecolor=fig.get_facecolor())
        plt.close()

    ###############################################
    ######## RETRIEVE GOOD ORDER OF RASTER ########
    ###############################################

    # Get the number of "true clusters" epochs with label = -1 are not in cluster
    if labels.max() + 1 > 0:
        n_clusters = labels.max() + 1
    if n_clusters > 0:
        n_epoch_by_cluster = [[len(np.where(labels == x)[0])] for x in np.arange(n_clusters)]

    # Keep all the epochs belongings to the same clusters
    kept_epochs = []
    for i in range(n_clusters):
        kept_epochs.append(np.where(labels == i))
    # print(f"Epochs kept to represent the clusters are {kept_epochs}")

    # Get 1 raster per cluster corresponding to the sum of all rasters from the epochs of the cluster
    patterns_raster = np.zeros((n_cells, len_epoch, n_clusters))
    for i in range(n_clusters):
        raster_cluster_i = np.zeros((n_cells, len_epoch))
        epochs_in_cluster_i = kept_epochs[i]
        epochs_in_cluster_i = epochs_in_cluster_i[0]
        n_epoch_in_cluster_i = n_epoch_by_cluster[i]
        n_epoch_in_cluster_i = n_epoch_in_cluster_i[0]
        # print(f" Epochs in cluster {i} are {epochs_in_cluster_i}")
        # print(f" Cluster {i} contain {n_epoch_in_cluster_i} epochs")
        for j in range(n_epoch_in_cluster_i):
            start_epoch = epochs_in_cluster_i[j] * len_epoch
            end_epoch = epochs_in_cluster_i[j] * len_epoch + len_epoch
            # print(f" Epoch {j} of cluster {i} starts at {start_epoch} ends at {end_epoch}")
            raster_cluster_i = raster_cluster_i + data[:, start_epoch:end_epoch]
        raster_cluster_i = np.true_divide(raster_cluster_i, n_epoch_in_cluster_i)
        patterns_raster[:, :, i] = raster_cluster_i

    for i in range(n_clusters):
        pattern_i = patterns_raster[:, :, i]
        # Plot this raster
        plot_spikes_raster(spike_nums=pattern_i, param=None,
                           file_name=f"raster_pattern_{i}",
                           # y_ticks_labels=np.arange(n_cells),
                           # y_ticks_labels_size=2,
                           save_raster=True,
                           show_raster=False,
                           without_activity_sum=True,
                           plot_with_amplitude=True,
                           path_results=path_results,
                           save_formats=["pdf", "png"])

    for i in range(n_clusters):
        pattern_i = patterns_raster[:, :, i]
        max_values_vector = np.amax(pattern_i, axis=1)
        order = np.argsort(- max_values_vector)
        sorted_pattern_i = np.copy(pattern_i)
        sorted_pattern_i = sorted_pattern_i[order, :]
        # Plot this raster
        plot_spikes_raster(spike_nums=sorted_pattern_i, param=None,
                           file_name=f"raster_ordered_pattern_{i}",
                           # y_ticks_labels=np.arange(n_cells),
                           # y_ticks_labels_size=2,
                           save_raster=True,
                           show_raster=False,
                           without_activity_sum=True,
                           plot_with_amplitude=True,
                           path_results=path_results,
                           save_formats=["pdf", "png"])


# spotdist_function(ms, param)









