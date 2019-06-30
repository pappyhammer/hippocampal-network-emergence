import numpy as np
from pattern_discovery.tools.misc import get_continous_time_periods
import seaborn as sns
import hdf5storage
import matplotlib.pyplot as plt
import scipy.stats
import hdbscan
from sklearn.manifold import TSNE as t_sne
import pandas as pd
from pattern_discovery.tools.misc import get_continous_time_periods


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


def sce_twitches(ms, before_extension=0, after_extension=15):
    spike_nums_dur = ms.spike_struct.spike_nums_dur
    spike_nums = ms.spike_struct.spike_nums
    n_cells_with_spikes = len(np.where(np.sum(spike_nums, axis=1) >= 1)[0])
    n_cells = spike_nums_dur.shape[0]
    # using twitch periods instead of SCE if info is available "shift_twitch"
    sce_times_bool = ms.shift_data_dict["shift_twitch"]

    n_frames = len(sce_times_bool)

    # period_extension
    extension_frames_after = after_extension
    extension_frames_before = before_extension
    shift_bool_tmp = np.copy(sce_times_bool)
    true_frames = np.where(sce_times_bool)[0]
    for frame in true_frames:
        first_frame = max(0, frame - extension_frames_before)
        last_frame = min(n_frames - 1, frame + extension_frames_after)
        shift_bool_tmp[first_frame:last_frame + 1] = True
    sce_times_bool[shift_bool_tmp] = True
    SCE_times = get_continous_time_periods(sce_times_bool.astype("int8"))
    sce_times_numbers = np.ones(len(sce_times_bool), dtype="int16")
    sce_times_numbers *= -1
    cells_in_twitches = np.zeros((n_cells, len(SCE_times)), dtype="int16")
    for index, period in enumerate(SCE_times):
        sce_times_numbers[period[0]:period[1] + 1] = index
        cells_in_twitches[:, index] = np.sum(spike_nums_dur[:, period[0]:period[1] + 1], axis=1)
        cells_in_twitches[cells_in_twitches[:, index] > 0, index] = 1

    return cells_in_twitches, n_cells_with_spikes


def get_twitches_intersection_matrix(m_sces):
    nb_events = np.shape(m_sces)[1]
    n_cells = np.shape(m_sces)[0]
    prop_common_cells = np.zeros((nb_events, nb_events))
    for i in np.arange(nb_events):
        for j in np.arange(nb_events):
            n_cell_twitch_i = np.sum(m_sces[:, i])
            n_cell_twitch_j = np.sum(m_sces[:, j])
            common_part_twitch_i_twitch_j = len(np.intersect1d(np.where(m_sces[:, i])[0], np.where(m_sces[:, j])[0],
                                                               assume_unique=True))
            prop_common_cells[i, j] = (common_part_twitch_i_twitch_j / np.min((n_cell_twitch_i, n_cell_twitch_j))) * 100
    return prop_common_cells


def covnorm(m_sces, use_pearson=False):
    nb_events = np.shape(m_sces)[1]
    co_var_matrix = np.zeros((nb_events, nb_events))
    for i in np.arange(nb_events):
        for j in np.arange(nb_events):
            if np.correlate(m_sces[:, i], m_sces[:, j]) == 0:
                co_var_matrix[i, j] = 0
            else:
                if use_pearson:
                    co_var_matrix[i, j] = scipy.stats.pearsonr(m_sces[:, i], m_sces[:, j])[0]
                else:
                    co_var_matrix[i, j] = np.correlate(m_sces[:, i] - np.mean(m_sces[:, i]), m_sces[:, j]
                                                       - np.mean(m_sces[:, j])) \
                                          / np.std(m_sces[:, i]) / np.std(m_sces[:, j]) / nb_events
    return co_var_matrix


def twitch_analysis(ms, n_surrogates, option="intersect"):
    param = ms.param
    cells_in_twitches, n_cells_with_spikes = sce_twitches(ms=ms, before_extension=0, after_extension=20)
    # print(f"{np.max(cells_in_twitches)}, {np.min(cells_in_twitches)}, {np.mean(cells_in_twitches)}")
    [n_cells, n_twitches] = cells_in_twitches.shape
    print(f"N total cells is {n_cells}, N twitches is {n_twitches}")
    print(f"N cells with spikes: {n_cells_with_spikes}")
    if option == "intersect":
        co_var_matrix = get_twitches_intersection_matrix(cells_in_twitches)
    else:
        co_var_matrix = covnorm(cells_in_twitches, use_pearson=True)

    # Plot heatmap figure
    svm = sns.heatmap(co_var_matrix)
    fig = svm.get_figure()
    # plt.show()
    save_formats = ["pdf", "png"]
    if isinstance(save_formats, str):
        save_formats = [save_formats]

    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/{ms.description}_twitches_covar_matrix'
                    f'.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())
    plt.close()

    # Get superior triangle to get distribution of correlation values
    var_matrix = np.triu(co_var_matrix, k=1)
    distrib = var_matrix[np.where(var_matrix)]

    # DO HDBSCAN ON DISTANCES MATRIX - CONSIDER PRECOMPUTED DISTANCES #
    # # clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    # #                             gen_min_span_tree=False, leaf_size=40,
    # #                             metric='precomputed', min_cluster_size=2, min_samples=None, p=None)
    # # # metric='precomputed' euclidean
    # #
    # # clusterer.fit(co_var_matrix)
    # #
    # # labels = clusterer.labels_
    # # # print(f"labels.shape: {labels.shape}")
    # # print(f"N clusters hdbscan: {labels.max()+1}")
    # # # print(f"labels: {labels}")
    # # # print(f"With no clusters hdbscan: {len(np.where(labels == -1)[0])}")
    # # n_clusters = 0
    # # if labels.max() + 1 > 0:
    # #     n_clusters = labels.max() + 1
    # #
    # # if n_clusters > 0:
    # #     n_epoch_by_cluster = [[len(np.where(labels == x)[0])] for x in np.arange(n_clusters)]
    # #     print(f"Number of epochs by clusters hdbscan: {' '.join(map(str, n_epoch_by_cluster))}")
    #
    # co_var_matrix_order = np.copy(co_var_matrix)
    # labels_indices_sorted = np.argsort(labels)
    # co_var_matrix_order = co_var_matrix_order[labels_indices_sorted, :]
    # co_var_matrix_order = co_var_matrix_order[:, labels_indices_sorted]

    # Generate figure: dissimilarity matrice ordered by cluster
    # svm = sns.heatmap(distances_order, annot=True)  # if you want the value
    # svm = sns.heatmap(co_var_matrix_order)
    # svm.set_yticklabels(labels_indices_sorted)
    # svm.set_xticklabels(labels_indices_sorted)
    # fig = svm.get_figure()
    # # plt.show()
    # save_formats = ["pdf", "png"]
    # if isinstance(save_formats, str):
    #     save_formats = [save_formats]
    #
    # for save_format in save_formats:
    #     fig.savefig(f'D:/Robin/data_hne/data/p5/p5_19_03_25_a001//test_twitches_analysis/co_var_matrix_hdbscan_ordered'
    #                 f'.{save_format}',
    #                 format=f"{save_format}",
    #                 facecolor=fig.get_facecolor())
    # plt.close()

    # DO TSNE CLUSTERING #
    # tsne = t_sne(n_components=2, verbose=1, perplexity=40, n_iter=300)
    # tsne_results = tsne.fit_transform(co_var_matrix)
    #
    # # first figure: plot t-sne without color
    # df_subset = pd.DataFrame()
    # df_subset['tsne-2d-one'] = tsne_results[:, 0]
    # df_subset['tsne-2d-two'] = tsne_results[:, 1]
    # df_subset['color'] = labels
    # plt.figure(figsize=(16, 10))
    # svm = sns.scatterplot(
    #     x="tsne-2d-one", y="tsne-2d-two",
    #     data=df_subset,
    #     legend="full",
    #     alpha=1
    # )
    # fig = svm.get_figure()
    #
    # for save_format in save_formats:
    #     fig.savefig(f'D:/Robin/data_hne/p5/p5_19_03_25_a001//test_twitches_analysis/tsne_cluster'
    #                 f'.{save_format}',
    #                 format=f"{save_format}",
    #                 facecolor=fig.get_facecolor())
    # plt.close()

    # GET RANDOM DISTRIBUTION OF CELLS IN TWITCHES #
    nbr_cells_per_twitches = np.sum(cells_in_twitches, axis=0)
    rnd_distrib_list = []
    rnd_co_var_matrix_list = []
    for surrogate in np.arange(n_surrogates):
        rnd_cells_in_twitches = np.zeros((n_cells, n_twitches))
        for i in range(n_twitches):
            random_cells_in_twitch_i = np.random.choice(n_cells_with_spikes, nbr_cells_per_twitches[i], replace=False)
            # print(f"{len(np.unique(random_cells_in_twitch_i))}")
            # print(f"cell in twitch {i} are {random_cells_in_twitch_i}")
            rnd_cells_in_twitches[random_cells_in_twitch_i, i] = 1
        # print(f"{np.sum(rnd_cells_in_twitches, axis=0)}")
        if option == "intersect":
            rnd_co_var_matrix = get_twitches_intersection_matrix(rnd_cells_in_twitches)
        else:
            rnd_co_var_matrix = covnorm(rnd_cells_in_twitches, use_pearson=True)

        rnd_co_var_matrix_list.append(rnd_co_var_matrix)

        # Get superior triangle to get distribution of correlation values
        rnd_var_matrix = np.triu(rnd_co_var_matrix, k=1)
        rnd_distrib = rnd_var_matrix[np.where(rnd_var_matrix)]
        rnd_distrib_list.append(rnd_distrib)

    return distrib, co_var_matrix, rnd_distrib_list, rnd_co_var_matrix_list
#
# twitch_analysis(n_surrogates=10)
