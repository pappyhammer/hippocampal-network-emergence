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

    return cells_in_twitches, n_cells_with_spikes, SCE_times


def get_twitches_intersection_matrix(m_sces):
    nb_events = np.shape(m_sces)[1]
    n_cells = np.shape(m_sces)[0]
    prop_common_cells = np.zeros((nb_events, nb_events))
    for i in np.arange(nb_events):
        for j in np.arange(nb_events):
            cell_in_twitch_i = np.where(m_sces[:, i])[0]
            cell_in_twitch_j = np.where(m_sces[:, j])[0]
            cells_in_twitch_i_and_j = np.concatenate((cell_in_twitch_i, cell_in_twitch_j), axis=None)
            cells_in_twitch_i_and_j_unique = np.unique(cells_in_twitch_i_and_j)
            n_cell_twitch_i = np.sum(m_sces[:, i])
            n_cell_twitch_j = np.sum(m_sces[:, j])
            common_part_twitch_i_twitch_j = len(np.intersect1d(np.where(m_sces[:, i])[0], np.where(m_sces[:, j])[0],
                                                               assume_unique=True))
            # prop_common_cells[i, j] = (common_part_twitch_i_twitch_j / np.min((n_cell_twitch_i, n_cell_twitch_j))) * 100
            prop_common_cells[i, j] = (common_part_twitch_i_twitch_j / len(cells_in_twitch_i_and_j_unique)) * 100
    return prop_common_cells


def covnorm(m_sces, use_pearson=False, full_with=None):
    nb_events = np.shape(m_sces)[1]
    if full_with is not None:
        co_var_matrix = np.ones((nb_events, nb_events))
        co_var_matrix = co_var_matrix * full_with
    else:
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


def plot_intersect_course_over_twitches(cells_in_twitches, twitches_times, ms):
    nb_twitches = np.shape(cells_in_twitches)[1]
    param = ms.param
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    # + 11 diverting
    brewer_colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                     '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
                     '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
                     '#74add1', '#4575b4', '#313695']
    # key is the first twitch
    # value is a list containing 2 list, x (delay) and y (n intersect cells)
    results_intersect = dict()

    for twitch in np.arange(nb_twitches - 1):
        twitch_1 = twitch
        intersect_cells = np.where(cells_in_twitches[:, twitch_1])[0]

        for twitch_2 in np.arange(twitch_1 + 1, nb_twitches):
            intersect_cells = np.intersect1d(intersect_cells,
                                             np.where(cells_in_twitches[:, twitch_2])[0],
                                             assume_unique=True)
            delay_bw_twitches = twitches_times[twitch_2][0] - twitches_times[twitch_1][0]
            if twitch not in results_intersect:
                results_intersect[twitch] = [[delay_bw_twitches], [len(intersect_cells)]]
            else:
                results_intersect[twitch][0].append(delay_bw_twitches)
                results_intersect[twitch][1].append(len(intersect_cells))
            twitch_1 = twitch_2
            if len(intersect_cells) == 0:
                break

    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(12, 12))
    background_color = "black"
    ax1.set_facecolor(background_color)

    fig.patch.set_facecolor(background_color)
    for twitch, delay_intersect in results_intersect.items():
        color = brewer_colors[twitch % len(brewer_colors)]
        ax1.scatter(delay_intersect[0], delay_intersect[1], color=color, marker="o",
                    edgecolors="white",
                    s=50, zorder=2)
        ax1.plot(delay_intersect[0], delay_intersect[1], color=color,
                 lw=1, zorder=1)

    labels_color = "white"
    ax1.set_xlabel("delay (frames)", fontsize=30, labelpad=20)
    ax1.set_ylabel("Intersecting cells course", fontsize=30, labelpad=20)
    ax1.xaxis.label.set_color(labels_color)
    ax1.yaxis.label.set_color(labels_color)
    ax1.tick_params(axis='y', colors=labels_color)
    ax1.tick_params(axis='x', colors=labels_color)

    save_formats = ["png", "pdf"]
    if isinstance(save_formats, str):
        save_formats = [save_formats]

    filename = f"{ms.description}_cells_intersecting_course"
    path_results = param.path_results
    for save_format in save_formats:
        fig.savefig(f'{path_results}/{filename}'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())

    plt.close()


def plot_twitches_delay_stat(distrib_similarity, twitches_times, ms):
    param = ms.param
    n_twitches = len(twitches_times)

    delay_bw_twitch_matrix = np.zeros((n_twitches, n_twitches), dtype="int16")

    # represent the number of twitches between 2 twitches
    n_twitches_bw_matrix = np.ones((n_twitches, n_twitches), dtype="int16")
    n_twitches_bw_matrix = n_twitches_bw_matrix * -1
    for twitch_1 in np.arange(n_twitches - 1):
        for twitch_2 in np.arange(twitch_1 + 1, n_twitches):
            delay_bw_twitch_matrix[twitch_1, twitch_2] = abs(twitches_times[twitch_2][0] - twitches_times[twitch_1][0])
            n_twitches_bw_matrix[twitch_1, twitch_2] = twitch_2 - twitch_1 - 1

    distrib_n_twitches_bw = delay_bw_twitch_matrix[np.where(n_twitches_bw_matrix) > -1]
    distrib_delay = delay_bw_twitch_matrix[np.where(delay_bw_twitch_matrix)]

    distrib_dict = ()
    distrib_dict["delay_bw_twitches"] = distrib_delay
    distrib_dict["n_twitches_bw"] = distrib_n_twitches_bw

    for key_str, distrib_value in distrib_dict.items():
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                gridspec_kw={'height_ratios': [1]},
                                figsize=(12, 12))
        background_color = "black"
        ax1.set_facecolor(background_color)

        fig.patch.set_facecolor(background_color)
        ax1.scatter(distrib_value, distrib_similarity, color="red", marker="o",
                    edgecolors="white",
                    s=50, zorder=1)
        labels_color = "white"
        ax1.set_xlabel("delay (frames)", fontsize=30, labelpad=20)
        ax1.set_ylabel("similarity (%)", fontsize=30, labelpad=20)
        ax1.xaxis.label.set_color(labels_color)
        ax1.yaxis.label.set_color(labels_color)
        ax1.tick_params(axis='y', colors=labels_color)
        ax1.tick_params(axis='x', colors=labels_color)

        save_formats = ["png", "pdf"]
        if isinstance(save_formats, str):
            save_formats = [save_formats]

        filename = f"{ms.description}_{key_str}"
        path_results = param.path_results
        for save_format in save_formats:
            fig.savefig(f'{path_results}/{filename}'
                        f'_{param.time_str}.{save_format}',
                        format=f"{save_format}",
                        facecolor=fig.get_facecolor())

        plt.close()


def twitch_analysis(ms, n_surrogates, option="intersect"):
    param = ms.param
    cells_in_twitches, n_cells_with_spikes, twitches_times = sce_twitches(ms=ms, before_extension=0, after_extension=20)
    n_cells_with_spikes_in_twitche = len(np.where(np.sum(cells_in_twitches, axis=1) >= 1)[0])
    # for cell in np.where(np.sum(cells_in_twitches, axis=1) == 0)[0]:
    #     print(f"{cell} = {np.sum(ms.spike_struct.spike_nums[cell, :])}")
    [n_cells, n_twitches] = cells_in_twitches.shape
    print(f"N total cells is {n_cells}, N twitches is {n_twitches}")
    print(f"N cells with spikes: {n_cells_with_spikes}")
    print(f"N cells with at least one spike after a twitch: {n_cells_with_spikes_in_twitche}")
    if option == "intersect":
        co_var_matrix = get_twitches_intersection_matrix(cells_in_twitches)
    else:
        co_var_matrix = covnorm(cells_in_twitches, use_pearson=True, full_with=-1)

    # Get superior triangle to get distribution of correlation values
    var_matrix = np.triu(co_var_matrix + 1, k=1)
    distrib = var_matrix[np.where((var_matrix - 1) > -1)]

    plot_twitches_delay_stat(distrib_similarity=distrib,
                             twitches_times=twitches_times,
                             ms=ms)
    plot_intersect_course_over_twitches(cells_in_twitches, twitches_times, ms)

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
            random_cells_in_twitch_i = np.random.choice(n_cells_with_spikes_in_twitche, nbr_cells_per_twitches[i],
                                                        replace=False)
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
        rnd_distrib = rnd_var_matrix[np.where(rnd_var_matrix > -1)]
        rnd_distrib_list.append(rnd_distrib)

    return distrib, co_var_matrix, rnd_distrib_list, rnd_co_var_matrix_list
#
# twitch_analysis(n_surrogates=10)
