import pandas as pd
# from scipy.io import loadmat
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

# important to avoid a bug when using virtualenv
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
from matplotlib.ticker import MultipleLocator
import seaborn as sns
# import copy
from datetime import datetime
from collections import Counter
from sklearn import metrics
# import keras
import os
import scipy.ndimage.morphology as morphology
import scipy.ndimage as ndimage
from collections import Counter
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from sortedcontainers import SortedList, SortedDict
# to add homemade package, go to preferences, then project interpreter, then click on the wheel symbol
# then show all, then select the interpreter and lick on the more right icon to display a list of folder and
# add the one containing the folder pattern_discovery
from pattern_discovery.seq_solver.markov_way import MarkovParameters
import pattern_discovery.tools.param as p_disc_param
import pattern_discovery.tools.misc as tools_misc
from pattern_discovery.display.raster import plot_spikes_raster
from pattern_discovery.display.raster import plot_sum_active_clusters
from pattern_discovery.display.raster import plot_dendogram_from_fca
from pattern_discovery.display.misc import plot_hist_clusters_by_sce
from pattern_discovery.tools.loss_function import loss_function_with_sliding_window
import pattern_discovery.tools.trains as trains_module
from pattern_discovery.seq_solver.markov_way import order_spike_nums_by_seq
from pattern_discovery.tools.sce_detection import get_sce_detection_threshold, detect_sce_with_sliding_window
from sortedcontainers import SortedList, SortedDict
from pattern_discovery.clustering.kmean_version.k_mean_clustering import co_var_first_and_clusters
from pattern_discovery.clustering.kmean_version.k_mean_clustering import show_co_var_first_matrix
from pattern_discovery.clustering.kmean_version.k_mean_clustering import compute_and_plot_clusters_raster_kmean_version
from pattern_discovery.clustering.fca.fca import functional_clustering_algorithm
import pattern_discovery.clustering.fca.fca as fca
from pattern_discovery.clustering.cluster_tools import detect_cluster_activations_with_sliding_window
from pattern_discovery.clustering.fca.fca import compute_and_plot_clusters_raster_fca_version


class HNEParameters(MarkovParameters):
    def __init__(self, path_results, time_str, time_inter_seq, min_duration_intra_seq, min_len_seq, min_rep_nb,
                 path_data,
                 max_branches, stop_if_twin, no_reverse_seq, error_rate, spike_rate_weight,
                 bin_size=1):
        super().__init__(time_inter_seq=time_inter_seq, min_duration_intra_seq=min_duration_intra_seq,
                         min_len_seq=min_len_seq, min_rep_nb=min_rep_nb, no_reverse_seq=no_reverse_seq,
                         max_branches=max_branches, stop_if_twin=stop_if_twin, error_rate=error_rate,
                         spike_rate_weight=spike_rate_weight,
                         bin_size=bin_size, path_results=path_results, time_str=time_str)
        self.path_data = path_data


class MouseSession:
    def __init__(self, age, session_id, param, nb_ms_by_frame, spike_nums=None, spike_nums_dur=None):
        # should be a list of int
        self.age = age
        self.session_id = str(session_id)
        self.nb_ms_by_frame = nb_ms_by_frame
        self.description = f"P{self.age}_{self.session_id}"
        self.spike_struct = HNESpikeStructure(mouse_session=self, spike_nums=spike_nums, spike_nums_dur=spike_nums_dur)
        # spike_nums represents the onsets of the neuron spikes
        self.traces = None
        self.coord = None
        self.param = param

    def load_data_from_file(self, file_name_to_load, variables_mapping):
        data = hdf5storage.loadmat(self.param.path_data + file_name_to_load)
        if "spike_nums" in variables_mapping:
            self.spike_struct.spike_nums = data[variables_mapping["spike_nums"]].astype(int)
            if self.spike_struct.labels is None:
                self.spike_struct.labels = np.arange(len(self.spike_struct.spike_nums))
        if "spike_nums_dur" in variables_mapping:
            self.spike_struct.spike_nums_dur = data[variables_mapping["spike_nums_dur"]].astype(int)
            if self.spike_struct.labels is None:
                self.spike_struct.labels = np.arange(len(self.spike_struct.spike_nums_dur))
        if "traces" in variables_mapping:
            self.traces = data[variables_mapping["traces"]].astype(float)
        if "coord" in variables_mapping:
            self.coord = data[variables_mapping["coord"]]

        self.spike_struct.set_spike_trains_from_spike_nums()


class HNESpikeStructure:

    def __init__(self, mouse_session, labels=None, spike_nums=None, spike_trains=None,
                 spike_nums_dur=None, activity_threshold=None,
                 title=None, ordered_indices=None, ordered_spike_data=None):
        self.mouse_session = mouse_session
        self.spike_nums = spike_nums
        self.spike_nums_dur = spike_nums_dur
        self.spike_trains = spike_trains
        self.ordered_spike_data = ordered_spike_data
        self.activity_threshold = activity_threshold
        self.title = title
        self.labels = labels
        self.ordered_indices = ordered_indices
        self.ordered_labels = None
        if self.ordered_indices is not None:
            if self.spike_nums is not None:
                self.ordered_spike_nums = np.copy(self.spike_nums[ordered_indices, :])
            else:
                self.ordered_spike_nums = None
            if self.spike_nums_dur is not None:
                self.ordered_spike_nums_dur = np.copy(self.spike_nums_dur[ordered_indices, :])
            else:
                self.ordered_spike_nums_dur = None
            if self.spike_trains is not None:
                self.ordered_spike_trains = []
                for index in ordered_indices:
                    self.ordered_spike_trains.append(self.spike_trains[index])
            else:
                self.ordered_spike_trains = None
            self.ordered_labels = []
            if self.labels is not None:
                for old_cell_index in self.ordered_indices:
                    self.ordered_labels.append(self.labels[old_cell_index])

    def set_order(self, ordered_indices):
        if ordered_indices is None:
            self.ordered_spike_nums = np.copy(self.spike_nums)
        else:
            if self.spike_nums is not None:
                self.ordered_spike_nums = np.copy(self.spike_nums[ordered_indices, :])
            # else:
            #     self.ordered_spike_nums = None
            if self.spike_trains is not None:
                self.ordered_spike_trains = []
                for index in ordered_indices:
                    self.ordered_spike_trains.append(self.spike_trains[index])
            # else:
            #     self.ordered_spike_trains = None
            self.ordered_indices = ordered_indices
            self.ordered_labels = []
            for old_cell_index in self.ordered_indices:
                self.ordered_labels.append(self.labels[old_cell_index])

    def set_spike_trains_from_spike_nums(self):
        # n_cells = len(self.spike_nums)
        # n_times = len(self.spike_nums[0, :])
        self.spike_trains = []
        for cell_spikes in self.spike_nums:
            self.spike_trains.append(np.where(cell_spikes)[0].astype(float))


class CoordClass:
    def __init__(self, coord, nb_lines, nb_col, mouse_session=None):
        # contour coord
        self.coord = coord
        self.nb_lines = nb_lines
        self.nb_col = nb_col
        # dict of tuples, key is the cell #, cell center coord x and y (x and y are inverted for imgshow)
        self.center_coord = list()
        self.img_filled = None
        self.img_contours = None
        # used in case some cells would be removed, to we can update the centers accordingly
        self.neurons_removed = []
        self.ms = mouse_session
        # compute_center_coord() will be called when a mouseSession will be created
        # self.compute_center_coord()

    def compute_center_coord(self):
        """
        Compute the center of each cell in the graph and build the image with contours and filled cells
        :return:
        """
        self.center_coord = list()

        test_img = np.zeros((self.nb_lines, self.nb_col), dtype="int8")
        early_born_img = np.zeros((self.nb_lines, self.nb_col), dtype="int8")

        # early_born cell
        early_born_cell = self.ms.early_born_cell
        c = self.coord[early_born_cell]
        c = c[0] - 1
        c_filtered = c.astype(int)
        # c = signal.medfilt(c)
        early_born_img[c_filtered[1, :], c_filtered[0, :]] = 1
        # test_img = morphology.binary_dilation(test_img)
        early_born_img = morphology.binary_fill_holes(early_born_img)
        # green value is -1
        early_born_img[c_filtered[1, :], c_filtered[0, :]] = 0
        # border value is 2
        # test_img[c_filtered[1, :], c_filtered[0, :]] = 2
        # self.img_filled = test_img

        for i, c in enumerate(self.coord):
            if i in self.neurons_removed:
                continue

            # it is necessary to remove one, as data comes from matlab, starting from 1 and not 0
            c = c[0] - 1

            if c.shape[0] == 0:
                print(f'Error: {i} c.shape {c.shape}')
                continue
            # c = signal.medfilt(c)
            c_filtered = c.astype(int)
            bw = np.zeros((self.nb_lines, self.nb_col), dtype="int8")
            # morphology.binary_fill_holes(input
            bw[c_filtered[1, :], c_filtered[0, :]] = 1
            # early born as been drawn earlier, but we need to update center_coord
            test_img[c_filtered[1, :], c_filtered[0, :]] = 1
            c_x, c_y = ndimage.center_of_mass(bw)
            self.center_coord.append((c_y, c_x))

        self.img_filled = np.zeros((self.nb_lines, self.nb_col), dtype="int8")
        # specifying output, otherwise binary_fill_holes return a boolean array
        morphology.binary_fill_holes(test_img, output=self.img_filled)

        with_borders = False

        # now putting contour to value 2
        for i, c in enumerate(self.coord):
            if i in self.neurons_removed:
                continue

            # it is necessary to remove one, as data comes from matlab, starting from 1 and not 0
            c = c[0] - 1

            if c.shape[0] == 0:
                continue
            c_filtered = c.astype(int)
            # border to 2, to be in black
            if with_borders:
                self.img_filled[c_filtered[1, :], c_filtered[0, :]] = 2
            else:
                self.img_filled[c_filtered[1, :], c_filtered[0, :]] = 0

        do_early_born_with_different_color = False
        if do_early_born_with_different_color:
            # filling early_born cell with value to -1
            for n, pixels in enumerate(early_born_img):
                # print(f"pixels > 0 {np.where(pixels > 0)}")
                self.img_filled[n, np.where(pixels > 0)[0]] = -1

        # if we dilate, some cells will fusion
        dilatation_version = False
        if (not with_borders) and dilatation_version:
            self.img_filled = morphology.binary_dilation(self.img_filled)

        self.img_contours = test_img


def compute_and_plot_clusters_raster_sarah_s_way(spike_struct, data_descr, mouse_session,
                                                 sliding_window_duration, sce_times_numbers,
                                                 SCE_times, perc_threshold,
                                                 n_surrogate_activity_threshold,
                                                 fca_early_stop=True):
    ms = mouse_session
    param = ms.param
    n_cells = len(spike_struct.spike_trains)
    sigma = 4
    n_surrogate_fca = 20

    merge_history, current_cluster = functional_clustering_algorithm(spike_struct.spike_trains,
                                                                     nsurrogate=n_surrogate_fca,
                                                                     sigma=sigma,
                                                                     early_stop=fca_early_stop,
                                                                     rolling_surrogate=False)
    print(f"merge_history {merge_history}")
    print(f"current_cluster {current_cluster}")
    if fca_early_stop:
        # each element is a list representing the cells of a cluster
        cells_in_clusters = []
        for element in current_cluster:
            if isinstance(element, int) or isinstance(element, np.int64):
                continue
            cells_in_clusters.append(fca.give_all_cells_from_cluster(element))
        n_cluster = len(cells_in_clusters)
    else:
        min_scale, max_scale = fca.get_min_max_scale_from_merge_history(merge_history)
        cluster_tree = fca.ClusterTree(clusters_lists=current_cluster[0], merge_history_list=merge_history, father=None,
                                       n_cells=n_cells, max_scale_value=max_scale, non_significant_color="white")

        n_cluster = len(cluster_tree.cluster_nb_list)

    print(f"n_cluster {n_cluster}")
    # each index correspond to a cell index, and the value is the cluster the cell belongs,
    # if -1, it means no cluster
    cluster_labels = np.zeros(n_cells, dtype="int16")
    cluster_labels = cluster_labels - 1
    for cluster in np.arange(n_cluster):
        if fca_early_stop:
            cells_in_cluster = np.array(cells_in_clusters[cluster])
        else:
            ct = cluster_tree.cluster_dict[cluster]
            cells_in_cluster = ct.get_cells_id()
            cells_in_cluster = np.array(cells_in_cluster)
        cluster_labels[cells_in_cluster] = cluster

    if fca_early_stop:
        axes_list_raster = None
    else:
        fig = plt.figure(figsize=(20, 14))
        fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'h_pad': 3})
        outer = gridspec.GridSpec(2, 1, height_ratios=[60, 40])

        # clusters display
        inner_top = gridspec.GridSpecFromSubplotSpec(1, 1,
                                                     subplot_spec=outer[0])

        inner_bottom = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                        subplot_spec=outer[1], height_ratios=[10, 2])

        # top is bottom and bottom is top, so the raster is under
        # ax1 contains raster
        ax1 = fig.add_subplot(inner_top[0])

        ax2 = fig.add_subplot(inner_bottom[0])
        # ax3 contains the peak activity diagram
        ax3 = fig.add_subplot(inner_bottom[1], sharex=ax2)
        axes_list_raster = [ax2, ax3]

    clustered_spike_nums = np.copy(spike_struct.spike_nums)
    cell_labels = []
    cluster_horizontal_thresholds = []
    cells_to_highlight = []
    cells_to_highlight_colors = []
    start = 0
    for k in np.arange(-1, np.max(cluster_labels) + 1):
        e = np.equal(cluster_labels, k)
        nb_k = np.sum(e)
        clustered_spike_nums[start:start + nb_k, :] = spike_struct.spike_nums[e, :]
        for index in np.where(e)[0]:
            cell_labels.append(spike_struct.labels[index])
        if k >= 0:
            color = cm.nipy_spectral(float(k + 1) / (n_cluster + 1))
            cell_indices = list(np.arange(start, start + nb_k))
            cells_to_highlight.extend(cell_indices)
            cells_to_highlight_colors.extend([color] * len(cell_indices))
        start += nb_k
        if (k + 1) < (np.max(cluster_labels) + 1):
            cluster_horizontal_thresholds.append(start)

    plot_spikes_raster(spike_nums=clustered_spike_nums, param=ms.param,
                       spike_train_format=False,
                       title=f"{n_cluster} clusters raster plot {ms.description}",
                       file_name=f"spike_nums_{data_descr}_{n_cluster}_clusters_hierarchical",
                       y_ticks_labels=cell_labels,
                       y_ticks_labels_size=4,
                       save_raster=True,
                       show_raster=False,
                       plot_with_amplitude=False,
                       activity_threshold=spike_struct.activity_threshold,
                       span_cells_to_highlight=False,
                       raster_face_color='black',
                       cell_spikes_color='white',
                       horizontal_lines=np.array(cluster_horizontal_thresholds) - 0.5,
                       horizontal_lines_colors=['white'] * len(cluster_horizontal_thresholds),
                       horizontal_lines_sytle="dashed",
                       horizontal_lines_linewidth=[1] * len(cluster_horizontal_thresholds),
                       vertical_lines=SCE_times,
                       vertical_lines_colors=['white'] * len(SCE_times),
                       vertical_lines_sytle="solid",
                       vertical_lines_linewidth=[0.2] * len(SCE_times),
                       cells_to_highlight=cells_to_highlight,
                       cells_to_highlight_colors=cells_to_highlight_colors,
                       sliding_window_duration=sliding_window_duration,
                       show_sum_spikes_as_percentage=True,
                       spike_shape="o",
                       spike_shape_size=2,
                       save_formats="pdf",
                       axes_list=axes_list_raster,
                       SCE_times=SCE_times,
                       ylabel="")

    if not fca_early_stop:
        plot_dendogram_from_fca(cluster_tree=cluster_tree, nb_cells=n_cells, save_plot=True,
                                file_name=f"dendogram_{data_descr}",
                                param=param,
                                cell_labels=spike_struct.labels,
                                axes_list=[ax1], fig_to_use=fig)

    result_detection = detect_cluster_activations_with_sliding_window(spike_nums=spike_struct.spike_nums,
                                                                      window_duration=sliding_window_duration,
                                                                      cluster_labels=cluster_labels,
                                                                      sce_times_numbers=sce_times_numbers)

    clusters_activations_by_cell, clusters_activations_by_cluster, cluster_particpation_to_sce, \
    clusters_corresponding_index = result_detection

    cell_labels = []
    cluster_horizontal_thresholds = []
    cells_to_highlight = []
    cells_to_highlight_colors = []
    start = 0
    for k in np.arange(np.max(cluster_labels) + 1):
        e = np.equal(cluster_labels, k)
        nb_k = np.sum(e)
        if nb_k == 0:
            continue
        for index in np.where(e)[0]:
            cell_labels.append(spike_struct.labels[index])
        if k >= 0:
            color = cm.nipy_spectral(float(k + 1) / (n_cluster + 1))
            cell_indices = list(np.arange(start, start + nb_k))
            cells_to_highlight.extend(cell_indices)
            cells_to_highlight_colors.extend([color] * len(cell_indices))
        start += nb_k
        if (k + 1) < (np.max(cluster_labels) + 1):
            cluster_horizontal_thresholds.append(start)

    fig = plt.figure(figsize=(20, 14))
    fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'h_pad': 3})
    outer = gridspec.GridSpec(1, 1)  # , height_ratios=[60, 40])

    # clusters display
    # inner_top = gridspec.GridSpecFromSubplotSpec(1, 1,
    #                                              subplot_spec=outer[0])

    inner_bottom = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                    subplot_spec=outer[0], height_ratios=[10, 2])

    # top is bottom and bottom is top, so the raster is under
    # ax1 contains raster
    ax1 = fig.add_subplot(inner_bottom[0])
    # ax3 contains the peak activity diagram
    ax2 = fig.add_subplot(inner_bottom[1], sharex=ax1)

    plot_spikes_raster(spike_nums=clusters_activations_by_cell, param=ms.param,
                       spike_train_format=False,
                       file_name=f"raster_clusters_detection_{ms.session_id}",
                       y_ticks_labels=cell_labels,
                       y_ticks_labels_size=4,
                       save_raster=True,
                       show_raster=False,
                       plot_with_amplitude=False,
                       span_cells_to_highlight=False,
                       raster_face_color='black',
                       cell_spikes_color='white',
                       horizontal_lines=np.array(cluster_horizontal_thresholds) - 0.5,
                       horizontal_lines_colors=['white'] * len(cluster_horizontal_thresholds),
                       horizontal_lines_sytle="dashed",
                       vertical_lines=SCE_times,
                       vertical_lines_colors=['white'] * len(SCE_times),
                       vertical_lines_sytle="solid",
                       vertical_lines_linewidth=[0.4] * len(SCE_times),
                       cells_to_highlight=cells_to_highlight,
                       cells_to_highlight_colors=cells_to_highlight_colors,
                       sliding_window_duration=sliding_window_duration,
                       show_sum_spikes_as_percentage=True,
                       spike_shape="|",
                       spike_shape_size=1,
                       save_formats="pdf",
                       axes_list=[ax1],
                       without_activity_sum=True,
                       ylabel="")

    plot_sum_active_clusters(clusters_activations=clusters_activations_by_cluster, param=param,
                             sliding_window_duration=sliding_window_duration,
                             data_str=f"raster_clusters_participation_{ms.session_id}",
                             axes_list=[ax2],
                             fig_to_use=fig)

    plot_hist_clusters_by_sce(cluster_particpation_to_sce, data_str="hist_percentage_of_network_events", param=param)

    plt.close()

    save_stat_SCE_and_cluster_fca_version(spike_nums_to_use=spike_struct.spike_nums,
                                          sigma=sigma,
                                          activity_threshold=spike_struct.activity_threshold,
                                          SCE_times=SCE_times, n_cluster=n_cluster, param=param,
                                          sliding_window_duration=sliding_window_duration,
                                          cluster_labels_for_neurons=cluster_labels,
                                          perc_threshold=perc_threshold,
                                          n_surrogate_FCA=n_surrogate_fca,
                                          n_surrogate_activity_threshold=n_surrogate_activity_threshold)


def save_stat_SCE_and_cluster_fca_version(spike_nums_to_use, activity_threshold, sigma,
                                          SCE_times, n_cluster, param, sliding_window_duration,
                                          cluster_labels_for_neurons, perc_threshold,
                                          n_surrogate_FCA, n_surrogate_activity_threshold):
    round_factor = 2
    file_name = f'{param.path_results}/stat_fca_v_{n_cluster}_clusters_{param.time_str}.txt'
    with open(file_name, "w", encoding='UTF-8') as file:
        file.write(f"Stat FCA version for {n_cluster} clusters" + '\n')
        file.write("" + '\n')
        file.write(f"cells {len(spike_nums_to_use)}, events {len(SCE_times)}" + '\n')
        file.write(f"Event participation threshold {activity_threshold}, {perc_threshold} percentile, "
                   f"{n_surrogate_activity_threshold} surrogates" + '\n')
        file.write(f"Sliding window duration {sliding_window_duration}" + '\n')
        file.write(f"Sigma {sigma}" + f", {n_surrogate_FCA} FCA surrogates " + '\n')
        file.write("" + '\n')
        file.write("" + '\n')

        for k in np.arange(n_cluster):
            e_cells = np.equal(cluster_labels_for_neurons, k)
            n_cells_in_cluster = np.sum(e_cells)

            file.write("#" * 10 + f"   cluster {k} / {n_cells_in_cluster} cells" +
                       "#" * 10 + '\n')
            file.write('\n')


def compute_and_plot_clusters_raster_arnaud_s_way(spike_struct, spike_nums_to_use, cellsinpeak, data_descr,
                                                  mouse_session,
                                                  sliding_window_duration, sce_times_numbers,
                                                  SCE_times, perc_threshold,
                                                  n_surrogate_activity_threshold):
    # perc_threshold is the number of percentile choosen to determine the threshold
    ms = mouse_session
    # -------- clustering params ------ -----
    range_n_clusters_k_mean = np.arange(2, 17)
    n_surrogate_k_mean = 100

    param = ms.param

    clusters_sce, best_kmeans_by_cluster, m_cov_sces, cluster_labels_for_neurons, surrogate_percentiles = \
        co_var_first_and_clusters(cells_in_sce=cellsinpeak, shuffling=True,
                                  n_surrogate=n_surrogate_k_mean,
                                  fct_to_keep_best_silhouettes=np.mean,
                                  range_n_clusters=range_n_clusters_k_mean,
                                  nth_best_clusters=-1,
                                  plot_matrix=False,
                                  data_str=data_descr,
                                  path_results=param.path_results,
                                  neurons_labels=spike_struct.labels)

    for n_cluster in range_n_clusters_k_mean:
        clustered_spike_nums = np.copy(spike_nums_to_use)
        cell_labels = []
        cluster_labels = cluster_labels_for_neurons[n_cluster]
        cluster_horizontal_thresholds = []
        cells_to_highlight = []
        cells_to_highlight_colors = []
        start = 0
        for k in np.arange(-1, np.max(cluster_labels) + 1):
            e = np.equal(cluster_labels, k)
            nb_k = np.sum(e)
            clustered_spike_nums[start:start + nb_k, :] = spike_nums_to_use[e, :]
            for index in np.where(e)[0]:
                cell_labels.append(spike_struct.labels[index])
            if k >= 0:
                color = cm.nipy_spectral(float(k + 1) / (n_cluster + 1))
                cell_indices = list(np.arange(start, start + nb_k))
                cells_to_highlight.extend(cell_indices)
                cells_to_highlight_colors.extend([color] * len(cell_indices))
            start += nb_k
            if (k + 1) < (np.max(cluster_labels) + 1):
                cluster_horizontal_thresholds.append(start)

        fig = plt.figure(figsize=(20, 14))
        fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'h_pad': 2})
        outer = gridspec.GridSpec(2, 1, height_ratios=[60, 40])

        inner_top = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                     subplot_spec=outer[1], height_ratios=[10, 2])

        # clusters display
        inner_bottom = gridspec.GridSpecFromSubplotSpec(1, 3,
                                                        subplot_spec=outer[0], width_ratios=[6, 10, 6])

        # top is bottom and bottom is top, so the raster is under
        # ax1 contains raster
        ax1 = fig.add_subplot(inner_top[0])
        # ax2 contains the peak activity diagram
        ax2 = fig.add_subplot(inner_top[1], sharex=ax1)

        ax3 = fig.add_subplot(inner_bottom[0])
        # ax2 contains the peak activity diagram
        ax4 = fig.add_subplot(inner_bottom[1])
        ax5 = fig.add_subplot(inner_bottom[2])

        plot_spikes_raster(spike_nums=clustered_spike_nums, param=ms.param,
                           spike_train_format=False,
                           title=f"{n_cluster} clusters raster plot {ms.session_id}",
                           file_name=f"spike_nums_{ms.session_id}_{n_cluster}_clusters",
                           y_ticks_labels=cell_labels,
                           y_ticks_labels_size=4,
                           save_raster=False,
                           show_raster=False,
                           plot_with_amplitude=False,
                           activity_threshold=spike_struct.activity_threshold,
                           span_cells_to_highlight=False,
                           raster_face_color='black',
                           cell_spikes_color='white',
                           horizontal_lines=np.array(cluster_horizontal_thresholds) - 0.5,
                           horizontal_lines_colors=['white'] * len(cluster_horizontal_thresholds),
                           horizontal_lines_sytle="dashed",
                           horizontal_lines_linewidth=[1] * len(cluster_horizontal_thresholds),
                           vertical_lines=SCE_times,
                           vertical_lines_colors=['white'] * len(SCE_times),
                           vertical_lines_sytle="solid",
                           vertical_lines_linewidth=[0.2] * len(SCE_times),
                           cells_to_highlight=cells_to_highlight,
                           cells_to_highlight_colors=cells_to_highlight_colors,
                           sliding_window_duration=sliding_window_duration,
                           show_sum_spikes_as_percentage=True,
                           spike_shape="|",
                           spike_shape_size=1,
                           save_formats="pdf",
                           axes_list=[ax1, ax2],
                           SCE_times=SCE_times)

        show_co_var_first_matrix(cells_in_peak=np.copy(cellsinpeak), m_sces=m_cov_sces,
                                 n_clusters=n_cluster, kmeans=best_kmeans_by_cluster[n_cluster],
                                 cluster_labels_for_neurons=cluster_labels_for_neurons[n_cluster],
                                 data_str=data_descr, path_results=param.path_results,
                                 show_silhouettes=True, neurons_labels=spike_struct.labels,
                                 surrogate_silhouette_avg=surrogate_percentiles[n_cluster],
                                 axes_list=[ax5, ax3, ax4], fig_to_use=fig, save_formats="pdf")
        plt.close()

        # ######### Plot that show cluster activation

        result_detection = detect_cluster_activations_with_sliding_window(spike_nums=spike_nums_to_use,
                                                                          window_duration=sliding_window_duration,
                                                                          cluster_labels=cluster_labels,
                                                                          sce_times_numbers=sce_times_numbers,
                                                                          debug_mode=False)

        clusters_activations_by_cell, clusters_activations_by_cluster, cluster_particpation_to_sce, \
        clusters_corresponding_index = result_detection
        # print(f"cluster_particpation_to_sce {cluster_particpation_to_sce}")

        cell_labels = []
        cluster_horizontal_thresholds = []
        cells_to_highlight = []
        cells_to_highlight_colors = []
        start = 0
        for k in np.arange(np.max(cluster_labels) + 1):
            e = np.equal(cluster_labels, k)
            nb_k = np.sum(e)
            if nb_k == 0:
                continue
            for index in np.where(e)[0]:
                cell_labels.append(spike_struct.labels[index])
            if k >= 0:
                color = cm.nipy_spectral(float(k + 1) / (n_cluster + 1))
                cell_indices = list(np.arange(start, start + nb_k))
                cells_to_highlight.extend(cell_indices)
                cells_to_highlight_colors.extend([color] * len(cell_indices))
            start += nb_k
            if (k + 1) < (np.max(cluster_labels) + 1):
                cluster_horizontal_thresholds.append(start)

        fig = plt.figure(figsize=(20, 14))
        fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'h_pad': 3})
        outer = gridspec.GridSpec(1, 1)  # , height_ratios=[60, 40])

        # clusters display
        # inner_top = gridspec.GridSpecFromSubplotSpec(1, 1,
        #                                              subplot_spec=outer[0])

        inner_bottom = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                        subplot_spec=outer[0], height_ratios=[10, 2])

        # top is bottom and bottom is top, so the raster is under
        # ax1 contains raster
        ax1 = fig.add_subplot(inner_bottom[0])
        # ax3 contains the peak activity diagram
        ax2 = fig.add_subplot(inner_bottom[1], sharex=ax1)

        plot_spikes_raster(spike_nums=clusters_activations_by_cell, param=ms.param,
                           spike_train_format=False,
                           file_name=f"raster_clusters_detection_{ms.session_id}",
                           y_ticks_labels=cell_labels,
                           y_ticks_labels_size=4,
                           save_raster=True,
                           show_raster=False,
                           plot_with_amplitude=False,
                           span_cells_to_highlight=False,
                           raster_face_color='black',
                           cell_spikes_color='white',
                           horizontal_lines=np.array(cluster_horizontal_thresholds) - 0.5,
                           horizontal_lines_colors=['white'] * len(cluster_horizontal_thresholds),
                           horizontal_lines_sytle="dashed",
                           vertical_lines=SCE_times,
                           vertical_lines_colors=['white'] * len(SCE_times),
                           vertical_lines_sytle="dashed",
                           vertical_lines_linewidth=[0.4] * len(SCE_times),
                           cells_to_highlight=cells_to_highlight,
                           cells_to_highlight_colors=cells_to_highlight_colors,
                           sliding_window_duration=sliding_window_duration,
                           show_sum_spikes_as_percentage=True,
                           spike_shape="|",
                           spike_shape_size=1,
                           save_formats="pdf",
                           axes_list=[ax1],
                           without_activity_sum=True,
                           ylabel="")

        # print(f"n_cluster {n_cluster} len(clusters_activations) {len(clusters_activations)}")

        plot_sum_active_clusters(clusters_activations=clusters_activations_by_cluster, param=param,
                                 sliding_window_duration=sliding_window_duration,
                                 data_str=f"raster_{n_cluster}_clusters_participation_{ms.session_id}",
                                 axes_list=[ax2],
                                 fig_to_use=fig)

        plt.close()

        save_stat_SCE_and_cluster_k_mean_version(spike_nums_to_use=spike_nums_to_use,
                                                 activity_threshold=spike_struct.activity_threshold,
                                                 k_means=best_kmeans_by_cluster[n_cluster],
                                                 SCE_times=SCE_times, n_cluster=n_cluster, param=param,
                                                 sliding_window_duration=sliding_window_duration,
                                                 cluster_labels_for_neurons=cluster_labels_for_neurons[n_cluster],
                                                 perc_threshold=perc_threshold,
                                                 n_surrogate_k_mean=n_surrogate_k_mean,
                                                 n_surrogate_activity_threshold=n_surrogate_activity_threshold
                                                 )


def save_stat_SCE_and_cluster_k_mean_version(spike_nums_to_use, activity_threshold, k_means,
                                             SCE_times, n_cluster, param, sliding_window_duration,
                                             cluster_labels_for_neurons, perc_threshold,
                                             n_surrogate_k_mean,
                                             n_surrogate_activity_threshold):
    round_factor = 2
    file_name = f'{param.path_results}/stat_k_mean_v_{n_cluster}_clusters_{param.time_str}.txt'
    with open(file_name, "w", encoding='UTF-8') as file:
        file.write(f"Stat k_mean version for {n_cluster} clusters" + '\n')
        file.write("" + '\n')
        file.write(f"cells {len(spike_nums_to_use)}, events {len(SCE_times)}" + '\n')
        file.write(f"Event participation threshold {activity_threshold}, {perc_threshold} percentile, "
                   f"{n_surrogate_activity_threshold} surrogates" + '\n')
        file.write(f"Sliding window duration {sliding_window_duration}" + '\n')
        file.write(f"{n_surrogate_k_mean} surrogates for kmean" + '\n')
        file.write("" + '\n')
        file.write("" + '\n')
        cluster_labels = k_means.labels_

        for k in np.arange(n_cluster):

            e = np.equal(cluster_labels, k)

            nb_sce_in_cluster = np.sum(e)
            sce_ids = np.where(e)[0]

            e_cells = np.equal(cluster_labels_for_neurons, k)
            n_cells_in_cluster = np.sum(e_cells)

            file.write("#" * 10 + f"   cluster {k} / {nb_sce_in_cluster} events / {n_cells_in_cluster} cells" +
                       "#" * 10 + '\n')
            file.write('\n')

            duration_values = np.zeros(nb_sce_in_cluster, dtype="uint16")
            max_activity_values = np.zeros(nb_sce_in_cluster, dtype="float")
            mean_activity_values = np.zeros(nb_sce_in_cluster, dtype="float")
            overall_activity_values = np.zeros(nb_sce_in_cluster, dtype="float")

            for n, sce_id in enumerate(sce_ids):
                duration_values[n], max_activity_values[n], \
                mean_activity_values[n], overall_activity_values[n] = \
                    give_stat_one_sce(sce_id=sce_id,
                                      spike_nums_to_use=spike_nums_to_use,
                                      SCE_times=SCE_times, sliding_window_duration=sliding_window_duration)
            file.write(f"Duration: mean {np.round(np.mean(duration_values), round_factor)}, "
                       f"std {np.round(np.std(duration_values), round_factor)}, "
                       f"median {np.round(np.median(duration_values), round_factor)}\n")
            file.write(f"Overall participation: mean {np.round(np.mean(overall_activity_values), round_factor)}, "
                       f"std {np.round(np.std(overall_activity_values), round_factor)}, "
                       f"median {np.round(np.median(overall_activity_values), round_factor)}\n")
            file.write(f"Max participation: mean {np.round(np.mean(max_activity_values), round_factor)}, "
                       f"std {np.round(np.std(max_activity_values), round_factor)}, "
                       f"median {np.round(np.median(max_activity_values), round_factor)}\n")
            file.write(f"Mean participation: mean {np.round(np.mean(mean_activity_values), round_factor)}, "
                       f"std {np.round(np.std(mean_activity_values), round_factor)}, "
                       f"median {np.round(np.median(mean_activity_values), round_factor)}\n")
            file.write('\n')

        file.write('\n')
        file.write('\n')
        file.write("#" * 50 + '\n')
        file.write('\n')
        file.write('\n')
        # for each SCE
        for sce_id in np.arange(len(SCE_times)):
            result = give_stat_one_sce(sce_id=sce_id,
                                       spike_nums_to_use=spike_nums_to_use,
                                       SCE_times=SCE_times,
                                       sliding_window_duration=sliding_window_duration)
            duration_in_frames, max_activity, mean_activity, overall_activity = result
            file.write(f"SCE {sce_id}" + '\n')
            file.write(f"Duration_in_frames {duration_in_frames}" + '\n')
            file.write(f"Overall participation {np.round(overall_activity, round_factor)}" + '\n')
            file.write(f"Max participation {np.round(max_activity, round_factor)}" + '\n')
            file.write(f"Mean participation {np.round(mean_activity, round_factor)}" + '\n')

            file.write('\n')
            file.write('\n')


def give_stat_one_sce(sce_id, spike_nums_to_use, SCE_times, sliding_window_duration):
    """

    :param sce_id:
    :param spike_nums_to_use:
    :param SCE_times:
    :param sliding_window_duration:
    :return: duration_in_frames: duration of the sce in frames
     max_activity: the max number of cells particpating during a window_duration to the sce
     mean_activity: the mean of cells particpating during the sum of window_duration
     overall_activity: the number of different cells participating to the SCE all along
     if duration == sliding_window duration, max_activity, mean_activity and overall_activity will be equal
    """
    time_tuple = SCE_times[sce_id]
    duration_in_frames = (time_tuple[1] - time_tuple[0]) + 1
    n_slidings = (duration_in_frames - sliding_window_duration) + 1
    sum_activity_for_each_frame = np.zeros(n_slidings)
    for n in np.arange(n_slidings):
        # see to use window_duration to find the amount of participation
        time_beg = time_tuple[0] + n
        sum_activity_for_each_frame[n] = len(np.where(np.sum(spike_nums_to_use[:,
                                                             time_beg:(time_beg + sliding_window_duration)],
                                                             axis=1))[0])
    max_activity = np.max(sum_activity_for_each_frame)
    mean_activity = np.mean(sum_activity_for_each_frame)
    overall_activity = len(np.where(np.sum(spike_nums_to_use[:,
                                           time_tuple[0]:(time_tuple[1] + 1)], axis=1))[0])

    return duration_in_frames, max_activity, mean_activity, overall_activity


def main():
    root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    path_data = root_path + "data/"
    path_results_raw = root_path + "results/"

    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    path_results = path_results_raw + f"{time_str}"
    os.mkdir(path_results)

    # --------------------------------------------------------------------------------
    # ------------------------------ param section ------------------------------
    # --------------------------------------------------------------------------------

    # param will be set later when the spike_nums will have been constructed
    param = HNEParameters(time_str=time_str, path_results=path_results, error_rate=2,
                          time_inter_seq=100, min_duration_intra_seq=-5, min_len_seq=10, min_rep_nb=4,
                          max_branches=20, stop_if_twin=False,
                          no_reverse_seq=False, spike_rate_weight=False, path_data=path_data)

    # loading data
    p7_171012_a000_ms = MouseSession(age=7, session_id="171012_a000", nb_ms_by_frame=100, param=param)
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital"}
    p7_171012_a000_ms.load_data_from_file(file_name_to_load="p7_171012_a000.mat", variables_mapping=variables_mapping)

    p12_171110_a000_ms = MouseSession(age=12, session_id="171110_a000", nb_ms_by_frame=100, param=param)
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital", "coord": "ContoursAll"}
    p12_171110_a000_ms.load_data_from_file(file_name_to_load="p12_171110_a000.mat", variables_mapping=variables_mapping)

    arnaud_ms = MouseSession(age=24, session_id="arnaud", nb_ms_by_frame=50, param=param)
    variables_mapping = {"spike_nums": "spikenums"}
    arnaud_ms.load_data_from_file(file_name_to_load="spikenumsarnaud.mat", variables_mapping=variables_mapping)

    available_ms = [p7_171012_a000_ms, p12_171110_a000_ms, arnaud_ms]
    ms_to_analyse = [p7_171012_a000_ms]
    ms_to_analyse = [p12_171110_a000_ms]

    # ------------------------------ param section ------------------------------

    for ms in ms_to_analyse:
        ###################################################################
        ###################################################################
        # ###########    SCE detection and clustering        ##############
        ###################################################################
        ###################################################################

        spike_struct = ms.spike_struct
        n_cells = len(spike_struct.spike_nums)
        # spike_struct.build_spike_trains()
        sarah_way = True
        if sarah_way:
            sliding_window_duration = 5
            spike_nums_to_use = spike_struct.spike_nums
        else:
            sliding_window_duration = 1
            spike_nums_to_use = spike_struct.spike_nums_dur

        perc_threshold = 95
        n_surrogate_activity_threshold = 100
        activity_threshold = get_sce_detection_threshold(spike_nums=spike_nums_to_use,
                                                         window_duration=sliding_window_duration,
                                                         spike_train_mode=False,
                                                         n_surrogate=n_surrogate_activity_threshold,
                                                         perc_threshold=perc_threshold,
                                                         debug_mode=True)
        print(f"perc_threshold {perc_threshold}, "
              f"activity_threshold {activity_threshold}, {np.round((activity_threshold/n_cells)*100, 2)}%")
        print(f"sliding_window_duration {sliding_window_duration}")
        spike_struct.activity_threshold = activity_threshold
        param.activity_threshold = activity_threshold

        print("plot_spikes_raster")

        if True:
            plot_spikes_raster(spike_nums=spike_nums_to_use, param=ms.param,
                               spike_train_format=False,
                               title=f"raster plot {ms.description}",
                               file_name=f"spike_nums_dur_{ms.description}",
                               y_ticks_labels=spike_struct.labels,
                               y_ticks_labels_size=4,
                               save_raster=True,
                               show_raster=False,
                               plot_with_amplitude=False,
                               activity_threshold=spike_struct.activity_threshold,
                               # 500 ms window
                               sliding_window_duration=sliding_window_duration,
                               show_sum_spikes_as_percentage=True,
                               spike_shape="|",
                               spike_shape_size=1,
                               save_formats="pdf")

        # TODO: detect_sce_with_sliding_window with spike_trains
        sce_detection_result = detect_sce_with_sliding_window(spike_nums=spike_nums_to_use,
                                                              window_duration=sliding_window_duration,
                                                              perc_threshold=perc_threshold,
                                                              activity_threshold=activity_threshold,
                                                              debug_mode=False,
                                                              no_redundancy=False)
        print(f"sce_with_sliding_window detected")
        cellsinpeak = sce_detection_result[2]
        SCE_times = sce_detection_result[1]
        sce_times_numbers = sce_detection_result[3]
        print(f"Nb SCE: {cellsinpeak.shape}")
        # print(f"Nb spikes by SCE: {np.sum(cellsinpeak, axis=0)}")
        display_isi_info = False
        if display_isi_info:
            cells_isi = tools_misc.get_isi(spike_data=spike_struct.spike_nums, spike_trains_format=False)
            for cell_index in np.arange(len(spike_struct.spike_nums)):
                print(f"{spike_struct.labels[cell_index]} median isi: {np.round(np.median(cells_isi[cell_index]), 2)}, "
                      f"mean isi {np.round(np.mean(cells_isi[cell_index]), 2)}")

        # return a dict of list of list of neurons, representing the best clusters
        # (as many as nth_best_clusters).
        # the key is the K from the k-mean

        data_descr = f"{ms.description}"

        if sarah_way:
            sigma=4
            n_surrogate_fca=20
            compute_and_plot_clusters_raster_fca_version(spike_trains=spike_struct.spike_trains,
                                                         spike_nums=spike_struct.spike_nums,
                                                         data_descr=data_descr, param=param,
                                                         sliding_window_duration=sliding_window_duration,
                                                         SCE_times=SCE_times, sce_times_numbers=sce_times_numbers,
                                                         perc_threshold=perc_threshold,
                                                         n_surrogate_activity_threshold=
                                                         n_surrogate_activity_threshold,
                                                         sigma=sigma, n_surrogate_fca=20,
                                                         labels=spike_struct.labels,
                                                         activity_threshold=activity_threshold,
                                                         fca_early_stop=True)
            #
            # compute_and_plot_clusters_raster_sarah_s_way(spike_struct=spike_struct,
            #                                              data_descr=data_descr, mouse_session=ms,
            #                                              sliding_window_duration=sliding_window_duration,
            #                                              SCE_times=SCE_times, sce_times_numbers=sce_times_numbers,
            #                                              fca_early_stop=True,
            #                                              perc_threshold=perc_threshold,
            #                                              n_surrogate_activity_threshold=n_surrogate_activity_threshold)
        else:
            range_n_clusters_k_mean = np.arange(2, 17)
            n_surrogate_k_mean = 50
            compute_and_plot_clusters_raster_kmean_version(labels=ms.spike_struct.labels,
                                                           activity_threshold=ms.spike_struct.activity_threshold,
                                                           range_n_clusters_k_mean=range_n_clusters_k_mean,
                                                           n_surrogate_k_mean=n_surrogate_k_mean,
                                                           with_shuffling = True,
                                                           spike_nums_to_use=spike_nums_to_use,
                                                           cellsinpeak=cellsinpeak,
                                                          data_descr=data_descr,
                                                           param=ms.param,
                                                           sliding_window_duration=sliding_window_duration,
                                                           SCE_times=SCE_times, sce_times_numbers=sce_times_numbers,
                                                           perc_threshold=perc_threshold,
                                                           n_surrogate_activity_threshold=
                                                           n_surrogate_activity_threshold)

            # compute_and_plot_clusters_raster_arnaud_s_way(spike_struct=ms.spike_struct,
            #                                               spike_nums_to_use=spike_nums_to_use,
            #                                               cellsinpeak=cellsinpeak,
            #                                               data_descr=data_descr, mouse_session=ms,
            #                                               sliding_window_duration=sliding_window_duration,
            #                                               SCE_times=SCE_times, sce_times_numbers=sce_times_numbers,
            #                                               perc_threshold=perc_threshold,
            #                                               n_surrogate_activity_threshold=n_surrogate_activity_threshold)

        ###################################################################
        ###################################################################
        # ##############    Sequences detection        ###################
        ###################################################################
        ###################################################################

    return


main()
