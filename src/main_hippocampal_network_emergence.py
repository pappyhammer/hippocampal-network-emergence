import pandas as pd
# from scipy.io import loadmat
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from bisect import bisect

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
from pattern_discovery.seq_solver.markov_way import find_sequences_in_ordered_spike_nums
from pattern_discovery.seq_solver.markov_way import give_me_stat_on_sorting_seq_results
from pattern_discovery.seq_solver.markov_way import order_spike_nums_by_seq
import pattern_discovery.tools.param as p_disc_param
import pattern_discovery.tools.misc as tools_misc
from pattern_discovery.display.raster import plot_spikes_raster
from pattern_discovery.tools.loss_function import loss_function_with_sliding_window
import pattern_discovery.tools.trains as trains_module
from pattern_discovery.tools.sce_detection import get_sce_detection_threshold, detect_sce_with_sliding_window
from sortedcontainers import SortedList, SortedDict
from pattern_discovery.clustering.kmean_version.k_mean_clustering import compute_and_plot_clusters_raster_kmean_version
from pattern_discovery.clustering.kmean_version.k_mean_clustering import give_stat_one_sce
from pattern_discovery.clustering.fca.fca import compute_and_plot_clusters_raster_fca_version
import pattern_discovery as pattern_discovery


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
        # for plotting ages
        self.markers = ['o', '*', 's', 'v', '<', '>', '^', 'x', '+', "."]  # d losange
        self.colors = ["darkmagenta", "white", "saddlebrown", "blue", "red", "darkgrey", "chartreuse", "cornflowerblue",
                       "pink", "darkgreen", "gold"]


class MouseSession:
    def __init__(self, age, session_id, param, nb_ms_by_frame, weight=None, spike_nums=None, spike_nums_dur=None):
        # should be a list of int
        self.age = age
        self.session_id = str(session_id)
        self.nb_ms_by_frame = nb_ms_by_frame
        self.description = f"P{self.age}_{self.session_id}"
        self.spike_struct = HNESpikeStructure(mouse_session=self, spike_nums=spike_nums, spike_nums_dur=spike_nums_dur)
        # spike_nums represents the onsets of the neuron spikes
        self.traces = None
        self.coord = None
        self.activity_threshold = None
        self.param = param
        self.weight = weight

    def set_inter_neurons(self, inter_neurons):
        self.spike_struct.inter_neurons = np.array(inter_neurons).astype(int)

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
        if "spike_durations" in variables_mapping:
            self.spike_struct.set_spike_durations(data[variables_mapping["spike_durations"]])
        if "spike_amplitudes" in variables_mapping:
            self.spike_struct.set_spike_amplitudes(data[variables_mapping["spike_amplitudes"]])

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

        # list of size n_cells, each list is array representing the duration (in frames) of each spike of the cell
        self.spike_durations = None
        self.inter_neurons = None
        # list of size n_cells, each list is array representing the amplitude of each spike of the cell
        self.spike_amplitudes = None

    def set_spike_durations(self, spike_durations_array):
        # print(f"np.shape(spike_durations_array) {np.shape(spike_durations_array)}")
        self.spike_durations = []
        avg_spike_duration_by_cell = np.zeros(len(spike_durations_array))
        for cell_id, spikes_d in enumerate(spike_durations_array):
            self.spike_durations.append(spikes_d[spikes_d > 0])
            if len(self.spike_durations[-1]) > 0:
                avg_spike_duration_by_cell[cell_id] = np.mean(self.spike_durations[-1])
            else:
                avg_spike_duration_by_cell[cell_id] = 0

        # inter_neurons can be manually added
        if self.inter_neurons is not None:
            return
        # otherwise, we select them throught a spike duration treshold

        # cells_ordered_by_duration = np.argsort(avg_spike_duration_by_cell)
        # avg_spike_duration_by_cell_ordered = np.sort(avg_spike_duration_by_cell)
        # n_cells = len(avg_spike_duration_by_cell)
        # interneurons_threshold_99 = avg_spike_duration_by_cell_ordered[int(n_cells * (1 - 0.01)]
        # interneurons_threshold_99_indices = cells_ordered_by_duration[int(n_cells * (1 - 0.01)]
        #

        # if self.mouse_session.session_id == "18_10_23_a001":
        #     for cell_id, avg in enumerate(avg_spike_duration_by_cell):
        #         print(f"cell_id {cell_id}, avg {avg}")

        interneurons_threshold_95 = np.percentile(avg_spike_duration_by_cell, 95)
        interneurons_threshold_99 = np.percentile(avg_spike_duration_by_cell, 99)

        self.inter_neurons = np.where(avg_spike_duration_by_cell >= interneurons_threshold_99)[0]

        print(f"{self.mouse_session.description}")
        inter_neurons_95 = np.where(avg_spike_duration_by_cell >= interneurons_threshold_95)[0]
        print(f"interneurons 95: {inter_neurons_95}")
        print(f"durations: {np.round(avg_spike_duration_by_cell[inter_neurons_95], 2)}")
        print("")

        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                gridspec_kw={'height_ratios': [1]},
                                figsize=(12, 12))
        ax1.set_facecolor("black")
        distribution = avg_spike_duration_by_cell
        bins = int(np.sqrt(len(distribution)))
        weights = (np.ones_like(distribution) / (len(distribution))) * 100
        hist_plt, edges_plt, patches_plt = plt.hist(distribution, bins=bins,
                                                    facecolor="blue",
                                                    edgecolor="white",
                                                    weights=weights, log=False)

        plt.scatter(x=interneurons_threshold_99, y=20, marker="*",
                    color=["white"], s=150, zorder=20)

        plt.title(f"{self.mouse_session.description}")
        # plt.legend()
        plt.show()
        plt.close()

    def set_spike_amplitudes(self, spike_amplitudes_array):
        self.spike_amplitudes = []
        for spikes_d in spike_amplitudes_array:
            self.spike_amplitudes.append(spikes_d[spikes_d > 0])

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


def sort_it_and_plot_it(spike_nums, param,
                        sliding_window_duration, activity_threshold, title_option="",
                        sce_times_bool=None,
                        spike_train_format=False,
                        debug_mode=False,
                        plot_all_best_seq_by_cell=False,
                        use_only_uniformity_method=False,
                        use_loss_score_to_keep_the_best_from_tree=
                        False
                        ):
    if spike_train_format:
        return
    # if sce_times_bool is not None, then we don't take in consideration SCE_time to do the pair-wise correlation
    result = order_spike_nums_by_seq(spike_nums,
                                     param, sce_times_bool=sce_times_bool,
                                     debug_mode=debug_mode,
                                     use_only_uniformity_method=use_only_uniformity_method,
                                     use_loss_score_to_keep_the_best_from_tree=
                                     use_loss_score_to_keep_the_best_from_tree)
    seq_dict_tmp, best_seq, all_best_seq = result

    if plot_all_best_seq_by_cell:
        for cell, each_best_seq in enumerate(all_best_seq):
            spike_nums_ordered = np.copy(spike_nums[each_best_seq, :])

            new_labels = np.arange(len(spike_nums))
            new_labels = new_labels[best_seq]
            loss_score = loss_function_with_sliding_window(spike_nums=spike_nums_ordered[::-1, :],
                                                           time_inter_seq=param.time_inter_seq,
                                                           min_duration_intra_seq=param.min_duration_intra_seq,
                                                           spike_train_mode=False,
                                                           debug_mode=True
                                                           )
            print(f'Cell {cell} loss_score ordered: {np.round(loss_score, 4)}')
            # saving the ordered spike_nums
            # micro_wires_ordered = micro_wires[best_seq]
            # np.savez(f'{param.path_results}/{channels_selection}_spike_nums_ordered_{patient_id}.npz',
            #          spike_nums_ordered=spike_nums_ordered, micro_wires_ordered=micro_wires_ordered)

            plot_spikes_raster(spike_nums=spike_nums_ordered, param=param,
                               title=f"cell {cell} raster plot ordered {title_option}",
                               spike_train_format=False,
                               file_name=f"cell_{cell}_spike_nums_ordered_{title_option}",
                               y_ticks_labels=new_labels,
                               y_ticks_labels_size=2,
                               save_raster=True,
                               show_raster=False,
                               sliding_window_duration=sliding_window_duration,
                               show_sum_spikes_as_percentage=True,
                               plot_with_amplitude=False,
                               activity_threshold=activity_threshold,
                               save_formats="pdf")
    else:
        spike_nums_ordered = np.copy(spike_nums[best_seq, :])

        new_labels = np.arange(len(spike_nums))
        new_labels = new_labels[best_seq]

        plot_spikes_raster(spike_nums=spike_nums_ordered, param=param,
                           title=f"raster plot ordered {title_option}",
                           spike_train_format=False,
                           file_name=f"spike_nums_ordered_{title_option}",
                           y_ticks_labels=new_labels,
                           y_ticks_labels_size=2,
                           save_raster=True,
                           show_raster=False,
                           sliding_window_duration=sliding_window_duration,
                           show_sum_spikes_as_percentage=True,
                           plot_with_amplitude=False,
                           activity_threshold=activity_threshold,
                           save_formats="pdf")

    #### test for coloring sequences
    spike_nums_ordered = np.copy(spike_nums[best_seq, :])
    print(f"starting finding sequences in orderered spike nums")
    seq_dict = find_sequences_in_ordered_spike_nums(spike_nums=spike_nums_ordered, param=param)
    print(f"Sequences in orderered spike nums found")
    # if debug_mode:
    #     print(f"best_seq {best_seq}")
    # if seq_dict_tmp is not None:
    #     if debug_mode:
    #         for key, value in seq_dict_tmp.items():
    #             print(f"seq: {key}, rep: {len(value)}")
    #
    #     best_seq_mapping_index = dict()
    #     for i, cell in enumerate(best_seq):
    #         best_seq_mapping_index[cell] = i
    #     # we need to replace the index by the corresponding one in best_seq
    #     seq_dict = dict()
    #     for key, value in seq_dict_tmp.items():
    #         new_key = []
    #         for cell in key:
    #             new_key.append(best_seq_mapping_index[cell])
    #         # checking if the list of cell is in the same order in best_seq
    #         # if the diff is only composed of one, this means all indices are following each other
    #         in_order = len(np.where(np.diff(new_key) != 1)[0]) == 0
    #         if in_order:
    #             print(f"in_order {new_key}")
    #             seq_dict[tuple(new_key)] = value
    #
    #     seq_colors = dict()
    #     len_seq = len(seq_dict)
    #     if debug_mode:
    #         print(f"nb seq to colors: {len_seq}")
    #     for index, key in enumerate(seq_dict.keys()):
    #         seq_colors[key] = cm.nipy_spectral(float(index + 1) / (len_seq + 1))
    #         if debug_mode:
    #             print(f"color {seq_colors[key]}, len(seq) {len(key)}")
    # else:
    #     seq_dict = None
    #     seq_colors = None
    # ordered_spike_nums = ordered_spike_data
    # spike_struct.ordered_spike_data = \
    #     trains_module.from_spike_nums_to_spike_trains(spike_struct.ordered_spike_data)
    new_labels = np.arange(len(spike_nums))
    new_labels = new_labels[best_seq]
    loss_score = loss_function_with_sliding_window(spike_nums=spike_nums_ordered,
                                                   time_inter_seq=param.time_inter_seq,
                                                   min_duration_intra_seq=param.min_duration_intra_seq,
                                                   spike_train_mode=False,
                                                   debug_mode=True
                                                   )
    print(f'total loss_score ordered: {np.round(loss_score, 4)}')
    # saving the ordered spike_nums
    # micro_wires_ordered = micro_wires[best_seq]
    # np.savez(f'{param.path_results}/{channels_selection}_spike_nums_ordered_{patient_id}.npz',
    #          spike_nums_ordered=spike_nums_ordered, micro_wires_ordered=micro_wires_ordered)

    colors_for_seq_list = ["blue", "red", "limegreen", "grey", "orange", "cornflowerblue", "yellow", "seagreen",
                           "magenta"]
    plot_spikes_raster(spike_nums=spike_nums_ordered, param=param,
                       title=f"raster plot ordered {title_option}",
                       spike_train_format=False,
                       file_name=f"spike_nums_ordered_seq_{title_option}",
                       y_ticks_labels=new_labels,
                       y_ticks_labels_size=2,
                       save_raster=True,
                       show_raster=False,
                       sliding_window_duration=sliding_window_duration,
                       show_sum_spikes_as_percentage=True,
                       plot_with_amplitude=False,
                       activity_threshold=activity_threshold,
                       save_formats="pdf",
                       seq_times_to_color_dict=seq_dict,
                       link_seq_color=colors_for_seq_list,
                       link_seq_line_width=0.8,
                       link_seq_alpha=0.9,
                       min_len_links_seq=3)
    # seq_colors=seq_colors)

    return best_seq, seq_dict


def use_new_pattern_package(spike_nums, param, activity_threshold, sliding_window_duration,
                            mouse_id, n_surrogate=2, extra_file_name="", debug_mode=False, without_raw_plot=True,
                            sce_times_bool=None, use_uniformity_method=False,
                            use_only_uniformity_method=False,
                            use_loss_score_to_keep_the_best_from_tree=
                            False):
    # around 250 ms
    # param.time_inter_seq
    # param.min_duration_intra_seq
    # -(10 ** (6 - decrease_factor)) // 40
    # a sequence should be composed of at least one third of the neurons
    # param.min_len_seq = len(spike_nums_struct.spike_data) // 4
    # param.min_len_seq = 5
    # param.error_rate = param.min_len_seq // 4

    labels = np.arange(len(spike_nums))

    if not without_raw_plot:
        plot_spikes_raster(spike_nums=spike_nums, param=param,
                           spike_train_format=False,
                           title=f"raster plot {mouse_id}",
                           file_name=f"raw_spike_nums_{mouse_id}{extra_file_name}",
                           y_ticks_labels=labels,
                           y_ticks_labels_size=4,
                           save_raster=True,
                           show_raster=False,
                           plot_with_amplitude=False,
                           activity_threshold=activity_threshold,
                           # 500 ms window
                           sliding_window_duration=sliding_window_duration,
                           show_sum_spikes_as_percentage=True,
                           spike_shape="|",
                           spike_shape_size=1,
                           save_formats="pdf")
    # continue

    # 2128885
    loss_score = loss_function_with_sliding_window(spike_nums=spike_nums,
                                                   time_inter_seq=param.time_inter_seq,
                                                   spike_train_mode=False,
                                                   min_duration_intra_seq=param.min_duration_intra_seq,
                                                   debug_mode=debug_mode)

    print(f'raw loss_score: {np.round(loss_score, 4)}')

    # spike_struct.spike_data = trains_module.from_spike_trains_to_spike_nums(spike_struct.spike_data)

    best_seq, seq_dict_real_data = sort_it_and_plot_it(spike_nums=spike_nums, param=param,
                                             sliding_window_duration=sliding_window_duration,
                                             activity_threshold=activity_threshold,
                                             title_option=f"{mouse_id}{extra_file_name}",
                                             spike_train_format=False,
                                             debug_mode=debug_mode,
                                             sce_times_bool=sce_times_bool,
                                             use_only_uniformity_method=use_only_uniformity_method,
                                        use_loss_score_to_keep_the_best_from_tree=
                                        use_loss_score_to_keep_the_best_from_tree)

    nb_cells = len(spike_nums)

    print("#### REAL DATA ####")
    print(f"best_seq {best_seq}")
    real_data_result_for_stat = SortedDict()
    neurons_sorted_real_data = np.zeros(nb_cells, dtype="uint16")
    if seq_dict_real_data is not None:
        for key, value in seq_dict_real_data.items():
            print(f"len: {len(key)}, seq: {key}, rep: {len(value)}")
            if len(key) not in real_data_result_for_stat:
                real_data_result_for_stat[len(key)] = []
            real_data_result_for_stat[len(key)].append(len(value))
            for cell in key:
                if neurons_sorted_real_data[cell] == 0:
                    neurons_sorted_real_data[cell] = 1

    n_times = len(spike_nums[0, :])

    print("#### SURROGATE DATA ####")
    # n_surrogate = 2
    surrogate_data_result_for_stat = SortedDict()
    neurons_sorted_surrogate_data = np.zeros(nb_cells, dtype="uint16")
    for surrogate_number in np.arange(n_surrogate):
        copy_spike_nums = np.copy(spike_nums)
        for n, neuron_spikes in enumerate(copy_spike_nums):
            # roll the data to a random displace number
            copy_spike_nums[n, :] = np.roll(neuron_spikes, np.random.randint(1, n_times))
        tmp_spike_nums = copy_spike_nums

        best_seq, seq_dict_surrogate = sort_it_and_plot_it(spike_nums=tmp_spike_nums, param=param,
                                                 sliding_window_duration=sliding_window_duration,
                                                 activity_threshold=activity_threshold,
                                                 title_option=f"surrogate {mouse_id}{extra_file_name}",
                                                 spike_train_format=False,
                                                 debug_mode=False,
                                                 use_only_uniformity_method=use_only_uniformity_method,
                                                 use_loss_score_to_keep_the_best_from_tree=
                                                 use_loss_score_to_keep_the_best_from_tree
                                                 )

        print(f"best_seq {best_seq}")

        mask = np.zeros(nb_cells, dtype="bool")
        if seq_dict_surrogate is not None:
            for key, value in seq_dict_surrogate.items():
                print(f"len: {len(key)}, seq: {key}, rep: {len(value)}")
                if len(key) not in surrogate_data_result_for_stat:
                    surrogate_data_result_for_stat[len(key)] = []
                surrogate_data_result_for_stat[len(key)].append(len(value))
                for cell in key:
                    mask[cell] = True
            neurons_sorted_surrogate_data[mask] += 1
    # min_time, max_time = trains_module.get_range_train_list(spike_nums)
    # surrogate_data_set = create_surrogate_dataset(train_list=spike_nums, nsurrogate=n_surrogate,
    #                                               min_value=min_time, max_value=max_time)
    print("")
    print("")

    give_me_stat_on_sorting_seq_results(results_dict=real_data_result_for_stat,
                                        neurons_sorted=neurons_sorted_real_data,
                                        title="%%%% DATA SET STAT %%%%%", param=param,
                                        results_dict_surrogate=surrogate_data_result_for_stat,
                                        neurons_sorted_surrogate=neurons_sorted_surrogate_data,
                                        extra_file_name=extra_file_name,
                                        n_surrogate=n_surrogate,
                                        use_sce_times_for_pattern_search=(sce_times_bool is not None),
                                        use_only_uniformity_method=use_only_uniformity_method,
                                        use_loss_score_to_keep_the_best_from_tree=
                                        use_loss_score_to_keep_the_best_from_tree
                                        )


# def give_me_stat_on_sorting_seq_results(results_dict, neurons_sorted, title, param,
#                                         use_sce_times_for_pattern_search,
#                                         results_dict_surrogate=None, neurons_sorted_surrogate=None,
#                                         extra_file_name=""):
#     """
#     Key will be the length of the sequence and value will be a list of int, representing the nb of rep
#     of the different lists
#     :param results_dict:
#     :return:
#     """
#     file_name = f'{param.path_results}/sorting_results{extra_file_name}_{param.time_str}.txt'
#     with open(file_name, "w", encoding='UTF-8') as file:
#         file.write(f"{title}" + '\n')
#         file.write("" + '\n')
#         file.write("Parameters" + '\n')
#         file.write("" + '\n')
#         file.write(f"error_rate {param.error_rate}" + '\n')
#         file.write(f"max_branches {param.max_branches}" + '\n')
#         file.write(f"time_inter_seq {param.time_inter_seq}" + '\n')
#         file.write(f"min_duration_intra_seq {param.min_duration_intra_seq}" + '\n')
#         file.write(f"min_len_seq {param.min_len_seq}" + '\n')
#         file.write(f"min_rep_nb {param.min_rep_nb}" + '\n')
#         file.write(f"use_sce_times_for_pattern_search {use_sce_times_for_pattern_search}" + '\n')
#
#         file.write("" + '\n')
#         min_len = 1000
#         max_len = 0
#         for key in results_dict.keys():
#             min_len = np.min((key, min_len))
#             max_len = np.max((key, max_len))
#         if results_dict_surrogate is not None:
#             for key in results_dict_surrogate.keys():
#                 min_len = np.min((key, min_len))
#                 max_len = np.max((key, max_len))
#
#         # key reprensents the length of a seq
#         for key in np.arange(min_len, max_len + 1):
#             nb_seq = None
#             nb_seq_surrogate = None
#             if key in results_dict:
#                 nb_seq = results_dict[key]
#             if key in results_dict_surrogate:
#                 nb_seq_surrogate = results_dict_surrogate[key]
#             str_to_write = ""
#             str_to_write += f"### Length {key}: \n"
#             real_data_in = False
#             if nb_seq is not None:
#                 real_data_in = True
#                 str_to_write += f"# Real data (x{len(nb_seq)}), length: mean {np.round(np.mean(nb_seq), 3)}"
#                 if np.std(nb_seq) > 0:
#                     str_to_write += f", std {np.round(np.std(nb_seq), 3)}"
#             if nb_seq_surrogate is not None:
#                 if real_data_in:
#                     str_to_write += f"\n"
#                 str_to_write += f"# Surrogate (x{len(nb_seq_surrogate)}), length: " \
#                                 f"mean {np.round(np.mean(nb_seq_surrogate), 3)}"
#                 if np.std(nb_seq_surrogate) > 0:
#                     str_to_write += f", std {np.round(np.std(nb_seq_surrogate), 3)}"
#             else:
#                 if not real_data_in:
#                     continue
#             str_to_write += '\n'
#             file.write(f"{str_to_write}")
#         file.write("" + '\n')
#         file.write("///// Neurons sorted /////" + '\n')
#         file.write("" + '\n')
#
#         for index in np.arange(len(neurons_sorted)):
#             go_for = False
#             if neurons_sorted_surrogate is not None:
#                 if neurons_sorted_surrogate[index] == 0:
#                     pass
#                 else:
#                     go_for = True
#             if (not go_for) and neurons_sorted[index] == 0:
#                 continue
#             str_to_write = f"Neuron {index}, x "
#             if neurons_sorted_surrogate is not None:
#                 str_to_write += f"{neurons_sorted_surrogate[index]} / "
#             str_to_write += f"{neurons_sorted[index]}"
#             if neurons_sorted_surrogate is not None:
#                 str_to_write += " (surrogate / real data)"
#             str_to_write += '\n'
#             file.write(f"{str_to_write}")


def save_stat_by_age(ratio_spikes_events_by_age, ratio_spikes_total_events_by_age, mouse_sessions,
                     interneurons_indices_by_age, param):
    file_name = f'{param.path_results}/stat_for_all_mice_{param.time_str}.txt'
    round_factor = 1

    with open(file_name, "w", encoding='UTF-8') as file:
        file.write(f"Stat by age" + '\n')
        file.write("" + '\n')

        for age in ratio_spikes_events_by_age.keys():
            ratio_spikes_events = np.array(ratio_spikes_events_by_age[age])
            inter_neurons = np.array(interneurons_indices_by_age[age])
            non_inter_neurons = np.setdiff1d(np.arange(len(ratio_spikes_events)), inter_neurons)
            file.write('\n')
            file.write('\n')
            file.write("#" * 50 + '\n')
            file.write('\n')
            file.write(f"p{age}")
            for ms in mouse_sessions:
                if ms.age == age:
                    file.write(f"{ms.description} + '\n'")
            file.write('\n')
            file.write("Ratio spikes in events vs all spikes for each non-interneuron cells:" + '\n')
            file.write(f"mean: {np.round(np.mean(ratio_spikes_events[non_inter_neurons]), round_factor)}, "
                       f"std: {np.round(np.std(ratio_spikes_events[non_inter_neurons]), round_factor)}" + '\n')
            file.write(f"median: {np.round(np.median(ratio_spikes_events[non_inter_neurons]), round_factor)}, "
                       f"5th percentile: {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 5), round_factor)}, "
                       f"25th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 25), round_factor)}, "
                       f"75th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 75), round_factor)}, "
                       f"95th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 95), round_factor)}" + '\n')
            file.write('\n')
            file.write('\n')

            if len(inter_neurons) == 0:
                file.write("No interneurons" + '\n')
            else:
                file.write("Ratio spikes in events vs all spikes for each interneuron cells:" + '\n')
                file.write(f"mean: {np.round(np.mean(ratio_spikes_events[inter_neurons]), round_factor)}, "
                           f"std: {np.round(np.std(ratio_spikes_events[inter_neurons]), round_factor)}" + '\n')
                file.write(f"median: {np.round(np.median(ratio_spikes_events[inter_neurons]), round_factor)}, "
                           f"5th percentile: {np.round(np.percentile(ratio_spikes_events[inter_neurons], 5), round_factor)}, "
                           f"25th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 25), round_factor)}, "
                           f"75th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 75), round_factor)}, "
                           f"95th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 95), round_factor)}" + '\n')
            file.write('\n')
            file.write('\n')

            file.write('\n')
            file.write('\n')
            ratio_spikes_events = np.array(ratio_spikes_total_events_by_age[age])
            file.write("Ratio spikes in events vs all events for non-interneuron cells:" + '\n')
            file.write(f"mean: {np.round(np.mean(ratio_spikes_events[non_inter_neurons]), round_factor)}, "
                       f"std: {np.round(np.std(ratio_spikes_events[non_inter_neurons]), round_factor)}" + '\n')
            file.write(f"median: {np.round(np.median(ratio_spikes_events[non_inter_neurons]), round_factor)}, "
                       f"5th percentile: {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 5), round_factor)}, "
                       f"25th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 25), round_factor)}, "
                       f"75th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 75), round_factor)}, "
                       f"95th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 95), round_factor)}" + '\n')
            file.write('\n')
            file.write('\n')
            if len(inter_neurons) == 0:
                file.write("No interneurons" + '\n')
            else:
                file.write("Ratio spikes in events vs all events for interneuron cells:" + '\n')
                file.write(f"mean: {np.round(np.mean(ratio_spikes_events[inter_neurons]), round_factor)}, "
                           f"std: {np.round(np.std(ratio_spikes_events[inter_neurons]), round_factor)}" + '\n')
                file.write(f"median: {np.round(np.median(ratio_spikes_events[inter_neurons]), round_factor)}, "
                           f"5th percentile: {np.round(np.percentile(ratio_spikes_events[inter_neurons], 5), round_factor)}, "
                           f"25th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 25), round_factor)}, "
                           f"75th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 75), round_factor)}, "
                           f"95th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 95), round_factor)}" + '\n')


def save_stat_sce_detection_methods(spike_nums_to_use, activity_threshold, ms,
                                    SCE_times, param, sliding_window_duration,
                                    perc_threshold, use_raster_dur,
                                    keep_max_each_surrogate, ratio_spikes_events,
                                    ratio_spikes_total_events,
                                    n_surrogate_activity_threshold):
    round_factor = 1
    raster_option = "raster_dur" if use_raster_dur else "onsets"
    technique_details = " max of each surrogate " if keep_max_each_surrogate else ""
    technique_details_file = "max_each_surrogate" if keep_max_each_surrogate else ""
    file_name = f'{param.path_results}/{ms.description}_stat_{raster_option}_{perc_threshold}_perc_' \
                f'{technique_details_file}_{param.time_str}.txt'

    n_sce = len(SCE_times)
    n_cells = len(spike_nums_to_use)
    with open(file_name, "w", encoding='UTF-8') as file:
        file.write(
            f"Stat {ms.description} using {raster_option}{technique_details} {perc_threshold} percentile " + '\n')
        file.write("" + '\n')
        file.write(f"{n_cells} cells, {n_sce} events" + '\n')
        file.write(f"Event participation threshold {activity_threshold} / "
                   f"{np.round((activity_threshold*100)/n_cells, 2)}%, "
                   f"{perc_threshold} percentile, "
                   f"{n_surrogate_activity_threshold} surrogates" + '\n')
        file.write(f"Sliding window duration {sliding_window_duration}" + '\n')
        file.write("" + '\n')
        file.write("" + '\n')

        inter_neurons = ms.spike_struct.inter_neurons
        non_inter_neurons = np.setdiff1d(np.arange(n_cells), inter_neurons)
        # ratio_spikes_events
        file.write("Ratio spikes in events vs all spikes for non interneuron cells:" + '\n')
        file.write(f"mean: {np.round(np.mean(ratio_spikes_events[non_inter_neurons]), round_factor)}, "
                   f"std: {np.round(np.std(ratio_spikes_events[non_inter_neurons]), round_factor)}" + '\n')
        file.write(f"median: {np.round(np.median(ratio_spikes_events[non_inter_neurons]), round_factor)}, "
                   f"5th percentile: {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 5), round_factor)}, "
                   f"25th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 25), round_factor)}, "
                   f"75th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 75), round_factor)}, "
                   f"95th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 95), round_factor)}" + '\n')
        file.write('\n')
        file.write('\n')

        if len(inter_neurons) == 0:
            file.write("No interneurons" + '\n')
        else:
            file.write("Ratio spikes in events vs all spikes for interneurons:" + '\n')
            file.write(f"mean: {np.round(np.mean(ratio_spikes_events[inter_neurons]), round_factor)}, "
                       f"std: {np.round(np.std(ratio_spikes_events[inter_neurons]), round_factor)}" + '\n')
            file.write(f"median: {np.round(np.median(ratio_spikes_events[inter_neurons]), round_factor)}, "
                       f"5th percentile: {np.round(np.percentile(ratio_spikes_events[inter_neurons], 5), round_factor)}, "
                       f"25th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 25), round_factor)}, "
                       f"75th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 75), round_factor)}, "
                       f"95th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 95), round_factor)}"
                       + '\n')
        file.write('\n')
        file.write('\n')
        file.write("#" * 50 + '\n')
        file.write('\n')
        file.write('\n')

        # ratio_spikes_events on total events
        file.write("Ratio spikes in events vs all events for non interneurons cell:" + '\n')
        file.write(f"mean: {np.round(np.mean(ratio_spikes_total_events[non_inter_neurons]), round_factor)}, "
                   f"std: {np.round(np.std(ratio_spikes_total_events[non_inter_neurons]), round_factor)}" + '\n')
        file.write(f"median: {np.round(np.median(ratio_spikes_total_events[non_inter_neurons]), round_factor)}, "
                   f"5th percentile: {np.round(np.percentile(ratio_spikes_total_events[non_inter_neurons], 5), round_factor)}, "
                   f"25th percentile {np.round(np.percentile(ratio_spikes_total_events[non_inter_neurons], 25), round_factor)}, "
                   f"75th percentile {np.round(np.percentile(ratio_spikes_total_events[non_inter_neurons], 75), round_factor)}, "
                   f"95th percentile {np.round(np.percentile(ratio_spikes_total_events[non_inter_neurons], 95), round_factor)}" + '\n')
        file.write('\n')
        file.write('\n')
        inter_neurons = ms.spike_struct.inter_neurons
        if len(inter_neurons) == 0:
            file.write("No interneurons" + '\n')
        else:
            file.write("Ratio spikes in events vs all events for interneurons:" + '\n')
            file.write(f"mean: {np.round(np.mean(ratio_spikes_total_events[inter_neurons]), round_factor)}, "
                       f"std: {np.round(np.std(ratio_spikes_total_events[inter_neurons]), round_factor)}" + '\n')
            file.write(f"median: {np.round(np.median(ratio_spikes_total_events[inter_neurons]), round_factor)}, "
                       f"5th percentile: {np.round(np.percentile(ratio_spikes_total_events[inter_neurons], 5), round_factor)}, "
                       f"25th percentile {np.round(np.percentile(ratio_spikes_total_events[inter_neurons], 25), round_factor)}, "
                       f"75th percentile {np.round(np.percentile(ratio_spikes_total_events[inter_neurons], 75), round_factor)}, "
                       f"95th percentile {np.round(np.percentile(ratio_spikes_total_events[inter_neurons], 95), round_factor)}"
                       + '\n')
        file.write('\n')
        file.write('\n')
        file.write("#" * 50 + '\n')
        file.write('\n')
        file.write('\n')

        if n_sce > 0:
            file.write("All events (SCEs) stat:" + '\n')
            file.write('\n')
            # duration in frames
            duration_values = np.zeros(n_sce, dtype="uint16")
            max_activity_values = np.zeros(n_sce)
            mean_activity_values = np.zeros(n_sce)
            overall_activity_values = np.zeros(n_sce)
            # Collecting data from each SCE
            for sce_id in np.arange(len(SCE_times)):
                result = give_stat_one_sce(sce_id=sce_id,
                                           spike_nums_to_use=spike_nums_to_use,
                                           SCE_times=SCE_times,
                                           sliding_window_duration=sliding_window_duration)
                duration_values[sce_id], max_activity_values[sce_id], mean_activity_values[sce_id], \
                overall_activity_values[sce_id] = result

            # percentage as the total number of cells
            max_activity_values = (max_activity_values / n_cells) * 100
            mean_activity_values = (mean_activity_values / n_cells) * 100
            overall_activity_values = (overall_activity_values / n_cells) * 100

            # # normalizing by duration
            # max_activity_values = max_activity_values / duration_values
            # mean_activity_values = mean_activity_values / duration_values
            # overall_activity_values = overall_activity_values / duration_values
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

            doing_each_sce_stat = False

            if doing_each_sce_stat:
                file.write('\n')
                file.write('\n')
                file.write("#" * 50 + '\n')
                file.write('\n')
                file.write('\n')
                # for each SCE
                for sce_id in np.arange(len(SCE_times)):
                    file.write(f"SCE {sce_id}" + '\n')
                    file.write(f"Duration_in_frames {duration_values[sce_id]}" + '\n')
                    file.write(
                        f"Overall participation {np.round(overall_activity_values[sce_id], round_factor)}" + '\n')
                    file.write(f"Max participation {np.round(max_activity_values[sce_id], round_factor)}" + '\n')
                    file.write(f"Mean participation {np.round(mean_activity_values[sce_id], round_factor)}" + '\n')

                    file.write('\n')
                    file.write('\n')

                file.write('\n')
                file.write('\n')
                file.write("#" * 50 + '\n')
                file.write('\n')

        file.write("ISI stats" '\n')
        file.write('\n')
        file.write('\n')

        cells_isi = tools_misc.get_isi(spike_data=ms.spike_struct.spike_nums, spike_trains_format=False)

        non_interneurons_isi = []
        interneurons_isi = []
        for cell_index in np.arange(n_cells):
            if cell_index in inter_neurons:
                interneurons_isi.extend(list(cells_isi[cell_index]))
            else:
                non_interneurons_isi.extend(list(cells_isi[cell_index]))

        file.write("ISI non interneurons:" + '\n')
        file.write(f"mean: {np.round(np.mean(non_interneurons_isi), round_factor)}, "
                   f"std: {np.round(np.std(non_interneurons_isi), round_factor)}" + '\n')
        file.write(f"median: {np.round(np.median(non_interneurons_isi), round_factor)}, "
                   f"5th percentile: {np.round(np.percentile(non_interneurons_isi, 5), round_factor)}, "
                   f"25th percentile {np.round(np.percentile(non_interneurons_isi, 25), round_factor)}, "
                   f"75th percentile {np.round(np.percentile(non_interneurons_isi, 75), round_factor)}, "
                   f"95th percentile {np.round(np.percentile(non_interneurons_isi, 95), round_factor)}" + '\n')
        if len(inter_neurons) == 0:
            file.write("No interneurons:" + '\n')
        else:
            file.write("ISI interneurons:" + '\n')
            file.write(f"mean: {np.round(np.mean(interneurons_isi), round_factor)}, "
                       f"std: {np.round(np.std(interneurons_isi), round_factor)}" + '\n')
            file.write(f"median: {np.round(np.median(interneurons_isi), round_factor)}, "
                       f"5th percentile: {np.round(np.percentile(interneurons_isi, 5), round_factor)}, "
                       f"25th percentile {np.round(np.percentile(interneurons_isi, 25), round_factor)}, "
                       f"75th percentile {np.round(np.percentile(interneurons_isi, 75), round_factor)}, "
                       f"95th percentile {np.round(np.percentile(interneurons_isi, 95), round_factor)}" + '\n')

        doing_each_isi_stat = False
        if doing_each_isi_stat:
            for cell_index in np.arange(len(ms.spike_struct.spike_nums)):
                file.write(f"cell {cell_index}" + '\n')
                file.write(f"median isi: {np.round(np.median(cells_isi[cell_index]), round_factor)}, " + '\n')
                file.write(f"mean isi {np.round(np.mean(cells_isi[cell_index]), round_factor)}" + '\n')
                file.write(f"std isi {np.round(np.std(cells_isi[cell_index]), round_factor)}" + '\n')

                file.write('\n')
                file.write('\n')

    return duration_values, max_activity_values, mean_activity_values, overall_activity_values


def plot_activity_duration_vs_age(mouse_sessions, ms_ages, duration_values_list,
                                  max_activity_values_list,
                                  mean_activity_values_list,
                                  overall_activity_values_list,
                                  param,
                                  save_formats="pdf"):
    duration_values_list_backup = duration_values_list
    overall_activity_values_list_backup = overall_activity_values_list
    # grouping mouses from the same age
    duration_dict = dict()
    max_activity_dict = dict()
    mean_activity_dict = dict()
    overall_activity_dict = dict()
    for i, age in enumerate(ms_ages):
        if age not in duration_dict:
            duration_dict[age] = duration_values_list[i]
            max_activity_dict[age] = max_activity_values_list[i]
            mean_activity_dict[age] = mean_activity_values_list[i]
            overall_activity_dict[age] = overall_activity_values_list[i]
        else:
            duration_dict[age] = np.concatenate((duration_dict[age], duration_values_list[i]))
            max_activity_dict[age] = np.concatenate((max_activity_dict[age], max_activity_values_list[i]))
            mean_activity_dict[age] = np.concatenate((mean_activity_dict[age], mean_activity_values_list[i]))
            overall_activity_dict[age] = np.concatenate((overall_activity_dict[age], overall_activity_values_list[i]))

    ms_ages = np.unique(ms_ages)
    duration_values_list = []
    max_activity_values_list = []
    mean_activity_values_list = []
    overall_activity_values_list = []
    for age in ms_ages:
        duration_values_list.append(duration_dict[age])
        max_activity_values_list.append(max_activity_dict[age])
        mean_activity_values_list.append(mean_activity_dict[age])
        overall_activity_values_list.append(overall_activity_dict[age])

    # normalizing by duration
    overall_activity_values_list_normalized = []
    for i, overall_activity_values in enumerate(overall_activity_values_list):
        overall_activity_values_list_normalized.append(overall_activity_values / duration_values_list[i])

    raw_y_datas = [duration_values_list, max_activity_values_list, mean_activity_values_list,
                   overall_activity_values_list,
                   overall_activity_values_list_normalized]
    y_data_labels = ["Duration", "Max participation", "Average participation", "Overall participation",
                     "Overall participation normalized"]

    # TODO: merge data from the same age
    # TODO: Show data with stds
    for index_raw, raw_y_data in enumerate(raw_y_datas):
        fcts_to_apply = [np.median]
        fcts_descr = ["median"]
        for index_fct, fct_to_apply in enumerate(fcts_to_apply):
            # raw_y_data is a list of np.array, we need to apply a fct to each array so we keep only one value
            y_data = np.zeros(len(ms_ages))
            stds = np.zeros(len(ms_ages))
            high_percentile_participation = np.zeros(len(ms_ages))
            low_percentile_participation = np.zeros(len(ms_ages))
            for index, data in enumerate(raw_y_data):
                y_data[index] = fct_to_apply(data)
                stds[index] = np.std(data)
                high_percentile_participation[index] = np.percentile(data, 95)
                low_percentile_participation[index] = np.percentile(data, 5)

            fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                    gridspec_kw={'height_ratios': [1]},
                                    figsize=(12, 12))
            ax1.set_facecolor("black")
            # ax1.scatter(ms_ages, y_data, color="blue", marker="o",
            #             s=12, zorder=20)
            # ax1.errorbar(ms_ages, y_data, yerr=stds, fmt='o', color="blue")
            # ax1.scatter(ms_ages, y_data, color="blue", marker="o",
            #             s=12, zorder=20)
            ax1.plot(ms_ages, y_data, color="blue")
            # stds = np.array(std_power_spectrum[freq_min_index:freq_max_index])
            ax1.fill_between(ms_ages, low_percentile_participation, high_percentile_participation,
                             alpha=0.5, facecolor="blue")

            ax1.set_ylabel(y_data_labels[index_raw] + f" ({fcts_descr[index_fct]})")
            ax1.set_xlabel("age")

            if isinstance(save_formats, str):
                save_formats = [save_formats]
            for save_format in save_formats:
                fig.savefig(f'{param.path_results}/{y_data_labels[index_raw]}_{fcts_descr[index_fct]}_vs_age'
                            f'_{param.time_str}.{save_format}',
                            format=f"{save_format}")

        plt.close()

    # then keeping only overall participation, non normalized
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(20, 20))
    ax1.set_facecolor("black")

    # markers = ['o', '*', 's', 'v', '<', '>', '^', 'x', '+', "."]  # d losange
    # colors = ["darkmagenta", "white", "saddlebrown", "blue", "red", "darkgrey", "chartreuse", "cornflowerblue",
    #           "pink", "darkgreen", "gold"]
    max_x = 0
    for index, age in enumerate(ms_ages):
        # color = cm.nipy_spectral(float(age-5) / (len(ms_ages)))
        # indices_rand used for jittering
        jitter_range_x = 0.5
        indices_rand = np.linspace(-jitter_range_x, jitter_range_x, len(duration_values_list[index]))
        jitter_range_y = 0.3
        indices_rand_y = np.linspace(-jitter_range_y, jitter_range_y, len(duration_values_list[index]))
        np.random.shuffle(indices_rand)
        np.random.shuffle(indices_rand_y)
        max_x = np.max((max_x, np.max(duration_values_list[index])))
        ax1.scatter(duration_values_list[index] + indices_rand,
                    overall_activity_values_list[index] + indices_rand_y,
                    color=param.colors[index % (len(param.colors))],
                    marker=param.markers[index % (len(param.markers))],
                    s=80, alpha=0.8, label=f"P{age}", edgecolors='none')

    ax1.set_xscale("log")
    ax1.set_xlim(0.4, max_x + 5)
    ax1.set_yscale("log")
    ax1.legend()
    ax1.set_ylabel("Participation (%)")
    ax1.set_xlabel("Duration (frames)")

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/participation_vs_duration'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")
    plt.close()

    ###################
    ###################
    # then the same with band of colors
    ###################

    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(20, 20))
    ax1.set_facecolor("black")

    # colors = ["darkmagenta", "white", "saddlebrown", "blue", "red", "darkgrey", "chartreuse", "cornflowerblue",
    #           "pink", "darkgreen", "gold"]
    max_x = 0
    # from 1 to 79, 15 bins
    ranges = np.logspace(0, 1.9, 15)
    for index, age in enumerate(ms_ages):
        # first we want to bin the duration values
        # gathering participation for each common duration
        duration_d = SortedDict()
        overall_activity_values = overall_activity_values_list[index]
        for i_duration, duration_value in enumerate(duration_values_list[index]):
            range_pos = bisect(ranges, duration_value) - 1
            range_value = ranges[range_pos]
            if range_value not in duration_d:
                duration_d[range_value] = [overall_activity_values[i_duration]]
            else:
                duration_d[range_value].append(overall_activity_values[i_duration])

        average_participation = np.zeros(len(duration_d))
        stds_participation = np.zeros(len(duration_d))
        high_percentile_participation = np.zeros(len(duration_d))
        low_percentile_participation = np.zeros(len(duration_d))
        durations = np.zeros(len(duration_d))

        for i, duration in enumerate(duration_d.keys()):
            durations[i] = duration
            participations = duration_d[duration]
            average_participation[i] = np.median(participations)
            stds_participation[i] = np.std(participations)
            high_percentile_participation[i] = np.percentile(participations, 75)
            low_percentile_participation[i] = np.percentile(participations, 25)

        max_x = np.max((max_x, np.max(duration_values_list[index])))

        ax1.plot(durations, average_participation, color=param.colors[index % (len(param.colors))], label=f"P{age}")
        # ax1.fill_between(durations, average_participation - stds_participation, average_participation + stds_participation,
        #                  alpha=0.5, facecolor=colors[index % (len(colors))])

        ax1.fill_between(durations, low_percentile_participation, high_percentile_participation,
                         alpha=0.5, facecolor=param.colors[index % (len(param.colors))])

    ax1.set_xscale("log")
    ax1.set_xlim(0.4, max_x + 5)
    ax1.set_yscale("log")
    ax1.legend()
    ax1.set_ylabel("Participation (%)")
    ax1.set_xlabel("Duration (frames)")

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/participation_vs_duration_bands'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")
    plt.close()

    # ##################################################################################
    # ########################## WEIGHT vs normalized participation ##################
    # ################################################################################

    # grouping mouses from the same weight
    duration_dict = dict()
    overall_activity_dict = dict()
    duration_values_list = duration_values_list_backup
    overall_activity_values_list = overall_activity_values_list_backup
    ms_weights = []
    for ms in mouse_sessions:
        ms_weights.append(ms.weight)

    for i, weight in enumerate(ms_weights):
        if weight is None:
            continue
        if weight not in duration_dict:
            duration_dict[weight] = duration_values_list[i]
            overall_activity_dict[weight] = overall_activity_values_list[i]
        else:
            duration_dict[weight] = np.concatenate((duration_dict[weight], duration_values_list[i]))
            overall_activity_dict[weight] = np.concatenate((overall_activity_dict[weight],
                                                            overall_activity_values_list[i]))
    filter_ms_weights = []
    for weight in ms_weights:
        if weight is None:
            continue
        filter_ms_weights.append(weight)
    ms_weights = filter_ms_weights
    ms_weights = np.unique(ms_weights)
    duration_values_list = []
    overall_activity_values_list = []
    for weight in ms_weights:
        duration_values_list.append(duration_dict[weight])
        overall_activity_values_list.append(overall_activity_dict[weight])

    # normalizing by duration
    overall_activity_values_list_normalized = []
    for i, overall_activity_values in enumerate(overall_activity_values_list):
        overall_activity_values_list_normalized.append(overall_activity_values / duration_values_list[i])

    y_data = np.zeros(len(ms_weights))
    stds = np.zeros(len(ms_weights))
    high_percentile_participation = np.zeros(len(ms_weights))
    low_percentile_participation = np.zeros(len(ms_weights))
    for index, data in enumerate(overall_activity_values_list_normalized):
        y_data[index] = np.median(data)
        stds[index] = np.std(data)
        high_percentile_participation[index] = np.percentile(data, 95)
        low_percentile_participation[index] = np.percentile(data, 5)

    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(12, 12))
    ax1.set_facecolor("black")
    ax1.plot(ms_weights, y_data, color="blue")
    # stds = np.array(std_power_spectrum[freq_min_index:freq_max_index])
    ax1.fill_between(ms_weights, low_percentile_participation, high_percentile_participation,
                     alpha=0.5, facecolor="blue")

    ax1.set_ylabel("Overall participation normalized")
    ax1.set_xlabel("weight (grams)")

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/Overall participation normalized_vs_weight'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")

    plt.close()


def plot_duration_spikes_by_age(mouse_sessions, ms_ages,
                                duration_spikes_by_age, param, save_formats="pdf"):
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(20, 20))
    ax1.set_facecolor("black")

    y_data = np.zeros(len(ms_ages))
    # stds = np.zeros(len(ms_ages))
    high_percentile_participation = np.zeros(len(ms_ages))
    low_percentile_participation = np.zeros(len(ms_ages))
    for index, age in enumerate(ms_ages):
        y_data[index] = np.median(duration_spikes_by_age[age])
        # stds[index] = np.std(data)
        high_percentile_participation[index] = np.percentile(duration_spikes_by_age[age], 95)
        low_percentile_participation[index] = np.percentile(duration_spikes_by_age[age], 5)

    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(12, 12))
    ax1.set_facecolor("black")
    # ax1.scatter(ms_ages, y_data, color="blue", marker="o",
    #             s=12, zorder=20)
    # ax1.errorbar(ms_ages, y_data, yerr=stds, fmt='o', color="blue")
    # ax1.scatter(ms_ages, y_data, color="blue", marker="o",
    #             s=12, zorder=20)
    ax1.plot(ms_ages, y_data, color="blue")
    # stds = np.array(std_power_spectrum[freq_min_index:freq_max_index])
    ax1.fill_between(ms_ages, low_percentile_participation, high_percentile_participation,
                     alpha=0.5, facecolor="blue")

    ax1.set_ylabel("Spikes duration (frames)")
    ax1.set_xlabel("age")

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/duration_spikes_by_age'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")
    plt.close()


def plot_hist_ratio_spikes_events(ratio_spikes_events, description, values_to_scatter,
                                  labels, scatter_shapes, colors, param, xlabel="", save_formats="pdf"):
    distribution = np.array(ratio_spikes_events)
    hist_color = "blue"
    edge_color = "white"
    max_range = np.max(distribution)
    weights = (np.ones_like(distribution) / (len(distribution))) * 100

    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(12, 12))
    ax1.set_facecolor("black")
    bins = int(np.sqrt(len(distribution)))
    hist_plt, edges_plt, patches_plt = plt.hist(distribution, bins=bins, range=(0, 100),
                                                facecolor=hist_color,
                                                edgecolor=edge_color,
                                                weights=weights, log=False)

    scatter_bins = np.ones(len(values_to_scatter), dtype="int16")
    scatter_bins *= -1

    for i, edge in enumerate(edges_plt):
        # print(f"i {i}, edge {edge}")
        if i >= len(hist_plt):
            # means that scatter left are on the edge of the last bin
            scatter_bins[scatter_bins == -1] = i - 1
            break

        if len(values_to_scatter[values_to_scatter <= edge]) > 0:
            if (i + 1) < len(edges_plt):
                bool_list = values_to_scatter < edge  # edges_plt[i + 1]
                for i_bool, bool_value in enumerate(bool_list):
                    if bool_value:
                        if scatter_bins[i_bool] == -1:
                            new_i = max(0, i - 1)
                            scatter_bins[i_bool] = new_i
            else:
                bool_list = values_to_scatter < edge
                for i_bool, bool_value in enumerate(bool_list):
                    if bool_value:
                        if scatter_bins[i_bool] == -1:
                            scatter_bins[i_bool] = i

    decay = np.linspace(1.1, 1.15, len(values_to_scatter))
    for i, value_to_scatter in enumerate(values_to_scatter):
        if i < len(labels):
            plt.scatter(x=value_to_scatter, y=hist_plt[scatter_bins[i]] * decay[i], marker=scatter_shapes[i],
                        color=colors[i], s=60, zorder=20, label=labels[i])
        else:
            plt.scatter(x=value_to_scatter, y=hist_plt[scatter_bins[i]] * decay[i], marker=scatter_shapes[i],
                        color=colors[i], s=60, zorder=20)

    plt.xlim(0, 100)
    ax1.set_ylabel("Distribution (%)")
    ax1.set_xlabel(xlabel)
    xticks = np.arange(0, 110, 10)
    ax1.set_xticks(xticks)
    # sce clusters labels
    ax1.set_xticklabels(xticks)
    ax1.legend()

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/{description}'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")

    plt.close()


def box_plot_data_by_age(data_dict, title, filename, y_label, param, save_formats="pdf"):
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(12, 12))
    ax1.set_facecolor("black")

    colorfull = True
    labels = []
    data_list = []
    for age, data in data_dict.items():
        data_list.append(data)
        labels.append(age)
    bplot = plt.boxplot(data_list, patch_artist=colorfull,
                        labels=labels, sym='', zorder=1)  # whis=[5, 95], sym='+'
    # color=["b", "cornflowerblue"],
    # fill with colors

    # edge_color="silver"

    for element in ['boxes', 'whiskers', 'fliers', 'caps']:
        plt.setp(bplot[element], color="white")

    for element in ['means', 'medians']:
        plt.setp(bplot[element], color="silver")

    if colorfull:
        colors = param.colors[:len(data_dict)]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    # plt.xlim(0, 100)
    plt.title(title)
    ax1.set_ylabel(f"{y_label}")
    ax1.set_xlabel("age")
    xticks = np.arange(0, len(data_dict))
    ax1.set_xticks(xticks)
    # sce clusters labels
    ax1.set_xticklabels(labels)

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/{filename}'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")

    plt.close()


def plot_psth_interneurons_events(ms, spike_nums_dur, spike_nums, SCE_times, sliding_window_duration,
                                  param, save_formats="pdf"):
    inter_neurons = ms.spike_struct.inter_neurons
    n_times = len(spike_nums[0, :])
    for inter_neuron in inter_neurons:
        # key is an int which reprensent the sum of spikes at a certain distance (in frames) of the event,
        # negative or positive
        spike_at_time_dict = dict()

        for sce_id, time_tuple in enumerate(SCE_times):
            time_tuple = SCE_times[sce_id]
            duration_in_frames = (time_tuple[1] - time_tuple[0]) + 1
            n_slidings = (duration_in_frames - sliding_window_duration) + 1

            sum_activity_for_each_frame = np.zeros(n_slidings)
            for n in np.arange(n_slidings):
                # see to use window_duration to find the amount of participation
                time_beg = time_tuple[0] + n
                sum_activity_for_each_frame[n] = len(np.where(np.sum(spike_nums_dur[:,
                                                                     time_beg:(time_beg + sliding_window_duration)],
                                                                     axis=1))[0])
            max_activity_index = np.argmax(sum_activity_for_each_frame)
            # time_max represent the time at which the event is at its top, we will center histogram on it
            time_max = time_tuple[0] + max_activity_index + int(sliding_window_duration / 2)
            # beg_time = 0 if sce_id == 0 else SCE_times[sce_id - 1][1]
            # end_time = n_times if sce_id == (len(SCE_times) - 1) else SCE_times[sce_id + 1][0]
            range_in_frames = 50
            min_SCE = 0 if sce_id == 0 else SCE_times[sce_id - 1][1]
            max_SCE = n_times if sce_id == (len(SCE_times) - 1) else SCE_times[sce_id + 1][0]
            beg_time = np.max((0, time_max - range_in_frames, min_SCE))
            end_time = np.min((n_times, time_max + range_in_frames, max_SCE))
            # print(f"time_max {time_max}, beg_time {beg_time}, end_time {end_time}")
            # before the event
            time_spikes = np.where(spike_nums[inter_neuron, beg_time:time_max])[0]
            # print(f"before time_spikes {time_spikes}")
            if len(time_spikes) > 0:
                time_spike = np.max(time_spikes)
                time_spike = time_spike - ((time_max + 1) - beg_time)
                # print(f"before time_spike {time_spike}")
                # for time_spike in time_spikes:
                spike_at_time_dict[time_spike] = spike_at_time_dict.get(time_spike, 0) + 1
            # after the event
            time_spikes = np.where(spike_nums[inter_neuron, time_max:end_time])[0]
            if len(time_spikes) > 0:
                time_spike = np.min(time_spikes)
                # for time_spike in time_spikes:
                spike_at_time_dict[time_spike] = spike_at_time_dict.get(time_spike, 0) + 1

        distribution = []
        for time, nb_spikes_at_time in spike_at_time_dict.items():
            # print(f"time {time}")
            distribution.extend([time] * nb_spikes_at_time)
        if len(distribution) == 0:
            continue
        distribution = np.array(distribution)
        hist_color = "blue"
        edge_color = "white"
        max_range = np.max((np.max(distribution), range_in_frames))
        min_range = np.min((np.min(distribution), -range_in_frames))
        weights = (np.ones_like(distribution) / (len(distribution))) * 100

        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                gridspec_kw={'height_ratios': [1]},
                                figsize=(15, 10))
        ax1.set_facecolor("black")
        # as many bins as time
        bins = (max_range - min_range) // 4
        # bins = int(np.sqrt(len(distribution)))
        hist_plt, edges_plt, patches_plt = plt.hist(distribution, bins=bins, range=(min_range, max_range),
                                                    facecolor=hist_color,
                                                    edgecolor=edge_color,
                                                    weights=weights, log=False)
        ax1.vlines(0, 0,
                   np.max(hist_plt), color="white", linewidth=2,
                   linestyles="dashed")
        plt.title(f"{ms.description} interneuron {inter_neuron}")
        ax1.set_ylabel(f"Spikes (%)")
        ax1.set_xlabel("time (frames)")
        ax1.set_ylim(0, 100)
        # xticks = np.arange(0, len(data_dict))
        # ax1.set_xticks(xticks)
        # # sce clusters labels
        # ax1.set_xticklabels(labels)

        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{param.path_results}/psth_{ms.description}_interneuron_{inter_neuron}'
                        f'_{param.time_str}.{save_format}',
                        format=f"{save_format}")

        plt.close()


def get_ratio_spikes_on_events_vs_total_events_by_cell(spike_nums,
                                                       spike_nums_dur,
                                                       sce_times_numbers):
    n_cells = len(spike_nums)
    result = np.zeros(n_cells)

    for cell in np.arange(n_cells):
        n_sces = np.max(sce_times_numbers) + 1
        spikes_index = np.where(spike_nums_dur[cell, :])[0]
        sce_numbers = sce_times_numbers[spikes_index]
        # will give us all sce in which the cell spikes
        sce_numbers = np.unique(sce_numbers)
        # removing the -1, which is when it spikes not in a SCE
        if len(np.where(sce_numbers == - 1)[0]) > 0:
            sce_numbers = sce_numbers[1:]
        # print(f"len sce_numbers {len(sce_numbers)}, sce_numbers {sce_numbers}, n_spikes {n_spikes}")
        # print(f"len sce_times_numbers {len(np.unique(sce_times_numbers))}, sce_numbers {np.unique(sce_times_numbers)}")
        result[cell] = np.min((((len(sce_numbers) / n_sces) * 100), 100))
    # result = result[result >= 0]
    return result


def get_ratio_spikes_on_events_vs_total_spikes_by_cell(spike_nums,
                                                       spike_nums_dur,
                                                       sce_times_numbers):
    n_cells = len(spike_nums)
    result = np.zeros(n_cells)

    for cell in np.arange(n_cells):
        n_spikes = np.sum(spike_nums[cell, :])
        spikes_index = np.where(spike_nums_dur[cell, :])[0]
        sce_numbers = sce_times_numbers[spikes_index]
        # will give us all sce in which the cell spikes
        sce_numbers = np.unique(sce_numbers)
        # removing the -1, which is when it spikes not in a SCE
        if len(np.where(sce_numbers == - 1)[0]) > 0:
            sce_numbers = sce_numbers[1:]
        # print(f"len sce_numbers {len(sce_numbers)}, sce_numbers {sce_numbers}, n_spikes {n_spikes}")
        # print(f"len sce_times_numbers {len(np.unique(sce_times_numbers))}, sce_numbers {np.unique(sce_times_numbers)}")
        if n_spikes == 0:
            result[cell] = 0
        else:
            result[cell] = np.min((((len(sce_numbers) / n_spikes) * 100), 100))
    # removing cells without spikes
    # result = result[result >= 0]
    return result


def main():
    root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    path_data = root_path + "data/"
    path_results_raw = root_path + "results_hne/"

    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    path_results = path_results_raw + f"{time_str}"
    os.mkdir(path_results)

    # --------------------------------------------------------------------------------
    # ------------------------------ param section ------------------------------
    # --------------------------------------------------------------------------------

    # param will be set later when the spike_nums will have been constructed
    param = HNEParameters(time_str=time_str, path_results=path_results, error_rate=2,
                          time_inter_seq=50, min_duration_intra_seq=-3, min_len_seq=10, min_rep_nb=4,
                          max_branches=20, stop_if_twin=False,
                          no_reverse_seq=False, spike_rate_weight=False, path_data=path_data)

    # loading data
    p6_18_02_07_a001_ms = MouseSession(age=6, session_id="18_02_07_a001", nb_ms_by_frame=100, param=param,
                                       weight=4.35)
    # calculated with 99th percentile on raster dur
    p6_18_02_07_a001_ms.activity_threshold = 17
    p6_18_02_07_a001_ms.set_inter_neurons([28, 36, 54, 75])
    # duration of those interneurons: [ 18.58 17.78   19.  17.67]
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p6_18_02_07_a001_ms.load_data_from_file(file_name_to_load="p6/p6_18_02_07_a001/RasterDur_p6_18_02_07_a001.mat",
                                            variables_mapping=variables_mapping)

    p6_18_02_07_a002_ms = MouseSession(age=6, session_id="18_02_07_a002", nb_ms_by_frame=100, param=param,
                                       weight=4.35)
    # calculated with 99th percentile on raster dur
    p6_18_02_07_a002_ms.activity_threshold = 9
    p6_18_02_07_a002_ms.set_inter_neurons([40, 90])
    # duration of those interneurons: 16.27  23.33
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p6_18_02_07_a002_ms.load_data_from_file(file_name_to_load="p6/p6_18_02_07_a002/RasterDur_p6_18_02_07_a002.mat",
                                            variables_mapping=variables_mapping)

    p7_171012_a000_ms = MouseSession(age=7, session_id="17_10_12_a000", nb_ms_by_frame=100, param=param,
                                     weight=None)
    # calculated with 99th percentile on raster dur
    p7_171012_a000_ms.activity_threshold = 22
    p7_171012_a000_ms.set_inter_neurons([305, 360, 398, 412])
    # duration of those interneurons: 13.23  12.48  10.8   11.88
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p7_171012_a000_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_12_a000/P7_17_10_12_a000_rasterdur.mat",
                                          variables_mapping=variables_mapping)

    p7_18_02_08_a000_ms = MouseSession(age=7, session_id="18_02_08_a000", nb_ms_by_frame=100, param=param,
                                       weight=3.85)
    # calculated with 99th percentile on raster dur
    p7_18_02_08_a000_ms.activity_threshold = 11
    p7_18_02_08_a000_ms.set_inter_neurons([56, 95, 178])
    # duration of those interneurons: 12.88  13.94  13.04
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p7_18_02_08_a000_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a000/p7_18_02_08_a000_RasterDur.mat",
                                            variables_mapping=variables_mapping)

    p7_18_02_08_a001_ms = MouseSession(age=7, session_id="18_02_08_a001", nb_ms_by_frame=100, param=param,
                                       weight=3.85)
    # calculated with 99th percentile on raster dur
    p7_18_02_08_a001_ms.activity_threshold = 13
    p7_18_02_08_a001_ms.set_inter_neurons([151])
    # duration of those interneurons: 22.11
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p7_18_02_08_a001_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a001/p7_18_02_08_a001_RasterDur.mat",
                                            variables_mapping=variables_mapping)

    p7_18_02_08_a002_ms = MouseSession(age=7, session_id="18_02_08_a002", nb_ms_by_frame=100, param=param,
                                       weight=3.85)
    # calculated with 99th percentile on raster dur
    p7_18_02_08_a002_ms.activity_threshold = 10
    p7_18_02_08_a002_ms.set_inter_neurons([207])
    # duration of those interneurons: 22.33
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p7_18_02_08_a002_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a002/p7_18_02_08_a002_RasterDur.mat",
                                            variables_mapping=variables_mapping)

    p7_18_02_08_a003_ms = MouseSession(age=7, session_id="18_02_08_a003", nb_ms_by_frame=100, param=param,
                                       weight=3.85)
    # calculated with 99th percentile on raster dur
    p7_18_02_08_a003_ms.activity_threshold = 8
    p7_18_02_08_a003_ms.set_inter_neurons([171])
    # duration of those interneurons: 14.92
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p7_18_02_08_a003_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a003/p7_18_02_08_a003_RasterDur.mat",
                                            variables_mapping=variables_mapping)

    p7_17_10_18_a002_ms = MouseSession(age=7, session_id="17_10_18_a002", nb_ms_by_frame=100, param=param,
                                       weight=None)
    # calculated with 99th percentile on raster dur
    p7_17_10_18_a002_ms.activity_threshold = 15
    p7_17_10_18_a002_ms.set_inter_neurons([51])
    # duration of those interneurons: 14.13
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p7_17_10_18_a002_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_18_a002/p7_17_10_18_a002_RasterDur.mat",
                                            variables_mapping=variables_mapping)

    p7_17_10_18_a004_ms = MouseSession(age=7, session_id="17_10_18_a004", nb_ms_by_frame=100, param=param,
                                       weight=None)
    # calculated with 99th percentile on raster dur
    p7_17_10_18_a004_ms.activity_threshold = 14
    p7_17_10_18_a004_ms.set_inter_neurons([298])
    # duration of those interneurons: 15.35
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p7_17_10_18_a004_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_18_a004/p7_17_10_18_a004_RasterDur.mat",
                                            variables_mapping=variables_mapping)

    p8_18_02_09_a000_ms = MouseSession(age=8, session_id="18_02_09_a000", nb_ms_by_frame=100, param=param,
                                       weight=None)
    # calculated with 99th percentile on raster dur
    p8_18_02_09_a000_ms.activity_threshold = 9
    p8_18_02_09_a000_ms.set_inter_neurons([64, 91])
    # duration of those interneurons: 12.48  11.47
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p8_18_02_09_a000_ms.load_data_from_file(file_name_to_load="p8/p8_18_02_09_a000/p8_18_02_09_a000_RasterDur.mat",
                                            variables_mapping=variables_mapping)

    p8_18_02_09_a001_ms = MouseSession(age=8, session_id="18_02_09_a001", nb_ms_by_frame=100, param=param,
                                       weight=None)
    # calculated with 99th percentile on raster dur
    p8_18_02_09_a001_ms.activity_threshold = 11
    p8_18_02_09_a001_ms.set_inter_neurons([])
    # duration of those interneurons:
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p8_18_02_09_a001_ms.load_data_from_file(file_name_to_load="p8/p8_18_02_09_a001/p8_18_02_09_a001_RasterDur.mat",
                                            variables_mapping=variables_mapping)

    # p9_17_11_29_a002 low participation comparing to other, dead shortly after the recording
    p9_17_11_29_a002_ms = MouseSession(age=9, session_id="17_11_29_a002", nb_ms_by_frame=100, param=param,
                                       weight=5.7)
    # calculated with 99th percentile on raster dur
    p9_17_11_29_a002_ms.activity_threshold = 10
    p9_17_11_29_a002_ms.set_inter_neurons([170])
    # limit ??
    # duration of those interneurons: 21
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p9_17_11_29_a002_ms.load_data_from_file(file_name_to_load="p9/p9_17_11_29_a002/p9_17_11_29_a002_RasterDur.mat",
                                            variables_mapping=variables_mapping)

    p9_17_11_29_a003_ms = MouseSession(age=9, session_id="17_11_29_a003", nb_ms_by_frame=100, param=param,
                                       weight=5.7)
    # calculated with 99th percentile on raster dur
    p9_17_11_29_a003_ms.activity_threshold = 7
    p9_17_11_29_a003_ms.set_inter_neurons([1, 13, 54])
    # duration of those interneurons: 21.1 22.75  23
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p9_17_11_29_a003_ms.load_data_from_file(file_name_to_load="p9/p9_17_11_29_a003/p9_17_11_29_a003_RasterDur.mat",
                                            variables_mapping=variables_mapping)

    p9_17_12_06_a001_ms = MouseSession(age=9, session_id="17_12_06_a001", nb_ms_by_frame=100, param=param,
                                       weight=5.6)
    # calculated with 99th percentile on raster dur
    p9_17_12_06_a001_ms.activity_threshold = 9
    p9_17_12_06_a001_ms.set_inter_neurons([72])
    # duration of those interneurons:15.88
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p9_17_12_06_a001_ms.load_data_from_file(file_name_to_load="p9/p9_17_12_06_a001/p9_17_12_06_a001_RasterDur.mat",
                                            variables_mapping=variables_mapping)

    p9_17_12_20_a001_ms = MouseSession(age=9, session_id="17_12_20_a001", nb_ms_by_frame=100, param=param,
                                       weight=5.05)
    # calculated with 99th percentile on raster dur
    p9_17_12_20_a001_ms.activity_threshold = 9
    p9_17_12_20_a001_ms.set_inter_neurons([32])
    # duration of those interneurons: 10.35
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p9_17_12_20_a001_ms.load_data_from_file(file_name_to_load="p9/p9_17_12_20_a001/p9_17_12_20_a001_RasterDur.mat",
                                            variables_mapping=variables_mapping)

    p10_17_11_16_a003_ms = MouseSession(age=10, session_id="17_11_16_a003", nb_ms_by_frame=100, param=param,
                                        weight=6.1)
    # calculated with 99th percentile on raster dur
    p10_17_11_16_a003_ms.activity_threshold = 6
    p10_17_11_16_a003_ms.set_inter_neurons([8])
    # duration of those interneurons: 28
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p10_17_11_16_a003_ms.load_data_from_file(file_name_to_load="p10/p10_17_11_16_a003/p10_17_11_16_a003_RasterDur.mat",
                                             variables_mapping=variables_mapping)

    p11_17_11_24_a001_ms = MouseSession(age=11, session_id="17_11_24_a001", nb_ms_by_frame=100, param=param,
                                        weight=6.7)
    # calculated with 99th percentile on raster dur
    p11_17_11_24_a001_ms.activity_threshold = 11
    p11_17_11_24_a001_ms.set_inter_neurons([])
    # duration of those interneurons:
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p11_17_11_24_a001_ms.load_data_from_file(file_name_to_load="p11/p11_17_11_24_a001/p11_17_11_24_a001_RasterDur.mat",
                                             variables_mapping=variables_mapping)

    p11_17_11_24_a000_ms = MouseSession(age=11, session_id="17_11_24_a000", nb_ms_by_frame=100, param=param,
                                        weight=6.7)
    # calculated with 99th percentile on raster dur
    p11_17_11_24_a000_ms.activity_threshold = 12
    p11_17_11_24_a000_ms.set_inter_neurons([193])
    # duration of those interneurons: 19.09
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p11_17_11_24_a000_ms.load_data_from_file(file_name_to_load="p11/p11_17_11_24_a000/p11_17_11_24_a000_RasterDur.mat",
                                             variables_mapping=variables_mapping)

    p12_17_11_10_a002_ms = MouseSession(age=12, session_id="17_11_10_a002", nb_ms_by_frame=100, param=param,
                                        weight=7)
    # calculated with 99th percentile on raster dur
    p12_17_11_10_a002_ms.activity_threshold = 13
    p12_17_11_10_a002_ms.set_inter_neurons([150, 252])
    # duration of those interneurons: 16.17, 24.8
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p12_17_11_10_a002_ms.load_data_from_file(file_name_to_load="p12/p12_17_11_10_a002/p12_17_11_10_a002_RasterDur.mat",
                                             variables_mapping=variables_mapping)

    p12_171110_a000_ms = MouseSession(age=12, session_id="171110_a000", nb_ms_by_frame=100, param=param,
                                      weight=7)
    # calculated with 99th percentile on raster dur
    p12_171110_a000_ms.activity_threshold = 10
    p12_171110_a000_ms.set_inter_neurons([106, 144])
    # duration of those interneurons: 18.29  14.4
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p12_171110_a000_ms.load_data_from_file(file_name_to_load="p12/P12_17_11_10_a000/p12_17_11_10_a000_rasterdur.mat",
                                           variables_mapping=variables_mapping)

    p14_18_10_23_a000_ms = MouseSession(age=14, session_id="18_10_23_a000", nb_ms_by_frame=100, param=param,
                                        weight=10.35)
    # calculated with 99th percentile on raster dur
    p14_18_10_23_a000_ms.activity_threshold = 9
    p14_18_10_23_a000_ms.set_inter_neurons([0])
    # duration of those interneurons: 24.33
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p14_18_10_23_a000_ms.load_data_from_file(file_name_to_load="p14/p14_18_10_23_a000/p14_18_10_23_a000_rasterdur.mat",
                                             variables_mapping=variables_mapping)

    # TODO : add weight
    p14_18_10_23_a001_ms = MouseSession(age=14, session_id="18_10_23_a001", nb_ms_by_frame=100, param=param)
    # calculated with 99th percentile on raster dur
    # p14_18_10_23_a001_ms.activity_threshold = 9
    # p14_18_10_23_a001_ms.set_inter_neurons(np.arange(31))
    p14_18_10_23_a001_ms.set_inter_neurons([])
    # duration of those interneurons: 24.33
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "PEAK_LOC_2"}
    p14_18_10_23_a001_ms.load_data_from_file(file_name_to_load="p14/p14_18_10_23_a001/p14_18_10_23_a001_RasterDur.mat",
                                             variables_mapping=variables_mapping)

    arnaud_ms = MouseSession(age=24, session_id="arnaud", nb_ms_by_frame=50, param=param)
    arnaud_ms.activity_threshold = 13
    variables_mapping = {"spike_nums": "spikenums"}
    arnaud_ms.load_data_from_file(file_name_to_load="spikenumsarnaud.mat", variables_mapping=variables_mapping)

    available_ms = [p6_18_02_07_a001_ms, p6_18_02_07_a002_ms,
                    p7_171012_a000_ms, p7_18_02_08_a000_ms,
                    p7_17_10_18_a002_ms, p7_17_10_18_a004_ms, p7_18_02_08_a001_ms, p7_18_02_08_a002_ms,
                    p7_18_02_08_a003_ms,
                    p8_18_02_09_a000_ms, p8_18_02_09_a001_ms,
                    p9_17_12_06_a001_ms, p9_17_12_20_a001_ms,
                    p10_17_11_16_a003_ms,
                    p11_17_11_24_a001_ms, p11_17_11_24_a000_ms,
                    p12_17_11_10_a002_ms, p12_171110_a000_ms,
                    p14_18_10_23_a000_ms]
    available_ms = [p14_18_10_23_a001_ms]
    # p9_17_11_29_a002_ms, p9_17_11_29_a003_ms removed because died just after
    # available_ms = [p6_18_02_07_a001_ms, p7_171012_a000_ms, p8_18_02_09_a000_ms, p9_17_12_20_a001_ms,
    #                 p10_17_11_16_a003_ms, p12_171110_a000_ms]

    ms_to_analyse = [arnaud_ms]
    # ms_to_analyse = [p6_18_02_07_a001_ms, p6_18_02_07_a002_ms]

    do_clustering = False
    # if False, clustering will be done using kmean
    do_fca_clustering = False
    with_cells_in_cluster_seq_sorted = False

    just_do_stat_on_event_detection_parameters = False

    # for events (sce) detection
    perc_threshold = 99
    use_max_of_each_surrogate = False
    n_surrogate_activity_threshold = 50
    use_raster_dur = False

    # for fca
    n_surrogate_fca = 20

    # for kmean
    with_shuffling = True
    print(f"use_raster_dur {use_raster_dur}")
    range_n_clusters_k_mean = np.arange(3, 10)
    n_surrogate_k_mean = 50

    do_pattern_search = True
    split_pattern_search = False
    use_only_uniformity_method = False
    use_loss_score_to_keep_the_best_from_tree = False
    use_sce_times_for_pattern_search = True
    n_surrogate_for_pattern_search = 0
    # seq params:
    param.error_rate = 4
    param.max_branches = 5
    param.time_inter_seq = 50
    param.min_duration_intra_seq = -5
    param.min_len_seq = 10
    param.min_rep_nb = 5

    # ------------------------------ end param section ------------------------------

    if just_do_stat_on_event_detection_parameters:
        # keep the value for each ms
        duration_values_list = []
        max_activity_values_list = []
        mean_activity_values_list = []
        overall_activity_values_list = []
        ms_ages = []
        ratio_spikes_events_by_age = dict()
        interneurons_indices_by_age = dict()
        duration_spikes_by_age = dict()
        ratio_spikes_total_events_by_age = dict()

        for ms in available_ms:
            ms_ages.append(ms.age)
            t = 1
            # for each session, we want to detect SCE using 6 methods:
            # 1) Keep the max of each surrogate and take the 95th percentile
            # 2) Keep the 95th percentile of n_times * n°surrogates events activity
            # 3) Keep the 99th percentile of n_times * n°surrogates events activity
            #  and for each with raster_dur and with_onsets for n surrogates

            spike_struct = ms.spike_struct
            n_cells = len(spike_struct.spike_nums)
            use_raster_durs = [True]
            # 0 : 99th percentile, 1 & 2: 95th percentile
            selection_options = [0]
            for use_raster_dur in use_raster_durs:
                for selection_option in selection_options:
                    if use_raster_dur:
                        sliding_window_duration = 1
                        spike_nums_to_use = spike_struct.spike_nums_dur
                    else:
                        sliding_window_duration = 5
                        spike_nums_to_use = spike_struct.spike_nums

                    # ######  parameters setting #########
                    # data_descr = f"{ms.description}"
                    n_surrogate_activity_threshold = 1000

                    if selection_option > 0:
                        perc_threshold = 95
                    else:
                        perc_threshold = 99

                    if selection_option == 2:
                        use_max_of_each_surrogate = True
                    else:
                        use_max_of_each_surrogate = False

                    if ms.activity_threshold is None:
                        activity_threshold = get_sce_detection_threshold(spike_nums=spike_nums_to_use,
                                                                         window_duration=sliding_window_duration,
                                                                         spike_train_mode=False,
                                                                         use_max_of_each_surrogate=use_max_of_each_surrogate,
                                                                         n_surrogate=n_surrogate_activity_threshold,
                                                                         perc_threshold=perc_threshold,
                                                                         debug_mode=False)

                        spike_struct.activity_threshold = activity_threshold
                        # param.activity_threshold = activity_threshold
                    else:
                        activity_threshold = ms.activity_threshold
                        spike_struct.activity_threshold = ms.activity_threshold

                    sce_detection_result = detect_sce_with_sliding_window(spike_nums=spike_nums_to_use,
                                                                          window_duration=sliding_window_duration,
                                                                          perc_threshold=perc_threshold,
                                                                          activity_threshold=activity_threshold,
                                                                          debug_mode=False,
                                                                          no_redundancy=False)

                    print(f"sce_with_sliding_window detected")
                    # tuple of times
                    SCE_times = sce_detection_result[1]
                    sce_times_numbers = sce_detection_result[3]

                    print(f"ms {ms.description}, {len(SCE_times)} sce "
                          f"activity threshold {activity_threshold}, "
                          f"use_raster_dur {use_raster_dur},  "
                          f"{perc_threshold} percentile, use_max_of_each_surrogate {use_max_of_each_surrogate}"
                          f", np.shape(spike_nums_to_use) {np.shape(spike_nums_to_use)}")

                    raster_option = "raster_dur" if use_raster_dur else "onsets"
                    technique_details_file = "max_each_surrogate" if use_max_of_each_surrogate else ""
                    file_name = f'{ms.description}_stat_{raster_option}_{perc_threshold}_perc_' \
                                f'{technique_details_file}'

                    spike_shape = '|' if use_raster_dur else 'o'
                    inter_neurons = ms.spike_struct.inter_neurons
                    plot_spikes_raster(spike_nums=spike_nums_to_use, param=ms.param,
                                       span_cells_to_highlight=inter_neurons,
                                       span_cells_to_highlight_colors=["red"] * len(inter_neurons),
                                       spike_train_format=False,
                                       title=file_name,
                                       file_name=file_name,
                                       y_ticks_labels=np.arange(n_cells),
                                       y_ticks_labels_size=2,
                                       save_raster=True,
                                       show_raster=False,
                                       plot_with_amplitude=False,
                                       activity_threshold=spike_struct.activity_threshold,
                                       # 500 ms window
                                       sliding_window_duration=sliding_window_duration,
                                       show_sum_spikes_as_percentage=True,
                                       # vertical_lines=SCE_times,
                                       # vertical_lines_colors=['white'] * len(SCE_times),
                                       # vertical_lines_sytle="solid",
                                       # vertical_lines_linewidth=[0.2] * len(SCE_times),
                                       span_area_coords=[SCE_times],
                                       span_area_colors=['white'],
                                       spike_shape=spike_shape,
                                       spike_shape_size=0.5,
                                       save_formats="pdf")

                    plot_psth_interneurons_events(ms=ms, spike_nums_dur=ms.spike_struct.spike_nums_dur,
                                                  spike_nums=ms.spike_struct.spike_nums,
                                                  SCE_times=SCE_times, sliding_window_duration=sliding_window_duration,
                                                  param=param)

                    # return an array of size n_cells, with each value the ratio as a float (percentage)
                    ratio_spikes_events = get_ratio_spikes_on_events_vs_total_spikes_by_cell(
                        spike_nums=spike_struct.spike_nums,
                        spike_nums_dur=spike_struct.spike_nums_dur,
                        sce_times_numbers=sce_times_numbers)

                    ratio_spikes_total_events = get_ratio_spikes_on_events_vs_total_events_by_cell(
                        spike_nums=spike_struct.spike_nums,
                        spike_nums_dur=spike_struct.spike_nums_dur,
                        sce_times_numbers=sce_times_numbers)
                    if ms.age not in ratio_spikes_events_by_age:
                        ratio_spikes_events_by_age[ms.age] = list(ratio_spikes_events)
                        interneurons_indices_by_age[ms.age] = list(ms.spike_struct.inter_neurons)
                        all_spike_duration = []
                        for spike_duration in ms.spike_struct.spike_durations:
                            all_spike_duration.extend(spike_duration)
                        duration_spikes_by_age[ms.age] = all_spike_duration
                    else:
                        nb_elem = len(ratio_spikes_events_by_age[ms.age])
                        ratio_spikes_events_by_age[ms.age].extend(list(ratio_spikes_events))
                        interneurons_indices_by_age[ms.age].extend(list(ms.spike_struct.inter_neurons +
                                                                        nb_elem))
                        all_spike_duration = []
                        for spike_duration in ms.spike_struct.spike_durations:
                            all_spike_duration.extend(spike_duration)
                        duration_spikes_by_age[ms.age].extend(all_spike_duration)

                    if ms.age not in ratio_spikes_total_events_by_age:
                        ratio_spikes_total_events_by_age[ms.age] = list(ratio_spikes_total_events)
                    else:
                        ratio_spikes_total_events_by_age[ms.age].extend(list(ratio_spikes_total_events))

                    res = save_stat_sce_detection_methods(spike_nums_to_use=spike_nums_to_use,
                                                          activity_threshold=activity_threshold,
                                                          ms=ms,
                                                          ratio_spikes_events=ratio_spikes_events,
                                                          ratio_spikes_total_events=ratio_spikes_total_events,
                                                          SCE_times=SCE_times, param=param,
                                                          sliding_window_duration=sliding_window_duration,
                                                          perc_threshold=perc_threshold,
                                                          use_raster_dur=use_raster_dur,
                                                          keep_max_each_surrogate=use_max_of_each_surrogate,
                                                          n_surrogate_activity_threshold=n_surrogate_activity_threshold)

                    duration_values_list.append(res[0])
                    max_activity_values_list.append(res[1])
                    mean_activity_values_list.append(res[2])
                    overall_activity_values_list.append(res[3])

                    values_to_scatter = []
                    non_inter_neurons = np.setdiff1d(np.arange(len(ratio_spikes_events)), inter_neurons)
                    ratio_interneurons = list(ratio_spikes_events[inter_neurons])
                    ratio_non_interneurons = list(ratio_spikes_events[non_inter_neurons])
                    values_to_scatter.append(np.mean(ratio_non_interneurons))
                    values_to_scatter.append(np.median(ratio_non_interneurons))
                    values_to_scatter.append(np.mean(ratio_interneurons))
                    values_to_scatter.append(np.median(ratio_interneurons))
                    values_to_scatter.extend(ratio_interneurons)
                    labels = ["mean", "median", "mean", "median", f"interneuron (x{len(inter_neurons)})"]
                    scatter_shapes = ["o", "s", "o", "s"]
                    scatter_shapes.extend(["*"] * len(inter_neurons))
                    colors = ["white", "white", "red", "red"]
                    colors.extend(["red"] * len(inter_neurons))

                    plot_hist_ratio_spikes_events(ratio_spikes_events=ratio_spikes_events,
                                                  description=f"{ms.description}_hist_spike_events_ratio",
                                                  values_to_scatter=np.array(values_to_scatter),
                                                  labels=labels,
                                                  scatter_shapes=scatter_shapes,
                                                  colors=colors,
                                                  xlabel="spikes in event vs total spikes (%)",
                                                  param=param)

                    values_to_scatter = []
                    ratio_interneurons = list(ratio_spikes_total_events[inter_neurons])
                    ratio_non_interneurons = list(ratio_spikes_total_events[non_inter_neurons])
                    values_to_scatter.append(np.mean(ratio_non_interneurons))
                    values_to_scatter.append(np.median(ratio_non_interneurons))
                    values_to_scatter.append(np.mean(ratio_interneurons))
                    values_to_scatter.append(np.median(ratio_interneurons))
                    values_to_scatter.extend(ratio_interneurons)
                    plot_hist_ratio_spikes_events(ratio_spikes_events=ratio_spikes_total_events,
                                                  description=f"{ms.description}_hist_spike_total_events_ratio",
                                                  values_to_scatter=np.array(values_to_scatter),
                                                  labels=labels,
                                                  scatter_shapes=scatter_shapes,
                                                  colors=colors,
                                                  xlabel="spikes in event vs total events (%)",
                                                  param=param)

        ratio_spikes_events_non_interneurons_by_age = dict()
        ratio_spikes_events_interneurons_by_age = dict()
        for age, ratio_spikes in ratio_spikes_events_by_age.items():
            inter_neurons = np.array(interneurons_indices_by_age[age]).astype(int)
            non_inter_neurons = np.setdiff1d(np.arange(len(ratio_spikes)), inter_neurons)
            ratio_spikes = np.array(ratio_spikes)
            values_to_scatter = []
            ratio_interneurons = list(ratio_spikes[inter_neurons])
            ratio_spikes_events_interneurons_by_age[age] = ratio_interneurons
            ratio_non_interneurons = list(ratio_spikes[non_inter_neurons])
            ratio_spikes_events_non_interneurons_by_age[age] = ratio_non_interneurons
            values_to_scatter.append(np.mean(ratio_non_interneurons))
            values_to_scatter.append(np.median(ratio_non_interneurons))
            values_to_scatter.append(np.mean(ratio_interneurons))
            values_to_scatter.append(np.median(ratio_interneurons))
            values_to_scatter.extend(ratio_interneurons)
            labels = ["mean", "median", "mean", "median", f"interneuron (x{len(inter_neurons)})"]
            scatter_shapes = ["o", "s", "o", "s"]
            scatter_shapes.extend(["*"] * len(inter_neurons))
            colors = ["white", "white", "red", "red"]
            colors.extend(["red"] * len(inter_neurons))
            plot_hist_ratio_spikes_events(ratio_spikes_events=ratio_spikes,
                                          description=f"p{age}_hist_spike_events_ratio",
                                          values_to_scatter=np.array(values_to_scatter),
                                          labels=labels,
                                          scatter_shapes=scatter_shapes,
                                          colors=colors,
                                          xlabel="spikes in event vs total spikes (%)",
                                          param=param)

        ratio_spikes_total_events_non_interneurons_by_age = dict()
        ratio_spikes_total_events_interneurons_by_age = dict()
        for age, ratio_spikes in ratio_spikes_total_events_by_age.items():
            inter_neurons = np.array(interneurons_indices_by_age[age]).astype(int)
            non_inter_neurons = np.setdiff1d(np.arange(len(ratio_spikes)), inter_neurons)
            ratio_spikes = np.array(ratio_spikes)
            values_to_scatter = []
            ratio_interneurons = list(ratio_spikes[inter_neurons])
            ratio_spikes_total_events_interneurons_by_age[age] = ratio_interneurons
            ratio_non_interneurons = list(ratio_spikes[non_inter_neurons])
            ratio_spikes_total_events_non_interneurons_by_age[age] = ratio_non_interneurons
            values_to_scatter.append(np.mean(ratio_non_interneurons))
            values_to_scatter.append(np.median(ratio_non_interneurons))
            values_to_scatter.append(np.mean(ratio_interneurons))
            values_to_scatter.append(np.median(ratio_interneurons))
            values_to_scatter.extend(ratio_interneurons)
            labels = ["mean", "median", "mean", "median", f"interneuron (x{len(inter_neurons)})"]
            scatter_shapes = ["o", "s", "o", "s"]
            scatter_shapes.extend(["*"] * len(inter_neurons))
            colors = ["white", "white", "red", "red"]
            colors.extend(["red"] * len(inter_neurons))
            plot_hist_ratio_spikes_events(ratio_spikes_events=ratio_spikes,
                                          description=f"p{age}_hist_spike_total_events_ratio",
                                          values_to_scatter=np.array(values_to_scatter),
                                          labels=labels,
                                          scatter_shapes=scatter_shapes,
                                          colors=colors,
                                          xlabel="spikes in event vs total events (%)",
                                          param=param)

        # plotting boxplots
        box_plot_data_by_age(data_dict=ratio_spikes_events_by_age,
                             title="Ratio spikes on events / total spikes for all cells",
                             filename="box_plots_ratio_spikes_events_total_spikes_by_age_all_cells",
                             y_label="",
                             param=param)
        box_plot_data_by_age(data_dict=ratio_spikes_events_non_interneurons_by_age,
                             title="Ratio spikes on events / total spikes for pyramidal cells",
                             filename="box_plots_ratio_spikes_events_total_spikes_by_age_pyramidal_cells",
                             y_label="",
                             param=param)
        box_plot_data_by_age(data_dict=ratio_spikes_events_interneurons_by_age,
                             title="Ratio spikes on events / total spikes for interneurons",
                             filename="box_plots_ratio_spikes_events_total_spikes_by_age_interneurons",
                             y_label="",
                             param=param)

        box_plot_data_by_age(data_dict=ratio_spikes_total_events_by_age,
                             title="Ratio spikes on events / total events for all cells",
                             filename="box_plots_ratio_spikes_events_total_events_by_age_all_cells",
                             y_label="",
                             param=param)
        box_plot_data_by_age(data_dict=ratio_spikes_total_events_non_interneurons_by_age,
                             title="Ratio spikes on events / total events for pyramidal cells",
                             filename="box_plots_ratio_spikes_events_total_events_by_age_pyramidal_cells",
                             y_label="",
                             param=param)
        box_plot_data_by_age(data_dict=ratio_spikes_total_events_interneurons_by_age,
                             title="Ratio spikes on events / total events for interneurons",
                             filename="box_plots_ratio_spikes_events_total_events_by_age_interneurons",
                             y_label="",
                             param=param)

        save_stat_by_age(ratio_spikes_events_by_age=ratio_spikes_events_by_age,
                         ratio_spikes_total_events_by_age=ratio_spikes_total_events_by_age,
                         interneurons_indices_by_age=interneurons_indices_by_age,
                         mouse_sessions=available_ms,
                         param=param)

        plot_activity_duration_vs_age(mouse_sessions=available_ms, ms_ages=ms_ages,
                                      duration_values_list=duration_values_list,
                                      max_activity_values_list=max_activity_values_list,
                                      mean_activity_values_list=mean_activity_values_list,
                                      overall_activity_values_list=overall_activity_values_list,
                                      param=param)

        plot_duration_spikes_by_age(mouse_sessions=available_ms, ms_ages=ms_ages,
                                    duration_spikes_by_age=duration_spikes_by_age, param=param)
        return

    for ms in ms_to_analyse:
        ###################################################################
        ###################################################################
        # ###########    SCE detection and clustering        ##############
        ###################################################################
        ###################################################################

        spike_struct = ms.spike_struct
        n_cells = len(spike_struct.spike_nums)
        # spike_struct.build_spike_trains()

        # ######  parameters setting #########
        data_descr = f"{ms.description}"
        debug_mode = True

        if do_fca_clustering:
            sliding_window_duration = 5
            spike_nums_to_use = spike_struct.spike_nums
            # sigma is the std of the random distribution used to jitter the data
            sigma = 20
        else:
            if use_raster_dur:
                sliding_window_duration = 1
                spike_nums_to_use = spike_struct.spike_nums_dur
                data_descr += "_raster_dur"
            else:
                sliding_window_duration = 5
                spike_nums_to_use = spike_struct.spike_nums

        # ######  end parameters setting #########
        if ms.activity_threshold is None:
            activity_threshold = get_sce_detection_threshold(spike_nums=spike_nums_to_use,
                                                             window_duration=sliding_window_duration,
                                                             spike_train_mode=False,
                                                             n_surrogate=n_surrogate_activity_threshold,
                                                             perc_threshold=perc_threshold,
                                                             use_max_of_each_surrogate=use_max_of_each_surrogate,
                                                             debug_mode=True)
        else:
            activity_threshold = ms.activity_threshold

        print(f"perc_threshold {perc_threshold}, "
              f"activity_threshold {activity_threshold}, {np.round((activity_threshold/n_cells)*100, 2)}%")
        print(f"sliding_window_duration {sliding_window_duration}")
        spike_struct.activity_threshold = activity_threshold
        # param.activity_threshold = activity_threshold

        print("plot_spikes_raster")

        plot_spikes_raster(spike_nums=spike_nums_to_use, param=ms.param,
                           spike_train_format=False,
                           title=f"raster plot {data_descr}",
                           file_name=f"spike_nums_{data_descr}",
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
        sce_times_bool = sce_detection_result[0]
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
        if do_clustering:
            if do_fca_clustering:
                compute_and_plot_clusters_raster_fca_version(spike_trains=spike_struct.spike_trains,
                                                             spike_nums=spike_struct.spike_nums,
                                                             data_descr=data_descr, param=param,
                                                             sliding_window_duration=sliding_window_duration,
                                                             SCE_times=SCE_times, sce_times_numbers=sce_times_numbers,
                                                             perc_threshold=perc_threshold,
                                                             n_surrogate_activity_threshold=
                                                             n_surrogate_activity_threshold,
                                                             sigma=sigma, n_surrogate_fca=n_surrogate_fca,
                                                             labels=spike_struct.labels,
                                                             activity_threshold=activity_threshold,
                                                             fca_early_stop=True,
                                                             with_cells_in_cluster_seq_sorted=
                                                             with_cells_in_cluster_seq_sorted,
                                                             use_uniform_jittering=True)
            else:
                compute_and_plot_clusters_raster_kmean_version(labels=ms.spike_struct.labels,
                                                               activity_threshold=ms.spike_struct.activity_threshold,
                                                               range_n_clusters_k_mean=range_n_clusters_k_mean,
                                                               n_surrogate_k_mean=n_surrogate_k_mean,
                                                               with_shuffling=with_shuffling,
                                                               spike_nums_to_use=spike_nums_to_use,
                                                               cellsinpeak=cellsinpeak,
                                                               data_descr=data_descr,
                                                               param=ms.param,
                                                               sliding_window_duration=sliding_window_duration,
                                                               SCE_times=SCE_times, sce_times_numbers=sce_times_numbers,
                                                               sce_times_bool=sce_times_bool,
                                                               perc_threshold=perc_threshold,
                                                               n_surrogate_activity_threshold=
                                                               n_surrogate_activity_threshold,
                                                               debug_mode=debug_mode,
                                                               fct_to_keep_best_silhouettes=np.median,
                                                               with_cells_in_cluster_seq_sorted=
                                                               with_cells_in_cluster_seq_sorted,
                                                               keep_only_the_best=True)

        ###################################################################
        ###################################################################
        # ##############    Sequences detection        ###################
        ###################################################################
        ###################################################################

        if do_pattern_search:
            sce_times_bool_to_use = sce_times_bool if use_sce_times_for_pattern_search else None
            if split_pattern_search:
                n_splits = 5
                splits_indices = np.linspace(0, len(spike_nums_to_use[0, :]), n_splits + 1).astype(int)
                for split_id in np.arange(n_splits):
                    use_new_pattern_package(spike_nums=
                                            spike_nums_to_use[:, splits_indices[split_id]:splits_indices[split_id + 1]],
                                            param=param,
                                            activity_threshold=activity_threshold,
                                            sliding_window_duration=sliding_window_duration,
                                            n_surrogate=n_surrogate_for_pattern_search,
                                            mouse_id=ms.description, debug_mode=True,
                                            extra_file_name=f"part_{split_id+1}",
                                            sce_times_bool=sce_times_bool_to_use,
                                            use_only_uniformity_method=use_only_uniformity_method,
                                            use_loss_score_to_keep_the_best_from_tree=
                                            use_loss_score_to_keep_the_best_from_tree
                                            )

            else:
                # TODO: split spikes_nums in 3 to 4 part
                print("Start of use_new_pattern_package")
                use_new_pattern_package(spike_nums=spike_nums_to_use, param=param,
                                        activity_threshold=activity_threshold,
                                        sliding_window_duration=sliding_window_duration,
                                        n_surrogate=n_surrogate_for_pattern_search,
                                        mouse_id=ms.description, debug_mode=True,
                                        extra_file_name="",
                                        sce_times_bool=sce_times_bool_to_use,
                                        use_only_uniformity_method=use_only_uniformity_method,
                                        use_loss_score_to_keep_the_best_from_tree=
                                        use_loss_score_to_keep_the_best_from_tree)

    return


main()
