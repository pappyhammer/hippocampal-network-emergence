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


def sort_it_and_plot_it(spike_nums, param,
                        sliding_window_duration, activity_threshold, title_option="",
                        spike_train_format=False,
                        debug_mode=False,
                        plot_all_best_seq_by_cell=False):
    if spike_train_format:
        return
    result = pattern_discovery.seq_solver.markov_way.order_spike_nums_by_seq(spike_nums,
                                                                             param,
                                                                             debug_mode=debug_mode)
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
    return best_seq, seq_dict_tmp

    #### test for coloring sequences
    spike_nums_ordered = np.copy(spike_nums[best_seq, :])

    if debug_mode:
        print(f"best_seq {best_seq}")
    if seq_dict_tmp is not None:
        if debug_mode:
            for key, value in seq_dict_tmp.items():
                print(f"seq: {key}, rep: {len(value)}")

        best_seq_mapping_index = dict()
        for i, cell in enumerate(best_seq):
            best_seq_mapping_index[cell] = i
        # we need to replace the index by the corresponding one in best_seq
        seq_dict = dict()
        for key, value in seq_dict_tmp.items():
            new_key = []
            for cell in key:
                new_key.append(best_seq_mapping_index[cell])
            seq_dict[tuple(new_key)] = value

        seq_colors = dict()
        len_seq = len(seq_dict)
        if debug_mode:
            print(f"nb seq to colors: {len_seq}")
        for index, key in enumerate(seq_dict.keys()):
            seq_colors[key] = cm.nipy_spectral(float(index + 1) / (len_seq + 1))
            if debug_mode:
                print(f"color {seq_colors[key]}, len(seq) {len(key)}")
    else:
        seq_dict = None
        seq_colors = None
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

    plot_spikes_raster(spike_nums=spike_nums_ordered, param=param,
                       title=f"raster plot ordered {title_option}",
                       spike_train_format=False,
                       file_name=f"spike_nums_ordered_{title_option}",
                       y_ticks_labels=new_labels,
                       y_ticks_labels_size=5,
                       save_raster=True,
                       show_raster=False,
                       sliding_window_duration=sliding_window_duration,
                       show_sum_spikes_as_percentage=True,
                       plot_with_amplitude=False,
                       activity_threshold=activity_threshold,
                       save_formats="png",
                       seq_times_to_color_dict=seq_dict,
                       seq_colors=seq_colors)

    return best_seq, seq_dict_tmp


def use_new_pattern_package(spike_nums, param, activity_threshold, sliding_window_duration,
                            mouse_id, n_surrogate=2, debug_mode=False, without_raw_plot=True):
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
                           file_name=f"raw_spike_nums_{mouse_id}",
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

    best_seq, seq_dict = sort_it_and_plot_it(spike_nums=spike_nums, param=param,
                                             sliding_window_duration=sliding_window_duration,
                                             activity_threshold=activity_threshold,
                                             title_option=f"{mouse_id}",
                                             spike_train_format=False,
                                             debug_mode=debug_mode)

    nb_cells = len(spike_nums)

    print("#### REAL DATA ####")
    print(f"best_seq {best_seq}")
    real_data_result_for_stat = SortedDict()
    neurons_sorted_real_data = np.zeros(nb_cells, dtype="uint16")
    if seq_dict is not None:
        for key, value in seq_dict.items():
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

        best_seq, seq_dict = sort_it_and_plot_it(spike_nums=tmp_spike_nums, param=param,
                                                 sliding_window_duration=sliding_window_duration,
                                                 activity_threshold=activity_threshold,
                                                 title_option=f"surrogate {mouse_id}",
                                                 spike_train_format=False,
                                                 debug_mode=False)

        print(f"best_seq {best_seq}")

        mask = np.zeros(nb_cells, dtype="bool")
        if seq_dict is not None:
            for key, value in seq_dict.items():
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
                                        neurons_sorted_surrogate=neurons_sorted_surrogate_data)


def give_me_stat_on_sorting_seq_results(results_dict, neurons_sorted, title, param,
                                        results_dict_surrogate=None, neurons_sorted_surrogate=None):
    """
    Key will be the length of the sequence and value will be a list of int, representing the nb of rep
    of the different lists
    :param results_dict:
    :return:
    """
    file_name = f'{param.path_results}/sorting_results_{param.time_str}.txt'
    with open(file_name, "w", encoding='UTF-8') as file:
        file.write(f"{title}" + '\n')
        file.write("" + '\n')
        min_len = 1000
        max_len = 0
        for key in results_dict.keys():
            min_len = np.min((key, min_len))
            max_len = np.max((key, max_len))
        if results_dict_surrogate is not None:
            for key in results_dict_surrogate.keys():
                min_len = np.min((key, min_len))
                max_len = np.max((key, max_len))

        # key reprensents the length of a seq
        for key in np.arange(min_len, max_len + 1):
            nb_seq = None
            nb_seq_surrogate = None
            if key in results_dict:
                nb_seq = results_dict[key]
            if key in results_dict_surrogate:
                nb_seq_surrogate = results_dict_surrogate[key]
            str_to_write = ""
            str_to_write += f"### Length {key}: \n"
            real_data_in = False
            if nb_seq is not None:
                real_data_in = True
                str_to_write += f"# Real data: mean {np.round(np.mean(nb_seq), 3)}"
                if np.std(nb_seq) > 0:
                    str_to_write += f", std {np.round(np.std(nb_seq), 3)}"
            if nb_seq_surrogate is not None:
                if real_data_in:
                    str_to_write += f"\n"
                str_to_write += f"# Surrogate: mean {np.round(np.mean(nb_seq_surrogate), 3)}"
                if np.std(nb_seq_surrogate) > 0:
                    str_to_write += f", std {np.round(np.std(nb_seq_surrogate), 3)}"
            else:
                if not real_data_in:
                    continue
            str_to_write += '\n'
            file.write(f"{str_to_write}")
        file.write("" + '\n')
        file.write("///// Neurons sorted /////" + '\n')
        file.write("" + '\n')

        for index in np.arange(len(neurons_sorted)):
            go_for = False
            if neurons_sorted_surrogate is not None:
                if neurons_sorted_surrogate[index] == 0:
                    pass
                else:
                    go_for = True
            if (not go_for) and neurons_sorted[index] == 0:
                continue
            str_to_write = f"Neuron {index}, x "
            if neurons_sorted_surrogate is not None:
                str_to_write += f"{neurons_sorted_surrogate[index]} / "
            str_to_write += f"{neurons_sorted[index]}"
            if neurons_sorted_surrogate is not None:
                str_to_write += " (surrogate / real data)"
            str_to_write += '\n'
            file.write(f"{str_to_write}")


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

    do_clustering = True
    # if False, clustering will be done using kmean
    do_fca_clustering = False
    with_cells_in_cluster_seq_sorted = False

    # for events (sce) detection
    perc_threshold = 99
    n_surrogate_activity_threshold = 100

    # for fca
    n_surrogate_fca = 20

    # for kmean
    with_shuffling = True
    use_raster_dur = False
    print(f"use_raster_dur {use_raster_dur}")
    range_n_clusters_k_mean = np.arange(2, 20)
    n_surrogate_k_mean = 50

    do_pattern_search = False
    # seq params:
    param.error_rate = 5
    param.max_branches = 5

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
                                                             with_cells_in_cluster_seq_sorted)
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
                                                               perc_threshold=perc_threshold,
                                                               n_surrogate_activity_threshold=
                                                               n_surrogate_activity_threshold,
                                                               debug_mode=debug_mode,
                                                               with_cells_in_cluster_seq_sorted=
                                                               with_cells_in_cluster_seq_sorted)

        ###################################################################
        ###################################################################
        # ##############    Sequences detection        ###################
        ###################################################################
        ###################################################################

        if do_pattern_search:
            print("Start of use_new_pattern_package")
            use_new_pattern_package(spike_nums=spike_nums_to_use, param=param, activity_threshold=activity_threshold,
                                    sliding_window_duration=sliding_window_duration, n_surrogate=2,
                                    mouse_id=ms.description, debug_mode=True)

    return


main()
