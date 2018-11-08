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
# import copy
from datetime import datetime
# import keras
import os
# to add homemade package, go to preferences, then project interpreter, then click on the wheel symbol
# then show all, then select the interpreter and lick on the more right icon to display a list of folder and
# add the one containing the folder pattern_discovery
from pattern_discovery.seq_solver.markov_way import MarkovParameters
from pattern_discovery.seq_solver.markov_way import find_significant_patterns
import pattern_discovery.tools.misc as tools_misc
from pattern_discovery.display.raster import plot_spikes_raster
from pattern_discovery.display.cells_map_module import CoordClass
from pattern_discovery.tools.sce_detection import get_sce_detection_threshold, detect_sce_with_sliding_window, \
    get_low_activity_events_detection_threshold
from sortedcontainers import SortedList, SortedDict
from pattern_discovery.clustering.kmean_version.k_mean_clustering import compute_and_plot_clusters_raster_kmean_version
from pattern_discovery.clustering.kmean_version.k_mean_clustering import give_stat_one_sce
from pattern_discovery.clustering.fca.fca import compute_and_plot_clusters_raster_fca_version
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats


class HNEParameters(MarkovParameters):
    def __init__(self, path_results, time_str, time_inter_seq, min_duration_intra_seq, min_len_seq, min_rep_nb,
                 path_data,
                 max_branches, stop_if_twin, no_reverse_seq, error_rate, spike_rate_weight,
                 bin_size=1, cell_assemblies_data_path=None):
        super().__init__(time_inter_seq=time_inter_seq, min_duration_intra_seq=min_duration_intra_seq,
                         min_len_seq=min_len_seq, min_rep_nb=min_rep_nb, no_reverse_seq=no_reverse_seq,
                         max_branches=max_branches, stop_if_twin=stop_if_twin, error_rate=error_rate,
                         spike_rate_weight=spike_rate_weight,
                         bin_size=bin_size, path_results=path_results, time_str=time_str)
        self.path_data = path_data
        self.cell_assemblies_data_path = cell_assemblies_data_path
        # for plotting ages
        self.markers = ['o', '*', 's', 'v', '<', '>', '^', 'x', '+', "."]  # d losange
        self.colors = ["darkmagenta", "white", "saddlebrown", "blue", "red", "darkgrey", "chartreuse", "cornflowerblue",
                       "pink", "darkgreen", "gold"]


class MouseSession:
    def __init__(self, age, session_id, param, nb_ms_by_frame, weight=None, spike_nums=None, spike_nums_dur=None,
                 percentile_for_low_activity_threshold=1):
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
        self.low_activity_threshold_by_percentile = dict()
        self.percentile_for_low_activity_threshold = percentile_for_low_activity_threshold
        self.low_activity_threshold = None
        self.param = param
        # list of list of int representing cell indices
        self.cell_assemblies = None
        if self.param.cell_assemblies_data_path is not None:
            self.load_cell_assemblies_data()
        self.weight = weight
        self.coord_obj = None

    def load_cell_assemblies_data(self):
        file_names = []

        # look for filenames in the fisrst directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(self.param.cell_assemblies_data_path):
            file_names.extend(local_filenames)
            break
        if len(file_names) == 0:
            return

        for file_name in file_names:
            file_name_original = file_name
            file_name = file_name.lower()
            descr = self.description.lower()
            if not file_name.startswith(descr):
                continue
            self.cell_assemblies = []
            with open(self.param.cell_assemblies_data_path + file_name_original, "r", encoding='UTF-8') as file:
                param_section = False
                cell_ass_section = False
                for nb_line, line in enumerate(file):
                    if line.startswith("#PARAM#"):
                        param_section = True
                        continue
                    if line.startswith("#CELL_ASSEMBLIES#"):
                        cell_ass_section = True
                        param_section = False
                        continue
                    if cell_ass_section:
                        if line.startswith("SCA_cluster"):
                            cells = []
                            line_list = line.split(':')
                            cells = line_list[2].split(" ")
                            self.cell_assemblies.append([int(cell) for cell in cells])

    def plot_each_inter_neuron_connect_map(self):
        # plot n_in and n_out map of the interneurons
        inter_neurons = self.spike_struct.inter_neurons
        n_inter_neurons = len(inter_neurons)
        if n_inter_neurons == 0:
            return

        for inter_neuron in inter_neurons:
            color_each_cells_link_to_interneuron = True

            connections_dict_in = dict()
            connections_dict_out = dict()
            n_in_matrix = self.spike_struct.n_in_matrix
            n_out_matrix = self.spike_struct.n_out_matrix
            at_least_on_in_link = False
            at_least_on_out_link = False

            connections_dict_in[inter_neuron] = dict()
            connections_dict_out[inter_neuron] = dict()

            for cell in np.where(n_in_matrix[inter_neuron, :])[0]:
                at_least_on_in_link = True
                connections_dict_in[inter_neuron][cell] = 1

            for cell in np.where(n_out_matrix[inter_neuron, :])[0]:
                at_least_on_out_link = True
                connections_dict_out[inter_neuron][cell] = 1

            cells_groups_colors = ["red"]
            cells_groups = [[inter_neuron]]
            if at_least_on_in_link and color_each_cells_link_to_interneuron:
                links_cells = list(connections_dict_in[inter_neuron].keys())
                # removing fellow inter_neurons
                links_cells = np.setdiff1d(np.array(links_cells), np.array(inter_neurons))
                if len(links_cells) > 0:
                    cells_groups.append(list(connections_dict_in[inter_neuron].keys()))
                    cells_groups_colors.append("cornflowerblue")

            self.coord_obj.compute_center_coord(cells_groups=cells_groups,
                                                cells_groups_colors=cells_groups_colors)

            self.coord_obj.plot_cells_map(param=self.param,
                                          data_id=self.description, show_polygons=False,
                                          title_option=f"n_in_interneuron_{inter_neuron}",
                                          connections_dict=connections_dict_in,
                                          with_cell_numbers=True)

            cells_groups_colors = ["red"]
            cells_groups = [[inter_neuron]]
            if at_least_on_out_link and color_each_cells_link_to_interneuron:
                links_cells = list(connections_dict_out[inter_neuron].keys())
                # removing fellow inter_neurons
                links_cells = np.setdiff1d(np.array(links_cells), np.array(inter_neurons))
                if len(links_cells) > 0:
                    cells_groups.append(list(connections_dict_out[inter_neuron].keys()))
                    cells_groups_colors.append("cornflowerblue")

            self.coord_obj.compute_center_coord(cells_groups=cells_groups,
                                                cells_groups_colors=cells_groups_colors)

            self.coord_obj.plot_cells_map(param=self.param,
                                          data_id=self.description, show_polygons=False,
                                          title_option=f"n_out_interneuron_{inter_neuron}",
                                          connections_dict=connections_dict_out,
                                          with_cell_numbers=True)

    def plot_all_inter_neurons_connect_map(self):
        # plot n_in and n_out map of the interneurons
        inter_neurons = self.spike_struct.inter_neurons
        n_inter_neurons = len(inter_neurons)
        if n_inter_neurons == 0:
            return

        color_each_cells_link_to_interneuron = True

        connections_dict_in = dict()
        connections_dict_out = dict()
        n_in_matrix = self.spike_struct.n_in_matrix
        n_out_matrix = self.spike_struct.n_out_matrix
        at_least_on_in_link = False
        at_least_on_out_link = False
        for inter_neuron in inter_neurons:
            connections_dict_in[inter_neuron] = dict()
            connections_dict_out[inter_neuron] = dict()

            for cell in np.where(n_in_matrix[inter_neuron, :])[0]:
                at_least_on_in_link = True
                connections_dict_in[inter_neuron][cell] = 1

            for cell in np.where(n_out_matrix[inter_neuron, :])[0]:
                at_least_on_out_link = True
                connections_dict_out[inter_neuron][cell] = 1

        cells_groups_colors = ["red"]
        cells_groups = [inter_neurons]
        if at_least_on_in_link and color_each_cells_link_to_interneuron:
            for index_inter_neuron, inter_neuron in enumerate(inter_neurons):
                links_cells = list(connections_dict_in[inter_neuron].keys())
                # removing fellow inter_neurons
                links_cells = np.setdiff1d(np.array(links_cells), np.array(inter_neurons))
                if len(links_cells) > 0:
                    cells_groups.append(list(connections_dict_in[inter_neuron].keys()))
                    cells_groups_colors.append(cm.nipy_spectral(float(index_inter_neuron + 1) / (n_inter_neurons + 1)))

        self.coord_obj.compute_center_coord(cells_groups=cells_groups,
                                            cells_groups_colors=cells_groups_colors)

        self.coord_obj.plot_cells_map(param=self.param,
                                      data_id=self.description, show_polygons=False,
                                      title_option=f"n_in_interneurons_x_{n_inter_neurons}",
                                      connections_dict=connections_dict_in,
                                      with_cell_numbers=True)

        cells_groups_colors = ["red"]
        cells_groups = [inter_neurons]
        if at_least_on_out_link and color_each_cells_link_to_interneuron:
            for index_inter_neuron, inter_neuron in enumerate(inter_neurons):
                links_cells = list(connections_dict_out[inter_neuron].keys())
                # removing fellow inter_neurons
                links_cells = np.setdiff1d(np.array(links_cells), np.array(inter_neurons))
                if len(links_cells) > 0:
                    cells_groups.append(list(connections_dict_out[inter_neuron].keys()))
                    cells_groups_colors.append(cm.nipy_spectral(float(index_inter_neuron + 1) / (n_inter_neurons + 1)))

        self.coord_obj.compute_center_coord(cells_groups=cells_groups,
                                            cells_groups_colors=cells_groups_colors)

        self.coord_obj.plot_cells_map(param=self.param,
                                      data_id=self.description, show_polygons=False,
                                      title_option=f"n_out_interneurons_x_{n_inter_neurons}",
                                      connections_dict=connections_dict_out,
                                      with_cell_numbers=True)

    def plot_cell_assemblies_on_map(self):
        if self.cell_assemblies is None:
            return

        n_assemblies = len(self.cell_assemblies)
        cells_groups_colors = []
        for i in np.arange(n_assemblies):
            # print(f"cm.nipy_spectral(float(i + 1) / (n_assemblies + 1)) "
            #       f"{cm.nipy_spectral(float(i + 1) / (n_assemblies + 1))}")
            cells_groups_colors.append(cm.nipy_spectral(float(i + 1) / (n_assemblies + 1)))

        self.coord_obj.compute_center_coord(cells_groups=self.cell_assemblies,
                                            cells_groups_colors=cells_groups_colors)

        self.coord_obj.plot_cells_map(param=self.param,
                                      data_id=self.description, show_polygons=True,
                                      title_option="cell_assemblies", connections_dict=None,
                                      with_cell_numbers=True)

    def set_low_activity_threshold(self, threshold, percentile_value):
        self.low_activity_threshold_by_percentile[percentile_value] = threshold
        if self.percentile_for_low_activity_threshold in self.low_activity_threshold_by_percentile:
            self.low_activity_threshold = \
                self.low_activity_threshold_by_percentile[self.percentile_for_low_activity_threshold]

    def set_inter_neurons(self, inter_neurons):
        self.spike_struct.inter_neurons = np.array(inter_neurons).astype(int)

    def load_data_from_file(self, file_name_to_load, variables_mapping, frames_filter=None):
        """

        :param file_name_to_load:
        :param variables_mapping:
        :param frames_filter: if not None, will keep only the frames in frames_filter
        :return:
        """
        data = hdf5storage.loadmat(self.param.path_data + file_name_to_load)
        if "spike_nums" in variables_mapping:
            self.spike_struct.spike_nums = data[variables_mapping["spike_nums"]].astype(int)
            if frames_filter is not None:
                self.spike_struct.spike_nums = self.spike_struct.spike_nums[:, frames_filter]
            if self.spike_struct.labels is None:
                self.spike_struct.labels = np.arange(len(self.spike_struct.spike_nums))
            if self.spike_struct.n_cells is None:
                self.spike_struct.n_cells = len(self.spike_struct.spike_nums)
            if self.spike_struct.n_in_matrix is None:
                self.spike_struct.n_in_matrix = np.zeros((self.spike_struct.n_cells, self.spike_struct.n_cells))
                self.spike_struct.n_out_matrix = np.zeros((self.spike_struct.n_cells, self.spike_struct.n_cells))
        if "spike_nums_dur" in variables_mapping:
            self.spike_struct.spike_nums_dur = data[variables_mapping["spike_nums_dur"]].astype(int)
            if frames_filter is not None:
                self.spike_struct.spike_nums_dur = self.spike_struct.spike_nums_dur[:, frames_filter]
            if self.spike_struct.labels is None:
                self.spike_struct.labels = np.arange(len(self.spike_struct.spike_nums_dur))
            if self.spike_struct.n_cells is None:
                self.spike_struct.n_cells = len(self.spike_struct.spike_nums_dur)
            if self.spike_struct.n_in_matrix is None:
                self.spike_struct.n_in_matrix = np.zeros((self.spike_struct.n_cells, self.spike_struct.n_cells))
                self.spike_struct.n_out_matrix = np.zeros((self.spike_struct.n_cells, self.spike_struct.n_cells))
        if "traces" in variables_mapping:
            self.traces = data[variables_mapping["traces"]].astype(float)
            if frames_filter is not None:
                self.traces = self.traces[:, frames_filter]
        if "coord" in variables_mapping:
            self.coord = data[variables_mapping["coord"]][0]
            self.coord_obj = CoordClass(coord=self.coord, nb_col=200,
                                        nb_lines=200)
            # print(f"self.coord {self.coord}")
        if "spike_durations" in variables_mapping:
            self.spike_struct.set_spike_durations(data[variables_mapping["spike_durations"]])
        elif self.spike_struct.spike_nums_dur is not None:
            self.spike_struct.set_spike_durations()
        if "spike_amplitudes" in variables_mapping:
            self.spike_struct.set_spike_amplitudes(data[variables_mapping["spike_amplitudes"]])

        self.spike_struct.set_spike_trains_from_spike_nums()

        # if (self.spike_struct.spike_nums_dur is not None) or (self.spike_struct.spike_nums is not None):
        #     self.detect_n_in_n_out()

    def detect_n_in_n_out(self):
        self.spike_struct.detect_n_in_n_out()


class HNESpikeStructure:

    def __init__(self, mouse_session, labels=None, spike_nums=None, spike_trains=None,
                 spike_nums_dur=None, activity_threshold=None,
                 title=None, ordered_indices=None, ordered_spike_data=None):
        self.mouse_session = mouse_session
        self.spike_nums = spike_nums
        self.spike_nums_dur = spike_nums_dur
        if (self.spike_nums is not None) or (self.spike_nums_dur is not None):
            if self.spike_nums is not None:
                self.n_cells = len(self.spike_nums)
            else:
                self.n_cells = len(self.spike_nums_dur)
        else:
            self.n_cells = None
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

        # nb frames (1 frame == 100 ms) to look for connection near a neuron that spike
        self.nb_frames_for_func_connect = 5
        # contain the list of neurons connected to the EB as keys, and the number of connection as values
        # first key is a dict of neuron, the second key is other neurons to which the first connect,
        # then the number of times is connected to it
        self.n_in_dict = dict()
        self.n_out_dict = dict()
        self.n_in_matrix = None
        self.n_out_matrix = None

    def detect_n_in_n_out(self):
        # look neuron by neuron, at each spike and make a pair wise for each other neurons according to the spike
        # distribution around 500ms before and after. If the distribution is not uniform then we look where is the max
        # and we add it to n_out or n_in if before or after. If it is at the same time, then we don't add it.
        nb_neurons = len(self.spike_nums)
        n_times = len(self.spike_nums[0, :])
        for neuron in np.arange(nb_neurons):
            self.n_in_dict[neuron] = dict()
            self.n_out_dict[neuron] = dict()
            neurons_to_consider = np.arange(len(self.spike_nums))
            mask = np.ones(len(self.spike_nums), dtype="bool")
            mask[neuron] = False
            neurons_to_consider = neurons_to_consider[mask]
            # look at onsets
            neuron_spikes, = np.where(self.spike_nums[neuron, :])
            # is_early_born = (neuron == ms.early_born_cell)

            if len(neuron_spikes) == 0:
                continue

            spike_nums_to_use = self.spike_nums

            distribution_array_2_d = np.zeros((nb_neurons, ((self.nb_frames_for_func_connect * 2) + 1)),
                                              dtype="int16")

            event_index = self.nb_frames_for_func_connect
            # looping on each spike of the main neuron
            for n, event in enumerate(neuron_spikes):
                # only taking in consideration events that are not too close from bottom range or upper range
                min_limit = max(event - self.nb_frames_for_func_connect, 0)
                max_limit = min((event + self.nb_frames_for_func_connect), (n_times - 1))
                mask = np.zeros((nb_neurons, ((self.nb_frames_for_func_connect * 2) + 1)),
                                dtype="bool")
                mask_start = 0
                if (event - self.nb_frames_for_func_connect) < 0:
                    mask_start = -1 * (event - self.nb_frames_for_func_connect)
                mask_end = mask_start + (max_limit - min_limit) + 1
                mask[:, mask_start:mask_end] = spike_nums_to_use[:, min_limit:(max_limit + 1)] > 0
                distribution_array_2_d[mask] += 1

            # going neuron by neuron
            for neuron_to_consider in neurons_to_consider:
                distribution_array = distribution_array_2_d[neuron_to_consider, :]
                distribution_for_test = np.zeros(np.sum(distribution_array))
                frames_time = np.arange(-self.nb_frames_for_func_connect, self.nb_frames_for_func_connect+1)
                i_n = 0
                for i_time, sum_spike in enumerate(distribution_array):
                    if sum_spike > 0:
                        distribution_for_test[i_n:i_n+sum_spike] = frames_time[i_time]
                        i_n += sum_spike
                if len(distribution_for_test) >= 20:
                    stat_n, p_value = stats.normaltest(distribution_for_test)
                    ks, p_ks = stats.kstest(distribution_for_test, stats.randint.cdf,
                                            args=(np.min(distribution_for_test),
                                                  np.max(distribution_for_test)))
                    is_normal_distribution = p_value >= 0.05
                    is_uniform_distribution = p_ks >= 0.05
                    # if the distribution is normal or uniform, we skip it
                    if is_normal_distribution or is_uniform_distribution:
                        continue

                n_in_sum = np.sum(distribution_array[:event_index])
                n_out_sum = np.sum(distribution_array[(event_index + 1):])
                # means we have the same number of spikes before and after
                if n_in_sum == n_out_sum:
                    continue
                max_value = max(n_in_sum, n_out_sum)
                min_value = min(n_in_sum, n_out_sum)
                # we should have at least twice more spikes on one side
                if max_value < (min_value * 2):
                    continue
                # and twice as more as the spikes at time 0
                if max_value < (distribution_array[event_index] * 2):
                    continue

                if n_in_sum > n_out_sum:
                    self.n_in_dict[neuron][neuron_to_consider] = 1
                    self.n_in_matrix[neuron][neuron_to_consider] = 1
                else:
                    self.n_out_dict[neuron][neuron_to_consider] = 1
                    self.n_out_matrix[neuron][neuron_to_consider] = 1

                # show_bar_chart = False
                # if show_bar_chart and len(distribution_for_test) >= 20:
                #     if (np.max(distribution_array) > 3):
                #         # if (neuron == self.early_born_cell):
                #         print(f'neuron {neuron}, neuron_to_consider {neuron_to_consider}')
                #
                #         print(f"### distribution_for_normal_test {distribution_for_test}")
                #
                #         print(f"stat_n {stat_n}, p_value {p_value} "
                #               f"{'Non normal distribution' if (p_value < 0.05) else 'Normal distribution'}")
                #
                #         print(f"ks {ks}, p_ks {p_ks} "
                #               f"{'Non uniform distribution' if (p_ks < 0.05) else 'Uniform distribution'}")
                #
                #         print("")
                #         plt.bar(np.arange(-1 * self.nb_frames_for_func_connect, self.nb_frames_for_func_connect + 1),
                #                 distribution_array, color="blue")
                #         plt.title(f"neuron {neuron} vs neuron {neuron_to_consider} "
                #                   f"{'Non normal distribution' if (p_value < 0.05) else 'Normal distribution'}, "
                #                   f"{'Non uniform distribution' if (p_ks < 0.05) else 'Uniform distribution'}")
                #         plt.xlabel("Cell")
                #         plt.ylabel("Nb of spikes")
                #         plt.show()
                #         plt.close()

    def set_spike_durations(self, spike_durations_array=None):
        if self.spike_durations is not None:
            return

        self.spike_durations = []
        n_cells = 0
        if self.spike_nums is not None:
            n_cells = len(self.spike_nums)
        elif self.spike_nums_dur is not None:
            n_cells = len(self.spike_nums_dur)
        if n_cells == 0:
            print("set_spike_durations no cell")
            return

        avg_spike_duration_by_cell = np.zeros(n_cells)

        if self.spike_nums_dur is None:
            for cell_id, spikes_d in enumerate(spike_durations_array):
                self.spike_durations.append(spikes_d[spikes_d > 0])
                if len(self.spike_durations[-1]) > 0:
                    avg_spike_duration_by_cell[cell_id] = np.mean(self.spike_durations[-1])
                else:
                    avg_spike_duration_by_cell[cell_id] = 0
        else:
            self.spike_durations = tools_misc.get_spikes_duration_from_raster_dur(spike_nums_dur=self.spike_nums_dur)
            # test
            # if spike_durations_array is not None:
            #     # test_spike_duration = []
            #     for cell_id, spikes_d in enumerate(spike_durations_array):
            #         # test_spike_duration.append(spikes_d[spikes_d > 0])
            #         print(f"cell {cell_id}")
            #         print(f"diff: {np.array(self.spike_durations[cell_id])}")
            #         print(f"Robin: {spikes_d[spikes_d > 0]}")
            #         print(f"raster: {np.where(self.spike_nums_dur[cell_id, :])[0]}")
            #         print("")
            #     raise Exception()

            for cell_id, spike_duration in enumerate(self.spike_durations):
                if len(spike_duration) > 0:
                    avg_spike_duration_by_cell[cell_id] = np.mean(spike_duration)
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
        if self.spike_nums is None:
            return
        self.spike_trains = []
        for cell_spikes in self.spike_nums:
            self.spike_trains.append(np.where(cell_spikes)[0].astype(float))


def connec_func_stat(mouse_sessions, data_descr, param):
    # print(f"connec_func_stat {mouse_session.session_numbers[0]}")
    interneurons_pos = np.zeros(0, dtype="uint16")
    total_nb_neurons = 0
    for ms in mouse_sessions:
        total_nb_neurons += ms.spike_struct.n_cells
    n_outs_total = np.zeros(total_nb_neurons)
    n_ins_total = np.zeros(total_nb_neurons)
    neurons_count_so_far = 0
    for ms_nb, ms in enumerate(mouse_sessions):
        nb_neurons = ms.spike_struct.n_cells
        n_ins = np.sum(ms.spike_struct.n_in_matrix, axis=1) / nb_neurons
        n_ins = np.round(n_ins * 100, 2)
        n_outs = np.sum(ms.spike_struct.n_out_matrix, axis=1) / nb_neurons
        n_outs = np.round(n_outs * 100, 2)
        if len(ms.spike_struct.inter_neurons) > 0:
            interneurons_pos = np.concatenate((interneurons_pos,
                                               np.array(ms.spike_struct.inter_neurons) + neurons_count_so_far))
        n_ins_total[neurons_count_so_far:(neurons_count_so_far + nb_neurons)] = n_ins
        n_outs_total[neurons_count_so_far:(neurons_count_so_far + nb_neurons)] = n_outs
        neurons_count_so_far += nb_neurons

    values_to_scatter = []
    labels = []
    scatter_shapes = []
    colors = []
    values_to_scatter.append(np.mean(n_outs_total))
    values_to_scatter.append(np.median(n_outs_total))
    labels.extend(["mean", "median"])
    scatter_shapes.extend(["o", "s"])
    colors.extend(["white", "white"])
    if len(interneurons_pos) > 0:
        values_to_scatter.extend(list(n_outs_total[interneurons_pos]))
        labels.extend([f"interneuron (x{len(interneurons_pos)})"])
        scatter_shapes.extend(["*"]*len(n_outs_total[interneurons_pos]))
        colors.extend(["red"]*len(n_outs_total[interneurons_pos]))
    # TODO add mean and median

    plot_hist_ratio_spikes_events(ratio_spikes_events=n_outs_total,
                                  description=f"{data_descr}_distribution_n_out",
                                  values_to_scatter=np.array(values_to_scatter),
                                  labels=labels,
                                  scatter_shapes=scatter_shapes,
                                  colors=colors,
                                  tight_x_range=True,
                                  xlabel="Active cells (%)",
                                  ylabel="Probability distribution (%)",
                                  param=param)

    values_to_scatter = []
    scatter_shapes = []
    colors = []
    values_to_scatter.append(np.mean(n_ins_total))
    values_to_scatter.append(np.median(n_ins_total))
    labels.extend(["mean", "median"])
    scatter_shapes.extend(["o", "s"])
    colors.extend(["white", "white"])
    if len(interneurons_pos) > 0:
        values_to_scatter.extend(list(n_ins_total[interneurons_pos]))
        scatter_shapes.extend(["*"]*len(n_ins_total[interneurons_pos]))
        colors.extend(["red"]*len(n_ins_total[interneurons_pos]))

    plot_hist_ratio_spikes_events(ratio_spikes_events=n_ins_total,
                                  description=f"{data_descr}_distribution_n_in",
                                  values_to_scatter=np.array(values_to_scatter),
                                  labels=labels,
                                  scatter_shapes=scatter_shapes,
                                  colors=colors,
                                  tight_x_range=True,
                                  xlabel="Active cells (%)",
                                  ylabel="Probability distribution (%)",
                                  param=param)
    return n_ins_total, n_outs_total


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
        if age not in duration_spikes_by_age:
            y_data[index] = 1
            high_percentile_participation[index] = 1
            low_percentile_participation[index] = 1
            continue

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
                                  labels, scatter_shapes, colors, param, tight_x_range=False,
                                  xlabel="", ylabel=None, save_formats="pdf"):
    distribution = np.array(ratio_spikes_events)
    hist_color = "blue"
    edge_color = "white"
    if tight_x_range:
        max_range = np.max(distribution)
        min_range = np.min(distribution)
    else:
        max_range = 100
        min_range = 0
    weights = (np.ones_like(distribution) / (len(distribution))) * 100

    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(12, 12))
    ax1.set_facecolor("black")
    bins = int(np.sqrt(len(distribution)))
    hist_plt, edges_plt, patches_plt = plt.hist(distribution, bins=bins, range=(min_range, max_range),
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
    # if tight_x_range:
    #     plt.xlim(min_range, max_range)
    plt.xlim(0, 100)
    xticks = np.arange(0, 110, 10)
    ax1.set_xticks(xticks)
    # sce clusters labels
    ax1.set_xticklabels(xticks)

    if ylabel is None:
        ax1.set_ylabel("Distribution (%)")
    else:
        ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)

    ax1.legend()

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/{description}'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")

    plt.close()


def compute_stat_about_significant_seq(files_path, param, save_formats="pdf"):
    file_names = []
    n_categories = 4
    marker_cat = ["*", "d", "o", "s"]
    # categories that should be displayed
    banned_categories = []
    # look for filenames in the fisrst directory, if we don't break, it will go through all directories
    for (dirpath, dirnames, local_filenames) in os.walk(files_path):
        file_names.extend(local_filenames)
        break

    # dict1: key age (int) value dict2
    # dict2: key category (int, from 1 to 4), value dict3
    # dict3: key length seq (int), value dict4
    # dict4: key repetitions (int), value nb of seq with this length and this repetition
    data_dict = SortedDict()
    # key is an int representing age, and value is an int representing the number of sessions for a given age
    nb_ms_by_age = dict()

    for file_name in file_names:
        file_name_original = file_name
        if not file_name.startswith("p"):
            if not file_name.startswith("significant_sorting_results"):
                continue
            # p_index = len("significant_sorting_results")+1
            file_name = file_name[len("significant_sorting_results"):]
        index_ = file_name.find("_")
        if index_ < 1:
            continue
        age = int(file_name[1:index_])
        nb_ms_by_age[age] = nb_ms_by_age.get(age, 0) + 1
        # print(f"age {age}")
        if age not in data_dict:
            data_dict[age] = dict()
            for cat in np.arange(1, 1 + n_categories):
                data_dict[age][cat] = dict()

        with open(files_path + file_name_original, "r", encoding='UTF-8') as file:
            for nb_line, line in enumerate(file):
                line_list = line.split(':')
                seq_n_cells = int(line_list[0])
                line_list = line_list[1].split("]")
                # we remove the '[' on the first position
                repetitions_str = line_list[0][1:].split(",")
                repetitions = []
                for rep in repetitions_str:
                    repetitions.append(int(rep))
                # we remove the ' [' on the first position
                categories_str = line_list[1][2:].split(",")
                categories = []
                for cat in categories_str:
                    categories.append(int(cat))

                for index, cat in enumerate(categories):
                    if seq_n_cells not in data_dict[age][cat]:
                        data_dict[age][cat][seq_n_cells] = dict()

                    rep = repetitions[index]
                    if rep not in data_dict[age][cat][seq_n_cells]:
                        data_dict[age][cat][seq_n_cells][rep] = 0
                    data_dict[age][cat][seq_n_cells][rep] += 1
                # print(f"{seq_n_cells} cells: {repetitions} {categories}")

    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(15, 15))
    ax1.set_facecolor("black")

    n_jitter = 10
    jitter_range_x = 0.4
    indices_rand = np.linspace(-jitter_range_x, jitter_range_x, n_jitter)
    jitter_range_y = 0.4
    indices_rand_y = np.linspace(-jitter_range_y, jitter_range_y, n_jitter)
    np.random.shuffle(indices_rand)
    np.random.shuffle(indices_rand_y)

    age_index = 0
    i_jitter = 0
    max_rep = 0
    min_rep = 100
    min_len = 100
    max_len = 0
    len_grid = tuple((3, 5))
    # key is a len, value dict2
    #  dict 2: key is rep, and value is np.array of n*n dimension, bool
    grid_dict = dict()
    for age, cat_dict in data_dict.items():
        for cat, len_dict in cat_dict.items():
            if cat in banned_categories:
                continue
            for len_seq, rep_dict in len_dict.items():
                if len_seq not in grid_dict:
                    grid_dict[len_seq] = dict()
                min_len = np.min((len_seq, min_len))
                max_len = np.max((len_seq, max_len))
                for rep, n_seq in rep_dict.items():
                    if rep not in grid_dict[len_seq]:
                        grid_dict[len_seq][rep] = np.ones(len_grid[0] * len_grid[1], dtype="bool")
                    max_rep = np.max((max_rep, rep))
                    min_rep = np.min((min_rep, rep))
                    grid = grid_dict[len_seq][rep]
                    free_spots = np.where(grid)[0]
                    if len(free_spots) == 0:
                        print("error no more free spots")
                        return
                    np.random.shuffle(free_spots)
                    spot = free_spots[0]
                    grid[spot] = False
                    grid_pos_y = int(spot / len_grid[1])
                    grid_pos_x = spot % len_grid[1]
                    x_pos = len_seq - 0.4 + (grid_pos_x * (0.8 / (len_grid[1] - 1)))
                    y_pos = rep - 0.35 + (grid_pos_y * (0.7 / (len_grid[0] - 1)))
                    n_seq_normalized = np.round((n_seq / nb_ms_by_age[age]), 1)
                    if n_seq_normalized % 1 == 0:
                        n_seq_normalized = int(n_seq_normalized)
                    ax1.scatter(x_pos,
                                y_pos,
                                color=param.colors[age_index % (len(param.colors))],
                                marker=param.markers[cat - 1],
                                s=15 + 5 * np.sqrt(n_seq_normalized), alpha=1, edgecolors='none')
                    ax1.text(x=x_pos, y=y_pos,
                             s=f"{n_seq_normalized}", color="black", zorder=22,
                             ha='center', va="center", fontsize=0.9, fontweight='bold')
                    i_jitter += 1
        age_index += 1

    for len_seq in np.arange(min_len, max_len + 2):
        ax1.vlines(len_seq - 0.5, 0,
                   max_rep + 1, color="white", linewidth=0.5,
                   linestyles="dashed", zorder=1)
    for rep in np.arange(min_rep, max_rep + 2):
        ax1.hlines(rep - 0.5, min_len - 1,
                   max_len + 1, color="white", linewidth=0.5,
                   linestyles="dashed", zorder=1)
    legend_elements = []
    # [Line2D([0], [0], color='b', lw=4, label='Line')
    age_index = 0
    for age, cat_dict in data_dict.items():
        legend_elements.append(Patch(facecolor=param.colors[age_index % (len(param.colors))],
                                     edgecolor='black', label=f'p{age}'))
        age_index += 1

    for cat in np.arange(1, n_categories + 1):
        if cat in banned_categories:
            continue
        legend_elements.append(Line2D([0], [0], marker=param.markers[cat - 1], color="w", lw=0, label="*" * cat,
                                      markerfacecolor='black', markersize=15))

    ax1.legend(handles=legend_elements)

    # plt.title(title)
    ax1.set_ylabel(f"Repetition (#)")
    ax1.set_xlabel("length")
    ax1.set_ylim(min_rep - 0.5, max_rep + 0.5)
    ax1.set_xlim(min_len - 0.5, max_len + 0.5)
    # xticks = np.arange(0, len(data_dict))
    # ax1.set_xticks(xticks)
    # # sce clusters labels
    # ax1.set_xticklabels(labels)

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/scatter_significant_seq'
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
    xticks = np.arange(1, len(data_dict) + 1)
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
    if spike_nums_dur is None:
        return
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
        if spike_nums_dur is not None:
            spikes_index = np.where(spike_nums_dur[cell, :])[0]
        else:
            spikes_index = np.where(spike_nums[cell, :])[0]
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
        if spike_nums_dur is not None:
            spikes_index = np.where(spike_nums_dur[cell, :])[0]
        else:
            spikes_index = np.where(spike_nums[cell, :])[0]

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
    cell_assemblies_data_path = path_data + "cell_assemblies/v1/"

    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    path_results = path_results_raw + f"{time_str}"
    os.mkdir(path_results)

    # --------------------------------------------------------------------------------
    # ------------------------------ param section ------------------------------
    # --------------------------------------------------------------------------------

    # param will be set later when the spike_nums will have been constructed
    param = HNEParameters(time_str=time_str, path_results=path_results, error_rate=2,
                          cell_assemblies_data_path=cell_assemblies_data_path,
                          time_inter_seq=50, min_duration_intra_seq=-3, min_len_seq=10, min_rep_nb=4,
                          max_branches=20, stop_if_twin=False,
                          no_reverse_seq=False, spike_rate_weight=False, path_data=path_data)

    just_compute_significant_seq_stat = False
    if just_compute_significant_seq_stat:
        compute_stat_about_significant_seq(files_path=f"{path_data}significant_seq/v3/", param=param)
        return

    # test
    # p11_17_11_24_a000_ms = MouseSession(age=11, session_id="17_11_24_a000", nb_ms_by_frame=100, param=param,
    #                                     weight=6.7)
    # # calculated with 99th percentile on raster dur
    # p11_17_11_24_a000_ms.activity_threshold = 12
    # p11_17_11_24_a000_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
    # p11_17_11_24_a000_ms.set_inter_neurons([193])
    # # duration of those interneurons: 19.09
    # variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
    #                      "spike_nums": "filt_Bin100ms_spikedigital",
    #                      "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    # p11_17_11_24_a000_ms.load_data_from_file(file_name_to_load="p11/p11_17_11_24_a000/p11_17_11_24_a000_RasterDur.mat",
    #                                          variables_mapping=variables_mapping)
    # variables_mapping = {"coord": "ContoursAll"}
    # p11_17_11_24_a000_ms.load_data_from_file(file_name_to_load="p11/p11_17_11_24_a000/p11_17_11_24_a000_CellDetect.mat",
    #                                          variables_mapping=variables_mapping)
    # # connec_func_stat([p11_17_11_24_a000_ms], data_descr="p11_17_11_24_a000", param=param)
    # p11_17_11_24_a000_ms.plot_inter_neurons_connect_map()
    # # p11_17_11_24_a000_ms.plot_cell_assemblies_on_map()
    # return

    # loading data
    p6_18_02_07_a001_ms = MouseSession(age=6, session_id="18_02_07_a001", nb_ms_by_frame=100, param=param,
                                       weight=4.35)
    # calculated with 99th percentile on raster dur
    p6_18_02_07_a001_ms.activity_threshold = 17
    p6_18_02_07_a001_ms.set_low_activity_threshold(threshold=3, percentile_value=1)
    p6_18_02_07_a001_ms.set_low_activity_threshold(threshold=5, percentile_value=5)
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
    p6_18_02_07_a002_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
    p6_18_02_07_a002_ms.set_low_activity_threshold(threshold=1, percentile_value=5)
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
    p7_171012_a000_ms.activity_threshold = 19
    p7_171012_a000_ms.set_low_activity_threshold(threshold=6, percentile_value=1)
    # p7_171012_a000_ms.set_low_activity_threshold(threshold=7, percentile_value=5)
    p7_171012_a000_ms.set_inter_neurons([305, 360, 398, 412])
    # duration of those interneurons: 13.23  12.48  10.8   11.88
    variables_mapping = {"spike_nums_dur": "rasterdur",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3"}
    p7_171012_a000_ms.load_data_from_file(
        file_name_to_load="p7/p7_17_10_12_a000/p7_17_10_12_a000_Corrected_RasterDur.mat",
        variables_mapping=variables_mapping)
    # variables_mapping = {"coord": "ContoursAll"} ContoursSoma ContoursIntNeur
    # p7_171012_a000_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_12_a000/p7_17_10_12_a000_CellDetect.mat",
    #                                          variables_mapping=variables_mapping)

    p7_18_02_08_a000_ms = MouseSession(age=7, session_id="18_02_08_a000", nb_ms_by_frame=100, param=param,
                                       weight=3.85)
    # calculated with 99th percentile on raster dur
    p7_18_02_08_a000_ms.activity_threshold = 11
    p7_18_02_08_a000_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
    p7_18_02_08_a000_ms.set_low_activity_threshold(threshold=2, percentile_value=5)
    p7_18_02_08_a000_ms.set_inter_neurons([56, 95, 178])
    # duration of those interneurons: 12.88  13.94  13.04
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p7_18_02_08_a000_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a000/p7_18_02_08_a000_RasterDur.mat",
                                            variables_mapping=variables_mapping)
    variables_mapping = {"coord": "ContoursAll"}
    p7_18_02_08_a000_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a000/p7_18_02_08_a000_CellDetect.mat",
                                            variables_mapping=variables_mapping)


    p7_18_02_08_a001_ms = MouseSession(age=7, session_id="18_02_08_a001", nb_ms_by_frame=100, param=param,
                                       weight=3.85)
    # calculated with 99th percentile on raster dur
    p7_18_02_08_a001_ms.activity_threshold = 13
    p7_18_02_08_a001_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
    p7_18_02_08_a001_ms.set_low_activity_threshold(threshold=3, percentile_value=5)
    p7_18_02_08_a001_ms.set_inter_neurons([151])
    # duration of those interneurons: 22.11
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p7_18_02_08_a001_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a001/p7_18_02_08_a001_RasterDur.mat",
                                            variables_mapping=variables_mapping)
    variables_mapping = {"coord": "ContoursAll"}
    p7_18_02_08_a001_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a001/p7_18_02_08_a001_CellDetect.mat",
                                            variables_mapping=variables_mapping)

    p7_18_02_08_a002_ms = MouseSession(age=7, session_id="18_02_08_a002", nb_ms_by_frame=100, param=param,
                                       weight=3.85)
    # calculated with 99th percentile on raster dur
    p7_18_02_08_a002_ms.activity_threshold = 10
    p7_18_02_08_a002_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
    p7_18_02_08_a002_ms.set_low_activity_threshold(threshold=1, percentile_value=5)
    p7_18_02_08_a002_ms.set_inter_neurons([207])
    # duration of those interneurons: 22.33
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p7_18_02_08_a002_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a002/p7_18_02_08_a002_RasterDur.mat",
                                            variables_mapping=variables_mapping)
    variables_mapping = {"coord": "ContoursAll"}
    p7_18_02_08_a002_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a002/p7_18_02_08_a002_CellDetect.mat",
                                            variables_mapping=variables_mapping)

    p7_18_02_08_a003_ms = MouseSession(age=7, session_id="18_02_08_a003", nb_ms_by_frame=100, param=param,
                                       weight=3.85)
    # calculated with 99th percentile on raster dur
    p7_18_02_08_a003_ms.activity_threshold = 8
    p7_18_02_08_a003_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
    p7_18_02_08_a003_ms.set_low_activity_threshold(threshold=0, percentile_value=5)
    p7_18_02_08_a003_ms.set_inter_neurons([171])
    # duration of those interneurons: 14.92
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p7_18_02_08_a003_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a003/p7_18_02_08_a003_RasterDur.mat",
                                            variables_mapping=variables_mapping)
    variables_mapping = {"coord": "ContoursAll"}
    p7_18_02_08_a003_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a003/p7_18_02_08_a003_CellDetect.mat",
                                            variables_mapping=variables_mapping)

    p7_17_10_18_a002_ms = MouseSession(age=7, session_id="17_10_18_a002", nb_ms_by_frame=100, param=param,
                                       weight=None)
    # calculated with 99th percentile on raster dur
    p7_17_10_18_a002_ms.activity_threshold = 15
    p7_17_10_18_a002_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
    p7_17_10_18_a002_ms.set_low_activity_threshold(threshold=4, percentile_value=5)
    p7_17_10_18_a002_ms.set_inter_neurons([51])
    # duration of those interneurons: 14.13
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p7_17_10_18_a002_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_18_a002/p7_17_10_18_a002_RasterDur.mat",
                                            variables_mapping=variables_mapping)
    variables_mapping = {"coord": "ContoursAll"}
    p7_17_10_18_a002_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_18_a002/p7_17_10_18_a002_CellDetect.mat",
                                            variables_mapping=variables_mapping)

    p7_17_10_18_a004_ms = MouseSession(age=7, session_id="17_10_18_a004", nb_ms_by_frame=100, param=param,
                                       weight=None)
    # calculated with 99th percentile on raster dur
    p7_17_10_18_a004_ms.activity_threshold = 14
    p7_17_10_18_a004_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
    p7_17_10_18_a004_ms.set_low_activity_threshold(threshold=3, percentile_value=5)
    p7_17_10_18_a004_ms.set_inter_neurons([298])
    # duration of those interneurons: 15.35
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p7_17_10_18_a004_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_18_a004/p7_17_10_18_a004_RasterDur.mat",
                                            variables_mapping=variables_mapping)
    variables_mapping = {"coord": "ContoursAll"}
    p7_17_10_18_a004_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_18_a004/p7_17_10_18_a004_CellDetect.mat",
                                            variables_mapping=variables_mapping)

    p8_18_02_09_a000_ms = MouseSession(age=8, session_id="18_02_09_a000", nb_ms_by_frame=100, param=param,
                                       weight=None)
    # calculated with 99th percentile on raster dur
    p8_18_02_09_a000_ms.activity_threshold = 9
    p8_18_02_09_a000_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
    p8_18_02_09_a000_ms.set_low_activity_threshold(threshold=1, percentile_value=5)
    p8_18_02_09_a000_ms.set_inter_neurons([64, 91])
    # duration of those interneurons: 12.48  11.47
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p8_18_02_09_a000_ms.load_data_from_file(file_name_to_load="p8/p8_18_02_09_a000/p8_18_02_09_a000_RasterDur.mat",
                                            variables_mapping=variables_mapping)
    variables_mapping = {"coord": "ContoursAll"}
    p8_18_02_09_a000_ms.load_data_from_file(file_name_to_load="p8/p8_18_02_09_a000/p8_18_02_09_a000_CellDetect.mat",
                                            variables_mapping=variables_mapping)

    p8_18_02_09_a001_ms = MouseSession(age=8, session_id="18_02_09_a001", nb_ms_by_frame=100, param=param,
                                       weight=None)
    # calculated with 99th percentile on raster dur
    p8_18_02_09_a001_ms.activity_threshold = 11
    p8_18_02_09_a001_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
    p8_18_02_09_a001_ms.set_low_activity_threshold(threshold=2, percentile_value=5)
    p8_18_02_09_a001_ms.set_inter_neurons([])
    # duration of those interneurons:
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p8_18_02_09_a001_ms.load_data_from_file(file_name_to_load="p8/p8_18_02_09_a001/p8_18_02_09_a001_RasterDur.mat",
                                            variables_mapping=variables_mapping)
    variables_mapping = {"coord": "ContoursAll"}
    p8_18_02_09_a001_ms.load_data_from_file(file_name_to_load="p8/p8_18_02_09_a001/p8_18_02_09_a001_CellDetect.mat",
                                            variables_mapping=variables_mapping)

    p8_18_10_17_a001_ms = MouseSession(age=8, session_id="18_10_17_a001", nb_ms_by_frame=100, param=param,
                                       weight=6)
    # calculated with 99th percentile on raster dur
    p8_18_10_17_a001_ms.activity_threshold = 10
    p8_18_10_17_a001_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
    p8_18_10_17_a001_ms.set_low_activity_threshold(threshold=1, percentile_value=5)
    p8_18_10_17_a001_ms.set_inter_neurons([117, 135, 217, 271])
    # duration of those interneurons: 32.33, 171, 144.5, 48.8
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p8_18_10_17_a001_ms.load_data_from_file(file_name_to_load="p8/p8_18_10_17_a001/p8_18_10_17_a001_RasterDur.mat",
                                            variables_mapping=variables_mapping)

    # 6.4
    p8_18_10_24_a005_ms = MouseSession(age=8, session_id="18_10_24_a005", nb_ms_by_frame=100, param=param,
                                       weight=6.4)
    # calculated with 99th percentile on raster dur
    p8_18_10_24_a005_ms.activity_threshold = 10
    p8_18_10_24_a005_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
    p8_18_10_24_a005_ms.set_low_activity_threshold(threshold=1, percentile_value=5)
    p8_18_10_24_a005_ms.set_inter_neurons([33, 112, 206])
    # duration of those interneurons: 18.92, 27.33, 20.55
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p8_18_10_24_a005_ms.load_data_from_file(file_name_to_load="p8/p8_18_10_24_a005/p8_18_10_24_a005_RasterDur.mat",
                                            variables_mapping=variables_mapping)

    #
    p8_18_10_17_a000_ms = MouseSession(age=8, session_id="18_10_17_a000", nb_ms_by_frame=100, param=param,
                                       weight=6)
    # calculated with 99th percentile on raster dur
    p8_18_10_17_a000_ms.activity_threshold = 11
    # p8_18_10_17_a000_ms.set_low_activity_threshold(threshold=, percentile_value=1)
    # p8_18_10_17_a000_ms.set_low_activity_threshold(threshold=, percentile_value=5)
    p8_18_10_17_a000_ms.set_inter_neurons([27, 70])
    # duration of those interneurons: 23.8, 43
    variables_mapping = {"spike_nums_dur": "rasterdur",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3"}
    p8_18_10_17_a000_ms.load_data_from_file(
        file_name_to_load="p8/p8_18_10_17_a000/P8_18_10_17_a000_Corrected_RasterDur.mat",
        variables_mapping=variables_mapping)

    # p9_17_11_29_a002 low participation comparing to other, dead shortly after the recording
    # p9_17_11_29_a002_ms = MouseSession(age=9, session_id="17_11_29_a002", nb_ms_by_frame=100, param=param,
    #                                    weight=5.7)
    # # calculated with 99th percentile on raster dur
    # p9_17_11_29_a002_ms.activity_threshold = 10
    # p9_17_11_29_a002_ms.set_inter_neurons([170])
    # # limit ??
    # # duration of those interneurons: 21
    # variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
    #                      "spike_nums": "filt_Bin100ms_spikedigital",
    #                      "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    # p9_17_11_29_a002_ms.load_data_from_file(file_name_to_load="p9/p9_17_11_29_a002/p9_17_11_29_a002_RasterDur.mat",
    #                                         variables_mapping=variables_mapping)

    # p9_17_11_29_a003_ms = MouseSession(age=9, session_id="17_11_29_a003", nb_ms_by_frame=100, param=param,
    #                                    weight=5.7)
    # # calculated with 99th percentile on raster dur
    # p9_17_11_29_a003_ms.activity_threshold = 7
    # p9_17_11_29_a003_ms.set_inter_neurons([1, 13, 54])
    # # duration of those interneurons: 21.1 22.75  23
    # variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
    #                      "spike_nums": "filt_Bin100ms_spikedigital",
    #                      "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    # p9_17_11_29_a003_ms.load_data_from_file(file_name_to_load="p9/p9_17_11_29_a003/p9_17_11_29_a003_RasterDur.mat",
    #                                         variables_mapping=variables_mapping)

    p9_17_12_06_a001_ms = MouseSession(age=9, session_id="17_12_06_a001", nb_ms_by_frame=100, param=param,
                                       weight=5.6)
    # calculated with 99th percentile on raster dur
    p9_17_12_06_a001_ms.activity_threshold = 9
    p9_17_12_06_a001_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
    p9_17_12_06_a001_ms.set_inter_neurons([72])
    # duration of those interneurons:15.88
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p9_17_12_06_a001_ms.load_data_from_file(file_name_to_load="p9/p9_17_12_06_a001/p9_17_12_06_a001_RasterDur.mat",
                                            variables_mapping=variables_mapping)
    variables_mapping = {"coord": "ContoursAll"}
    p9_17_12_06_a001_ms.load_data_from_file(file_name_to_load="p9/p9_17_12_06_a001/p9_17_12_06_a001_CellDetect.mat",
                                            variables_mapping=variables_mapping)

    p9_17_12_20_a001_ms = MouseSession(age=9, session_id="17_12_20_a001", nb_ms_by_frame=100, param=param,
                                       weight=5.05)
    # calculated with 99th percentile on raster dur
    p9_17_12_20_a001_ms.activity_threshold = 9
    p9_17_12_20_a001_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
    p9_17_12_20_a001_ms.set_inter_neurons([32])
    # duration of those interneurons: 10.35
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p9_17_12_20_a001_ms.load_data_from_file(file_name_to_load="p9/p9_17_12_20_a001/p9_17_12_20_a001_RasterDur.mat",
                                            variables_mapping=variables_mapping)
    variables_mapping = {"coord": "ContoursAll"}
    p9_17_12_20_a001_ms.load_data_from_file(file_name_to_load="p9/p9_17_12_20_a001/p9_17_12_20_a001_CellDetect.mat",
                                            variables_mapping=variables_mapping)

    #
    p9_18_09_27_a003_ms = MouseSession(age=9, session_id="18_09_27_a003", nb_ms_by_frame=100, param=param,
                                       weight=6.65)
    # calculated with 99th percentile on raster dur
    p9_18_09_27_a003_ms.activity_threshold = 10
    # p9_18_09_27_a003_ms.set_low_activity_threshold(threshold=, percentile_value=1)
    p9_18_09_27_a003_ms.set_inter_neurons([2, 9, 67, 206])
    # duration of those interneurons: 59.1, 32, 28, 35.15
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p9_18_09_27_a003_ms.load_data_from_file(file_name_to_load="p9/p9_18_09_27_a003/p9_18_09_27_a003_RasterDur.mat",
                                            variables_mapping=variables_mapping)

    p10_17_11_16_a003_ms = MouseSession(age=10, session_id="17_11_16_a003", nb_ms_by_frame=100, param=param,
                                        weight=6.1)
    # calculated with 99th percentile on raster dur
    p10_17_11_16_a003_ms.activity_threshold = 6
    p10_17_11_16_a003_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
    p10_17_11_16_a003_ms.set_inter_neurons([8])
    # duration of those interneurons: 28
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p10_17_11_16_a003_ms.load_data_from_file(file_name_to_load="p10/p10_17_11_16_a003/p10_17_11_16_a003_RasterDur.mat",
                                             variables_mapping=variables_mapping)
    variables_mapping = {"coord": "ContoursAll"}
    p10_17_11_16_a003_ms.load_data_from_file(file_name_to_load="p10/p10_17_11_16_a003/p10_17_11_16_a003_CellDetect.mat",
                                            variables_mapping=variables_mapping)

    p11_17_11_24_a001_ms = MouseSession(age=11, session_id="17_11_24_a001", nb_ms_by_frame=100, param=param,
                                        weight=6.7)
    # calculated with 99th percentile on raster dur
    p11_17_11_24_a001_ms.activity_threshold = 11
    p11_17_11_24_a001_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
    p11_17_11_24_a001_ms.set_inter_neurons([])
    # duration of those interneurons:
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p11_17_11_24_a001_ms.load_data_from_file(file_name_to_load="p11/p11_17_11_24_a001/p11_17_11_24_a001_RasterDur.mat",
                                             variables_mapping=variables_mapping)
    variables_mapping = {"coord": "ContoursAll"}
    p11_17_11_24_a001_ms.load_data_from_file(file_name_to_load="p11/p11_17_11_24_a001/p11_17_11_24_a001_CellDetect.mat",
                                            variables_mapping=variables_mapping)

    p11_17_11_24_a000_ms = MouseSession(age=11, session_id="17_11_24_a000", nb_ms_by_frame=100, param=param,
                                        weight=6.7)
    # calculated with 99th percentile on raster dur
    p11_17_11_24_a000_ms.activity_threshold = 12
    p11_17_11_24_a000_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
    p11_17_11_24_a000_ms.set_inter_neurons([193])
    # duration of those interneurons: 19.09
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p11_17_11_24_a000_ms.load_data_from_file(file_name_to_load="p11/p11_17_11_24_a000/p11_17_11_24_a000_RasterDur.mat",
                                             variables_mapping=variables_mapping)
    variables_mapping = {"coord": "ContoursAll"}
    p11_17_11_24_a000_ms.load_data_from_file(file_name_to_load="p11/p11_17_11_24_a000/p11_17_11_24_a000_CellDetect.mat",
                                             variables_mapping=variables_mapping)
    # p11_17_11_24_a000_ms.plot_cell_assemblies_on_map()

    p12_17_11_10_a002_ms = MouseSession(age=12, session_id="17_11_10_a002", nb_ms_by_frame=100, param=param,
                                        weight=7)
    # calculated with 99th percentile on raster dur
    p12_17_11_10_a002_ms.activity_threshold = 13
    p12_17_11_10_a002_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
    p12_17_11_10_a002_ms.set_inter_neurons([150, 252])
    # duration of those interneurons: 16.17, 24.8
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p12_17_11_10_a002_ms.load_data_from_file(file_name_to_load="p12/p12_17_11_10_a002/p12_17_11_10_a002_RasterDur.mat",
                                             variables_mapping=variables_mapping)
    variables_mapping = {"coord": "ContoursAll"}
    p12_17_11_10_a002_ms.load_data_from_file(file_name_to_load="p12/p12_17_11_10_a002/p12_17_11_10_a002_CellDetect.mat",
                                            variables_mapping=variables_mapping)

    p12_171110_a000_ms = MouseSession(age=12, session_id="171110_a000", nb_ms_by_frame=100, param=param,
                                      weight=7)
    # calculated with 99th percentile on raster dur
    p12_171110_a000_ms.activity_threshold = 10
    p12_171110_a000_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
    p12_171110_a000_ms.set_inter_neurons([106, 144])
    # duration of those interneurons: 18.29  14.4
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p12_171110_a000_ms.load_data_from_file(file_name_to_load="p12/p12_17_11_10_a000/p12_17_11_10_a000_rasterdur.mat",
                                           variables_mapping=variables_mapping)
    variables_mapping = {"coord": "ContoursAll"}
    p12_171110_a000_ms.load_data_from_file(file_name_to_load="p12/p12_17_11_10_a000/p12_17_11_10_a000_CellDetect.mat",
                                            variables_mapping=variables_mapping)

    p13_18_10_29_a000_ms = MouseSession(age=13, session_id="18_10_29_a000", nb_ms_by_frame=100, param=param,
                                        weight=9.4)
    # calculated with 99th percentile on raster dur
    p13_18_10_29_a000_ms.activity_threshold = 16
    # p13_18_10_29_a000_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
    p13_18_10_29_a000_ms.set_inter_neurons([5, 26, 27, 35, 38])
    # duration of those interneurons: 13.57, 16.8, 22.4, 12, 14.19
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p13_18_10_29_a000_ms.load_data_from_file(file_name_to_load="p13/p13_18_10_29_a000/p13_18_10_29_a000_RasterDur.mat",
                                             variables_mapping=variables_mapping)

    p13_18_10_29_a001_ms = MouseSession(age=13, session_id="18_10_29_a001", nb_ms_by_frame=100, param=param,
                                        weight=9.4)
    # calculated with 99th percentile on raster dur
    p13_18_10_29_a001_ms.activity_threshold = 13
    # p13_18_10_29_a001_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
    p13_18_10_29_a001_ms.set_inter_neurons([68])
    # duration of those interneurons: 13.31
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p13_18_10_29_a001_ms.load_data_from_file(file_name_to_load="p13/p13_18_10_29_a001/p13_18_10_29_a001_RasterDur.mat",
                                             variables_mapping=variables_mapping)
    # ,
    # frames_filter=np.arange(5000))

    p14_18_10_23_a000_ms = MouseSession(age=14, session_id="18_10_23_a000", nb_ms_by_frame=100, param=param,
                                        weight=10.35)
    # calculated with 99th percentile on raster dur
    p14_18_10_23_a000_ms.activity_threshold = 9
    p14_18_10_23_a000_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
    p14_18_10_23_a000_ms.set_inter_neurons([0])
    # duration of those interneurons: 24.33
    variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                         "spike_nums": "filt_Bin100ms_spikedigital",
                         "spike_durations": "LOC3", "spike_amplitudes": "MAX"}
    p14_18_10_23_a000_ms.load_data_from_file(file_name_to_load="p14/p14_18_10_23_a000/p14_18_10_23_a000_rasterdur.mat",
                                             variables_mapping=variables_mapping)
    variables_mapping = {"coord": "ContoursAll"}
    p14_18_10_23_a000_ms.load_data_from_file(file_name_to_load="p14/p14_18_10_23_a000/p14_18_10_23_a000_CellDetect.mat",
                                            variables_mapping=variables_mapping)

    # TODO : add weight
    # only interneurons in p14_18_10_23_a001_ms
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
    variables_mapping = {"coord": "ContoursAll"}
    p14_18_10_23_a001_ms.load_data_from_file(file_name_to_load="p14/p14_18_10_23_a001/p14_18_10_23_a001_CellDetect.mat",
                                            variables_mapping=variables_mapping)

    arnaud_ms = MouseSession(age=24, session_id="arnaud", nb_ms_by_frame=50, param=param)
    arnaud_ms.activity_threshold = 13
    arnaud_ms.set_inter_neurons([])
    variables_mapping = {"spike_nums": "spikenums"}
    arnaud_ms.load_data_from_file(file_name_to_load="spikenumsarnaud.mat", variables_mapping=variables_mapping)

    available_ms = [p6_18_02_07_a001_ms, p6_18_02_07_a002_ms,
                    p7_171012_a000_ms, p7_18_02_08_a000_ms,
                    p7_17_10_18_a002_ms, p7_17_10_18_a004_ms, p7_18_02_08_a001_ms, p7_18_02_08_a002_ms,
                    p7_18_02_08_a003_ms,
                    p8_18_02_09_a000_ms, p8_18_02_09_a001_ms,
                    p8_18_10_24_a005_ms, p8_18_10_17_a001_ms,
                    p8_18_10_17_a000_ms,  # new
                    p9_17_12_06_a001_ms, p9_17_12_20_a001_ms,
                    p9_18_09_27_a003_ms,  # new
                    p10_17_11_16_a003_ms,
                    p11_17_11_24_a001_ms, p11_17_11_24_a000_ms,
                    p12_17_11_10_a002_ms, p12_171110_a000_ms,
                    p13_18_10_29_a000_ms,  # new
                    p13_18_10_29_a001_ms,
                    p14_18_10_23_a000_ms]
    # arnaud_ms]
    # available_ms = [p13_18_10_29_a001_ms]
    interneurons_ms = [p14_18_10_23_a001_ms]
    available_ms = interneurons_ms
    still_to_cluster = [p7_18_02_08_a001_ms, p7_18_02_08_a002_ms,
                        p7_18_02_08_a003_ms,
                        p8_18_02_09_a000_ms, p8_18_02_09_a001_ms,
                        p8_18_10_24_a005_ms, p8_18_10_17_a001_ms,
                        p9_17_12_06_a001_ms, p9_17_12_20_a001_ms,
                        p10_17_11_16_a003_ms,
                        p13_18_10_29_a001_ms,
                        p14_18_10_23_a000_ms]
    # p9_17_11_29_a002_ms, p9_17_11_29_a003_ms removed because died just after
    # available_ms = [p6_18_02_07_a001_ms, p7_171012_a000_ms, p8_18_02_09_a000_ms, p9_17_12_20_a001_ms,
    #                 p10_17_11_16_a003_ms, p12_171110_a000_ms]

    # ms_to_analyse = [p7_18_02_08_a000_ms,
    #                     p7_17_10_18_a002_ms, p7_17_10_18_a004_ms, p7_18_02_08_a001_ms, p7_18_02_08_a002_ms,
    #                     p7_18_02_08_a003_ms,
    #                     p8_18_02_09_a000_ms, p8_18_02_09_a001_ms,
    #                     p8_18_10_24_a005_ms, p8_18_10_17_a001_ms]
    corrected_ms_from_robin = [p7_171012_a000_ms, p8_18_10_17_a000_ms]
    # available_ms = corrected_ms_from_robin
    ms_to_test_clustering = [p11_17_11_24_a000_ms]
    ms_to_analyse = ms_to_test_clustering  # corrected_ms_from_robin

    just_do_stat_on_event_detection_parameters = False

    # for events (sce) detection
    perc_threshold = 99
    use_max_of_each_surrogate = False
    n_surrogate_activity_threshold = 50
    use_raster_dur = True
    determine_low_activity_by_variation = True

    do_plot_interneurons_connect_maps = False
    do_plot_connect_hist = True

    # ##########################################################################################
    # #################################### CLUSTERING ###########################################
    # ##########################################################################################
    do_clustering = False
    # if False, clustering will be done using kmean
    do_fca_clustering = False
    with_cells_in_cluster_seq_sorted = True

    # ##### for fca #####
    n_surrogate_fca = 20

    # #### for kmean  #####
    with_shuffling = True
    print(f"use_raster_dur {use_raster_dur}")
    # range_n_clusters_k_mean = np.arange(2, 5)
    range_n_clusters_k_mean = np.array([3])
    n_surrogate_k_mean = 10
    keep_only_the_best_kmean_cluster = False

    # ##########################################################################################
    # ################################ PATTERNS SEARCH #########################################
    # ##########################################################################################
    do_pattern_search = False
    split_pattern_search = False
    use_only_uniformity_method = True
    use_loss_score_to_keep_the_best_from_tree = False
    use_sce_times_for_pattern_search = True
    use_ordered_spike_nums_for_surrogate = True
    n_surrogate_for_pattern_search = 100
    # seq params:
    # TODO: error_rate that change with the number of element in the sequence
    param.error_rate = 0.25
    param.max_branches = 10
    param.time_inter_seq = 50
    param.min_duration_intra_seq = 0
    param.min_len_seq = 5
    param.min_rep_nb = 5

    debug_mode = False

    # ------------------------------ end param section ------------------------------

    ms_by_age = dict()
    for ms in available_ms:
        if ms.age not in ms_by_age:
            ms_by_age[ms.age] = []
        ms_by_age[ms.age].append(ms)

        if do_plot_interneurons_connect_maps or do_plot_connect_hist:
            ms.detect_n_in_n_out()

        if do_plot_connect_hist:
            connec_func_stat([ms], data_descr=ms.description, param=param)

        if do_plot_interneurons_connect_maps:
            if ms.coord is None:
                continue
            ms.plot_each_inter_neuron_connect_map()

    if do_plot_connect_hist:
        n_ins_by_age = dict()
        n_outs_by_age = dict()
        for age, ms_list in ms_by_age.items():
            n_ins_by_age[age], n_outs_by_age[age] = connec_func_stat(ms_list, data_descr=f"p{age}", param=param)
        box_plot_data_by_age(data_dict=n_ins_by_age,
                             title="Connectivity in by age",
                             filename="box_plots_connectivity_in_by_age",
                             y_label="Active cells (%)",
                             param=param)
        box_plot_data_by_age(data_dict=n_outs_by_age,
                             title="Connectivity out by age",
                             filename="box_plots_connectivity_out_by_age",
                             y_label="Active cells (%)",
                             param=param)

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
            if ms.spike_struct.spike_nums_dur is not None:
                use_raster_durs = [True]
            else:
                use_raster_durs = [False]
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

                    ###############  TEST    ##################
                    if not determine_low_activity_by_variation:
                        perc_low_activity_threshold = 5
                        if perc_low_activity_threshold not in ms.low_activity_threshold_by_percentile:
                            low_activity_events_thsld = get_low_activity_events_detection_threshold(
                                spike_nums=spike_nums_to_use,
                                window_duration=sliding_window_duration,
                                spike_train_mode=False,
                                use_min_of_each_surrogate=False,
                                n_surrogate=n_surrogate_activity_threshold,
                                perc_threshold=perc_low_activity_threshold,
                                debug_mode=False)
                            print(f"ms {ms.description}")
                            print(f"low_activity_events_thsld {low_activity_events_thsld}, "
                                  f"{np.round((low_activity_events_thsld/n_cells), 3)}%")
                            continue
                    else:
                        pass
                    # ########### END TEST ###########

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
                        if ms.spike_struct.spike_durations is not None:
                            for spike_duration in ms.spike_struct.spike_durations:
                                all_spike_duration.extend(spike_duration)
                            duration_spikes_by_age[ms.age] = all_spike_duration
                    else:
                        nb_elem = len(ratio_spikes_events_by_age[ms.age])
                        ratio_spikes_events_by_age[ms.age].extend(list(ratio_spikes_events))
                        interneurons_indices_by_age[ms.age].extend(list(ms.spike_struct.inter_neurons +
                                                                        nb_elem))
                        all_spike_duration = []
                        if ms.spike_struct.spike_durations is not None:
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
                    labels = []
                    scatter_shapes = []
                    colors = []
                    if len(ratio_non_interneurons) > 0:
                        values_to_scatter.append(np.mean(ratio_non_interneurons))
                        values_to_scatter.append(np.median(ratio_non_interneurons))
                        labels.extend(["mean", "median"])
                        scatter_shapes.extend(["o", "s"])
                        colors.extend(["white", "white"])
                    if len(ratio_interneurons) > 0:
                        values_to_scatter.append(np.mean(ratio_interneurons))
                        values_to_scatter.append(np.median(ratio_interneurons))
                        values_to_scatter.extend(ratio_interneurons)
                        labels.extend(["mean", "median", f"interneuron (x{len(inter_neurons)})"])
                        scatter_shapes.extend(["o", "s"])
                        scatter_shapes.extend(["*"] * len(inter_neurons))
                        colors.extend(["red", "red"])
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
                    if len(ratio_non_interneurons) > 0:
                        values_to_scatter.append(np.mean(ratio_non_interneurons))
                        values_to_scatter.append(np.median(ratio_non_interneurons))
                    if len(ratio_interneurons) > 0:
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
            labels = []
            scatter_shapes = []
            colors = []
            if len(ratio_non_interneurons) > 0:
                values_to_scatter.append(np.mean(ratio_non_interneurons))
                values_to_scatter.append(np.median(ratio_non_interneurons))
                labels.extend(["mean", "median"])
                scatter_shapes.extend(["o", "s"])
                colors.extend(["white", "white"])
            if len(ratio_interneurons) > 0:
                values_to_scatter.append(np.mean(ratio_interneurons))
                values_to_scatter.append(np.median(ratio_interneurons))
                values_to_scatter.extend(ratio_interneurons)
                labels.extend(["mean", "median", f"interneuron (x{len(inter_neurons)})"])
                scatter_shapes.extend(["o", "s"])
                scatter_shapes.extend(["*"] * len(inter_neurons))
                colors.extend(["red", "red"])
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
            labels = []
            scatter_shapes = []
            colors = []
            if len(ratio_non_interneurons) > 0:
                values_to_scatter.append(np.mean(ratio_non_interneurons))
                values_to_scatter.append(np.median(ratio_non_interneurons))
                labels.extend(["mean", "median"])
                scatter_shapes.extend(["o", "s"])
                colors.extend(["white", "white"])
            if len(ratio_interneurons) > 0:
                values_to_scatter.append(np.mean(ratio_interneurons))
                values_to_scatter.append(np.median(ratio_interneurons))
                values_to_scatter.extend(ratio_interneurons)
                labels.extend(["mean", "median", f"interneuron (x{len(inter_neurons)})"])
                scatter_shapes.extend(["o", "s"])
                scatter_shapes.extend(["*"] * len(inter_neurons))
                colors.extend(["red", "red"])
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
        print(f"ms: {data_descr}")

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
                                                               keep_only_the_best=keep_only_the_best_kmean_cluster)

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
                    find_significant_patterns(spike_nums=
                                              spike_nums_to_use[:,
                                              splits_indices[split_id]:splits_indices[split_id + 1]],
                                              param=param,
                                              activity_threshold=activity_threshold,
                                              sliding_window_duration=sliding_window_duration,
                                              n_surrogate=n_surrogate_for_pattern_search,
                                              use_ordered_spike_nums_for_surrogate=use_ordered_spike_nums_for_surrogate,
                                              data_id=ms.description, debug_mode=False,
                                              extra_file_name=f"part_{split_id+1}",
                                              sce_times_bool=sce_times_bool_to_use,
                                              use_only_uniformity_method=use_only_uniformity_method,
                                              use_loss_score_to_keep_the_best_from_tree=
                                              use_loss_score_to_keep_the_best_from_tree,
                                              spike_shape="|",
                                              spike_shape_size=10
                                              )

            else:
                # TODO: split spikes_nums in 3 to 4 part
                print("Start of use_new_pattern_package")
                find_significant_patterns(spike_nums=spike_nums_to_use, param=param,
                                          activity_threshold=activity_threshold,
                                          sliding_window_duration=sliding_window_duration,
                                          n_surrogate=n_surrogate_for_pattern_search,
                                          data_id=ms.description, debug_mode=False,
                                          use_ordered_spike_nums_for_surrogate=use_ordered_spike_nums_for_surrogate,
                                          extra_file_name="",
                                          sce_times_bool=sce_times_bool_to_use,
                                          use_only_uniformity_method=use_only_uniformity_method,
                                          use_loss_score_to_keep_the_best_from_tree=
                                          use_loss_score_to_keep_the_best_from_tree,
                                          spike_shape="|",
                                          spike_shape_size=5)

    return


main()
