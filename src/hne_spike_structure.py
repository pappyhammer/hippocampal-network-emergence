# important to avoid a bug when using virtualenv
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import numpy as np
# to add homemade package, go to preferences, then project interpreter, then click on the wheel symbol
# then show all, then select the interpreter and lick on the more right icon to display a list of folder and
# add the one containing the folder pattern_discovery
import pattern_discovery.tools.misc as tools_misc
from pattern_discovery.tools.misc import get_continous_time_periods
from scipy import stats
import networkx as nx

class HNESpikeStructure:

    def __init__(self, mouse_session, labels=None, spike_nums=None, spike_trains=None,
                 spike_nums_dur=None, activity_threshold=None,
                 title=None, ordered_indices=None, ordered_spike_data=None):
        self.mouse_session = mouse_session
        self.spike_nums = spike_nums
        self.spike_nums_dur = spike_nums_dur
        self.peak_nums = None
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
        self.nb_frames_for_func_connect = 15
        # contain the list of neurons connected to the EB as keys, and the number of connection as values
        # first key is a dict of neuron, the second key is other neurons to which the first connect,
        # then the number of times is connected to it
        self.n_in_dict = dict()
        self.n_out_dict = dict()
        # initialized when loading rasters
        self.n_in_matrix = None
        self.n_out_matrix = None
        self.graph_in = None
        self.graph_out = None

    def clean_data_using_cells_to_remove(self, cells_to_remove):
        if len(cells_to_remove) == 0:
            return

        new_n_cells = self.n_cells - len(cells_to_remove)
        cells_to_remove = np.array(cells_to_remove)
        mask = np.ones(self.n_cells, dtype="bool")
        mask[cells_to_remove] = False

        if self.spike_nums is not None:
            self.spike_nums = self.spike_nums[mask]

        if self.spike_nums_dur is not None:
            self.spike_nums_dur = self.spike_nums_dur[mask]

        if self.peak_nums is not None:
            self.peak_nums = self.peak_nums[mask]

        if self.n_in_matrix is not None:
            self.n_in_matrix = self.n_in_matrix[mask]

        if self.n_out_matrix is not None:
            self.n_out_matrix = self.n_out_matrix[mask]

        if self.graph_in is not None:
            self.graph_in = self.graph_in[mask]

        if self.graph_out is not None:
            self.graph_out = self.graph_out[mask]

        # if self.spike_nums is not None:
        #     new_spike_nums = np.zeros((new_n_cells, self.spike_nums.shape[1]), dtype="int8")
        # if self.spike_nums_dur is not None:
        #     new_spike_nums_dur = np.zeros((new_n_cells, self.spike_nums_dur.shape[1]), dtype="int8")
        # if self.peak_nums is not None:
        #     new_peak_nums = np.zeros((new_n_cells, self.spike_nums_dur.shape[1]), dtype="int8")
        # if self.n_in_matrix is not None:
        #     new_n_in_matrix = np.zeros((new_n_cells, self.n_in_matrix.shape[1]))
        # if self.n_out_matrix is not None:
        #     new_n_out_matrix = np.zeros((new_n_cells, self.n_out_matrix.shape[1]))
        # if self.graph_in is not None:
        #     new_graph_in = np.zeros((new_n_cells, self.graph_in.shape[1]))
        # if self.graph_out is not None:
        #     new_graph_out = np.zeros((new_n_cells, self.graph_out.shape[1]))

        if self.labels is not None:
            new_labels = []
        # cell_count = 0
        for cell in np.arange(self.n_cells):
            if cell in cells_to_remove:
                continue
            # if self.spike_nums is not None:
            #     new_spike_nums[cell_count] = self.spike_nums[cell]
            # if self.spike_nums_dur is not None:
            #     new_spike_nums_dur[cell_count] = self.spike_nums_dur[cell]
            # if self.peak_nums is not None:
            #     new_peak_nums[cell_count] = self.peak_nums[cell]
            #
            # if self.n_in_matrix is not None:
            #     self.n_in_matrix[cell] = np.zeros((new_n_cells, self.n_in_matrix.shape[1]))
            # if self.n_out_matrix is not None:
            #     self.n_out_matrix[cell] = np.zeros((new_n_cells, self.n_out_matrix.shape[1]))
            # if self.graph_in is not None:
            #     self.graph_in[cell] = np.zeros((new_n_cells, self.graph_in.shape[1]))
            # if self.graph_out is not None:
            #     self.graph_out[cell] = np.zeros((new_n_cells, self.graph_out.shape[1]))

            if self.labels is not None:
                new_labels.append(self.labels[cell])

            # cell_count += 1

        # if self.spike_nums is not None:
        #     self.spike_nums = new_spike_nums
        # if self.spike_nums_dur is not None:
        #     self.spike_nums_dur = new_spike_nums_dur
        # if self.peak_nums is not None:
        #     self.peak_nums = new_peak_nums
        if self.labels is not None:
            self.labels = new_labels
        self.n_cells = new_n_cells

    def clean_raster_at_concatenation(self):
        """
        Movies of 2500 frames are concatenated, we need to clean around the concatenation times
        :return:
        """

        if self.spike_nums_dur is None:
            return

        if (self.spike_nums_dur.shape[1] != 12500) and (self.spike_nums_dur.shape[1] != 10000) :
            return
        mask_frames = np.zeros(self.spike_nums_dur.shape[1], dtype="bool")
        concatenation_times = [2500, 5000, 7500, 10000]
        if self.spike_nums_dur.shape[1] == 10000:
            concatenation_times = [2500, 5000, 7500]
        for i in concatenation_times:
            mask_frames[i:i+5] = True

        if self.spike_nums is not None:
            self.spike_nums[:, mask_frames] = 0
        if self.spike_nums_dur is not None:
            self.spike_nums_dur[:, mask_frames] = 0
        if self.peak_nums is not None:
            self.peak_nums[:, mask_frames] = 0
        print("clean_raster_at_concatenation done")

    def build_spike_nums_dur(self):
        if (self.spike_nums is None) or (self.peak_nums is None):
            return
        print(f"{self.mouse_session.description} build_spike_nums_dur from spike_nums and peak_nums")
        n_cells = len(self.spike_nums)
        n_frames = self.spike_nums.shape[1]
        ms = self.mouse_session
        self.spike_nums_dur = np.zeros((n_cells, n_frames), dtype="int8")
        for cell in np.arange(n_cells):
            peaks_index = np.where(self.peak_nums[cell, :])[0]
            onsets_index = np.where(self.spike_nums[cell, :])[0]

            for onset_index in onsets_index:
                peaks_after = np.where(peaks_index > onset_index)[0]
                if len(peaks_after) == 0:
                    continue
                peaks_after = peaks_index[peaks_after]
                peak_after = peaks_after[0]
                if (peak_after - onset_index) > 200:
                    print(f"{ms.description} long transient in cell {cell} of "
                          f"duration {peak_after - onset_index} frames at frame {onset_index}")

                self.spike_nums_dur[cell, onset_index:peak_after+1] = 1

    def build_spike_nums_and_peak_nums(self):
        if self.spike_nums_dur is None:
            return

        n_cells = len(self.spike_nums_dur)
        n_frames = self.spike_nums_dur.shape[1]
        ms = self.mouse_session
        self.spike_nums = np.zeros((n_cells, n_frames), dtype="int8")
        self.peak_nums = np.zeros((n_cells, n_frames), dtype="int8")
        for cell in np.arange(n_cells):
            transient_periods = get_continous_time_periods(self.spike_nums_dur[cell])
            for transient_period in transient_periods:
                onset = transient_period[0]
                peak = transient_period[1]
                # if onset == peak:
                #     print("onset == peak")
                self.spike_nums[cell, onset] = 1
                self.peak_nums[cell, peak] = 1

    def detect_n_in_n_out(self, save_graphs=True):
        if self.spike_nums is None:
            print(f"{self.mouse_session.description} spike_nums is None in detect_n_in_n_out")
            return
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
            # to build the distribution of each other neurons around the spikes of this one
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
                frames_time = np.arange(-self.nb_frames_for_func_connect, self.nb_frames_for_func_connect + 1)
                i_n = 0
                for i_time, sum_spike in enumerate(distribution_array):
                    if sum_spike > 0:
                        distribution_for_test[i_n:i_n + sum_spike] = frames_time[i_time]
                        i_n += sum_spike

                # print(f"neuron {neuron} to {neuron_to_consider}: len {len(distribution_for_test)}: "
                #       f"{distribution_array_2_d[neuron_to_consider, :]}")
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
                    else:
                        n_in_sum = np.sum(distribution_array[:event_index])
                        n_out_sum = np.sum(distribution_array[(event_index + 1):])
                        if n_in_sum > n_out_sum:
                            self.n_in_dict[neuron][neuron_to_consider] = 1
                            # self.n_in_matrix[neuron][neuron_to_consider] = 1
                        else:
                            self.n_out_dict[neuron][neuron_to_consider] = 1
                            # self.n_out_matrix[neuron][neuron_to_consider] = 1
                else:
                    continue
                # # means we have the same number of spikes before and after
                # if n_in_sum == n_out_sum:
                #     continue
                # max_value = max(n_in_sum, n_out_sum)
                # min_value = min(n_in_sum, n_out_sum)
                # # we should have at least twice more spikes on one side
                # if max_value < (min_value * 2):
                #     continue
                # # and twice as more as the spikes at time 0
                # if max_value < (distribution_array[event_index] * 2):
                #     continue

        # building graph using Networkx package
        # DiGraph means directed graph
        self.graph_in = nx.DiGraph()
        self.graph_in.add_nodes_from(np.arange(self.n_cells))
        self.graph_out = nx.DiGraph()
        self.graph_out.add_nodes_from(np.arange(self.n_cells))

        for cell, cells_connected in self.n_in_dict.items():
            for cell_connected in cells_connected.keys():
                self.graph_in.add_edge(cell, cell_connected)

        for cell, cells_connected in self.n_out_dict.items():
            for cell_connected in cells_connected.keys():
                self.graph_out.add_edge(cell, cell_connected)

        if save_graphs:
            ms = self.mouse_session
            param = ms.param
            nx.write_graphml(self.graph_in, f"{param.path_data}/p{ms.age}/{ms.description.lower()}/"
            f"{ms.description}_graph_in.graphml")
            nx.write_graphml(self.graph_out, f"{param.path_data}/p{ms.age}/{ms.description.lower()}/"
            f"{ms.description}_graph_out.graphml")
            nx.write_gexf(self.graph_in, f"{param.path_data}/p{ms.age}/{ms.description.lower()}/"
            f"{ms.description}_graph_in.gexf")
            nx.write_gexf(self.graph_out, f"{param.path_data}/p{ms.age}/{ms.description.lower()}/"
            f"{ms.description}_graph_out.gexf")
            # nx.write_gpickle(self.graph_in,f"{param.path_data}/p{ms.age}/{ms.description.lower()}/"
            # f"{ms.description}_graph_in.gpickle")
            # nx.write_gpickle(self.graph_out,f"{param.path_data}/p{ms.age}/{ms.description.lower()}/"
            # f"{ms.description}_graph_out.gpickle")


        # raise Exception("testing")
        # best_cell = -1
        # best_score = 0
        # for cell in np.arange(nb_neurons):
        #     score = np.sum(self.n_out_matrix[cell])
        #     if best_score < score:
        #         best_cell = cell
        #         best_score = score
        #     print(f"cell {cell}: {score}")
        # print(f"Most connected cell: {best_cell} with {best_score} connections")
        #
        # raise Exception("connnnnecc")

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

            for cell_id, spike_duration in enumerate(self.spike_durations):
                if len(spike_duration) > 0:
                    avg_spike_duration_by_cell[cell_id] = np.mean(spike_duration)
                else:
                    avg_spike_duration_by_cell[cell_id] = 0

        # inter_neurons can be manually added
        if self.inter_neurons is not None:
            return
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


