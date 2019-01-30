import numpy as np
import classification_stat as cs
import hdf5storage
from datetime import datetime
import os
import matplotlib.pyplot as plt

from matplotlib.figure import SubplotParams
import matplotlib.gridspec as gridspec
from sortedcontainers import SortedDict


class BenchmarkRasterDur:
    def __init__(self, description, ground_truth_raster_dur, predicted_raster_dur_dict, cells):
        self.description = description
        self.ground_truth_raster_dur = ground_truth_raster_dur
        # cells on which base the ground truth
        self.cells = cells
        # first key describe the data, value is raster_dur (2D array, cells vs frames)
        self.predicted_raster_dur_dict = predicted_raster_dur_dict
        # first key is the cell, value a dict with
        # same keys as raster_dur_dict, value will be a list of dict with results from benchmarks
        self.results_frames_dict_by_cell = dict()
        self.results_transients_dict_by_cell = dict()
        # same keys as raster_dur_dict, value will be a list of dict with results from benchmarks
        self.results_dict_global = dict()

    def compute_stats_on_onsets(self):
        print(f"{self.description} stats on onsets")
        for cell in self.cells:
            print(f"Cell {cell}")
            for key, raster_dur in self.predicted_raster_dur_dict.items():
                gt_rd = self.ground_truth_raster_dur[cell]
                p_rd = raster_dur[cell]
                frames_stat = cs.compute_stats_on_onsets(spike_nums=gt_rd, predicted_spike_nums=p_rd)
                # frames stats
                print(f"raster {key}")
                print(f"Onsets stat:")
                for k, value in frames_stat.items():
                    print(f"{k}: {str(np.round(value, 4))}")
            print("")
            print("/////////////////")
            print("")
        print("All cells")
        for key, raster_dur in self.predicted_raster_dur_dict.items():
            gt_rd = self.ground_truth_raster_dur[self.cells]
            p_rd = raster_dur[self.cells]
            frames_stat = cs.compute_stats_on_onsets(gt_rd, p_rd)
            # frames stats
            print(f"raster {key}")
            print(f"Onsets stat:")
            for k, value in frames_stat.items():
                print(f"{k}: {str(np.round(value, 4))}")

    def compute_stats(self):
        print(f"{self.description} stats on raster dur")
        for cell in self.cells:
            print(f"Cell {cell}")
            self.results_frames_dict_by_cell[cell] = SortedDict()
            self.results_transients_dict_by_cell[cell] = SortedDict()
            for key, raster_dur in self.predicted_raster_dur_dict.items():
                gt_rd = self.ground_truth_raster_dur[cell]
                p_rd = raster_dur[cell]
                frames_stat, transients_stat = cs.compute_stats(gt_rd, p_rd)
                self.results_frames_dict_by_cell[cell][key] = frames_stat
                self.results_transients_dict_by_cell[cell][key] = transients_stat
                # frames stats
                print(f"raster {key}")
                print(f"Frames stat:")
                for k, value in frames_stat.items():
                    print(f"{k}: {str(np.round(value, 4))}")

                print(f"###")
                print(f"Transients stat:")
                for k, value in transients_stat.items():
                    print(f"{k}: {str(np.round(value, 4))}")
                print("")
            print("")
            print("/////////////////")
            print("")
        print("All cells")
        for key, raster_dur in self.predicted_raster_dur_dict.items():
            gt_rd = self.ground_truth_raster_dur[self.cells]
            p_rd = raster_dur[self.cells]
            frames_stat, transients_stat = cs.compute_stats(gt_rd, p_rd)
            # frames stats
            print(f"raster {key}")
            print(f"Frames stat:")
            for k, value in frames_stat.items():
                print(f"{k}: {str(np.round(value, 4))}")

            print(f"###")
            print(f"Transients stat:")
            for k, value in transients_stat.items():
                print(f"{k}: {str(np.round(value, 4))}")
            print("")

    def plot_boxplots_for_frames_stat(self, path_results, description, time_str, save_formats="pdf"):
        stats_to_show = ["sensitivity", "specificity", "PPV", "NPV"]
        colors = ["cornflowerblue", "blue", "steelblue", "skyblue"]

        stat_fig, axes = plt.subplots(nrows=2, ncols=2, squeeze=True,
                                      gridspec_kw={'height_ratios': [0.5, 0.5],
                                                   'width_ratios': [0.5, 0.5]},
                                      figsize=(10, 10))

        stat_fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'w_pad': 1, 'h_pad': 5})
        axes = np.ndarray.flatten(axes)
        fig_patch = stat_fig.patch
        # rgba = c_map(0)
        face_color = "black"
        text_color = "white"
        title_color = "red"
        fig_patch.set_facecolor(face_color)

        for stat_index, stat_name in enumerate(stats_to_show):
            ax = axes[stat_index]
            n_cells = len(self.results_frames_dict_by_cell)

            ax.set_facecolor(face_color)
            ax.xaxis.set_ticks_position('none')
            ax.xaxis.label.set_color(text_color)
            ax.tick_params(axis='x', colors=text_color)
            ax.yaxis.label.set_color(text_color)
            ax.tick_params(axis='y', colors=text_color)
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            # ax.get_yaxis().set_visible(False)
            # ax.get_xaxis().set_visible(False)

            ax.set_frame_on(False)
            n_box_plots = None
            labels = None
            values_by_prediction = None
            for cell_index, cell_to_display in enumerate(self.results_frames_dict_by_cell.keys()):
                if n_box_plots is None:
                    n_box_plots = len(self.results_frames_dict_by_cell[cell_to_display])
                    labels = list(self.results_frames_dict_by_cell[cell_to_display].keys())
                    values_by_prediction = [[] for n in np.arange(n_box_plots)]
                for label_index, label in enumerate(labels):
                    values_by_prediction[label_index].\
                        append(self.results_frames_dict_by_cell[cell_to_display][label][stat_name])

            colorfull = True

            bplot = ax.boxplot(values_by_prediction, patch_artist=colorfull,
                               labels=labels, sym='', zorder=1)  # whis=[5, 95], sym='+'

            for element in ['boxes', 'whiskers', 'fliers', 'caps']:
                plt.setp(bplot[element], color="white")

            for element in ['means', 'medians']:
                plt.setp(bplot[element], color="silver")

            if colorfull:
                colors = colors[:n_box_plots]
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_facecolor(color)

            # ax.set_ylabel(f"proportion")
            # ax.set_xlabel("age")
            xticks = np.arange(1, n_box_plots + 1)
            ax.set_xticks(xticks)
            # sce clusters labels
            ax.set_xticklabels(labels)

            ax.set_title(stat_name, color=title_color, pad=20)


        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            stat_fig.savefig(f'{path_results}/'
                             f'{description}_box_plots_predictions_frames_on_{n_cells}_cells'
                             f'_{time_str}.{save_format}',
                             format=f"{save_format}",
                             facecolor=stat_fig.get_facecolor(), edgecolor='none')

    def plot_boxplots_for_transients_stat(self, path_results, description, time_str, save_formats="pdf"):
        stats_to_show = ["sensitivity"]
        colors = ["cornflowerblue", "blue", "steelblue", "skyblue"]

        stat_fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=True,
                                      gridspec_kw={'height_ratios': [1],
                                                   'width_ratios': [1]},
                                      figsize=(8, 8))

        stat_fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'w_pad': 1, 'h_pad': 5})
        # axes = np.ndarray.flatten(axes)
        fig_patch = stat_fig.patch
        # rgba = c_map(0)
        face_color = "black"
        text_color = "white"
        title_color = "red"
        fig_patch.set_facecolor(face_color)

        for stat_index, stat_name in enumerate(stats_to_show):
            ax = axes
            n_cells = len(self.results_transients_dict_by_cell)

            # now adding as many suplots as need, depending on how many overlap has the cell

            ax.set_facecolor(face_color)
            ax.xaxis.set_ticks_position('none')
            ax.xaxis.label.set_color(text_color)
            ax.tick_params(axis='x', colors=text_color)
            ax.yaxis.label.set_color(text_color)
            ax.tick_params(axis='y', colors=text_color)
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            # ax.get_yaxis().set_visible(False)
            # ax.get_xaxis().set_visible(False)

            ax.set_frame_on(False)
            n_box_plots = None
            labels = None
            values_by_prediction = None
            for cell_index, cell_to_display in enumerate(self.results_transients_dict_by_cell.keys()):
                if n_box_plots is None:
                    n_box_plots = len(self.results_transients_dict_by_cell[cell_to_display])
                    labels = list(self.results_transients_dict_by_cell[cell_to_display].keys())
                    values_by_prediction = [[] for n in np.arange(n_box_plots)]
                for label_index, label in enumerate(labels):
                    values_by_prediction[label_index].\
                        append(self.results_transients_dict_by_cell[cell_to_display][label][stat_name])

            colorfull = True

            bplot = ax.boxplot(values_by_prediction, patch_artist=colorfull,
                               labels=labels, sym='', zorder=1)  # whis=[5, 95], sym='+'

            for element in ['boxes', 'whiskers', 'fliers', 'caps']:
                plt.setp(bplot[element], color="white")

            for element in ['means', 'medians']:
                plt.setp(bplot[element], color="silver")

            if colorfull:
                colors = colors[:n_box_plots]
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_facecolor(color)

            # ax.set_ylabel(f"proportion")
            # ax.set_xlabel("age")
            xticks = np.arange(1, n_box_plots + 1)
            ax.set_xticks(xticks)
            # sce clusters labels
            ax.set_xticklabels(labels)

            ax.set_title(stat_name, color=title_color, pad=20)

        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            stat_fig.savefig(f'{path_results}/'
                             f'{description}_box_plots_predictions_transients_on_{n_cells}_cell'
                             f'_{time_str}.{save_format}',
                             format=f"{save_format}",
                             facecolor=stat_fig.get_facecolor(), edgecolor='none')


def build_spike_nums_dur(spike_nums, peak_nums):
    n_cells = len(spike_nums)
    n_frames = spike_nums.shape[1]
    spike_nums_dur = np.zeros((n_cells, n_frames), dtype="int8")
    for cell in np.arange(n_cells):
        peaks_index = np.where(peak_nums[cell, :])[0]
        onsets_index = np.where(spike_nums[cell, :])[0]

        for onset_index in onsets_index:
            peaks_after = np.where(peaks_index > onset_index)[0]
            if len(peaks_after) == 0:
                continue
            peaks_after = peaks_index[peaks_after]
            peak_after = peaks_after[0]

            spike_nums_dur[cell, onset_index:peak_after + 1] = 1
    return spike_nums_dur


def main_benchmark():
    root_path = None
    with open("param_hne.txt", "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split('=')
            root_path = line_list[1]
    if root_path is None:
        raise Exception("Root path is None")

    path_data = root_path + "data/"
    path_results_raw = root_path + "results_hne/"

    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    path_results = path_results_raw + f"{time_str}"
    os.mkdir(path_results)

    # ########### options ###################
    ms_to_benchmark = "p12_17_11_10_a000"
    do_onsets_benchmarks = False
    # ########### end options ###################

    data_dict = dict()
    if ms_to_benchmark == "p12_17_11_10_a000":
        # gt as ground_truth
        data_dict["gt"] = dict()
        # p12
        data_dict["gt"]["path"] = "p12/p12_17_11_10_a000"
        data_dict["gt"]["gui_file"] = "p12_17_11_10_a000_GUI_JD.mat"
        data_dict["gt"]["cnn"] = "p12_17_11_10_a000_cell_to_suppress_ground_truth.txt"
        data_dict["gt"]["cells"] = np.arange(7)

        data_dict["rnn"] = dict()
        data_dict["rnn"]["path"] = "p12/p12_17_11_10_a000"
        data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_01_26.19-22-21.mat"
        data_dict["rnn"]["var_name"] = "spike_nums_dur_predicted"
        data_dict["rnn"]["predictions"] = "predictions"

        data_dict["caiman_raw"] = dict()
        data_dict["caiman_raw"]["path"] = "p12/p12_17_11_10_a000"
        data_dict["caiman_raw"]["file_name"] = "p12_17_11_10_a000_RasterDur.mat"
        data_dict["caiman_raw"]["file_name_onsets"] = "Robin_28_01_19/p12_17_11_10_a000_Spikenums_caiman.mat"
        data_dict["caiman_raw"]["onsets_var_name"] = "spikenums"
        data_dict["caiman_raw"]["to_bin"] = True
        data_dict["caiman_raw"]["var_name"] = "rasterdur"

        data_dict["caiman_filt"] = dict()
        data_dict["caiman_filt"]["path"] = "p12/p12_17_11_10_a000"
        data_dict["caiman_filt"]["file_name"] = "p12_17_11_10_a000_filt_RasterDur_caiman.mat"
        data_dict["caiman_filt"]["file_name_onsets"] = "Robin_28_01_19/p12_17_11_10_a000_Bin100ms_spikedigital.mat"
        data_dict["caiman_filt"]["onsets_var_name"] = "Bin100ms_spikedigital"
        data_dict["caiman_filt"]["var_name"] = "rasterdur"

    # ground truth
    data_file = hdf5storage.loadmat(os.path.join(path_data, data_dict["gt"]["path"], data_dict["gt"]["gui_file"]))
    peak_nums = data_file['LocPeakMatrix_Python'].astype(int)
    spike_nums = data_file['Bin100ms_spikedigital_Python'].astype(int)
    inter_neurons = data_file['inter_neurons'].astype(int)
    cells_to_remove = data_file['cells_to_remove'].astype(int)
    ground_truth_raster_dur = build_spike_nums_dur(spike_nums, peak_nums)
    print(f"ground_truth_raster_dur.shape {ground_truth_raster_dur.shape}")

    cell_cnn_predictions = []
    with open(os.path.join(path_data, data_dict["gt"]["path"], data_dict["gt"]["cnn"]), "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split()
            cells_list = [float(i) for i in line_list]
            cell_cnn_predictions.extend(cells_list)
    cells_predicted_as_false = np.array(cell_cnn_predictions).astype(int)
    cells_for_benchmark = data_dict["gt"]["cells"]
    # adding cells not selected by cnn
    cells_for_benchmark = np.setdiff1d(cells_for_benchmark, cells_to_remove)
    # not taking into consideration cells that are not predicted as true from the cell classifier
    cells_for_benchmark = np.setdiff1d(cells_for_benchmark, cells_predicted_as_false)

    predicted_raster_dur_dict = dict()
    predicted_spike_nums_dict = dict()
    # value is a dict
    for key, value in data_dict.items():
        if key == "gt":
            continue
        data_file = hdf5storage.loadmat(os.path.join(path_data, value["path"], value["file_name"]))
        raster_dur = data_file[value['var_name']].astype(int)
        predicted_raster_dur_dict[key] = raster_dur

        if do_onsets_benchmarks:
            # onsets
            data_file = hdf5storage.loadmat(os.path.join(path_data, value["path"], value["file_name_onsets"]))
            predicted_spike_nums = data_file[value['onsets_var_name']].astype(int)
            if "to_bin" in value:
                # we need to bin predicted_spike_nums
                new_predicted_spike_nums = np.zeros((predicted_spike_nums.shape[0], predicted_spike_nums.shape[1] // 2),
                                                    dtype="int8")
                for cell in np.arange(predicted_spike_nums.shape[0]):
                    binned_cell = predicted_spike_nums[cell].reshape(-1, 2).mean(axis=1)
                    binned_cell[binned_cell > 0] = 1
                    new_predicted_spike_nums[cell] = binned_cell.astype("int")
                predicted_spike_nums = new_predicted_spike_nums
            print(f"predicted_spike_nums.shape {predicted_spike_nums.shape}")
            predicted_spike_nums_dict[key] = predicted_spike_nums

    benchmarks = BenchmarkRasterDur(description=ms_to_benchmark, ground_truth_raster_dur=ground_truth_raster_dur,
                                    predicted_raster_dur_dict=predicted_raster_dur_dict, cells=cells_for_benchmark)

    benchmarks.compute_stats()

    benchmarks.plot_boxplots_for_frames_stat(description=ms_to_benchmark, time_str=time_str, path_results=path_results,
                                             save_formats="pdf")
    benchmarks.plot_boxplots_for_transients_stat(description=ms_to_benchmark, time_str=time_str, path_results=path_results,
                                             save_formats="pdf")

    if do_onsets_benchmarks:
        print("")
        print("#######################################")
        print("#######################################")
        print("#######################################")
        print("")

        benchmarks_onsets = BenchmarkRasterDur(description=ms_to_benchmark, ground_truth_raster_dur=spike_nums,
                                               predicted_raster_dur_dict=predicted_spike_nums_dict,
                                               cells=cells_for_benchmark)

        benchmarks_onsets.compute_stats_on_onsets()


main_benchmark()
