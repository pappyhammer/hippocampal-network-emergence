import numpy as np
import classification_stat as cs
import hdf5storage
from datetime import datetime
import os
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal as signal
from pattern_discovery.tools.misc import get_continous_time_periods
from pattern_discovery.display.raster import plot_spikes_raster

from matplotlib.figure import SubplotParams
import matplotlib.gridspec as gridspec
from sortedcontainers import SortedDict
from pattern_discovery.tools.signal import smooth_convolve


class BenchmarkRasterDur:
    def __init__(self, description, ground_truth_raster_dur, predicted_raster_dur_dict, cells,
                 traces, debug_mode=True):
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
        self.debug_mode = debug_mode
        self.traces = traces

    def compute_stats_on_onsets(self):
        if self.debug_mode:
            print(f"{self.description} stats on onsets")
        for cell in self.cells:
            if self.debug_mode:
                print(f"Cell {cell}")
            for key, raster_dur in self.predicted_raster_dur_dict.items():
                gt_rd = self.ground_truth_raster_dur[cell]
                p_rd = raster_dur[cell]
                frames_stat = cs.compute_stats_on_onsets(spike_nums=gt_rd, predicted_spike_nums=p_rd)
                # frames stats
                if self.debug_mode:
                    print(f"raster {key}")
                    print(f"Onsets stat:")
                    for k, value in frames_stat.items():
                        print(f"{k}: {str(np.round(value, 4))}")
            if self.debug_mode:
                print("")
                print("/////////////////")
                print("")
        if self.debug_mode:
            print("All cells")
        for key, raster_dur in self.predicted_raster_dur_dict.items():
            gt_rd = self.ground_truth_raster_dur[self.cells]
            p_rd = raster_dur[self.cells]
            frames_stat = cs.compute_stats_on_onsets(gt_rd, p_rd)
            # frames stats
            if self.debug_mode:
                print(f"raster {key}")
                print(f"Onsets stat:")
                for k, value in frames_stat.items():
                    print(f"{k}: {str(np.round(value, 4))}")

    def compute_stats(self):
        if self.debug_mode:
            print(f"{self.description} stats on raster dur")
        for cell in self.cells:
            if self.debug_mode:
                print(f"Cell {cell}")
            self.results_frames_dict_by_cell[cell] = SortedDict()
            self.results_transients_dict_by_cell[cell] = SortedDict()
            for key, raster_dur in self.predicted_raster_dur_dict.items():
                gt_rd = self.ground_truth_raster_dur[cell]
                p_rd = raster_dur[cell]
                if self.traces is not None:
                    traces = self.traces[cell]
                else:
                    traces = None
                frames_stat, transients_stat = cs.compute_stats(gt_rd, p_rd,
                                                                traces=traces)
                self.results_frames_dict_by_cell[cell][key] = frames_stat
                self.results_transients_dict_by_cell[cell][key] = transients_stat
                if self.debug_mode:
                    # frames stats
                    print(f"raster {key}")
                    print(f"Frames stat:")
                    for k, value in frames_stat.items():
                        print(f"{k}: {str(np.round(value, 4))}")
                if self.debug_mode:
                    print(f"###")
                    print(f"Transients stat:")
                    for k, value in transients_stat.items():
                        print(f"{k}: {str(np.round(value, 4))}")
                    print("")
            if self.debug_mode:
                print("")
                print("/////////////////")
                print("")
        if self.debug_mode:
            print("All cells")
        for key, raster_dur in self.predicted_raster_dur_dict.items():
            gt_rd = self.ground_truth_raster_dur[self.cells]
            p_rd = raster_dur[self.cells]
            if self.traces is not None:
                traces = self.traces[self.cells]
            else:
                traces = None
            frames_stat, transients_stat = cs.compute_stats(gt_rd, p_rd,
                                                            traces=traces)
            # frames stats
            if self.debug_mode:
                print(f"raster {key}")
                print(f"Frames stat:")
                for k, value in frames_stat.items():
                    print(f"{k}: {str(np.round(value, 4))}")
            if self.debug_mode:
                print(f"###")
                print(f"Transients stat:")
                for k, value in transients_stat.items():
                    print(f"{k}: {str(np.round(value, 4))}")
                print("")

    def plot_boxplots_full_stat(self, path_results, description, time_str, for_frames=True, save_formats="pdf"):
        result_dict_to_use = self.results_frames_dict_by_cell
        if not for_frames:
            result_dict_to_use = self.results_transients_dict_by_cell
        stats_to_show = ["sensitivity", "specificity", "PPV", "NPV"]
        colors = ["cornflowerblue", "blue", "steelblue", "red", "orange"]

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
            for cell_index, cell_to_display in enumerate(result_dict_to_use.keys()):
                if n_box_plots is None:
                    n_box_plots = len(result_dict_to_use[cell_to_display])
                    labels = list(result_dict_to_use[cell_to_display].keys())
                    values_by_prediction = [[] for n in np.arange(n_box_plots)]
                for label_index, label in enumerate(labels):
                    values_by_prediction[label_index]. \
                        append(result_dict_to_use[cell_to_display][label][stat_name])

            colorfull = True
            outliers = dict(markerfacecolor='white', marker='D')

            bplot = ax.boxplot(values_by_prediction, patch_artist=colorfull,
                               flierprops=outliers,
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

        str_details = "frames"
        if not for_frames:
            str_details = "transients"
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            stat_fig.savefig(f'{path_results}/'
                             f'{description}_box_plots_predictions_{str_details}_on_{n_cells}_cells'
                             f'_{time_str}.{save_format}',
                             format=f"{save_format}",
                             facecolor=stat_fig.get_facecolor(), edgecolor='none')

    def plot_boxplots_for_transients_stat(self, path_results, description, time_str, save_formats="pdf"):
        stats_to_show = ["sensitivity"]
        colors = ["cornflowerblue", "blue", "steelblue", "red", "orange"]

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
                    values_by_prediction[label_index]. \
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


def manually_boost_rnn(title, path_data, raster_dur_file_name, trace_file_name, path_results, trace_var_name,
                       smooth_the_trace=True):
    data_traces = hdf5storage.loadmat(path_data + trace_file_name)
    traces = data_traces[trace_var_name].astype(float)

    data_raster_dur = hdf5storage.loadmat(path_data + raster_dur_file_name)
    raster_dur_predicted = data_raster_dur["spike_nums_dur_predicted"].astype(int)

    n_cells = traces.shape[0]
    n_times = traces.shape[1]

    for i in np.arange(n_cells):
        traces[i, :] = (traces[i, :] - np.mean(traces[i, :])) / np.std(traces[i, :])
        if smooth_the_trace:
            # smoothing the trace
            windows = ['hanning', 'hamming', 'bartlett', 'blackman']
            i_w = 1
            window_length = 11
            smooth_signal = smooth_convolve(x=traces[i], window_len=window_length,
                                            window=windows[i_w])
            beg = (window_length - 1) // 2
            traces[i] = smooth_signal[beg:-beg]

    peak_nums = np.zeros((n_cells, n_times), dtype="int8")
    for cell in np.arange(n_cells):
        peaks, properties = signal.find_peaks(x=traces[cell], distance=2)
        peak_nums[cell, peaks] = 1
    spike_nums = np.zeros((n_cells, n_times), dtype="int8")
    for cell in np.arange(n_cells):
        onsets = []
        diff_values = np.diff(traces[cell])
        for index, value in enumerate(diff_values):
            if index == (len(diff_values) - 1):
                continue
            if value < 0:
                if diff_values[index + 1] >= 0:
                    onsets.append(index + 1)
        spike_nums[cell, np.array(onsets)] = 1

    spike_nums_dur = np.zeros((n_cells, n_times), dtype="int8")
    for cell in np.arange(n_cells):
        peaks_index = np.where(peak_nums[cell, :])[0]
        onsets_index = np.where(spike_nums[cell, :])[0]

        for onset_index in onsets_index:
            peaks_after = np.where(peaks_index > onset_index)[0]
            if len(peaks_after) == 0:
                continue
            peaks_after = peaks_index[peaks_after]
            peak_after = peaks_after[0]
            if (peak_after - onset_index) > 200:
                print(f"long transient in cell {cell} of "
                      f"duration {peak_after - onset_index} frames at frame {onset_index}")

            spike_nums_dur[cell, onset_index:peak_after + 1] = 1

    for cell in np.arange(n_cells):
        predicted_periods = get_continous_time_periods(raster_dur_predicted[cell])

        for period in predicted_periods:
            # if the predicted period corresponds to no active frames in the spike_nums_dur
            # then it is considered false and we remove it
            beg = period[0]
            end = period[1]
            if np.sum(spike_nums_dur[cell, beg:end + 1]) == 0:
                raster_dur_predicted[cell, beg:end + 1] = 0

    path_results += "/"
    sio.savemat(path_results + f"{title}boost_rnn_raster_dur.mat", {'spike_nums_dur_predicted': raster_dur_predicted})


def get_raster_dur_from_caiman_25000_frames_onsets_new_version(caiman_spike_nums, traces):
    # we need to bin it, to get 12500 frames
    spike_nums_bin = np.zeros((caiman_spike_nums.shape[0], caiman_spike_nums.shape[1] // 2),
                              dtype="int8")
    for cell in np.arange(spike_nums_bin.shape[0]):
        binned_cell = caiman_spike_nums[cell].reshape(-1, 2).mean(axis=1)
        binned_cell[binned_cell > 0] = 1
        spike_nums_bin[cell] = binned_cell.astype("int")
    caiman_spike_nums = spike_nums_bin

    n_cells = traces.shape[0]
    n_times = traces.shape[1]

    for i in np.arange(n_cells):
        traces[i, :] = (traces[i, :] - np.mean(traces[i, :])) / np.std(traces[i, :])

    spike_nums_all = np.zeros((n_cells, n_times), dtype="int8")
    for cell in np.arange(n_cells):
        onsets = []
        diff_values = np.diff(traces[cell])
        for index, value in enumerate(diff_values):
            if index == (len(diff_values) - 1):
                continue
            if value < 0:
                if diff_values[index + 1] >= 0:
                    onsets.append(index + 1)
        if len(onsets) > 0:
            spike_nums_all[cell, np.array(onsets)] = 1

    peak_nums = np.zeros((n_cells, n_times), dtype="int8")
    for cell in np.arange(n_cells):
        peaks, properties = signal.find_peaks(x=traces[cell])
        peak_nums[cell, peaks] = 1

    # for cell in np.arange(n_cells):
    #     n_onsets = np.sum(spike_nums_all[cell])
    #     n_peaks = np.sum(peak_nums[cell])
    #     if n_onsets != n_peaks:
    #         print(f"n_onsets {n_onsets}, n_peaks {n_peaks}")

    spike_nums_dur = build_spike_nums_dur(spike_nums_all, peak_nums)
    #
    # plot_spikes_raster(spike_nums=spike_nums_dur[:20, :],
    #                    param=None,
    #                    traces=traces[:20, :],
    #                    display_traces=True,
    #                    spike_train_format=False,
    #                    title="traces raster",
    #                    file_name=f"traces raster",
    #                    y_ticks_labels_size=2,
    #                    save_raster=False,
    #                    show_raster=True,
    #                    plot_with_amplitude=False,
    #                    raster_face_color="white",
    #                    without_activity_sum=True,
    #                    span_area_only_on_raster=False,
    #                    spike_shape_size=1,
    #                    spike_shape="o",
    #                    cell_spikes_color='black',
    #                    display_spike_nums=True,
    #                    save_formats="pdf")

    # raise Exception("test toto")
    caiman_spike_nums_dur = np.zeros((spike_nums_dur.shape[0], spike_nums_dur.shape[1]), dtype="int8")
    for cell in np.arange(n_cells):
        periods = get_continous_time_periods(spike_nums_dur[cell])
        for period in periods:
            if np.sum(caiman_spike_nums[cell, period[0]:period[1] + 1]) > 0:
                caiman_spike_nums_dur[cell, period[0]:period[1] + 1] = 1

    return caiman_spike_nums_dur


def get_boost_rnn_raster_dur(rnn_raster_dur, traces):
    n_cells = traces.shape[0]
    n_times = traces.shape[1]

    for i in np.arange(n_cells):
        traces[i, :] = (traces[i, :] - np.mean(traces[i, :])) / np.std(traces[i, :])

    spike_nums_all = np.zeros((n_cells, n_times), dtype="int8")
    for cell in np.arange(n_cells):
        onsets = []
        diff_values = np.diff(traces[cell])
        for index, value in enumerate(diff_values):
            if index == (len(diff_values) - 1):
                continue
            if value < 0:
                if diff_values[index + 1] >= 0:
                    onsets.append(index + 1)
        if len(onsets) > 0:
            spike_nums_all[cell, np.array(onsets)] = 1

    peak_nums = np.zeros((n_cells, n_times), dtype="int8")
    for cell in np.arange(n_cells):
        peaks, properties = signal.find_peaks(x=traces[cell])
        peak_nums[cell, peaks] = 1

    spike_nums_dur = build_spike_nums_dur(spike_nums_all, peak_nums)

    new_rnn_raster_dur = np.copy(rnn_raster_dur)
    for cell in np.arange(n_cells):
        periods = get_continous_time_periods(spike_nums_dur[cell])
        for period in periods:
            if np.sum(rnn_raster_dur[cell, period[0]:period[1] + 1]) > 0:
                new_rnn_raster_dur[cell, period[0]:period[1] + 1] = 1

    return new_rnn_raster_dur


def build_raster_dur_from_caiman_25000_frames_onsets(file_name, var_name, trace_file_name, trace_var_name,
                                                     description, path_results):
    data_onsets = hdf5storage.loadmat(file_name)
    # print(f"data_onsets {list(data_onsets.keys())}")
    # return
    caiman_spike_nums = data_onsets[var_name].astype(float)

    print(f"caiman_spike_nums.shape {caiman_spike_nums.shape}")

    # we need to bin it, to get 12500 frames
    spike_nums_bin = np.zeros((caiman_spike_nums.shape[0], caiman_spike_nums.shape[1] // 2),
                              dtype="int8")
    for cell in np.arange(spike_nums_bin.shape[0]):
        binned_cell = caiman_spike_nums[cell].reshape(-1, 2).mean(axis=1)
        binned_cell[binned_cell > 0] = 1
        spike_nums_bin[cell] = binned_cell.astype("int")
    caiman_spike_nums = spike_nums_bin

    print(f"caiman_spike_nums.shape {caiman_spike_nums.shape}")

    data_traces = hdf5storage.loadmat(trace_file_name)
    traces = data_traces[trace_var_name].astype(float)

    n_cells = traces.shape[0]
    n_times = traces.shape[1]

    for i in np.arange(n_cells):
        traces[i, :] = (traces[i, :] - np.mean(traces[i, :])) / np.std(traces[i, :])
        # if smooth_the_trace:
        #     # smoothing the trace
        #     windows = ['hanning', 'hamming', 'bartlett', 'blackman']
        #     i_w = 1
        #     window_length = 11
        #     smooth_signal = smooth_convolve(x=traces[i], window_len=window_length,
        #                                     window=windows[i_w])
        #     beg = (window_length - 1) // 2
        #     traces[i] = smooth_signal[beg:-beg]

    # detecting all onsets
    spike_nums_all = np.zeros((n_cells, n_times), dtype="int8")
    for cell in np.arange(n_cells):
        onsets = []
        diff_values = np.diff(traces[cell])
        for index, value in enumerate(diff_values):
            if index == (len(diff_values) - 1):
                continue
            if value < 0:
                if diff_values[index + 1] >= 0:
                    onsets.append(index + 1)
        spike_nums_all[cell, np.array(onsets)] = 1

    caiman_peak_nums = np.zeros((n_cells, n_times), dtype="int8")
    for cell in np.arange(n_cells):
        peaks, properties = signal.find_peaks(x=traces[cell], distance=2)
        caiman_peak_nums[cell, peaks] = 1

    spike_nums_dur = np.zeros((n_cells, n_times), dtype="int8")

    for cell in np.arange(n_cells):
        peaks_index = np.where(caiman_peak_nums[cell])[0]
        # for each peaks we look at onsets before, until an onset is at higher amplitude than the previous one
        # if none of the onset are from caiman, we don't consider this peak as part of a transient
        # otherwise we add the transient to the raster_dur, using the caiman onset with the lowest amplitudes
        caiman_onsets_index = np.where(caiman_spike_nums[cell])[0]
        all_onsets_index = np.where(spike_nums_all[cell])[0]
        n_transients = 0
        for peak_index in peaks_index:
            onsets_before_peak = all_onsets_index[all_onsets_index < peak_index]
            if len(onsets_before_peak) == 0:
                continue
            # looking for the lowest amplitude onsets
            last_amplitude = None
            lowest_onset = None
            peak_amplitude = traces[cell, peak_index]
            last_peak_index = peak_index
            onset_found = True
            for onset_index in onsets_before_peak[::-1]:
                if last_amplitude is None:
                    last_amplitude = traces[cell, onset_index]
                    lowest_onset = onset_index
                    if last_amplitude > peak_amplitude:
                        onset_found = False
                        break
                else:
                    amplitude = traces[cell, onset_index]
                    if amplitude > last_amplitude:
                        # then it's over
                        break
                    else:
                        # first we want to check if no peak between the last peak and the index is higher than the
                        # lowest peak encounter
                        new_peaks_index = peaks_index[np.logical_and(peaks_index > onset_index,
                                                                     peaks_index < last_peak_index)]
                        if len(new_peaks_index) > 0:
                            if traces[cell, new_peaks_index[0]] > traces[cell, last_peak_index]:
                                # print(f"cell {cell}: break peak > last_peak")
                                break
                            last_peak_index = new_peaks_index[0]
                        last_amplitude = amplitude
                        lowest_onset = onset_index
            if not onset_found:
                continue
            else:
                # now we want to see if there is a caiman onset between this onset and the peak
                # taking the furthest one
                # we allow a 2 frames imprecision, due to the bin operation
                if lowest_onset > 1:
                    lowest_onset -= 2
                # print(f"cell {cell}: len lowest to peak {peak_index + 1 - lowest_onset}")
                caiman_champion = np.where(np.logical_and(caiman_onsets_index >= lowest_onset,
                                                          caiman_onsets_index < peak_index))[0]
                if len(caiman_champion) > 0:
                    spike_nums_dur[cell, caiman_onsets_index[caiman_champion[0]]:peak_index + 1] = 1
                    # print(f"cell {cell}: len transient {peak_index + 1 - caiman_onsets_index[caiman_champion[0]]}")
                    n_transients += 1
        print(f"cell {cell}: n_transients {n_transients}")
        periods = get_continous_time_periods(spike_nums_dur[cell])
        print(f"cell {cell}: real n_transients {len(periods)}")
    path_results += "/"
    sio.savemat(path_results + f"{description}_caiman_raster_dur_JD_version.mat", {'rasterdur': spike_nums_dur})


def build_p7_17_10_12_a000_raster_dur_caiman(path_data, path_results):
    path_data += "p7/p7_17_10_12_a000/"
    file_name_onsets = "Robin_30_01_19/p7_17_10_12_a000_filt_Bin100ms_spikedigital.mat"
    file_name_trace = "p7_17_10_12_a000_Traces.mat"

    data_onsets = hdf5storage.loadmat(path_data + file_name_onsets)
    spike_nums = data_onsets["filt_Bin100ms_spikedigital"].astype(float)
    # raster_dur with just 1 sec filter
    data_raster_dur = hdf5storage.loadmat(path_data + "Robin_30_01_19/p7_17_10_12_a000_RasterDur.mat")
    raster_dur_non_filt = data_raster_dur["rasterdur"].astype(int)

    n_cells = len(spike_nums)
    n_frames = spike_nums.shape[1]

    # building peak_nums

    peak_nums = np.zeros((n_cells, n_frames), dtype="int8")
    for cell in np.arange(n_cells):
        time_periods = get_continous_time_periods(raster_dur_non_filt[cell])
        for time_period in time_periods:
            peak_nums[cell, time_period[1]] = 1

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
            if (peak_after - onset_index) > 200:
                print(f"long transient in cell {cell} of "
                      f"duration {peak_after - onset_index} frames at frame {onset_index}")

            spike_nums_dur[cell, onset_index:peak_after + 1] = 1

    path_results += "/"
    sio.savemat(path_results + "p7_17_10_12_a000_caiman_raster_dur.mat", {'rasterdur': spike_nums_dur})


def build_raster_dur_from_predictions(predictions, predictions_threshold, cells, n_total_cells, n_frames):
    predicted_raster_dur_dict = np.zeros((n_total_cells, n_frames), dtype="int8")
    for cell in cells:
        pred = predictions[cell]

        # predicted_raster_dur_dict[cell, pred >= predictions_threshold] = 1
        if len(pred.shape) == 1:
            predicted_raster_dur_dict[cell, pred >= predictions_threshold] = 1
        elif (len(pred.shape) == 2) and (pred.shape[1] == 1):
            pred = pred[:, 0]
            predicted_raster_dur_dict[cell, pred >= predictions_threshold] = 1
        elif (len(pred.shape) == 2) and (pred.shape[1] == 3):
            # real transient, fake ones, other (neuropil, decay etc...)
            # keeping predictions about real transient when superior
            # to other prediction on the same frame
            max_pred_by_frame = np.max(pred, axis=1)
            real_transient_frames = (pred[:, 0] == max_pred_by_frame)
            predicted_raster_dur_dict[cell, real_transient_frames] = 1
        elif pred.shape[1] == 2:
            # real transient, fake ones
            # keeping predictions about real transient superior to the threshold
            # and superior to other prediction on the same frame
            max_pred_by_frame = np.max(pred, axis=1)
            real_transient_frames = np.logical_and((pred[:, 0] >= predictions_threshold),
                                                   (pred[:, 0] == max_pred_by_frame))
            predicted_raster_dur_dict[cell, real_transient_frames] = 1

    return predicted_raster_dur_dict


def plot_roc_predictions(ground_truth_raster_dur, rnn_predictions, cells,
                         path_results, description, time_str,
                         save_formats):
    n_frames = ground_truth_raster_dur.shape[1]
    n_cells = ground_truth_raster_dur.shape[0]
    sensitivity_values = []
    specificity_values = []
    threshold_values = np.arange(0, 1.05, 0.05)

    for predictions_threshold in threshold_values:
        # building the raster_dur
        predicted_raster_dur_dict = build_raster_dur_from_predictions(predictions=rnn_predictions,
                                                                      predictions_threshold=predictions_threshold,
                                                                      cells=cells,
                                                                      n_total_cells=n_cells, n_frames=n_frames)

        raster_dict = {"raster_dur": predicted_raster_dur_dict}
        benchmarks = BenchmarkRasterDur(description=description, ground_truth_raster_dur=ground_truth_raster_dur,
                                        predicted_raster_dur_dict=raster_dict, cells=cells, traces=None,
                                        debug_mode=False)

        benchmarks.compute_stats()
        avg_sensitivity = []
        avg_specificity = []
        for cell in cells:
            avg_sensitivity.append(benchmarks.results_frames_dict_by_cell[cell]["raster_dur"]["sensitivity"])
            avg_specificity.append(benchmarks.results_frames_dict_by_cell[cell]["raster_dur"]["specificity"])
        sensitivity_values.append(np.mean(avg_sensitivity))
        specificity_values.append(np.mean(avg_specificity))

    specificity_values = np.array(specificity_values)

    roc_fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True,
                               gridspec_kw={'height_ratios': [1],
                                            'width_ratios': [1]},
                               figsize=(8, 8))

    roc_fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'w_pad': 1, 'h_pad': 5})
    # axes = np.ndarray.flatten(axes)
    fig_patch = roc_fig.patch
    # rgba = c_map(0)
    face_color = "black"
    text_color = "white"
    title_color = "red"
    fig_patch.set_facecolor(face_color)

    ax.set_facecolor(face_color)
    # ax.xaxis.set_ticks_position('none')
    ax.xaxis.label.set_color(text_color)
    ax.tick_params(axis='x', colors=text_color)
    ax.yaxis.label.set_color(text_color)
    ax.tick_params(axis='y', colors=text_color)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.get_yaxis().set_visible(False)
    # ax.get_xaxis().set_visible(False)

    ax.set_frame_on(False)

    ax.plot(1 - specificity_values, sensitivity_values, color="red", lw=2)  # whis=[5, 95], sym='+'
    for index, predictions_threshold in enumerate(threshold_values):
        ax.text(x=1 - specificity_values[index], y=sensitivity_values[index],
                s=f"{str(np.round(predictions_threshold, 2))}", color="white", zorder=22,
                ha='center', va="center", fontsize=5, fontweight='bold')

    ax.set_ylabel(f"sensitivity")
    ax.set_xlabel("1 - specificity")
    # xticks = np.arange(1, n_box_plots + 1)
    # ax.set_xticks(xticks)
    # sce clusters labels
    # ax.set_xticklabels(labels)

    ax.set_title("ROC", color=title_color, pad=20)
    # ax.set_xscale("log")

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        roc_fig.savefig(f'{path_results}/'
                        f'{description}_ROC_threshold_predictions_on_{len(cells)}_cells'
                        f'_{time_str}.{save_format}',
                        format=f"{save_format}",
                        facecolor=roc_fig.get_facecolor(), edgecolor='none')


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

    do_build_raster_dur_on_25000 = False
    if do_build_raster_dur_on_25000:
        build_p12 = False
        build_p7 = True
        if build_p12:
            path_data = path_data + "p12/p12_17_11_10_a000/"
            file_name = path_data + "robin_28_01_19/" + "p12_17_11_10_a000_Spikenums_caiman.mat"
            var_name = "spikenums"
            trace_file_name = path_data + "p12_17_11_10_a000_Traces.mat"
            trace_var_name = "C_df"
            description = "p12_17_11_10_a000"
        if build_p7:
            path_data = path_data + "p7/p7_17_10_12_a000/"
            file_name = path_data + "Robin_30_01_19/" + "p7_17_10_12_a000_spikenums.mat"
            var_name = "spikenums"
            trace_file_name = path_data + "p7_17_10_12_a000_Traces.mat"
            trace_var_name = "C_df"
            description = "p7_17_10_12_a000"

        build_raster_dur_from_caiman_25000_frames_onsets(file_name=file_name, var_name=var_name,
                                                         trace_file_name=trace_file_name, trace_var_name=trace_var_name,
                                                         description=description,
                                                         path_results=path_results)
        return

    build_p7 = False
    if build_p7:
        build_p7_17_10_12_a000_raster_dur_caiman(path_data=path_data, path_results=path_results)
        return

    boost_rnn = False
    if boost_rnn:
        title = "p7_17_10_12_a000"
        path_data = path_data + "p7/p7_17_10_12_a000/"
        raster_dur_file_name = "P7_17_10_12_a000_predictions_2019_01_31.19-26-49.mat"

        # trace_file_name = "p7_17_10_12_a000_raw_Traces.mat"
        # trace_var_name = "raw_traces"
        # smooth_the_trace = True

        trace_file_name = "p7_17_10_12_a000_Traces.mat"
        trace_var_name = "C_df"
        smooth_the_trace = False

        manually_boost_rnn(title=title, path_data=path_data, raster_dur_file_name=raster_dur_file_name,
                           trace_file_name=trace_file_name, path_results=path_results,
                           trace_var_name=trace_var_name, smooth_the_trace=smooth_the_trace)
        return

    # ########### options ###################
    ms_to_benchmark = "p12_17_11_10_a000"
    # ms_to_benchmark = "p7_17_10_12_a000"
    # ms_to_benchmark = "p13_18_10_29_a001_ms"
    # ms_to_benchmark = "p8_18_10_24_a005_ms"
    # ms_to_benchmark = "artificial_ms"
    do_onsets_benchmarks = False
    do_plot_roc_predictions = False
    # ########### end options ###################

    data_dict = dict()
    if ms_to_benchmark == "p12_17_11_10_a000":
        # gt as ground_truth
        data_dict["gt"] = dict()
        data_dict["gt"]["path"] = "p12/p12_17_11_10_a000"
        data_dict["gt"]["gui_file"] = "p12_17_11_10_a000_GUI_JD.mat"
        data_dict["gt"]["gt_file"] = "p12_17_11_10_a000_cell_to_suppress_ground_truth.txt"
        data_dict["gt"]["cnn"] = "cell_classifier_results_txt/cell_classifier_cnn_results_P12_17_11_10_a000.txt"
        data_dict["gt"]["cnn_threshold"] = 0.5
        # TODO: with cell 10, we need to re-make the prediction
        data_dict["gt"]["cells"] =  np.concatenate((np.arange(11), [14])) # np.array([14])

        data_dict["rnn"] = dict()
        data_dict["rnn"]["path"] = "p12/p12_17_11_10_a000"
        # if traces is given, then rnn will be boosted
        data_dict["rnn"]["trace_file_name"] = "p12_17_11_10_a000_Traces.mat"
        data_dict["rnn"]["trace_var_name"] = "C_df"
        data_dict["rnn"]["boost_rnn"] = False
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_03.19-16-43.mat"

        # "P12_17_11_10_a000_predictions_2019_02_03.19-16-43.mat" based on best 2 p12 cells predictions
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_01_26.19-22-21.mat"

        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_from_5_sessions_2019_02_05.23-37-09.mat"

        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_06.22-06-13_from_p8_training.mat"

        # trained on 0 & 3 cell, with just the cell mask, on 50 epochs. trained on 13/02/2019 19:39:49
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_13.21-40-46.mat"

        # ## trained on 0,3 cell, with 2 inputs (cell masked + all), on 20 epochs. trained on 13/02/2019 12:24:23
        # BO so far
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_14.19-07-26.mat"
        # trained on 0,3 cell, with 2 inputs (cell masked + all), on 34 epochs. trained on 13/02/2019 14:34:45
        # with predictions up to cell 9
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_14.19-19-05.mat"
        # with predictions up to cell 10+ cell 14
        data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_19.14-20-01.mat"
        # with predictions up to cell 10+ cell 14 without data_augmentation, just overlap 0.8
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_19.14-54-24.mat"

        # ## trained on p12 0,3 cell, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 16/02/2019 11:21:11 BO equality
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_16.14-57-49.mat"

        # trained on 5 sessions and 15 cells, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 15/02/2019 00:01:55, on epoch 13 with overlap at 0.8
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_16.15-17-17.mat"

        # trained on 5 sessions and 15 cells, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 15/02/2019 00:01:55, on epoch 13 with overlap at 0.5
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_16.16-38-24.mat"

        # trained on 5 sessions and 15 cells, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 15/02/2019 00:01:55, on epoch 13 with overlap at 0.9
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_16.16-38-24.mat"
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_16.16-46-58.mat"

        # trained from 2019_02_16.18-23-24_p12_0_3_2_inputs_buffer_1_dropout
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_16.21-39-50.mat"

        # ## trained on p12 0,3, 6 cell, with 3 inputs (cell masked + cells masked + neuropil mask),
        # with bin et al. v, without attention, trained on 19/02/2019 00-56-47, up to cell 10 + 14
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_19.13-36-27.mat"

        # ## trained on p12 0 cell, with 1 inputs (cell masked),
        # with bin et al. v, with attention before, trained on 19/02/2019 15-33-47, up to cell 10 + 14
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_19.15-42-02.mat"

        # ## trained on p12 0 cell, with 1 inputs (cell masked), 8 epochs
        # with bin et al. v, with attention before, trained on 19/02/2019 15-33-47, up to cell 10 + 14
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_19.16-28-12.mat"

        # ## trained on p12 0 cell, with 1 inputs (cell masked), 10 epochs, multi-class
        # with bin et al. v, with attention before, trained on 19/02/2019 16-52-15, up to cell 10 + 14
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_19.18-21-44.mat"

        # ## trained on p12 0,3 cell, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 20/02/2019 12:20:24 with bin et al. version + atttention before
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_20.15-39-51.mat"

        # ## trained on p12 0,3 cell, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 20/02/2019 12:20:24 with bin et al. version + atttention before, using last epoch
        data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_20.16-29-06.mat"

        # ## trained on p12 0,3, 6, 9 cell, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 20/02/2019 21-57-23 with bin et al. version + atttention before
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_20.16-09-29.mat"

        # ## trained on p12 0,3 cell, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 20/02/2019  with bin et al. version + atttention before, multi-class
        data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_20.19-50-17.mat"

        # ## trained on p12 0,3 cell, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 20/02/2019 12:20:24 with bin et al. version + atttention before, using avt-derniere epoch
        data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_20.20-15-26.mat"

        # ## trained on 4 sessions, 13 cells, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 21/02/2019 00-47-30 with bin et al. version + atttention before
        data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_21.18-16-21.mat"

        # ## trained on artificial data , 4 cells, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 21/02/2019 21-11-31 with bin et al. version + atttention before predictions up to cell 9 + 14
        data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_22.14-40-26.mat"

        # ## trained on artificial data , 4 cells + p12 2 cells,
        # with 3 inputs (cell masked + cells masked + neuropil mask), BO
        # trained on 21/02/2019 22-24-00 with bin et al. version + atttention before predictions up to cell 9 + 14
        data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_22.16-08-44.mat"

        # ## trained on artificial data , 4 cells + p12 1 cell, with 1 inputs (global + contour),
        # trained on 23/02/2019 18-36-53 with bin et al. version + atttention before predictions up to cell 10 + 14
        data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_23.19-10-46.mat"

        # ## trained on artificial data , 4 cells + p12 2 cells,
        # with 3 inputs (cell masked + cells masked + neuropil mask), BO
        # trained on 21/02/2019 22-24-00 with bin et al. version + atttention before predictions for all cnn valide cell
        data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_26.08-43-06_all_cnn_cells_arti_p12_2_cells.mat"

        # trained on 50 cells + artificial data, 3 inputs, with overlap 0.7 and 2 transformations
        # rnn trained on 26/02/2019 17-20-11 on 391 cells
        # data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_03_14.08-16-02_391_cells.mat"

        # trained on 50 cells + artificial data, 3 inputs, with overlap 0.9 and 3 transformations
        # rnn trained on 26/02/2019 17-20-11 on 391 cells
        data_dict["rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_03_14.20-19-48.mat"

        data_dict["rnn"]["var_name"] = "spike_nums_dur_predicted"
        data_dict["rnn"]["predictions"] = "predictions"
        data_dict["rnn"]["prediction_threshold"] = 0.5

        data_dict["last_rnn"] = dict()
        data_dict["last_rnn"]["path"] = "p12/p12_17_11_10_a000"
        # ## trained on p12 0,3 cell, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 16/02/2019 11:21:11 BO equality, predictions up to cell 10 + 14
        # data_dict["last_rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_19.14-20-01.mat"
        # ## trained on artificial data , 4 cells + p12 2 cells,
        # with 3 inputs (cell masked + cells masked + neuropil mask), BO
        # trained on 21/02/2019 22-24-00 with bin et al. version + atttention before predictions for all cnn valide cell
        data_dict["last_rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_26.08-43-06_all_cnn_cells_arti_p12_2_cells.mat"
        # # ## trained on artificial data , 4 cells + p12 2 cells, with 3 inputs (cell masked + cells masked + neuropil mask),
        # # trained on 21/02/2019 222-24-00 with bin et al. version + atttention before predictions up to cell 9 + 14
        # data_dict["last_rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_22.16-08-44.mat"

        data_dict["last_rnn"]["predictions"] = "predictions"
        data_dict["last_rnn"]["prediction_threshold"] = 0.6

        # data_dict["old_rnn"] = dict()
        # data_dict["old_rnn"]["path"] = "p12/p12_17_11_10_a000"
        # data_dict["old_rnn"]["file_name"] = "P12_17_11_10_a000_predictions_2019_02_03.19-16-43.mat"
        # # "P12_17_11_10_a000_predictions_2019_02_03.19-16-43.mat" based on best 2 p12 cells predictions
        # data_dict["old_rnn"]["var_name"] = "spike_nums_dur_predicted"
        # data_dict["old_rnn"]["predictions"] = "predictions"
        # data_dict["old_rnn"]["prediction_threshold"] = 0.4

        data_dict["caiman_raw"] = dict()
        data_dict["caiman_raw"]["path"] = "p12/p12_17_11_10_a000"
        data_dict["caiman_raw"]["file_name"] = "p12_17_11_10_a000_RasterDur.mat"
        data_dict["caiman_raw"]["file_name_onsets"] = "robin_28_01_19/p12_17_11_10_a000_Spikenums_caiman.mat"
        data_dict["caiman_raw"]["onsets_var_name"] = "spikenums"
        data_dict["caiman_raw"]["to_bin"] = True
        data_dict["caiman_raw"]["var_name"] = "rasterdur"
        data_dict["caiman_raw"]["trace_file_name"] = "p12_17_11_10_a000_Traces.mat"
        data_dict["caiman_raw"]["trace_var_name"] = "C_df"
        # "p12_17_11_10_a000_caiman_raster_dur_JD_version.mat"

        # data_dict["caiman_jd"] = dict()
        # data_dict["caiman_jd"]["path"] = "p12/p12_17_11_10_a000"
        # data_dict["caiman_jd"]["file_name"] = "p12_17_11_10_a000_caiman_raster_dur_JD_version.mat"
        # data_dict["caiman_jd"]["var_name"] = "rasterdur"

        data_dict["caiman_filt"] = dict()
        data_dict["caiman_filt"]["path"] = "p12/p12_17_11_10_a000"
        data_dict["caiman_filt"]["file_name"] = "p12_17_11_10_a000_filt_RasterDur_caiman.mat"
        data_dict["caiman_filt"]["file_name_onsets"] = "robin_28_01_19/p12_17_11_10_a000_Bin100ms_spikedigital.mat"
        data_dict["caiman_filt"]["onsets_var_name"] = "Bin100ms_spikedigital"
        data_dict["caiman_filt"]["var_name"] = "rasterdur"

    elif ms_to_benchmark == "artificial_ms":
        data_dict["gt"] = dict()
        data_dict["gt"]["path"] = "artificial_movies"
        data_dict["gt"]["gui_file"] = "gui_data.mat"
        data_dict["gt"]["cells"] = np.array([0, 13, 23, 30, 45, 53, 63, 71, 84, 94, 101, 106, 119, 128, 133, 144])

        data_dict["rnn"] = dict()
        data_dict["rnn"]["path"] = "artificial_movies"
        # if traces is given, then rnn will be boosted
        # data_dict["rnn"]["trace_file_name"] = "p12_17_11_10_a000_Traces.mat"
        # data_dict["rnn"]["trace_var_name"] = "C_df"
        data_dict["rnn"]["boost_rnn"] = False
        data_dict["rnn"]["file_name"] = "P10_artificial_predictions_2019_02_22.12-49-09.mat"

        data_dict["rnn"]["var_name"] = "spike_nums_dur_predicted"
        data_dict["rnn"]["predictions"] = "predictions"
        data_dict["rnn"]["prediction_threshold"] = 0.6
    elif ms_to_benchmark == "p13_18_10_29_a001_ms":
        # gt as ground_truth
        data_dict["gt"] = dict()
        data_dict["gt"]["path"] = "p13/p13_18_10_29_a001"
        data_dict["gt"]["gui_file"] = "p13_18_10_29_a001_GUI_transients_RD.mat"
        data_dict["gt"]["cnn"] = "cell_classifier_results_txt/cell_classifier_cnn_results_P13_18_10_29_a001.txt"
        data_dict["gt"]["cnn_threshold"] = 0.5
        data_dict["gt"]["cells"] = np.array([0, 5, 12, 13, 31, 42, 44, 48, 51, 77, 117])

        data_dict["rnn"] = dict()
        data_dict["rnn"]["path"] = "p13/p13_18_10_29_a001"
        data_dict["rnn"]["file_name"] = "P13_18_10_29_a001_predictions_2019_02_05.22-54-05.mat"
        # P13_18_10_29_a001_predictions_2019_02_05.22-54-05.mat trained on 5 sessions, 10 cells
        data_dict["rnn"]["var_name"] = "spike_nums_dur_predicted"
        data_dict["rnn"]["predictions"] = "predictions"
        data_dict["rnn"]["prediction_threshold"] = 0.4

        # data_dict["caiman_filt"] = dict()
        # data_dict["caiman_filt"]["path"] = "p13/p13_18_10_29_a001"
        # data_dict["caiman_filt"]["file_name"] = "p12_17_11_10_a000_filt_RasterDur_caiman.mat"
        # data_dict["caiman_filt"]["file_name_onsets"] = "Robin_28_01_19/p12_17_11_10_a000_Bin100ms_spikedigital.mat"
        # data_dict["caiman_filt"]["onsets_var_name"] = "Bin100ms_spikedigital"
        # data_dict["caiman_filt"]["var_name"] = "rasterdur"
    elif ms_to_benchmark == "p7_17_10_12_a000":
        # gt as ground_truth
        data_dict["gt"] = dict()
        data_dict["gt"]["path"] = "p7/p7_17_10_12_a000"
        data_dict["gt"]["gui_file"] = "p7_17_10_12_a000_GUI_transients_RD.mat"
        data_dict["gt"]["gt_file"] = "p7_17_10_12_a000_cell_to_suppress_ground_truth.txt"
        data_dict["gt"]["cnn"] = "cell_classifier_results_txt/cell_classifier_cnn_results_P7_17_10_12_a000.txt"
        data_dict["gt"]["cnn_threshold"] = 0.5
        data_dict["gt"]["cells"] = np.arange(100)
        # data_dict["gt"]["cells_to_remove"] = np.array([52, 75])


        data_dict["rnn"] = dict()
        data_dict["rnn"]["path"] = "p7/p7_17_10_12_a000"
        data_dict["rnn"]["trace_file_name"] = "p7_17_10_12_a000_Traces.mat"
        data_dict["rnn"]["trace_var_name"] = "C_df"
        data_dict["rnn"]["boost_rnn"] = False
        # not of these two better than caiman
        # data_dict["rnn"]["file_name"] = "P7_17_10_12_a000_predictions_2019_02_01.15-56-10.mat"
        # BO ?
        data_dict["rnn"]["file_name"] = "P7_17_10_12_a000_predictions_2019_01_31.19-26-49.mat"
        # bad results
        # data_dict["rnn"]["file_name"] = "P7_17_10_12_a000_predictions_2019_02_06.12-53-02_on_6_cells_overfitting.mat"
        # data_dict["rnn"]["file_name"] = "P7_17_10_12_a000_predictions_2019_02_06.14-58-40_5_sessions_training.mat"
        # not good
        # data_dict["rnn"]["file_name"] = "P7_17_10_12_a000_predictions_2019_02_06.20-48-56_on_2_cells_02_02_19_1_30_26.mat"
        #  P7_17_10_12_a000_predictions_2019_02_06.20-48-56_on_2_cells_02_02_19_1_30_26.mat
        # trained on 5 sessions and 15 cells, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 15/02/2019 00:01:55, on epoch 13 with overlap at 0.8
        data_dict["rnn"]["file_name"] = "P7_17_10_12_a000_predictions_2019_02_16.17-29-37.mat"
        # trained on p7 3 cells + artificial, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 22/02/2019 19:16:53, on epoch 14 with overlap at 0.8 (from cell 20 to 50)
        data_dict["rnn"]["file_name"] = "P7_17_10_12_a000_predictions_2019_02_23.12-16-46.mat"
        # trained on 50 cells (p11, p12, p13) + artificial, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 26/02/2019 17-20-11, on epoch 21 with overlap at 0.8 (until cell 99 included)
        data_dict["rnn"]["file_name"] = "P7_17_10_12_a000_predictions_2019_03_08.14-15-54.mat"
        data_dict["rnn"]["var_name"] = "spike_nums_dur_predicted"
        data_dict["rnn"]["predictions"] = "predictions"
        data_dict["rnn"]["prediction_threshold"] = 0.35

        data_dict["last_rnn"] = dict()
        data_dict["last_rnn"]["path"] = "p7/p7_17_10_12_a000"
        # BO before
        data_dict["last_rnn"]["file_name"] = "P7_17_10_12_a000_predictions_2019_01_31.19-26-49.mat"
        data_dict["last_rnn"]["var_name"] = "spike_nums_dur_predicted"
        data_dict["last_rnn"]["predictions"] = "predictions"
        data_dict["last_rnn"]["prediction_threshold"] = 0.4

        # data_dict["caiman_jd"] = dict()
        # data_dict["caiman_jd"]["path"] = "p7/p7_17_10_12_a000"
        # data_dict["caiman_jd"]["file_name"] = "p7_17_10_12_a000_caiman_raster_dur_JD_version.mat"
        # data_dict["caiman_jd"]["var_name"] = "rasterdur"

        data_dict["caiman_raw"] = dict()
        data_dict["caiman_raw"]["path"] = "p7/p7_17_10_12_a000"
        data_dict["caiman_raw"]["file_name_onsets"] = "Robin_30_01_19/p7_17_10_12_a000_spikenums.mat"
        data_dict["caiman_raw"]["onsets_var_name"] = "spikenums"
        data_dict["caiman_raw"]["to_bin"] = True
        data_dict["caiman_raw"]["trace_file_name"] = "p7_17_10_12_a000_Traces.mat"
        data_dict["caiman_raw"]["trace_var_name"] = "C_df"

        data_dict["caiman_filt"] = dict()
        data_dict["caiman_filt"]["path"] = "p7/p7_17_10_12_a000"
        data_dict["caiman_filt"]["file_name"] = "Robin_30_01_19/p7_17_10_12_a000_caiman_raster_dur.mat"
        data_dict["caiman_filt"]["var_name"] = "rasterdur"
    elif ms_to_benchmark == "p8_18_10_24_a005_ms":
        # gt as ground_truth
        data_dict["gt"] = dict()
        data_dict["gt"]["path"] = "p8/p8_18_10_24_a005"
        data_dict["gt"]["gui_file"] = "p8_18_10_24_a005_GUI_transientsRD.mat"
        data_dict["gt"]["cnn"] = "cell_classifier_results_txt/cell_classifier_cnn_results_P8_18_10_24_a005.txt"
        data_dict["gt"]["cnn_threshold"] = 0.5
        data_dict["gt"]["cells"] = np.array([9, 10, 13, 28, 41, 42, 207, 321, 110])

        data_dict["rnn"] = dict()
        data_dict["rnn"]["path"] = "p8/p8_18_10_24_a005"
        data_dict["rnn"]["trace_file_name"] = "p8_18_10_24_a005_Traces.mat"
        data_dict["rnn"]["trace_var_name"] = "C_df"
        data_dict["rnn"]["boost_rnn"] = False
        # train on 2 of the cell of Robin
        # trained on 2 cells of p8
        # data_dict["rnn"]["file_name"] = "P8_18_10_24_a005_predictions_2019_02_06.20-29-38_9_cells_from_Robin.mat"
        # data_dict["rnn"]["file_name"] = "P8_18_10_24_a005_predictions_2019_02_06.22-18-43_from_p12_training.mat"
        # data_dict["rnn"]["file_name"] = "P8_18_10_24_a005_predictions_2019_02_06.22-33-03_trained_on_5_sessions.mat"
        # data_dict["rnn"]["file_name"] = "P8_18_10_24_a005_predictions_2019_02_07.13-31-43_train_on_3_cells_p8.mat"
        # trained on p12 0,3 cell, with 2 inputs (cell masked + all), on 20 epochs. trained on 13/02/2019 12:24:23
        # data_dict["rnn"]["file_name"] = "P8_18_10_24_a005_predictions_2019_02_14.20-10-56_from_new_p12_training.mat"
        # ## trained on p12 0,3 cell, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 16/02/2019 11:21:11
        # data_dict["rnn"]["file_name"] = "P8_18_10_24_a005_predictions_2019_02_16.15-43-26.mat"
        # trained on 5 sessions and 15 cells, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 15/02/2019 00:01:55, on epoch 13
        data_dict["rnn"]["file_name"] = "P8_18_10_24_a005_predictions_2019_02_16.16-06-00.mat"
        # ## trained on 4 sessions, 13 cells, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 21/02/2019 00-47-30 with bin et al. version + atttention before
        data_dict["rnn"]["file_name"] = "P8_18_10_24_a005_predictions_2019_02_21.19-07-23.mat"
        # ## trained on 3 cells p8, 8 artificial, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 25/02/2019 11-01-02 with bin et al. version + atttention before
        data_dict["rnn"]["file_name"] = "P8_18_10_24_a005_predictions_2019_02_25.17-40-58.mat"
        # trained on 50 cells (p11, p12, p13) + artificial, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 26/02/2019 , on epoch 21 with overlap at 0.8 (until cell 99 included)
        data_dict["rnn"]["file_name"] = "P8_18_10_24_a005_predictions_2019_03_08.13-53-35.mat"


        data_dict["rnn"]["var_name"] = "spike_nums_dur_predicted"
        data_dict["rnn"]["predictions"] = "predictions"
        data_dict["rnn"]["prediction_threshold"] = 0.5

        data_dict["old_rnn"] = dict()
        data_dict["old_rnn"]["path"] = "p8/p8_18_10_24_a005"
        # ## trained on p12 0,3 cell, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 16/02/2019 11:21:11
        data_dict["old_rnn"]["file_name"] = "P8_18_10_24_a005_predictions_2019_02_16.15-43-26.mat"
        # ## trained on 4 sessions, 13 cells, with 3 inputs (cell masked + cells masked + neuropil mask),
        # trained on 21/02/2019 00-47-30 with bin et al. version + atttention before
        data_dict["old_rnn"]["file_name"] = "P8_18_10_24_a005_predictions_2019_02_21.19-07-23.mat"
        data_dict["old_rnn"]["var_name"] = "spike_nums_dur_predicted"
        data_dict["old_rnn"]["predictions"] = "predictions"
        data_dict["old_rnn"]["prediction_threshold"] = 0.5

        data_dict["caiman_raw"] = dict()
        data_dict["caiman_raw"]["path"] = "p8/p8_18_10_24_a005"
        data_dict["caiman_raw"]["file_name_onsets"] = "p8_18_10_24_a005_MCMC.mat"
        data_dict["caiman_raw"]["onsets_var_name"] = "spikenums"
        data_dict["caiman_raw"]["to_bin"] = True
        data_dict["caiman_raw"]["trace_file_name"] = "p8_18_10_24_a005_Traces.mat"
        data_dict["caiman_raw"]["trace_var_name"] = "C_df"

        # data_dict["caiman_filt"] = dict()
        # data_dict["caiman_filt"]["path"] = "p8/p8_18_10_24_a005"
        # data_dict["caiman_filt"]["file_name"] = "p8_18_10_24_a005_filt_RasterDur.mat"
        # data_dict["caiman_filt"]["var_name"] = "rasterdur"

    # ground truth
    data_file = hdf5storage.loadmat(os.path.join(path_data, data_dict["gt"]["path"], data_dict["gt"]["gui_file"]))
    peak_nums = data_file['LocPeakMatrix_Python'].astype(int)
    spike_nums = data_file['Bin100ms_spikedigital_Python'].astype(int)
    # inter_neurons = data_file['inter_neurons'].astype(int)
    if 'cells_to_remove' in data_file:
        cells_to_remove = data_file['cells_to_remove'].astype(int)
    else:
        cells_to_remove = None
    ground_truth_raster_dur = build_spike_nums_dur(spike_nums, peak_nums)
    print(f"ground_truth_raster_dur.shape {ground_truth_raster_dur.shape}")
    n_cells = ground_truth_raster_dur.shape[0]
    n_frames = ground_truth_raster_dur.shape[1]

    using_gt_cell_classifier = False
    cells_false_gt = []
    if using_gt_cell_classifier and (path_data, "gt_file" in data_dict["gt"]):
        with open(os.path.join(path_data, data_dict["gt"]["path"], data_dict["gt"]["gt_file"]), "r",
                  encoding='UTF-8') as file:
            for nb_line, line in enumerate(file):
                line_list = line.split()
                cells_list = [float(i) for i in line_list]
                cells_false_gt.extend(cells_list)
        cells_false_gt = np.array(cells_false_gt).astype(int)

    cells_for_benchmark = data_dict["gt"]["cells"]

    if "cnn" in data_dict["gt"]:
        cell_cnn_predictions = []
        with open(os.path.join(path_data, data_dict["gt"]["cnn"]), "r", encoding='UTF-8') as file:
            for nb_line, line in enumerate(file):
                line_list = line.split()
                cells_list = [float(i) for i in line_list]
                cell_cnn_predictions.extend(cells_list)
        cell_cnn_predictions = np.array(cell_cnn_predictions)
        cells_predicted_as_false = np.where(cell_cnn_predictions < data_dict["gt"]["cnn_threshold"])[0]
        # print(f"cells_predicted_as_false {cells_predicted_as_false}")

        # not taking into consideration cells that are not predicted as true from the cell classifier
        cells_for_benchmark = np.setdiff1d(cells_for_benchmark, cells_predicted_as_false)

    # adding cells not selected by cnn
    if cells_to_remove is not None:
        cells_for_benchmark = np.setdiff1d(cells_for_benchmark, cells_to_remove)

    if "cells_to_remove" in data_dict["gt"]:
        cells_for_benchmark = np.setdiff1d(cells_for_benchmark, data_dict["gt"]["cells_to_remove"])
    # print(f"cells_for_benchmark {cells_for_benchmark}")
    # return

    data_file = hdf5storage.loadmat(os.path.join(path_data, data_dict["rnn"]["path"], data_dict["rnn"]["file_name"]))
    rnn_predictions = data_file[data_dict["rnn"]['predictions']]
    if do_plot_roc_predictions:
        plot_roc_predictions(ground_truth_raster_dur=ground_truth_raster_dur, rnn_predictions=rnn_predictions,
                             cells=cells_for_benchmark,
                             time_str=time_str, description=ms_to_benchmark,
                             path_results=path_results, save_formats="pdf")
        return

    predicted_raster_dur_dict = dict()
    predicted_spike_nums_dict = dict()
    traces = None
    # value is a dict
    for key, value in data_dict.items():
        if key == "gt":
            continue
        if key == "rnn" and ("prediction_threshold" in value):
            data_file = hdf5storage.loadmat(os.path.join(path_data, value["path"], value["file_name"]))
            rnn_raster_dur = \
                build_raster_dur_from_predictions(predictions=data_file[value["predictions"]],
                                                  predictions_threshold=value["prediction_threshold"],
                                                  cells=cells_for_benchmark,
                                                  n_total_cells=n_cells,
                                                  n_frames=n_frames)
            if "trace_file_name" in value:
                data_file = hdf5storage.loadmat(os.path.join(path_data, value["path"], value["trace_file_name"]))
                traces = data_file[value['trace_var_name']]
            if ("boost_rnn" in value) and value["boost_rnn"]:
                rnn_raster_dur = get_boost_rnn_raster_dur(rnn_raster_dur, traces)
                predicted_raster_dur_dict["rnn_boost"] = rnn_raster_dur
            else:
                predicted_raster_dur_dict[key] = rnn_raster_dur
        elif ("rnn" in key) and ("prediction_threshold" in value):
            data_file = hdf5storage.loadmat(os.path.join(path_data, value["path"], value["file_name"]))
            predicted_raster_dur_dict[key] = \
                build_raster_dur_from_predictions(predictions=data_file[value["predictions"]],
                                                  predictions_threshold=value["prediction_threshold"],
                                                  cells=cells_for_benchmark,
                                                  n_total_cells=n_cells,
                                                  n_frames=n_frames)
        else:
            if "to_bin" in value:
                # onsets
                data_file = hdf5storage.loadmat(os.path.join(path_data, value["path"], value["file_name_onsets"]))
                caiman_spike_nums = data_file[value['onsets_var_name']].astype(int)
                data_file = hdf5storage.loadmat(os.path.join(path_data, value["path"], value["trace_file_name"]))
                traces = data_file[value['trace_var_name']]
                raster_dur = get_raster_dur_from_caiman_25000_frames_onsets_new_version(caiman_spike_nums, traces)
                predicted_raster_dur_dict[key] = raster_dur

                # # we need to bin predicted_spike_nums, because there are 50 000 frames
                # new_predicted_spike_nums = np.zeros(
                #     (predicted_spike_nums.shape[0], predicted_spike_nums.shape[1] // 2),
                #     dtype="int8")
                # for cell in np.arange(predicted_spike_nums.shape[0]):
                #     binned_cell = predicted_spike_nums[cell].reshape(-1, 2).mean(axis=1)
                #     binned_cell[binned_cell > 0] = 1
                #     new_predicted_spike_nums[cell] = binned_cell.astype("int")
                # predicted_spike_nums = new_predicted_spike_nums
                # print(f"predicted_spike_nums.shape {predicted_spike_nums.shape}")
                # predicted_spike_nums_dict[key] = predicted_spike_nums
            else:
                data_file = hdf5storage.loadmat(os.path.join(path_data, value["path"], value["file_name"]))
                raster_dur = data_file[value['var_name']].astype(int)
                predicted_raster_dur_dict[key] = raster_dur

    benchmarks = BenchmarkRasterDur(description=ms_to_benchmark, ground_truth_raster_dur=ground_truth_raster_dur,
                                    predicted_raster_dur_dict=predicted_raster_dur_dict, cells=cells_for_benchmark,
                                    traces=traces)

    benchmarks.compute_stats()

    description = ms_to_benchmark
    if "prediction_threshold" in data_dict["rnn"]:
        threshold_value = data_dict["rnn"]["prediction_threshold"]
        description += f"_thr_{threshold_value}_"

    benchmarks.plot_boxplots_full_stat(description=description, time_str=time_str, path_results=path_results,
                                       for_frames=True, save_formats="pdf")
    benchmarks.plot_boxplots_full_stat(description=description, time_str=time_str, path_results=path_results,
                                       for_frames=False, save_formats="pdf")
    # benchmarks.plot_boxplots_for_transients_stat(description=description, time_str=time_str,
    #                                              path_results=path_results,
    #                                              save_formats="pdf")

    # if do_onsets_benchmarks:
    #     print("")
    #     print("#######################################")
    #     print("#######################################")
    #     print("#######################################")
    #     print("")
    #
    #     benchmarks_onsets = BenchmarkRasterDur(description=ms_to_benchmark, ground_truth_raster_dur=spike_nums,
    #                                            predicted_raster_dur_dict=predicted_spike_nums_dict,
    #                                            cells=cells_for_benchmark)
    #
    #     benchmarks_onsets.compute_stats_on_onsets()


main_benchmark()
