import numpy as np
import classification_stat as cs
import hdf5storage
from datetime import datetime
import os
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal as signal
from pattern_discovery.tools.misc import get_continous_time_periods
import sys
# from pattern_discovery.display.raster import plot_spikes_raster
#
# from matplotlib.figure import SubplotParams
# import matplotlib.gridspec as gridspec
from sortedcontainers import SortedDict
from pattern_discovery.tools.signal import smooth_convolve


def do_traces_smoothing(traces):
    # smoothing the trace
    windows = ['hanning', 'hamming', 'bartlett', 'blackman']
    i_w = 1
    window_length = 7  # 11
    for i in np.arange(traces.shape[0]):
        smooth_signal = smooth_convolve(x=traces[i], window_len=window_length,
                                        window=windows[i_w])
        beg = (window_length - 1) // 2
        traces[i] = smooth_signal[beg:-beg]


class BenchmarkRasterDur:
    def __init__(self, description, ground_truth_raster_dur, predicted_raster_dur_dict, cells,
                 traces, rnn_predictions, debug_mode=False, cells_to_color=None, cells_to_remove_by_key=None,
                 from_fusion=False):
        self.description = description
        # matrix of 2 dimensions, first being the cells, second the frames, and values the predictions
        # (float between 0 and 1)
        self.rnn_predictions = rnn_predictions
        self.ground_truth_raster_dur = ground_truth_raster_dur
        # cells on which base the ground truth
        self.cells = cells
        # first key describe the data, value is raster_dur (2D array, cells vs frames)
        self.predicted_raster_dur_dict = predicted_raster_dur_dict
        if from_fusion:
            print(f"init keys {list(self.predicted_raster_dur_dict.keys())}")
        # first key is the cell, value a dict with
        # same keys as raster_dur_dict, value will be a list of dict with results from benchmarks
        self.results_frames_dict_by_cell = dict()
        self.results_transients_dict_by_cell = dict()
        # first key is the cell, value a dict with
        # same keys as raster_dur_dict, value will be a dict with results from benchmarks. Each key of the dict
        # each key being a string like "fp_transients_predictions" or "fp_frames_predictions"
        self.results_predictions_dict_by_cell = dict()
        # same keys as raster_dur_dict, value will be a list of dict with results from benchmarks
        self.results_dict_global = dict()
        self.debug_mode = debug_mode
        self.traces = traces
        if cells_to_color is None:
            self.cells_to_color = dict()
        else:
            self.cells_to_color = cells_to_color
        if cells_to_remove_by_key is None:
            self.cells_to_remove_by_key = dict()
        else:
            self.cells_to_remove_by_key = cells_to_remove_by_key
        if self.traces is not None:
            do_traces_smoothing(self.traces)
        self.from_fusion = from_fusion

    # def compute_stats_on_onsets(self):
    #     if self.debug_mode:
    #         print(f"{self.description} stats on onsets")
    #     for cell in self.cells:
    #         if self.debug_mode:
    #             print(f"Cell {cell}")
    #         for key, raster_dur in self.predicted_raster_dur_dict.items():
    #             gt_rd = self.ground_truth_raster_dur[cell]
    #             p_rd = raster_dur[cell]
    #             frames_stat = cs.compute_stats_on_onsets(spike_nums=gt_rd, predicted_spike_nums=p_rd)
    #             # frames stats
    #             if self.debug_mode:
    #                 print(f"raster {key}")
    #                 print(f"Onsets stat:")
    #                 for k, value in frames_stat.items():
    #                     print(f"{k}: {str(np.round(value, 4))}")
    #         if self.debug_mode:
    #             print("")
    #             print("/////////////////")
    #             print("")
    #     if self.debug_mode:
    #         print("All cells")
    #     for key, raster_dur in self.predicted_raster_dur_dict.items():
    #         gt_rd = self.ground_truth_raster_dur[self.cells]
    #         p_rd = raster_dur[self.cells]
    #         frames_stat = cs.compute_stats_on_onsets(gt_rd, p_rd)
    #         # frames stats
    #         if self.debug_mode:
    #             print(f"raster {key}")
    #             print(f"Onsets stat:")
    #             for k, value in frames_stat.items():
    #                 print(f"{k}: {str(np.round(value, 4))}")

    def fusion(self, other):
        """
        Do fusion with another benchmark, return a new benchmark object
        :param other:
        :return:
        """
        print(f"Fusion self: {self.description}, with other: {other.description}")
        print(f"fusion self keys {list(self.predicted_raster_dur_dict.keys())}, "
              f"other keys {list(other.predicted_raster_dur_dict.keys())}")
        make_special_case_for_MP = True

        ground_truth_raster_dur = np.copy(self.ground_truth_raster_dur)
        # print(f"ground_truth_raster_dur.shape {ground_truth_raster_dur.shape}")
        raster_dict = {}
        for key, value in self.predicted_raster_dur_dict.items():
            raster_dict[key] = np.copy(value)
            if key == "MP" and make_special_case_for_MP:
                self.cells_to_color[tuple(self.cells)] = "white"
        if make_special_case_for_MP:
            if "MP" not in self.predicted_raster_dur_dict:
                raster_dict["MP"] = np.zeros(value.shape)
                if "MP" not in self.cells_to_remove_by_key:
                    self.cells_to_remove_by_key["MP"] = self.cells
                else:
                    self.cells_to_remove_by_key["MP"] = np.concatenate((self.cells_to_remove_by_key["MP"], self.cells))
        cells = np.copy(self.cells)
        if self.traces is not None:
            traces = np.copy(self.traces)
        else:
            traces = None

        description = self.description
        # print(f"len(ground_truth_raster_dur) {len(ground_truth_raster_dur)}")
        # print(f"len(other.ground_truth_raster_dur) {len(other.ground_truth_raster_dur)}")
        ground_truth_raster_dur = np.concatenate((ground_truth_raster_dur, other.ground_truth_raster_dur))
        for key, value in other.predicted_raster_dur_dict.items():
            # print(f"key {key}") MP
            # only keeping the key that are in previous BenchmarkRasterDur otherwise it will mess up the cells indices
            if key in raster_dict:
                raster_dict[key] = np.concatenate((raster_dict[key], value))
                if key == "MP" and make_special_case_for_MP:
                    for cell_index, cell_to_color in enumerate(other.cells + len(self.ground_truth_raster_dur)):
                        print(f"np.sum(value[cell_index]) {np.sum(value[other.cells[cell_index]])}")
                        if np.sum(value[other.cells[cell_index]]) > 0:
                            self.cells_to_color[cell_to_color] = "white"
            # elif make_special_case_for_MP:
            #     # we add fake values, negative ones, just not to mess up with the cells indices
            #     raster_dict[key] = np.array(([-1] * len(self.cells), value))

        if make_special_case_for_MP:
            if "MP" not in other.predicted_raster_dur_dict:
                raster_dict["MP"] = np.concatenate((raster_dict["MP"], np.zeros(value.shape)))
                if "MP" not in self.cells_to_remove_by_key:
                    self.cells_to_remove_by_key["MP"] = other.cells + len(self.ground_truth_raster_dur)
                else:
                    self.cells_to_remove_by_key["MP"] = np.concatenate((self.cells_to_remove_by_key["MP"],
                                                                        other.cells + len(self.ground_truth_raster_dur)))
        else:
            keys_to_remove = []
            for key in raster_dict.keys():
                if key not in other.predicted_raster_dur_dict:
                    # the key should on all sessions
                    keys_to_remove.append(key)
            # print(f"{self.description} {other.description}, keys_to_remove: {keys_to_remove}")
            for key in keys_to_remove:
                raster_dict.pop(key, None)

        rnn_predictions = np.concatenate((self.rnn_predictions, other.rnn_predictions))
        # print(f"cells {cells}")
        # print(f"other.cells {other.cells}")
        cells = np.copy(np.concatenate((cells, other.cells + len(self.ground_truth_raster_dur))))
        # if from_fusion:
        #     cells = np.copy(np.concatenate((cells,
        #                                     np.arange(np.max(cells)+1, np.max(cells)+1 + len(other.cells)))))
        # else:
        #     cells = np.copy(np.concatenate((np.arange(len(cells)),
        #                                     np.arange(np.max(cells)+1, np.max(cells)+1 + len(other.cells))))

        if other.traces is not None:
            if traces is None:
                traces = np.copy(other.traces)
            else:
                traces = np.concatenate((traces, other.traces))

        description += "_" + other.description

        print(f"end fusion keys {list(raster_dict.keys())}")

        return BenchmarkRasterDur(description=description, ground_truth_raster_dur=ground_truth_raster_dur,
                                  predicted_raster_dur_dict=raster_dict, cells=cells, traces=traces,
                                  debug_mode=False, rnn_predictions=rnn_predictions,
                                  cells_to_color=self.cells_to_color,
                                  cells_to_remove_by_key=self.cells_to_remove_by_key, from_fusion=True)

    def compute_stats(self):
        if self.debug_mode:
            print(f"{self.description} stats on raster dur")

        # first we compute for each cell, the lowest peak predicted, in order to fix a low threshold for
        # for transients used for benchmarks (possible transients), transients below the threshold are not considered
        traces_threshold = None
        # if self.traces is not None:
        #     traces_threshold = np.zeros(len(self.traces))
        #     rd_list = []
        #     rd_list.append(self.ground_truth_raster_dur)
        #     for key, raster_dur in self.predicted_raster_dur_dict.items():
        #         rd_list.append(raster_dur)
        #     for cell in np.arange(len(self.traces)):
        #         min_value = None
        #         for raster_dur in rd_list:
        #             periods = get_continous_time_periods(raster_dur[cell])
        #             for period in periods:
        #                 if period[0] == period[1]:
        #                     peak_amplitude = self.traces[cell, period[0]:period[1]+1]
        #                 else:
        #                     peak_amplitude = np.max(self.traces[cell, period[0]:period[1]+1])
        #                 if min_value is None:
        #                     min_value = peak_amplitude
        #                 else:
        #                     min_value = min(peak_amplitude, min_value)
        #         traces_threshold[cell] = min_value
        for cell in self.cells:
            if self.debug_mode:
                print(f"Cell {cell}")
            self.results_frames_dict_by_cell[cell] = SortedDict()
            self.results_transients_dict_by_cell[cell] = SortedDict()
            self.results_predictions_dict_by_cell[cell] = SortedDict()
            for key, raster_dur in self.predicted_raster_dur_dict.items():
                gt_rd = self.ground_truth_raster_dur[cell]
                gt_predictions = self.rnn_predictions[cell]
                # predicted raster_dur
                # if self.from_fusion:
                #     print(f"key {key} {raster_dur.shape}")
                p_rd = raster_dur[cell]
                # if no predictions were made for this cell, then we pass
                # that's supposed we don't want to benchmarks cells with no transients
                # this allow a different number of cell for each boxplot
                if np.sum(p_rd) == 0:
                    # print(f"{self.description} {cell} {key} np.sum(p_rd) == 0")
                    continue
                if self.traces is not None:
                    traces = self.traces[cell]
                else:
                    traces = None
                frames_stat, transients_stat, predictions_stat_dict = cs.compute_stats(gt_rd, p_rd,
                                                                                       gt_predictions=gt_predictions,
                                                                                       traces=traces,
                                                                                       with_threshold=traces_threshold)
                self.results_predictions_dict_by_cell[cell][key] = predictions_stat_dict
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
            # just for display
            print("All cells")
            for key, raster_dur in self.predicted_raster_dur_dict.items():
                gt_rd = self.ground_truth_raster_dur[self.cells]
                p_rd = raster_dur[self.cells]
                # if no predictions were made for this cell, then we pass
                # that's supposed we don't want to benchmarks cells with no transients
                # this allow a different number of cell for each boxplot
                if np.sum(p_rd) == 0:
                    continue
                if self.traces is not None:
                    traces = self.traces[self.cells]
                else:
                    traces = None
                frames_stat, transients_stat, predictions_stat_dict = cs.compute_stats(gt_rd, p_rd,
                                                                                       traces=traces,
                                                                                       gt_predictions=gt_predictions)
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

    def plot_boxplot_predictions_stat_by_metrics(self, path_results, description, time_str,
                                                 save_formats="pdf", dpi=500):
        """

        Args:
            path_results:
            description:
            time_str:
            save_formats:
            dpi:

        Returns:

        """

        type_of_activity_list = ["frames", "transients"]
        metrics_to_show_list = ["tp", "tn", "fp", "fn"]

        # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12 + 12 blue range
        colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
                  '#ff7f00', '#cab2d6', '#fdbf6f', '#6a3d9a', '#ffff99', '#b15928', '#ffffd9', '#edf8b1', '#c7e9b4',
                  '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84']
        colors = ['cornflowerblue'] + colors[3:]

        with_scatter = True

        predictions_by_cell_stat = dict()
        predictions_stat = dict()
        # we take just the first data, as it's supposed to be same results for each, and we put them as
        for type_of_activity in type_of_activity_list:
            predictions_by_cell_stat[type_of_activity] = dict()
            predictions_stat[type_of_activity] = dict()
            for metrics_to_show in metrics_to_show_list:
                predictions_by_cell_stat[type_of_activity][metrics_to_show] = dict()
                predictions_stat[type_of_activity][metrics_to_show] = []
                for cell, data_pred_dict in self.results_predictions_dict_by_cell.items():
                    predictions_by_cell_stat[type_of_activity][metrics_to_show][cell] = []
                    for data_key, pred_dict in data_pred_dict.items():
                        # pred_dict take as keys something like "fp_transients_prediction"
                        string_key = metrics_to_show + "_" + type_of_activity + "_" + "predictions"
                        predictions_by_cell_stat[type_of_activity][metrics_to_show][cell] = pred_dict[string_key]
                        predictions_stat[type_of_activity][metrics_to_show].extend(pred_dict[string_key])
                        # first one is enough
                        break

        for by_cell in [True, False]:
            for type_of_activity in type_of_activity_list:
                if by_cell:
                    if len(predictions_by_cell_stat[type_of_activity][metrics_to_show]) > 15:
                        # if more than 15 boxplots, we don't plot it
                        continue
                stat_fig, axes = plt.subplots(nrows=2, ncols=2, squeeze=True,
                                              gridspec_kw={'height_ratios': [0.5, 0.5],
                                                           'width_ratios': [0.5, 0.5]},
                                              figsize=(10, 10), dpi=dpi)

                stat_fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'w_pad': 1, 'h_pad': 5})
                axes = np.ndarray.flatten(axes)
                fig_patch = stat_fig.patch
                black_and_white_skin = False
                # rgba = c_map(0)
                if black_and_white_skin:
                    face_color = "white"
                    text_color = "black"
                    title_color = "black"
                else:
                    face_color = "black"
                    text_color = "white"
                    title_color = "red"
                fig_patch.set_facecolor(face_color)

                for metrics_index, metrics_to_show in enumerate(metrics_to_show_list):
                    ax = axes[metrics_index]

                    ax.set_facecolor(face_color)

                    ax.set_frame_on(False)
                    if by_cell:
                        by_cell_dict = predictions_by_cell_stat[type_of_activity][metrics_to_show]
                        n_box_plots = len(by_cell_dict)
                        labels = []
                        for cell_key in by_cell_dict.keys():
                            labels.append(f"{cell_key} ({len(by_cell_dict[cell_key])})")
                        # labels = list(by_cell_dict.keys())
                        values_by_prediction = [by_cell_dict[cell_key] for cell_key in list(by_cell_dict.keys())]
                        if with_scatter:
                            for label_index, cell_key in enumerate(list(by_cell_dict.keys())):
                                y_pos = by_cell_dict[cell_key]
                                x_pos = []
                                for ii in range(len(y_pos)):
                                    # Adding jitter
                                    x_pos.append(1 + label_index + ((np.random.random_sample() - 0.5) * 0.8))
                                if black_and_white_skin:
                                    edgecolors = "black"
                                else:
                                    edgecolors = "white"
                                ax.scatter(x_pos, y_pos,
                                           color=colors[label_index % len(colors)],
                                           marker="o",
                                           edgecolors=edgecolors,
                                           s=20, zorder=21, alpha=0.5)

                    else:
                        n_cells = len(predictions_by_cell_stat[type_of_activity][metrics_to_show])
                        labels = [f"{n_cells} cells, "
                                  f"{len(predictions_stat[type_of_activity][metrics_to_show])} {type_of_activity}"]
                        n_box_plots = 1
                        values_by_prediction = [predictions_stat[type_of_activity][metrics_to_show]]

                        if with_scatter:
                            y_pos = predictions_stat[type_of_activity][metrics_to_show]
                            x_pos = []
                            for ii in range(len(y_pos)):
                                # Adding jitter
                                x_pos.append(1 + ((np.random.random_sample() - 0.5) * 0.8))
                            if black_and_white_skin:
                                edgecolors = "black"
                            else:
                                edgecolors = "white"
                            ax.scatter(x_pos, y_pos,
                                       color=colors[0 % len(colors)],
                                       marker="o",
                                       edgecolors=edgecolors,
                                       s=20, zorder=21, alpha=0.5)

                    colorfull = True
                    outliers = dict(markerfacecolor='white', marker='D')
                    # if not for_frames:
                    #     print(f"plot_boxplots_full_stat: {stat_name}: values_by_prediction {values_by_prediction}")
                    if with_scatter:
                        sym = ''
                    else:
                        sym = '+'
                    bplot = ax.boxplot(values_by_prediction, patch_artist=colorfull,
                                       flierprops=outliers, widths=[0.7] * len(values_by_prediction),
                                       labels=labels, sym=sym, zorder=1)  # whis=[5, 95], sym='+'

                    for element in ['boxes', 'whiskers', 'fliers', 'caps']:
                        if black_and_white_skin:
                            plt.setp(bplot[element], color="black")
                        else:
                            plt.setp(bplot[element], color="white")

                    for element in ['means', 'medians']:
                        if black_and_white_skin:
                            plt.setp(bplot[element], color="black")
                        else:
                            plt.setp(bplot[element], color="silver")

                    if colorfull:
                        colors = colors[:n_box_plots]
                        for patch, color in zip(bplot['boxes'], colors):
                            patch.set_facecolor(color)

                    ax.xaxis.set_ticks_position('none')
                    ax.xaxis.label.set_color(text_color)
                    ax.tick_params(axis='x', colors=text_color)
                    if n_box_plots <= 2:
                        ax.xaxis.set_tick_params(labelsize=15)
                    elif n_box_plots <= 6:
                        ax.xaxis.set_tick_params(labelsize=9)
                    else:
                        ax.xaxis.set_tick_params(labelsize=3)
                    ax.yaxis.label.set_color(text_color)
                    ax.tick_params(axis='y', colors=text_color)
                    # ax.set_xticklabels([])
                    # ax.set_yticklabels([])
                    # ax.get_yaxis().set_visible(False)
                    # ax.get_xaxis().set_visible(False)
                    # ax.set_ylabel(f"proportion")
                    # ax.set_xlabel("age")
                    xticks = np.arange(1, n_box_plots + 1)
                    ax.set_xticks(xticks)
                    # sce clusters labels
                    ax.set_xticklabels(labels)
                    # fixing the limits
                    # if stat_name == "sensitivity":
                    #     ax.set_ylim(0, 1)
                    # elif stat_name == "specificity":
                    #     ax.set_ylim(0.85, 1)
                    # elif stat_name == "PPV":
                    #     ax.set_ylim(0, 1)
                    # elif stat_name == "NPV":
                    #     ax.set_ylim(0.6, 1.1)

                    ax.set_title(metrics_to_show.upper(), color=title_color, pad=20, fontsize=20)

                str_details = "and_cells_" if by_cell else ""
                if isinstance(save_formats, str):
                    save_formats = [save_formats]
                for save_format in save_formats:
                    stat_fig.savefig(f'{path_results}/'
                                     f'{description}_box_plots_predictions_by_metrics_{str_details}'
                                     f'for_{type_of_activity}'
                                     f'_{time_str}.{save_format}',
                                     format=f"{save_format}",
                                     facecolor=stat_fig.get_facecolor(), edgecolor='none')
                plt.close()

    def plot_boxplots_full_stat(self, path_results, description, time_str, for_frames=True, with_cells=False,
                                save_formats="pdf", dpi=500):
        """

        :param path_results:
        :param description:
        :param time_str:
        :param for_frames:
        :param with_cells: if True, display a scatter for each cell
        :param save_formats:
        :return:
        """
        result_dict_to_use = self.results_frames_dict_by_cell
        if not for_frames:
            result_dict_to_use = self.results_transients_dict_by_cell
        stats_to_show = ["sensitivity", "specificity", "PPV", "NPV"]
        # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12 + 12 blue range
        colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                  '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#ffffd9', '#edf8b1', '#c7e9b4',
                  '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84']
        colors = ['cornflowerblue'] + colors[3:]

        stat_fig, axes = plt.subplots(nrows=2, ncols=2, squeeze=True,
                                      gridspec_kw={'height_ratios': [0.5, 0.5],
                                                   'width_ratios': [0.5, 0.5]},
                                      figsize=(10, 10), dpi=dpi)

        stat_fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'w_pad': 1, 'h_pad': 5})
        axes = np.ndarray.flatten(axes)
        fig_patch = stat_fig.patch
        black_and_white_skin = False
        # rgba = c_map(0)
        if black_and_white_skin:
            face_color = "white"
            text_color = "black"
            title_color = "black"
        else:
            face_color = "black"
            text_color = "white"
            title_color = "red"
        fig_patch.set_facecolor(face_color)

        for stat_index, stat_name in enumerate(stats_to_show):
            ax = axes[stat_index]
            n_cells = len(self.results_frames_dict_by_cell)

            ax.set_facecolor(face_color)

            ax.set_frame_on(False)
            n_box_plots = None
            labels = None
            values_by_prediction = None

            # determining the cell with the max number of label
            max_key = 0
            max_n_values = 0
            for cell_to_display in result_dict_to_use.keys():
                n_values = len(result_dict_to_use[cell_to_display])
                if n_values > max_n_values:
                    max_n_values = n_values
                    max_key = cell_to_display
            n_box_plots = len(result_dict_to_use[max_key])
            labels = list(result_dict_to_use[max_key].keys())
            values_by_prediction = [[] for n in np.arange(n_box_plots)]

            for cell_index, cell_to_display in enumerate(result_dict_to_use.keys()):
                # if n_box_plots is None:
                #     n_box_plots = len(result_dict_to_use[cell_to_display])
                #     labels = list(result_dict_to_use[cell_to_display].keys())
                #     values_by_prediction = [[] for n in np.arange(n_box_plots)]
                for label_index, label in enumerate(labels):
                    # all label might not have the same cells
                    if label in result_dict_to_use[cell_to_display]:
                        values_by_prediction[label_index]. \
                            append(result_dict_to_use[cell_to_display][label][stat_name])
                    else:
                        continue
                    if with_cells:
                        # Adding jitter
                        x_pos = 1 + label_index + ((np.random.random_sample() - 0.5) * 0.8)
                        y_pos = result_dict_to_use[cell_to_display][label][stat_name]
                        font_size = 3
                        if black_and_white_skin:
                            edgecolors = "black"
                        else:
                            edgecolors = "white"
                        if cell_to_display not in self.cells_to_color:
                            color_cell = colors[label_index % len(colors)]
                        else:
                            color_cell = self.cells_to_color[cell_to_display]
                        ax.scatter(x_pos, y_pos,
                                   color=color_cell,
                                   marker="o",
                                   edgecolors=edgecolors,
                                   s=60, zorder=21)
                        ax.text(x=x_pos, y=y_pos,
                                s=f"{cell_to_display}", color="black", zorder=22,
                                ha='center', va="center", fontsize=font_size, fontweight='bold')

            colorfull = True
            outliers = dict(markerfacecolor='white', marker='D')
            # if not for_frames:
            #     print(f"plot_boxplots_full_stat: {stat_name}: values_by_prediction {values_by_prediction}")
            bplot = ax.boxplot(values_by_prediction, patch_artist=colorfull,
                               flierprops=outliers, widths=[0.7] * len(values_by_prediction),
                               labels=labels, sym='', zorder=1)  # whis=[5, 95], sym='+'

            for element in ['boxes', 'whiskers', 'fliers', 'caps']:
                if black_and_white_skin:
                    plt.setp(bplot[element], color="black")
                else:
                    plt.setp(bplot[element], color="white")

            for element in ['means', 'medians']:
                if black_and_white_skin:
                    plt.setp(bplot[element], color="black")
                else:
                    plt.setp(bplot[element], color="silver")

            if colorfull:
                colors = colors[:n_box_plots]
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_facecolor(color)

            ax.xaxis.set_ticks_position('none')
            ax.xaxis.label.set_color(text_color)
            ax.tick_params(axis='x', colors=text_color)
            if n_box_plots <= 2:
                ax.xaxis.set_tick_params(labelsize=15)
            elif n_box_plots <= 6:
                ax.xaxis.set_tick_params(labelsize=9)
            else:
                ax.xaxis.set_tick_params(labelsize=3)
            ax.yaxis.label.set_color(text_color)
            ax.tick_params(axis='y', colors=text_color)
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            # ax.get_yaxis().set_visible(False)
            # ax.get_xaxis().set_visible(False)
            # ax.set_ylabel(f"proportion")
            # ax.set_xlabel("age")
            xticks = np.arange(1, n_box_plots + 1)
            ax.set_xticks(xticks)
            # sce clusters labels
            ax.set_xticklabels(labels)
            # fixing the limits
            if stat_name == "sensitivity":
                ax.set_ylim(0.2, 1.1)
            # elif stat_name == "specificity":
            #     ax.set_ylim(0.85, 1)
            elif stat_name == "PPV":
                ax.set_ylim(0.5, 1.1)
            # elif stat_name == "NPV":
            #     ax.set_ylim(0.6, 1.1)

            ax.set_title(stat_name, color=title_color, pad=20, fontsize=20)

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
        plt.close()

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

    def plot_boxplots_f1_score(self, path_results, description, time_str, for_frames=True, with_cells=False,
                                save_formats="pdf", dpi=500):
        """

        :param path_results:
        :param description:
        :param time_str:
        :param for_frames:
        :param with_cells: if True, display a scatter for each cell
        :param save_formats:
        :return:
        """
        result_dict_to_use = self.results_frames_dict_by_cell
        if not for_frames:
            result_dict_to_use = self.results_transients_dict_by_cell
        # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12 + 12 blue range
        colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                  '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#ffffd9', '#edf8b1', '#c7e9b4',
                  '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84']
        colors = ['cornflowerblue'] + colors[3:]

        stat_fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True,
                                      gridspec_kw={'height_ratios': [1],
                                                   'width_ratios': [1]},
                                      figsize=(10, 10), dpi=dpi)

        stat_fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'w_pad': 1, 'h_pad': 5})
        fig_patch = stat_fig.patch
        black_and_white_skin = False
        # rgba = c_map(0)
        if black_and_white_skin:
            face_color = "white"
            text_color = "black"
            title_color = "black"
        else:
            face_color = "black"
            text_color = "white"
            title_color = "red"
        fig_patch.set_facecolor(face_color)

        n_cells = len(self.results_frames_dict_by_cell)

        ax.set_facecolor(face_color)

        ax.set_frame_on(False)

        # determining the cell with the max number of label
        max_key = 0
        max_n_values = 0
        for cell_to_display in result_dict_to_use.keys():
            n_values = len(result_dict_to_use[cell_to_display])
            if n_values > max_n_values:
                max_n_values = n_values
                max_key = cell_to_display
        n_box_plots = len(result_dict_to_use[max_key])
        labels = list(result_dict_to_use[max_key].keys())
        values_by_prediction = [[] for n in np.arange(n_box_plots)]

        for cell_index, cell_to_display in enumerate(result_dict_to_use.keys()):
            for label_index, label in enumerate(labels):
                # all label might not have the same cells
                if label in result_dict_to_use[cell_to_display]:
                    recall = result_dict_to_use[cell_to_display][label]["sensitivity"]
                    precision = result_dict_to_use[cell_to_display][label]["PPV"]
                    f1_score = 2 * ((precision * recall) / (precision + recall))
                    values_by_prediction[label_index]. \
                        append(f1_score)
                else:
                    continue
                if with_cells:
                    # Adding jitter
                    x_pos = 1 + label_index + ((np.random.random_sample() - 0.5) * 0.8)
                    y_pos = f1_score
                    font_size = 3
                    if black_and_white_skin:
                        edgecolors = "black"
                    else:
                        edgecolors = "white"
                    if cell_to_display not in self.cells_to_color:
                        color_cell = colors[label_index % len(colors)]
                    else:
                        color_cell = self.cells_to_color[cell_to_display]
                    ax.scatter(x_pos, y_pos,
                               color=color_cell,
                               marker="o",
                               edgecolors=edgecolors,
                               s=60, zorder=21)
                    ax.text(x=x_pos, y=y_pos,
                            s=f"{cell_to_display}", color="black", zorder=22,
                            ha='center', va="center", fontsize=font_size, fontweight='bold')

        colorfull = True
        outliers = dict(markerfacecolor='white', marker='D')
        # if not for_frames:
        #     print(f"plot_boxplots_full_stat: {stat_name}: values_by_prediction {values_by_prediction}")
        bplot = ax.boxplot(values_by_prediction, patch_artist=colorfull,
                           flierprops=outliers, widths=[0.7] * len(values_by_prediction),
                           labels=labels, sym='', zorder=1)  # whis=[5, 95], sym='+'

        for element in ['boxes', 'whiskers', 'fliers', 'caps']:
            if black_and_white_skin:
                plt.setp(bplot[element], color="black")
            else:
                plt.setp(bplot[element], color="white")

        for element in ['means', 'medians']:
            if black_and_white_skin:
                plt.setp(bplot[element], color="black")
            else:
                plt.setp(bplot[element], color="silver")

        if colorfull:
            colors = colors[:n_box_plots]
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

        ax.xaxis.set_ticks_position('none')
        ax.xaxis.label.set_color(text_color)
        ax.tick_params(axis='x', colors=text_color)
        if n_box_plots <= 2:
            ax.xaxis.set_tick_params(labelsize=15)
        elif n_box_plots <= 6:
            ax.xaxis.set_tick_params(labelsize=9)
        else:
            ax.xaxis.set_tick_params(labelsize=3)
        ax.yaxis.label.set_color(text_color)
        ax.tick_params(axis='y', colors=text_color)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.get_yaxis().set_visible(False)
        # ax.get_xaxis().set_visible(False)
        # ax.set_ylabel(f"proportion")
        # ax.set_xlabel("age")
        xticks = np.arange(1, n_box_plots + 1)
        ax.set_xticks(xticks)
        # sce clusters labels
        ax.set_xticklabels(labels)
        # fixing the limits
        ax.set_ylim(0, 1.1)

        ax.set_title("F1 SCORE", color=title_color, pad=20, fontsize=20)

        str_details = "frames"
        if not for_frames:
            str_details = "transients"
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            stat_fig.savefig(f'{path_results}/'
                             f'{description}_box_plots_f1_score_{str_details}_on_{n_cells}_cells'
                             f'_{time_str}.{save_format}',
                             format=f"{save_format}",
                             facecolor=stat_fig.get_facecolor(), edgecolor='none')
        plt.close()

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


def build_raster_dur_from_predictions(predictions, predictions_threshold, cells, n_total_cells, n_frames):
    if predictions_threshold == 1:
        # print("predictions_threshold == 1")
        # predictions_threshold -= sys.float_info.epsilon
        predictions_threshold = 0.99
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


def plot_recall_vs_precision_curve(ground_truth_raster_dur, rnn_predictions, cells,
                                   path_results, description, time_str, traces,
                                   use_transients,
                                   save_formats, for_suite2p=False, with_text=False):
    n_frames = ground_truth_raster_dur.shape[1]
    n_cells = ground_truth_raster_dur.shape[0]
    sensitivity_values = []
    precision_values = []
    if for_suite2p:
        threshold_values = np.arange(0, np.max(rnn_predictions) + 2, 5)
    else:
        threshold_values = np.arange(0, 1.05, 0.05)

    for predictions_threshold in threshold_values:
        # building the raster_dur
        predicted_raster_dur_dict = build_raster_dur_from_predictions(predictions=rnn_predictions,
                                                                      predictions_threshold=predictions_threshold,
                                                                      cells=cells,
                                                                      n_total_cells=n_cells, n_frames=n_frames)

        raster_dict = {"raster_dur": predicted_raster_dur_dict}
        benchmarks = BenchmarkRasterDur(description=description, ground_truth_raster_dur=ground_truth_raster_dur,
                                        predicted_raster_dur_dict=raster_dict, cells=cells, traces=traces,
                                        debug_mode=False, rnn_predictions=rnn_predictions)

        benchmarks.compute_stats()
        avg_sensitivity = []
        avg_precision = []
        for cell in cells:
            # all cell might not have been predicted
            if use_transients:
                if "raster_dur" in benchmarks.results_transients_dict_by_cell[cell]:
                    avg_sensitivity.append(
                        benchmarks.results_transients_dict_by_cell[cell]["raster_dur"]["sensitivity"])
                    avg_precision.append(benchmarks.results_transients_dict_by_cell[cell]["raster_dur"]["PPV"])
            else:
                if "raster_dur" in benchmarks.results_frames_dict_by_cell[cell]:
                    avg_sensitivity.append(benchmarks.results_frames_dict_by_cell[cell]["raster_dur"]["sensitivity"])
                    avg_precision.append(benchmarks.results_frames_dict_by_cell[cell]["raster_dur"]["PPV"])
        if len(avg_sensitivity) > 0:
            sensitivity_values.append(np.mean(avg_sensitivity))
            precision_values.append(np.mean(avg_precision))
        else:
            sensitivity_values.append(0)
            precision_values.append(0)

    precision_values = np.array(precision_values)

    roc_fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True,
                               gridspec_kw={'height_ratios': [1],
                                            'width_ratios': [1]},
                               figsize=(8, 8), dpi=500)

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

    ax.plot(sensitivity_values, precision_values, color="red", lw=2)  # whis=[5, 95], sym='+'
    if with_text:
        for index, predictions_threshold in enumerate(threshold_values):
            ax.text(x=sensitivity_values[index], y=precision_values[index],
                    s=f"{str(np.round(predictions_threshold, 2))}", color="white", zorder=22,
                    ha='center', va="center", fontsize=5, fontweight='bold')

    ax.set_ylabel(f"Precision")
    ax.set_xlabel("Recall")
    # xticks = np.arange(1, n_box_plots + 1)
    # ax.set_xticks(xticks)
    # sce clusters labels
    # ax.set_xticklabels(labels)

    # ax.set_title("", color=title_color, pad=20)
    # ax.set_xscale("log")

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        roc_fig.savefig(f'{path_results}/'
                        f'{description}_precision_vs_recall_predictions_on_{len(cells)}_cells'
                        f'_{time_str}.{save_format}',
                        format=f"{save_format}",
                        facecolor=roc_fig.get_facecolor(), edgecolor='none')


def plot_roc_predictions(ground_truth_raster_dur, rnn_predictions, cells,
                         path_results, description, time_str,
                         save_formats, for_suite2p=False):
    n_frames = ground_truth_raster_dur.shape[1]
    n_cells = ground_truth_raster_dur.shape[0]
    sensitivity_values = []
    specificity_values = []
    if for_suite2p:
        threshold_values = np.arange(0, np.max(rnn_predictions) + 2, 5)
    else:
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
            # all cell might not have been predicted
            if "raster_dur" in benchmarks.results_frames_dict_by_cell[cell]:
                avg_sensitivity.append(benchmarks.results_frames_dict_by_cell[cell]["raster_dur"]["sensitivity"])
                avg_specificity.append(benchmarks.results_frames_dict_by_cell[cell]["raster_dur"]["specificity"])
        if len(avg_sensitivity) > 0:
            sensitivity_values.append(np.mean(avg_sensitivity))
            specificity_values.append(np.mean(avg_specificity))
        else:
            sensitivity_values.append(0)
            specificity_values.append(0)

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


def clean_raster_at_concatenation(spike_nums_dur):
    """
    Movies of 2500 frames are concatenated, we need to clean around the concatenation times
    :return:
    """

    if spike_nums_dur is None:
        return

    if (spike_nums_dur.shape[1] != 12500) and (spike_nums_dur.shape[1] != 10000):
        return
    mask_frames = np.zeros(spike_nums_dur.shape[1], dtype="bool")
    concatenation_times = [2500, 5000, 7500, 10000]
    if spike_nums_dur.shape[1] == 10000:
        concatenation_times = [2500, 5000, 7500]
    for i in concatenation_times:
        mask_frames[i:i + 5] = True

    # if spike_nums is not None:
    #     spike_nums[:, mask_frames] = 0
    if spike_nums_dur is not None:
        spike_nums_dur[:, mask_frames] = 0
    # if peak_nums is not None:
    #     peak_nums[:, mask_frames] = 0
    # print("clean_raster_at_concatenation done")


def load_data_dict(ms_to_benchmark, data_dict, version=None):
    if ms_to_benchmark == "p12_17_11_10_a000":
        # gt as ground_truth
        data_dict["gt"] = dict()
        data_dict["gt"]["path"] = "p12/p12_17_11_10_a000"
        data_dict["gt"]["gui_file"] = "p12_17_11_10_a000_GUI_fusion_validation.mat"
        # p12_17_11_10_a000_GUI_JD.mat

        data_dict["gt"]["trace_file_name"] = "p12_17_11_10_a000_raw_Traces.mat"
        data_dict["gt"]["trace_var_name"] = "raw_traces"
        # data_dict["gt"]["gt_file"] = "p12_17_11_10_a000_cell_to_suppress_ground_truth.txt"
        # data_dict["gt"]["cnn"] = "cell_classifier_results_txt/cell_classifier_cnn_results_P12_17_11_10_a000.txt"
        # data_dict["gt"]["cnn_threshold"] = 0.5
        data_dict["gt"]["cells"] = np.array([9, 10])
        # 9, 10np.array([0, 3, 6, 7, 9, 10, 12, 14, 15, 19]) [9, 10]

        data_dict["caiman"] = dict()
        data_dict["caiman"]["path"] = "p12/p12_17_11_10_a000"
        data_dict["caiman"]["file_name"] = "p12_17_11_10_a000_RasterDur.mat"
        data_dict["caiman"]["file_name_onsets"] = "robin_28_01_19/p12_17_11_10_a000_Spikenums_caiman.mat"
        data_dict["caiman"]["onsets_var_name"] = "spikenums"
        data_dict["caiman"]["to_bin"] = True
        # data_dict["caiman"]["var_name"] = "rasterdur"
        data_dict["caiman"]["trace_file_name"] = "p12_17_11_10_a000_Traces.mat"
        data_dict["caiman"]["trace_var_name"] = "C_df"
        # "p12_17_11_10_a000_caiman_raster_dur_JD_version.mat"

        # data_dict["suite2p"] = dict()
        # data_dict["suite2p"]["path"] = "p12/p12_17_11_10_a000/suite2p/"
        # data_dict["suite2p"]["caiman_suite2p_mapping"] = "P12_17_11_10_a000_suite2p_vs_caiman.npy"
        # data_dict["suite2p"]["threshold"] = 120  # 50
        # #
        data_dict["MP"] = dict()
        data_dict["MP"]["path"] = "p12/p12_17_11_10_a000/"
        data_dict["MP"]["gui_file"] = "p12_17_11_10_a000_GUI_JDMP.mat"

        data_dict["RD"] = dict()
        data_dict["RD"]["path"] = "p12/p12_17_11_10_a000/"
        data_dict["RD"]["gui_file"] = "p12_17_11_10_a000_GUI_transientsRD_2.mat"

        data_dict["JD"] = dict()
        data_dict["JD"]["path"] = "p12/p12_17_11_10_a000/"
        data_dict["JD"]["gui_file"] = "p12_17_11_10_a000_GUI_JD.mat"

        # data_dict["caiman_filt"] = dict()
        # data_dict["caiman_filt"]["path"] = "p12/p12_17_11_10_a000"
        # data_dict["caiman_filt"]["file_name"] = "p12_17_11_10_a000_filt_RasterDur_caiman.mat"
        # data_dict["caiman_filt"]["file_name_onsets"] = "robin_28_01_19/p12_17_11_10_a000_Bin100ms_spikedigital.mat"
        # data_dict["caiman_filt"]["onsets_var_name"] = "Bin100ms_spikedigital"
        # data_dict["caiman_filt"]["var_name"] = "rasterdur"
    elif ms_to_benchmark == "p6_19_02_18_a000_ms":
        data_dict["gt"] = dict()
        data_dict["gt"]["path"] = "p6/p6_19_02_18_a000"
        # single expert labeling
        data_dict["gt"]["gui_file"] = "p6_19_02_18_a000_ground_truth_cell_0_1_2_3.mat"
        data_dict["gt"]["cells"] = np.array([3])  # np.array([0, 1, 2, 3]) # 3 not seen by the network
        # data_dict["gt"]["cells"] = np.array([3])
        data_dict["gt"]["trace_file_name"] = "p6_19_02_18_a000_raw_traces.npy"
        data_dict["gt"]["trace_var_name"] = "raw_traces"

        data_dict["caiman"] = dict()
        data_dict["caiman"]["path"] = "p6/p6_19_02_18_a000"
        # data_dict["caiman"]["file_name_onsets"] = "caiman_matlab/p6_19_02_18_a000_MCMC_(6).mat"
        data_dict["caiman"]["file_name_onsets"] = "caiman_matlab/p6_19_02_18_a000_MCMC_new.mat"
        data_dict["caiman"]["onsets_var_name"] = "spikenums"
        data_dict["caiman"]["to_bin"] = True
        data_dict["caiman"]["caiman_fiji_mapping"] = "P6_19_02_18_a000_fiji_vs_caiman.npy"
        data_dict["caiman"]["trace_file_name"] = "caiman_matlab/p6_19_02_18_a000_Traces.mat"
        data_dict["caiman"]["trace_var_name"] = "C_df"

        data_dict["MP"] = dict()
        data_dict["MP"]["path"] = "p6/p6_19_02_18_a000/"
        data_dict["MP"]["gui_file"] = "p6_19_02_18_a000_Transient MP.mat"

        data_dict["EQ"] = dict()
        data_dict["EQ"]["path"] = "p6/p6_19_02_18_a000/"
        data_dict["EQ"]["gui_file"] = "p6_19_02_18_a000_ground_truth_ele.mat"

        data_dict["RD"] = dict()
        data_dict["RD"]["path"] = "p6/p6_19_02_18_a000/"
        data_dict["RD"]["gui_file"] = "p6_19_02_18_a000_Transients_selection_RD.mat"

        data_dict["JD"] = dict()
        data_dict["JD"]["path"] = "p6/p6_19_02_18_a000/"
        data_dict["JD"]["gui_file"] = "p6_19_02_18_a000_ground_truth_JD.mat"

    elif ms_to_benchmark == "p11_19_04_30_a001_ms":
        data_dict["gt"] = dict()
        data_dict["gt"]["path"] = "p11/p11_19_04_30_a001"
        # single expert labeling
        data_dict["gt"]["gui_file"] = "p11_19_04_30_a001_gound_truth.mat"
        data_dict["gt"]["cells"] = np.array([0, 2, 3, 4])  # 4 not seen by the network
        # data_dict["gt"]["cells"] = np.array([4])  # 4 not seen by the network
        data_dict["gt"]["trace_file_name"] = "p11_19_04_30_a001_raw_traces.npy"
        data_dict["gt"]["trace_var_name"] = "raw_traces"

        data_dict["caiman"] = dict()
        data_dict["caiman"]["path"] = "p11/p11_19_04_30_a001"
        # data_dict["caiman"]["file_name_onsets"] = "caiman_matlab/p11_19_04_30_a001_MCMC_(5).mat"
        data_dict["caiman"]["file_name_onsets"] = "caiman_matlab/p11_19_04_30_a001_MCMC_new.mat"
        data_dict["caiman"]["onsets_var_name"] = "spikenums"
        data_dict["caiman"]["to_bin"] = True
        data_dict["caiman"]["caiman_fiji_mapping"] = "P11_19_04_30_a001_fiji_vs_caiman.npy"
        data_dict["caiman"]["trace_file_name"] = "caiman_matlab/p11_19_04_30_a001_Traces.mat"
        data_dict["caiman"]["trace_var_name"] = "C_df"


    elif ms_to_benchmark == "p8_18_10_24_a006_ms":
        data_dict["gt"] = dict()
        data_dict["gt"]["path"] = "p8/p8_18_10_24_a006"
        # single expert labeling
        data_dict["gt"]["gui_file"] = "p8_18_10_24_a006_GUI_transients_RD.mat"
        # "p8_18_10_24_a006_GUI_transients_RD.mat" "p8_18_10_24_a006_fusion_validation.mat"
        data_dict["gt"]["cells"] = np.array([28, 32, 33])
        # np.array([6, 7, 9, 10, 11, 18, 24, 28, 32, 33]) 28, 32, 33
        data_dict["gt"]["trace_file_name"] = "p8_18_10_24_a006_raw_Traces.mat"
        data_dict["gt"]["trace_var_name"] = "raw_traces"

        data_dict["caiman"] = dict()
        data_dict["caiman"]["path"] = "p8/p8_18_10_24_a006"
        data_dict["caiman"]["file_name_onsets"] = "p8_18_10_24_a006_Spikenums_caiman.mat"
        data_dict["caiman"]["onsets_var_name"] = "spikenums"
        data_dict["caiman"]["to_bin"] = True
        data_dict["caiman"]["trace_file_name"] = "p8_18_10_24_a006_Traces.mat"
        data_dict["caiman"]["trace_var_name"] = "C_df"

        # data_dict["caiman_filt"] = dict()
        # data_dict["caiman_filt"]["path"] = "p8/p8_18_10_24_a006"
        # data_dict["caiman_filt"]["file_name"] = "p8_18_10_24_a006_Spikenums_caiman.mat"
        # data_dict["caiman_filt"]["var_name"] = "rasterdur"

    elif ms_to_benchmark == "p11_17_11_24_a000_ms":
        data_dict["gt"] = dict()
        data_dict["gt"]["path"] = "p11/p11_17_11_24_a000"
        # single expert labeling
        data_dict["gt"]["gui_file"] = "p11_17_11_24_a000_fusion_validation.mat"

        data_dict["gt"]["trace_file_name"] = "p11_17_11_24_a000_raw_Traces.mat"
        data_dict["gt"]["trace_var_name"] = "raw_traces"
        # "p11_17_11_24_a000_GUI_transientsRD.mat" "p11_17_11_24_a000_fusion_validation.mat"
        data_dict["gt"]["cells"] = np.array(
            [3, 45])  # np.array([3, 17, 22, 24, 25, 29, 30, 33, 45]) # np.array([3, 45])

        # data_dict["caiman"] = dict()
        # data_dict["caiman"]["path"] = "p11/p11_17_11_24_a000"
        # data_dict["caiman"]["file_name_onsets"] = "p11_17_11_24_a000_Corrected_RasterDur.mat"
        # data_dict["caiman"]["onsets_var_name"] = "filt_Bin100ms_spikedigital"
        # data_dict["caiman"]["to_bin"] = True
        # data_dict["caiman"]["trace_file_name"] = "p11_17_11_24_a000_Traces.mat"
        # data_dict["caiman"]["trace_var_name"] = "C_df"
        #
        # data_dict["caiman_filt"] = dict()
        # data_dict["caiman_filt"]["path"] = "p11/p11_17_11_24_a000"
        # data_dict["caiman_filt"]["file_name"] = "p11_17_11_24_a000_Corrected_RasterDur.mat"
        # data_dict["caiman_filt"]["var_name"] = "corrected_rasterdur"
        # no caiman available yet

        data_dict["caiman"] = dict()
        data_dict["caiman"]["path"] = "p11/p11_17_11_24_a000"
        data_dict["caiman"]["file_name_onsets"] = "p11_17_11_24_a000_spikenums_MCMC.mat"
        data_dict["caiman"]["onsets_var_name"] = "spikenums"
        data_dict["caiman"]["to_bin"] = True
        data_dict["caiman"]["trace_file_name"] = "p11_17_11_24_a000_Traces.mat"
        data_dict["caiman"]["trace_var_name"] = "C_df"

        # data_dict["suite2p"] = dict()
        # data_dict["suite2p"]["path"] = "p11/p11_17_11_24_a000/suite2p/"
        # data_dict["suite2p"]["caiman_suite2p_mapping"] = "P11_17_11_24_a000_suite2p_vs_caiman.npy"
        # data_dict["suite2p"]["threshold"] = 100 # best compromise for transients, for better sensibility try 70
        #
        data_dict["RD"] = dict()
        data_dict["RD"]["path"] = "p11/p11_17_11_24_a000/"
        data_dict["RD"]["gui_file"] = "p11_17_11_24_a000_GUI_transientsRD.mat"

        data_dict["JD"] = dict()
        data_dict["JD"]["path"] = "p11/p11_17_11_24_a000/"
        data_dict["JD"]["gui_file"] = "p11_17_11_24_a000_transients_GUI_JD.mat"

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
        # single expert labeling
        data_dict["gt"]["gui_file"] = "p13_18_10_29_a001_fusion_validation.mat"
        data_dict["gt"]["trace_file_name"] = "p13_18_10_29_a001_raw_Traces.mat"
        data_dict["gt"]["trace_var_name"] = "raw_traces"
        # data_dict["gt"]["gui_file"] = "p13_18_10_29_a001_GUI_transientsRD.mat"
        # data_dict["gt"]["cnn"] = "cell_classifier_results_txt/cell_classifier_cnn_results_P13_18_10_29_a001.txt"
        # data_dict["gt"]["cnn_threshold"] = 0.5
        # not done yet, JD still need to do the GT and then fusion need to be done with RD
        data_dict["gt"]["cells"] = np.array([77, 117])  # np.array([0, 5, 12, 13, 31, 42, 44, 48, 51, 77, 117])

        data_dict["caiman"] = dict()
        data_dict["caiman"]["path"] = "p13/p13_18_10_29_a001"
        data_dict["caiman"]["file_name_onsets"] = "p13_18_10_29_a001_spikenums_MCMC.mat"
        data_dict["caiman"]["onsets_var_name"] = "spikenums"
        data_dict["caiman"]["to_bin"] = True
        data_dict["caiman"]["trace_file_name"] = "p13_18_10_29_a001_Traces.mat"
        data_dict["caiman"]["trace_var_name"] = "C_df"
        # no CAIMAN results available
    elif ms_to_benchmark == "p7_17_10_12_a000":
        # gt as ground_truth
        data_dict["gt"] = dict()
        data_dict["gt"]["path"] = "p7/p7_17_10_12_a000"
        data_dict["gt"]["gui_file"] = "p7_17_10_12_a000_fusion_validation.mat"
        # if 117 cells:
        # data_dict["gt"]["gui_file"] = "p7_17_10_12_a000_GUI_transients_RD.mat"

        data_dict["gt"]["trace_file_name"] = "p7_17_10_12_a000_Traces.mat"
        data_dict["gt"]["trace_var_name"] = "C_df"
        # data_dict["gt"]["gt_file"] = "p7_17_10_12_a000_cell_to_suppress_ground_truth.txt"
        # data_dict["gt"]["cnn"] = "cell_classifier_results_txt/cell_classifier_cnn_results_P7_17_10_12_a000.txt"
        # data_dict["gt"]["cnn_threshold"] = 0.5
        data_dict["gt"]["cells"] = np.array([2, 25])  # np.array([2, 25]) # np.array([2, 3, 8, 11, 12, 14, 17, 18, 24, 25])  #  np.arange(117)
        # data_dict["gt"]["cells_to_remove"] = np.array([52, 75])

        data_dict["caiman"] = dict()
        data_dict["caiman"]["path"] = "p7/p7_17_10_12_a000"
        data_dict["caiman"]["file_name_onsets"] = "Robin_30_01_19/p7_17_10_12_a000_spikenums.mat"
        data_dict["caiman"]["onsets_var_name"] = "spikenums"
        data_dict["caiman"]["to_bin"] = True
        data_dict["caiman"]["trace_file_name"] = "p7_17_10_12_a000_Traces.mat"
        data_dict["caiman"]["trace_var_name"] = "C_df"
        #
        # data_dict["suite2p"] = dict()
        # data_dict["suite2p"]["path"] = "p7/p7_17_10_12_a000/suite2p/"
        # data_dict["suite2p"]["caiman_suite2p_mapping"] = "P7_17_10_12_a000_suite2p_vs_caiman.npy"
        # data_dict["suite2p"]["threshold"] = 5  # try 30, 70
        #
        data_dict["RD"] = dict()
        data_dict["RD"]["path"] = "p7/p7_17_10_12_a000"
        data_dict["RD"]["gui_file"] = "p7_17_10_12_a000_GUI_transients_RD.mat"

        data_dict["JD"] = dict()
        data_dict["JD"]["path"] = "p7/p7_17_10_12_a000"
        data_dict["JD"]["gui_file"] = "p7_17_10_12_a000_GUI_transients_JD.mat"

        # data_dict["caiman_filt"] = dict()
        # data_dict["caiman_filt"]["path"] = "p7/p7_17_10_12_a000"
        # data_dict["caiman_filt"]["file_name"] = "Robin_30_01_19/p7_17_10_12_a000_caiman_raster_dur.mat"
        # data_dict["caiman_filt"]["var_name"] = "rasterdur"
    elif ms_to_benchmark == "p8_18_10_24_a005_ms":
        # gt as ground_truth
        data_dict["gt"] = dict()
        data_dict["gt"]["path"] = "p8/p8_18_10_24_a005"
        data_dict["gt"]["gui_file"] = "p8_18_10_24_a005_fusion_validation.mat"
        # data_dict["gt"]["gui_file"] = "p8_18_10_24_a005_GUI_transientsRD.mat"
        # data_dict["gt"]["gui_file"] = "p8_18_10_24_a005_GUI_Transiant MP.mat"
        # data_dict["gt"]["cnn"] = "cell_classifier_results_txt/cell_classifier_cnn_results_P8_18_10_24_a005.txt"
        # data_dict["gt"]["cnn_threshold"] = 0.5
        data_dict["gt"]["cells"] = np.array([0, 1, 9, 10, 13, 15, 28, 41, 42, 110, 207, 321])
        # JD, RD: np.array([0, 1, 9, 10, 13, 15, 28, 41, 42, 110, 207, 321])
        # done by Michel: 0, 1, 9, 10, 13, 15
        data_dict["gt"]["trace_file_name"] = "p8_18_10_24_a005_raw_Traces.mat"
        data_dict["gt"]["trace_var_name"] = "raw_traces"

        data_dict["caiman"] = dict()
        data_dict["caiman"]["path"] = "p8/p8_18_10_24_a005"
        data_dict["caiman"]["file_name_onsets"] = "p8_18_10_24_a005_MCMC.mat"
        data_dict["caiman"]["onsets_var_name"] = "spikenums"
        data_dict["caiman"]["to_bin"] = True
        data_dict["caiman"]["trace_file_name"] = "p8_18_10_24_a005_Traces.mat"
        data_dict["caiman"]["trace_var_name"] = "C_df"

        data_dict["RD"] = dict()
        data_dict["RD"]["path"] = "p8/p8_18_10_24_a005"
        data_dict["RD"]["gui_file"] = "p8_18_10_24_a005_GUI_transientsRD.mat"

        data_dict["JD"] = dict()
        data_dict["JD"]["path"] = "p8/p8_18_10_24_a005"
        data_dict["JD"]["gui_file"] = "p8_18_10_24_a005_GUI_transientsJD.mat"

        data_dict["MP"] = dict()
        data_dict["MP"]["path"] = "p8/p8_18_10_24_a005"
        data_dict["MP"]["gui_file"] = "p8_18_10_24_a005_GUI_Transiant MP.mat"
        data_dict["MP"]["cells_not_predicted"] = np.array([28, 41, 42, 110, 207, 321])
        #
        # data_dict["suite2p"] = dict()
        # data_dict["suite2p"]["path"] = "p8/p8_18_10_24_a005/suite2p/"
        # data_dict["suite2p"]["caiman_suite2p_mapping"] = "P8_18_10_24_a005_suite2p_vs_caiman.npy"
        # data_dict["suite2p"]["threshold"] = 45  # try 30, 70

        # data_dict["caiman_filt"] = dict()
        # data_dict["caiman_filt"]["path"] = "p8/p8_18_10_24_a005"
        # data_dict["caiman_filt"]["file_name"] = "p8_18_10_24_a005_filt_RasterDur.mat"
        # data_dict["caiman_filt"]["var_name"] = "rasterdur"


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
    # ms_to_benchmarks = ["p7_17_10_12_a000", "p8_18_10_24_a005_ms", "p8_18_10_24_a006_ms",
    #                     "p12_17_11_10_a000"]
    # "p11_17_11_24_a000_ms", "p13_18_10_29_a001_ms"
    # ms_to_benchmarks = ["p12_17_11_10_a000"]
    # ms_to_benchmarks = ["p7_17_10_12_a000"]
    # ms_to_benchmarks = ["p13_18_10_29_a001_ms"]
    # ms_to_benchmarks = ["p8_18_10_24_a006_ms"]
    # ms_to_benchmarks = ["p12_17_11_10_a000"]
    # ms_to_benchmarks = ["p7_17_10_12_a000"]
    ms_to_benchmarks = ["p8_18_10_24_a006_ms"]
    # ms_to_benchmarks = ["p8_18_10_24_a005_ms"]
    # ms_to_benchmark = "artificial_ms"
    # no benchmarks yet for p13
    # ms_to_benchmarks = ["p13_18_10_29_a001_ms"]
    # ms_to_benchmarks = ["p7_17_10_12_a000", "p8_18_10_24_a005_ms", "p8_18_10_24_a006_ms",
    #                     "p12_17_11_10_a000", "p11_17_11_24_a000_ms"]
    # ms_to_benchmarks = ["p7_17_10_12_a000", "p8_18_10_24_a005_ms",
    #                     "p12_17_11_10_a000", "p11_17_11_24_a000_ms"]
    # ms_to_benchmarks = ["p7_17_10_12_a000", "p8_18_10_24_a005_ms", "p11_17_11_24_a000_ms"]
    # ms_to_benchmarks = ["p8_18_10_24_a006_ms"]
    # ms_to_benchmarks = ["p7_17_10_12_a000", "p8_18_10_24_a005_ms"]
    # ms_to_benchmarks = ["p8_18_10_24_a006_ms",
    #                     "p12_17_11_10_a000", "p11_17_11_24_a000_ms", "p13_18_10_29_a001_ms"]
    # ms_to_benchmarks = ["p7_17_10_12_a000", "p8_18_10_24_a005_ms", "p8_18_10_24_a006_ms",
    #                     "p11_17_11_24_a000_ms", "p12_17_11_10_a000"]
    # ms_to_benchmarks = ["p7_17_10_12_a000"]
    # ms_to_benchmarks = ["p11_17_11_24_a000_ms"]
    # ms_to_benchmarks = ["p8_18_10_24_a006_ms"]
    # gad-cre + oriens
    # ms_to_benchmarks = ["p6_19_02_18_a000_ms", "p8_18_10_24_a006_ms",
    #                     "p11_19_04_30_a001_ms"]
    # ms_to_benchmarks = ["p6_19_02_18_a000_ms", "p11_19_04_30_a001_ms"]
    # gad-cre
    # ms_to_benchmarks = ["p6_19_02_18_a000_ms", "p11_19_04_30_a001_ms"]
    # ms_to_benchmarks = ["p6_19_02_18_a000_ms"]
    do_onsets_benchmarks = False
    do_plot_roc_predictions = False
    do_plot_recall_vs_precision_curve = True
    produce_separate_benchmarks = True
    do_plot_roc_predictions_for_suite_2p = False
    # ########### end options ###################
    global_benchmarks = None
    description = ""
    boost_predictions = False
    predictions_threshold = 0.5

    # predictions_to_load = ["epoch_11", "meso_2", "meso_3", "meso_4", "meso_8", "meso_6", "meso_7", "meso_8", "meso_9",
    #                        "meso_10", "meso_11", "meso_12", "meso_13", "meso_14"]
    predictions_to_load = ["meso_9", "v2_epoch_8", "v2_epoch_12", "v2_epoch_14",
                           "v2_epoch_17", "v2_epoch_19"]
    predictions_to_load = ["meso_9", "v2_epoch_19"]
    predictions_to_load = ["v2_epoch_19"]
    # gad-cre
    # predictions_to_load = ["meso_9", "cre_v1_epoch_10", "cre_v1_epoch_15", "cre_v1_epoch_19",
    #                        "cre_v1_epoch_22", "cre_v1_epoch_23"]
    # predictions_to_load = ["v1_epoch_15"]
    # TO PLOT predictions distribution for each metrics, use only one predictions_to_load
    # predictions_to_load = []
    for ms_to_benchmark in ms_to_benchmarks:
        print(f"ms_to_benchmark {ms_to_benchmark}")
        data_dict = dict()
        load_data_dict(ms_to_benchmark, data_dict)
        # ground truth
        data_file = hdf5storage.loadmat(os.path.join(path_data, data_dict["gt"]["path"], data_dict["gt"]["gui_file"]))
        peak_nums = data_file['LocPeakMatrix_Python'].astype(int)
        spike_nums = data_file['Bin100ms_spikedigital_Python'].astype(int)
        if data_dict["gt"]["trace_file_name"].endswith(".mat"):
            data_file_traces = hdf5storage.loadmat(os.path.join(path_data, data_dict["gt"]["path"],
                                                                data_dict["gt"]["trace_file_name"]))
            traces = data_file_traces[data_dict["gt"]['trace_var_name']]
        else:
            # npy format
            traces = np.load((os.path.join(path_data, data_dict["gt"]["path"],
                                           data_dict["gt"]["trace_file_name"])))
        print(f"{ms_to_benchmark} traces.shape {traces.shape}")
        # inter_neurons = data_file['inter_neurons'].astype(int)
        if 'cells_to_remove' in data_file:
            cells_to_remove = data_file['cells_to_remove'].astype(int)
        else:
            cells_to_remove = None
        ground_truth_raster_dur = build_spike_nums_dur(spike_nums, peak_nums)
        # print(f"ground_truth_raster_dur.shape {ground_truth_raster_dur.shape}")
        n_cells = ground_truth_raster_dur.shape[0]
        n_frames = ground_truth_raster_dur.shape[1]

        cells_for_benchmark = data_dict["gt"]["cells"]

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
            cells_for_benchmark = np.setdiff1d(cells_for_benchmark, cells_false_gt)

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
            print(f"cells_to_remove {cells_to_remove}")
            # print(f"cells_for_benchmark before removing {cells_for_benchmark}")
            cells_for_benchmark = np.setdiff1d(cells_for_benchmark, cells_to_remove)
            # print(f"cells_for_benchmark after removing {cells_for_benchmark}")

        if "cells_to_remove" in data_dict["gt"]:
            cells_for_benchmark = np.setdiff1d(cells_for_benchmark, data_dict["gt"]["cells_to_remove"])
        # print(f"cells_for_benchmark {cells_for_benchmark}")
        # return

        # data_file = hdf5storage.loadmat(os.path.join(path_data, data_dict["rnn"]["path"], data_dict["rnn"]["file_name"]))
        # rnn_predictions = data_file[data_dict["rnn"]['predictions']]
        #
        # # we remove cell for which predictions was not done, aka those with sum predictions == 0
        # cell_predictions_count = np.sum(rnn_predictions, axis=1)
        # cells_to_remove = np.where(cell_predictions_count == 0)[0]
        # cells_for_benchmark = np.setdiff1d(cells_for_benchmark, cells_to_remove)

        if do_plot_roc_predictions_for_suite_2p:
            if "suite2p" in data_dict:
                value = data_dict["suite2p"]
                spks = np.load(os.path.join(path_data, value["path"], 'spks.npy'))
                is_cell = np.load(os.path.join(path_data, value["path"], 'iscell.npy'))
                caiman_suite2p_mapping = np.load(
                    os.path.join(path_data, value["path"], value["caiman_suite2p_mapping"]))
                suite2p_predictions = np.zeros((n_cells, n_frames))
                cell_mapping_index = 0
                # print(f"len(caiman_suite2p_mapping) {len(caiman_suite2p_mapping)}, "
                #       f"len(spks) {len(spks)}")
                for suite2p_cell in np.arange(len(spks)):
                    if is_cell[suite2p_cell][0] == 0:
                        continue
                    if caiman_suite2p_mapping[cell_mapping_index] >= 0:
                        map_cell = caiman_suite2p_mapping[cell_mapping_index]
                        # using deconvolution value, cell is active if value > 0
                        # TODO: see to use a threshold superior than 0
                        suite2p_predictions[map_cell] = spks[suite2p_cell]
                    cell_mapping_index += 1
                cells_to_keep = []
                for cell in cells_for_benchmark:
                    if cell not in caiman_suite2p_mapping:
                        print(f"Cell {cell} has no match in suite2p segmentation")
                    else:
                        cells_to_keep.append(cell)
                cells_for_benchmark = np.array(cells_to_keep)
                plot_roc_predictions(ground_truth_raster_dur=ground_truth_raster_dur,
                                     rnn_predictions=suite2p_predictions,
                                     cells=cells_for_benchmark,
                                     time_str=time_str, description=ms_to_benchmark + "_suite2p",
                                     path_results=path_results, save_formats="pdf", for_suite2p=True)
            # return

        predicted_raster_dur_dict = dict()
        predicted_spike_nums_dict = dict()

        # loading predictions
        file_names = []
        # look for filenames in the fisrst directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(path_data, data_dict["gt"]["path"],
                                                                         "predictions")):
            file_names.extend(local_filenames)
            break

        if len(file_names) > 0:
            for file_name in file_names:
                for prediction_key in predictions_to_load:
                    if (prediction_key in file_name) and ("filtered_predicted_raster_dur" not in file_name):
                        data_file = hdf5storage.loadmat(os.path.join(path_data, data_dict["gt"]["path"],
                                                                     "predictions", file_name))
                        predicted_raster_dur = \
                            build_raster_dur_from_predictions(predictions=data_file["predictions"],
                                                              predictions_threshold=predictions_threshold,
                                                              cells=cells_for_benchmark,
                                                              n_total_cells=n_cells,
                                                              n_frames=n_frames)
                        # removing predictions around concatenations
                        clean_raster_at_concatenation(predicted_raster_dur)

                        rnn_predictions = data_file['predictions']
                        # we remove cell for which predictions was not done, aka those with sum predictions == 0
                        cell_predictions_count = np.sum(rnn_predictions, axis=1)
                        cells_to_remove_with_no_predictions = np.where(cell_predictions_count == 0)[0]
                        cells_for_benchmark = np.setdiff1d(cells_for_benchmark, cells_to_remove_with_no_predictions)

                        if boost_predictions:
                            rnn_raster_dur = get_boost_rnn_raster_dur(rnn_raster_dur, traces)
                            predicted_raster_dur_dict[prediction_key + "_boost"] = predicted_raster_dur
                        else:
                            predicted_raster_dur_dict[prediction_key] = predicted_raster_dur

                        if do_plot_roc_predictions:
                            plot_roc_predictions(ground_truth_raster_dur=ground_truth_raster_dur,
                                                 rnn_predictions=rnn_predictions,
                                                 cells=cells_for_benchmark,
                                                 time_str=time_str, description=ms_to_benchmark + "_" + prediction_key,
                                                 path_results=path_results, save_formats="pdf")
                        if do_plot_recall_vs_precision_curve:
                            plot_recall_vs_precision_curve(ground_truth_raster_dur=ground_truth_raster_dur,
                                                           rnn_predictions=rnn_predictions, traces=traces,
                                                           cells=cells_for_benchmark,
                                                           use_transients=False,
                                                           time_str=time_str,
                                                           description=ms_to_benchmark + "_" + prediction_key
                                                                       + "_frames",
                                                           path_results=path_results, save_formats="png")
                            plot_recall_vs_precision_curve(ground_truth_raster_dur=ground_truth_raster_dur,
                                                           rnn_predictions=rnn_predictions, traces=traces,
                                                           cells=cells_for_benchmark,
                                                           use_transients=True,
                                                           time_str=time_str,
                                                           description=ms_to_benchmark + "_" + prediction_key
                                                                       + "_transients",
                                                           path_results=path_results, save_formats="png")


        # value is a dict
        for key, value in data_dict.items():
            if key == "gt":
                continue
            if key == "suite2p":
                spks = np.load(os.path.join(path_data, value["path"], 'spks.npy'))
                is_cell = np.load(os.path.join(path_data, value["path"], 'iscell.npy'))
                caiman_suite2p_mapping = np.load(
                    os.path.join(path_data, value["path"], value["caiman_suite2p_mapping"]))
                suite2p_raster_dur = np.zeros((n_cells, n_frames), dtype="int8")
                cell_mapping_index = 0
                threshold = value["threshold"]
                for cell in np.arange(len(spks)):
                    if is_cell[cell][0] == 0:
                        continue
                    if caiman_suite2p_mapping[cell_mapping_index] >= 0:
                        map_cell = caiman_suite2p_mapping[cell_mapping_index]
                        # using deconvolution value, cell is active if value > 0
                        suite2p_raster_dur[map_cell, spks[cell] > threshold] = 1
                    cell_mapping_index += 1
                cells_to_keep = []
                for cell in cells_for_benchmark:
                    if cell not in caiman_suite2p_mapping:
                        print(f"Cell {cell} has no match in suite2p segmentation")
                    else:
                        cells_to_keep.append(cell)
                cells_for_benchmark = np.array(cells_to_keep)
                clean_raster_at_concatenation(suite2p_raster_dur)
                predicted_raster_dur_dict[key] = suite2p_raster_dur
                # removing predictions around concatenations
            # elif key == "rnn" and ("prediction_threshold" in value):
            #     data_file = hdf5storage.loadmat(os.path.join(path_data, value["path"], value["file_name"]))
            #     rnn_raster_dur = \
            #         build_raster_dur_from_predictions(predictions=data_file[value["predictions"]],
            #                                           predictions_threshold=value["prediction_threshold"],
            #                                           cells=cells_for_benchmark,
            #                                           n_total_cells=n_cells,
            #                                           n_frames=n_frames)
            #     if "trace_file_name" in value:
            #         data_file = hdf5storage.loadmat(os.path.join(path_data, value["path"], value["trace_file_name"]))
            #         traces = data_file[value['trace_var_name']]
            #     if ("boost_rnn" in value) and value["boost_rnn"]:
            #         rnn_raster_dur = get_boost_rnn_raster_dur(rnn_raster_dur, traces)
            #         predicted_raster_dur_dict["rnn_boost"] = rnn_raster_dur
            #     else:
            #         predicted_raster_dur_dict[key] = rnn_raster_dur
            elif "gui_file" in value:
                data_file = hdf5storage.loadmat(
                    os.path.join(path_data, value["path"], value["gui_file"]))
                peak_nums = data_file['LocPeakMatrix_Python'].astype(int)
                spike_nums = data_file['Bin100ms_spikedigital_Python'].astype(int)
                if "cells_not_predicted" in value:
                    cells_not_predicted = value["cells_not_predicted"]
                    peak_nums[cells_not_predicted] = 0
                    spike_nums[cells_not_predicted] = 0
                predicted_raster_dur_dict[key] = build_spike_nums_dur(spike_nums, peak_nums)


            elif "prediction_threshold" in value:  # ("rnn" in key) and
                data_file = hdf5storage.loadmat(os.path.join(path_data, value["path"], value["file_name"]))
                predicted_raster_dur_dict[key] = \
                    build_raster_dur_from_predictions(predictions=data_file[value["predictions"]],
                                                      predictions_threshold=value["prediction_threshold"],
                                                      cells=cells_for_benchmark,
                                                      n_total_cells=n_cells,
                                                      n_frames=n_frames)
                # removing predictions around concatenations
                clean_raster_at_concatenation(predicted_raster_dur_dict[key])
            else:
                if "to_bin" in value:
                    # onsets
                    data_file = hdf5storage.loadmat(os.path.join(path_data, value["path"], value["file_name_onsets"]))
                    caiman_spike_nums = data_file[value['onsets_var_name']].astype(int)
                    if value["trace_file_name"].endswith(".mat"):
                        data_file = hdf5storage.loadmat(os.path.join(path_data, value["path"],
                                                                     value["trace_file_name"]))
                        traces_caiman = data_file[value['trace_var_name']]
                    else:
                        # npy
                        traces_caiman = np.load(os.path.join(path_data, value["path"],
                                                             value["trace_file_name"]))

                    raster_dur = get_raster_dur_from_caiman_25000_frames_onsets_new_version(caiman_spike_nums,
                                                                                            traces_caiman)
                    predicted_raster_dur_dict[key] = raster_dur
                    # removing predictions around concatenations
                    clean_raster_at_concatenation(predicted_raster_dur_dict[key])

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

                    # removing predictions around concatenations
                    clean_raster_at_concatenation(predicted_raster_dur_dict[key])

                # mapping fiji coord to caiman coord if necessary
                if "caiman_fiji_mapping" in value:
                    # print("caiman_fiji_mapping in value")
                    caiman_fiji_mapping = np.load(
                        os.path.join(path_data, value["path"], value["caiman_fiji_mapping"]))
                    caiman_raster_dur = np.zeros((n_cells, n_frames), dtype="int8")
                    cells_mapped = []
                    # caiman_fiji_mapping take a cell index from caiman segmentation
                    # and return a value > 0 if a cell of Fiji matches (the value is the index of this cell in fiji)
                    # -1 otherwise
                    for cell in np.arange(predicted_raster_dur_dict[key].shape[0]):
                        if caiman_fiji_mapping[cell] >= 0:
                            # print(f"{ms_to_benchmark}: Cell {cell} -> {caiman_fiji_mapping[cell]}")
                            map_cell = caiman_fiji_mapping[cell]
                            # using deconvolution value, cell is active if value > 0
                            caiman_raster_dur[map_cell] = predicted_raster_dur_dict[key][cell]
                            cells_mapped.append(map_cell)
                    predicted_raster_dur_dict[key] = caiman_raster_dur
                    cells_to_keep = []
                    for cell in cells_for_benchmark:
                        if cell not in cells_mapped:  # caiman_fiji_mapping:
                            print(f"Cell {cell} has no match in caiman segmentation")
                        else:
                            cells_to_keep.append(cell)
                    cells_for_benchmark = np.array(cells_to_keep)

        benchmarks = BenchmarkRasterDur(description=ms_to_benchmark, ground_truth_raster_dur=ground_truth_raster_dur,
                                        predicted_raster_dur_dict=predicted_raster_dur_dict, cells=cells_for_benchmark,
                                        traces=traces, rnn_predictions=rnn_predictions)
        if produce_separate_benchmarks:
            tmp_description = ms_to_benchmark
            tmp_description += f"_thr_{predictions_threshold}_"
            benchmarks.compute_stats()
            benchmarks.plot_boxplot_predictions_stat_by_metrics(path_results=path_results,
                                                                description=tmp_description, time_str=time_str,
                                                                save_formats=["png"])
            # benchmarks.plot_boxplots_full_stat(description=tmp_description, time_str=time_str,
            #                                    path_results=path_results, with_cells=True,
            #                                    for_frames=True, save_formats=["pdf", "png"])
            benchmarks.plot_boxplots_full_stat(description=tmp_description, time_str=time_str,
                                               path_results=path_results, with_cells=True,
                                               for_frames=False, save_formats=["png"])
            benchmarks.plot_boxplots_f1_score(description=tmp_description, time_str=time_str,
                                               path_results=path_results, with_cells=True,
                                               for_frames=False, save_formats=["png"])
        if global_benchmarks is None:
            global_benchmarks = benchmarks
        else:
            global_benchmarks = global_benchmarks.fusion(benchmarks)

        description += ms_to_benchmark
        description += f"_thr_{predictions_threshold}_"
    if len(ms_to_benchmarks) > 1:
        # print(f"GLOBAL BENCH {global_benchmarks.description}")
        global_benchmarks.compute_stats()
        global_benchmarks.plot_boxplot_predictions_stat_by_metrics(path_results=path_results,
                                                                   description=description, time_str=time_str,
                                                                   save_formats=["png"])
        # global_benchmarks.plot_boxplots_full_stat(description=description, time_str=time_str, path_results=path_results,
        #                                           for_frames=True, save_formats=["pdf", "png"], with_cells=True)
        global_benchmarks.plot_boxplots_full_stat(description=description, time_str=time_str, path_results=path_results,
                                                  for_frames=False, save_formats=["png"], with_cells=True)
        global_benchmarks.plot_boxplots_f1_score(description=description, time_str=time_str, path_results=path_results,
                                                  for_frames=False, save_formats=["png"], with_cells=True)
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
