import matplotlib.pyplot as plt
import numpy as np
from sortedcontainers import SortedList, SortedDict
from matplotlib.patches import Patch

class MvtSelectionGui:
    def __init__(self, mouse_session):
        self.ms = mouse_session

        self.piezo = self.ms.raw_piezo
        # times_for_1_ms = self.abf_sampling_rate / 1000
        # window_in_times = int(times_for_1_ms * window_in_ms)
        self.n_times = len(self.piezo)

        # key will be a tuple corresponding to the beginning and end (included) of a mvt in frames
        # the value will be the category, as int
        self.mvt_categories = dict()

        self.original_mvt_categories = dict()
        self.original_mvt_categories_onsets = SortedDict()

        self.categories_code = dict()
        self.categories_code["twitch"] = 0
        self.categories_code["short lasting mvt"] = 1
        self.categories_code["noise mvt"] = 2

        self.categories_color = dict()
        self.categories_color[0] = "blue"
        self.categories_color[1] = "yellow"
        self.categories_color[2] = "grey"

        self.categories_name = dict()
        for name, category in self.categories_code.items():
            self.categories_name[category] = name

        self.n_categories = len(self.categories_code)

        # feeling the original categories
        for twitches_period in self.ms.twitches_frames_periods:
            t = (twitches_period[0], twitches_period[1])
            self.original_mvt_categories_onsets[twitches_period[0]] = t
            self.original_mvt_categories[t] = self.categories_code["twitch"]

        for short_lasting_mvt in self.ms.short_lasting_mvt:
            t = (short_lasting_mvt[0], short_lasting_mvt[1])
            self.original_mvt_categories_onsets[short_lasting_mvt[0]] = t
            self.original_mvt_categories[t] = self.categories_code["short lasting mvt"]

        self.onset_mvts = list(self.original_mvt_categories_onsets.keys())
        self.n_mvt_periods = len(self.original_mvt_categories_onsets)
        self.mvt_index_to_display = 0

        # making an array containing the index of complex and intermediate mvt to find them in each window
        # self.ms.complex_mvt
        # self.ms.intermediate_behavourial_events
        self.complex_mvt_numbers = np.ones(self.n_times, dtype="int16")
        self.complex_mvt_numbers *= -1
        self.intermediate_behavourial_events_numbers = np.ones(self.n_times, dtype="int16")
        self.intermediate_behavourial_events_numbers *= -1

        for complex_mvt_index, complex_mvt in enumerate(self.ms.complex_mvt):
            self.complex_mvt_numbers[complex_mvt[0]:complex_mvt[1]+1] = complex_mvt_index

        for index, intermediate_behavourial_event in enumerate(self.ms.intermediate_behavourial_events):
            self.intermediate_behavourial_events_numbers[intermediate_behavourial_event[0]:intermediate_behavourial_event[1]+1] = index

        self.ax = None
        self.fig = None
        self.plot_graph(first_time=True)
        # self.ms.twitches_frames
        # self.ms.short_lasting_mvt
        # self.ms.noise_mvt

    def plot_graph(self, first_time=False):
        # TODO: do like in Claire, flush canvas at each iteration
        """
        will display mvt that match the mvt_index_to_display
        :return:
        """
        period_times = self.original_mvt_categories_onsets[self.onset_mvts[self.mvt_index_to_display]]
        print(f"period_times {period_times} self.mvt_index_to_display {self.mvt_index_to_display}")
        if period_times in self.mvt_categories:
            category = self.mvt_categories[period_times]
        else:
            category = self.original_mvt_categories[period_times]

        # first_derivative = np.diff(self.piezo) / np.diff(np.arange(self.n_times))
        window_size = self.ms.abf_sampling_rate * 4
        min_time = np.max((0, period_times[0]-window_size))
        max_time = np.min((self.n_times, period_times[1]+1+window_size))
        print(f"min_time {min_time}, max_time {max_time}")
        if period_times[0]-window_size < 0:
            min_local_period_times = window_size
        else:
            min_local_period_times = period_times[0]
        local_period_times = (min_local_period_times, (min_local_period_times+period_times[1]-period_times[0]))

        times_to_display = np.arange(min_time, max_time)

        if first_time:
            self.fig, self.ax = plt.subplots(nrows=1, ncols=1,
                                   gridspec_kw={'height_ratios': [1]},
                                   figsize=(20, 8))
        ax = self.ax
        self.fig.canvas.mpl_connect('key_press_event', self.key_release_action)
        ax.plot(self.ms.abf_times_in_sec[times_to_display], self.piezo[times_to_display], lw=.5, color="black")
        # plt.plot(self.ms.abf_times_in_sec[:-1], np.abs(first_derivative), lw=.5, zorder=10, color="green")
        if self.ms.lowest_std_in_piezo is not None:
            ax.hlines(self.ms.lowest_std_in_piezo, self.ms.abf_times_in_sec[times_to_display[0]],
                      self.ms.abf_times_in_sec[times_to_display[-1]], color="red", linewidth=1,
                      linestyles="dashed", zorder=1)
        # plt.scatter(x=self.abf_times_in_sec[peaks], y=piezo[peaks], marker="*",
        #             color=["black"], s=5, zorder=15)

        beh_mvt_index = np.where(self.intermediate_behavourial_events_numbers[times_to_display] >= 0)[0]
        if len(beh_mvt_index) > 0:
            beh_mvt_index += times_to_display[0]
            beh_mvt_indices = np.unique(self.intermediate_behavourial_events_numbers[beh_mvt_index])
            for index in beh_mvt_indices:
                period = self.ms.intermediate_behavourial_events[index]
                beg_pos = np.max((times_to_display[0], period[0]))
                end_pos = np.min((times_to_display[-1], period[1]))
                ax.axvspan(beg_pos / self.ms.abf_sampling_rate,
                           end_pos / self.ms.abf_sampling_rate,
                           alpha=0.5, facecolor="green", zorder=1)

        complex_mvt_index = np.where(self.complex_mvt_numbers[times_to_display] >= 0)[0]
        if len(complex_mvt_index) > 0:
            complex_mvt_index += times_to_display[0]
            complex_mvt_indices = np.unique(self.complex_mvt_numbers[complex_mvt_index])
            for index in complex_mvt_indices:
                period = self.ms.complex_mvt[index]
                beg_pos = np.max((times_to_display[0], period[0]))
                end_pos = np.min((times_to_display[-1], period[1]))
                ax.axvspan(beg_pos / self.ms.abf_sampling_rate,
                           end_pos / self.ms.abf_sampling_rate,
                           alpha=0.5, facecolor="red", zorder=1)

        ax.axvspan(period_times[0] / self.ms.abf_sampling_rate, period_times[1] / self.ms.abf_sampling_rate,
                        alpha=0.5, facecolor=self.categories_color[category], zorder=1)

        pos = period_times[0] + np.argmax(self.piezo[period_times[0]:period_times[1] + 1])
        # TODO: display scatter for all short movement, twith and noise around
        ax.scatter(x=self.ms.abf_times_in_sec[pos], y=self.piezo[pos], marker="*",
                    color=self.categories_color[category], s=30, zorder=20)

        # TODO: display a line between two period of mvt with the length in ms between both
        # TODO: display piezo in non absolute value

        plt.title(f"piezo {self.ms.description} mvt {self.mvt_index_to_display}/{self.n_mvt_periods}")

        legend_elements = []

        for category in self.categories_name.keys():
            legend_elements.append(Patch(facecolor=self.categories_color[category],
                                         edgecolor='black', label=f'{self.categories_name[category]}'))

        legend_elements.append(Patch(facecolor="red",
                                     edgecolor='black', label="complex mvt"))
        legend_elements.append(Patch(facecolor="green",
                                     edgecolor='black', label="intermediate behavourial events"))

        # legend_elements.append(Line2D([0], [0], marker="o", color="w", lw=0, label="twitches",
        #                               markerfacecolor='blue', markersize=10))

        ax.legend(handles=legend_elements)

        if first_time is False:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        if first_time:
            plt.show()

    # def plot_graph(self, first_time=False):
    #     pass

    def change_actual_mvt(self, code_name):
        period_times = self.original_mvt_categories_onsets[self.onset_mvts[self.mvt_index_to_display]]

        self.mvt_categories[period_times] = self.categories_code[code_name]
        self.original_mvt_categories[period_times] = self.categories_code[code_name]

        self.go_to_next_mvt()

    def save_new_data_and_quit(self):
        pass

    def go_to_next_mvt(self):
        if self.mvt_index_to_display == (self.n_mvt_periods-1):
            self.save_new_data_and_quit()
        else:
            self.mvt_index_to_display += 1
            # self.fig.clf()
            self.ax.clear()
            # plt.close()
            self.plot_graph()

    def go_to_previous_mvt(self):
        if self.mvt_index_to_display > 0:
            self.mvt_index_to_display -= 1
            # self.fig.clf()
            self.ax.clear()
            # plt.close()
            self.plot_graph()

    def key_release_action(self, event):
        print(f"event.key {event.key}")
        if event.key in ["t", "T"]:
            self.change_actual_mvt(code_name="twitch")
        elif event.key in ["m", "M"]:
            self.change_actual_mvt(code_name="short lasting mvt")
        elif event.key in ["n", "N"]:
            self.change_actual_mvt(code_name="noise mvt")
        if event.key == 'right':
            self.go_to_next_mvt()
        elif event.key == 'left':
            self.go_to_previous_mvt()
