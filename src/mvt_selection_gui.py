import matplotlib.pyplot as plt
import numpy as np
from sortedcontainers import SortedList, SortedDict
from matplotlib.patches import Patch

class MvtSelectionGui:
    def __init__(self, mouse_session):
        self.ms = mouse_session

        self.piezo = self.ms.raw_piezo_without_abs
        # times_for_1_ms = self.abf_sampling_rate / 1000
        # window_in_times = int(times_for_1_ms * window_in_ms)
        self.n_times = len(self.piezo)

        # key will be a tuple corresponding to the beginning and end (included) of a mvt in frames
        # the value will be the category, as int
        self.mvt_categories = dict()

        self.original_mvt_categories = dict()
        self.original_mvt_categories_onsets = SortedDict()

        self.categories_code = dict()
        self.categories_code["twitches"] = 0
        self.categories_code["short lasting mvt"] = 1
        self.categories_code["noise"] = 2
        self.categories_code["behavourial events"] = 3


        self.keyboard_code = dict()
        self.keyboard_code[0] = "t"
        self.keyboard_code[1] = "m"
        self.keyboard_code[2] = "n"
        self.keyboard_code[3] = "b"

        self.categories_color = dict()
        self.categories_color[0] = "blue"
        self.categories_color[1] = "yellow"
        self.categories_color[2] = "grey"
        self.categories_color[3] = "green"

        self.categories_name = dict()
        for name, category in self.categories_code.items():
            self.categories_name[category] = name

        self.n_categories = len(self.categories_code)

        # feeling the original categories
        for twitches_period in self.ms.twitches_frames_periods:
            t = (twitches_period[0], twitches_period[1])
            self.original_mvt_categories_onsets[twitches_period[0]] = t
            self.original_mvt_categories[t] = self.categories_code["twitches"]

        for short_lasting_mvt in self.ms.short_lasting_mvt:
            t = (short_lasting_mvt[0], short_lasting_mvt[1])
            self.original_mvt_categories_onsets[short_lasting_mvt[0]] = t
            self.original_mvt_categories[t] = self.categories_code["short lasting mvt"]

        for intermediate_behavourial_events in self.ms.intermediate_behavourial_events:
            t = (intermediate_behavourial_events[0], intermediate_behavourial_events[1])
            self.original_mvt_categories_onsets[intermediate_behavourial_events[0]] = t
            self.original_mvt_categories[t] = self.categories_code["behavourial events"]

        self.onset_mvts = list(self.original_mvt_categories_onsets.keys())
        self.n_mvt_periods = len(self.original_mvt_categories_onsets)
        self.mvt_index_to_display = 0

        # making an array containing the index of complex and intermediate mvt to find them in each window
        # self.ms.complex_mvt
        # self.ms.intermediate_behavourial_events
        self.complex_mvt_numbers = np.ones(self.n_times, dtype="int16")
        self.complex_mvt_numbers *= -1
        # self.intermediate_behavourial_events_numbers = np.ones(self.n_times, dtype="int16")
        # self.intermediate_behavourial_events_numbers *= -1

        for complex_mvt_index, complex_mvt in enumerate(self.ms.complex_mvt):
            self.complex_mvt_numbers[complex_mvt[0]:complex_mvt[1]+1] = complex_mvt_index

        # for index, intermediate_behavourial_event in enumerate(self.ms.intermediate_behavourial_events):
        #     self.intermediate_behavourial_events_numbers[intermediate_behavourial_event[0]:intermediate_behavourial_event[1]+1] = index

        self.mvts_count = dict()
        self.mvts_count[self.categories_code["twitches"]] = len(self.ms.twitches_frames_periods)
        self.mvts_count[self.categories_code["short lasting mvt"]] = len(self.ms.short_lasting_mvt)
        self.mvts_count[self.categories_code["noise"]] = len(self.ms.noise_mvt)
        self.mvts_count[self.categories_code["behavourial events"]] = len(self.ms.intermediate_behavourial_events)

        self.ax = None
        self.fig = None
        self.plot_graph(first_time=True)
        # self.ms.twitches_frames
        # self.ms.short_lasting_mvt
        # self.ms.noise_mvt

    def plot_graph(self, first_time=False):
        """
        will display mvt that match the mvt_index_to_display
        :return:
        """
        period_times = self.original_mvt_categories_onsets[self.onset_mvts[self.mvt_index_to_display]]
        # print(f"period_times {period_times} self.mvt_index_to_display {self.mvt_index_to_display}")
        if period_times in self.mvt_categories:
            category = self.mvt_categories[period_times]
        else:
            category = self.original_mvt_categories[period_times]

        # first_derivative = np.diff(self.piezo) / np.diff(np.arange(self.n_times))
        window_size = self.ms.abf_sampling_rate * 4
        min_time = np.max((0, period_times[0]-window_size))
        max_time = np.min((self.n_times, period_times[1]+1+window_size))
        # print(f"min_time {min_time}, max_time {max_time}")
        if period_times[0]-window_size < 0:
            min_local_period_times = window_size
        else:
            min_local_period_times = period_times[0]
        local_period_times = (min_local_period_times, (min_local_period_times+period_times[1]-period_times[0]))

        times_to_display = np.arange(min_time, max_time)

        closet_mvt_before_pos = None
        closet_mvt_after_pos = None

        other_period_times = []
        other_categories = []
        i = 1
        while (self.mvt_index_to_display-i) >= 0:
            other_period = self.original_mvt_categories_onsets[self.onset_mvts[self.mvt_index_to_display-i]]
            if other_period[1] > min_time:
                other_period_times.append(other_period)
                if other_period in self.mvt_categories:
                    other_category = self.mvt_categories[other_period]
                else:
                    other_category = self.original_mvt_categories[other_period]
                other_categories.append(other_category)
            else:
                break
            i += 1
        i = 1
        while (self.mvt_index_to_display+i) < len(self.onset_mvts):
            other_period = self.original_mvt_categories_onsets[self.onset_mvts[self.mvt_index_to_display+i]]
            if other_period[0] < max_time:
                other_period_times.append(other_period)
                if other_period in self.mvt_categories:
                    other_category = self.mvt_categories[other_period]
                else:
                    other_category = self.original_mvt_categories[other_period]
                other_categories.append(other_category)
            else:
                break
            i += 1

        if first_time:
            self.fig, self.ax = plt.subplots(nrows=1, ncols=1,
                                   gridspec_kw={'height_ratios': [1]},
                                   figsize=(20, 8))
        ax = self.ax
        self.fig.canvas.mpl_connect('key_press_event', self.key_release_action)

        min_piezo_value = np.min(self.piezo[times_to_display])
        max_piezo_value = np.max(self.piezo[times_to_display])

        ax.plot(self.ms.abf_times_in_sec[times_to_display], self.piezo[times_to_display], lw=.5, color="black")
        # plt.plot(self.ms.abf_times_in_sec[:-1], np.abs(first_derivative), lw=.5, zorder=10, color="green")

        if self.ms.lowest_std_in_piezo is not None:
            ax.hlines(self.ms.lowest_std_in_piezo, self.ms.abf_times_in_sec[times_to_display[0]],
                      self.ms.abf_times_in_sec[times_to_display[-1]], color="red", linewidth=1,
                      linestyles="dashed", zorder=1)
            ax.hlines(-self.ms.lowest_std_in_piezo, self.ms.abf_times_in_sec[times_to_display[0]],
                      self.ms.abf_times_in_sec[times_to_display[-1]], color="red", linewidth=1,
                      linestyles="dashed", zorder=1)

        # plt.scatter(x=self.abf_times_in_sec[peaks], y=piezo[peaks], marker="*",
        #             color=["black"], s=5, zorder=15)

        # beh_mvt_index = np.where(self.intermediate_behavourial_events_numbers[times_to_display] >= 0)[0]
        # if len(beh_mvt_index) > 0:
        #     beh_mvt_index += times_to_display[0]
        #     beh_mvt_indices = np.unique(self.intermediate_behavourial_events_numbers[beh_mvt_index])
        #     for index in beh_mvt_indices:
        #         period = self.ms.intermediate_behavourial_events[index]
        #         beg_pos = np.max((times_to_display[0], period[0]))
        #         end_pos = np.min((times_to_display[-1], period[1]))
        #         if end_pos < period_times[0]:
        #             if closet_mvt_before_pos is None:
        #                 closet_mvt_before_pos = end_pos
        #             else:
        #                 closet_mvt_before_pos = np.max((end_pos, closet_mvt_before_pos))
        #         if beg_pos > period_times[1]:
        #             if closet_mvt_after_pos is None:
        #                 closet_mvt_after_pos = beg_pos
        #             else:
        #                 closet_mvt_after_pos = np.min((beg_pos, closet_mvt_after_pos))
        #         ax.axvspan(beg_pos / self.ms.abf_sampling_rate,
        #                    end_pos / self.ms.abf_sampling_rate,
        #                    alpha=0.5, facecolor="green", zorder=1)

        complex_mvt_index = np.where(self.complex_mvt_numbers[times_to_display] >= 0)[0]
        if len(complex_mvt_index) > 0:
            complex_mvt_index += times_to_display[0]
            complex_mvt_indices = np.unique(self.complex_mvt_numbers[complex_mvt_index])
            for index in complex_mvt_indices:
                period = self.ms.complex_mvt[index]
                beg_pos = np.max((times_to_display[0], period[0]))
                end_pos = np.min((times_to_display[-1], period[1]))
                if end_pos < period_times[0]:
                    if closet_mvt_before_pos is None:
                        closet_mvt_before_pos = end_pos
                    else:
                        closet_mvt_before_pos = np.max((end_pos, closet_mvt_before_pos))
                if beg_pos > period_times[1]:
                    if closet_mvt_after_pos is None:
                        closet_mvt_after_pos = beg_pos
                    else:
                        closet_mvt_after_pos = np.min((beg_pos, closet_mvt_after_pos))
                ax.axvspan(beg_pos / self.ms.abf_sampling_rate,
                           end_pos / self.ms.abf_sampling_rate,
                           alpha=0.5, facecolor="red", zorder=1)

        if len(other_period_times) > 0:
            for index, other_period in enumerate(other_period_times):
                beg_pos = np.max((times_to_display[0], other_period[0]))
                end_pos = np.min((times_to_display[-1], other_period[1]))

                if end_pos < period_times[0]:
                    if closet_mvt_before_pos is None:
                        closet_mvt_before_pos = end_pos
                    else:
                        closet_mvt_before_pos = np.max((end_pos, closet_mvt_before_pos))
                if beg_pos > period_times[1]:
                    if closet_mvt_after_pos is None:
                        closet_mvt_after_pos = beg_pos
                    else:
                        closet_mvt_after_pos = np.min((beg_pos, closet_mvt_after_pos))

                other_category = other_categories[index]
                ax.axvspan(beg_pos / self.ms.abf_sampling_rate,
                           end_pos / self.ms.abf_sampling_rate,
                           alpha=0.5, facecolor=self.categories_color[other_category], zorder=1)
                pos = other_period[0] + np.argmax(self.piezo[other_period[0]:other_period[1] + 1])
                ax.scatter(x=self.ms.abf_times_in_sec[pos], y=self.piezo[pos], marker="*",
                           color=self.categories_color[other_category], s=30, zorder=20)

        # main event
        ax.axvspan(period_times[0] / self.ms.abf_sampling_rate, period_times[1] / self.ms.abf_sampling_rate,
                        alpha=0.5, facecolor=self.categories_color[category], zorder=1)

        pos = period_times[0] + np.argmax(self.piezo[period_times[0]:period_times[1] + 1])
        ax.scatter(x=self.ms.abf_times_in_sec[pos], y=self.piezo[pos], marker="*",
                   color=self.categories_color[category], s=30, zorder=20)

        n_ms = int(((period_times[1] - period_times[0]) / self.ms.abf_sampling_rate) * 1000)
        if n_ms >= 300:
            ax.text(x=((period_times[1]+period_times[0])/2) / self.ms.abf_sampling_rate, y=0,
                    s=f"{n_ms} ms", color="dimgrey", zorder=22,
                    ha='center', va="center", fontsize=6, fontweight='bold')

        ax.vlines(((period_times[1]+period_times[0])/2) / self.ms.abf_sampling_rate,
                  min_piezo_value,
                  max_piezo_value, color="black", linewidth=1,
                  linestyles="dashed", zorder=20)

        if closet_mvt_before_pos is not None:
            ax.hlines(max_piezo_value, closet_mvt_before_pos / self.ms.abf_sampling_rate,
                      period_times[0] / self.ms.abf_sampling_rate, color="dimgrey", linewidth=1,
                      linestyles=":", zorder=1)
            x_pos = ((closet_mvt_before_pos + period_times[0]) / 2) / self.ms.abf_sampling_rate
            y_pos = max_piezo_value * 1.02
            n_ms = int(((period_times[0] - closet_mvt_before_pos) / self.ms.abf_sampling_rate) * 1000)
            ax.text(x=x_pos, y=y_pos,
                     s=f"{n_ms} ms", color="dimgrey", zorder=22,
                     ha='center', va="center", fontsize=6, fontweight='bold')
        if closet_mvt_after_pos is not None:
            ax.hlines(max_piezo_value, period_times[1] / self.ms.abf_sampling_rate,
                      closet_mvt_after_pos / self.ms.abf_sampling_rate,
                      color="dimgrey", linewidth=1,
                      linestyles=":", zorder=1)
            x_pos = ((closet_mvt_after_pos + period_times[1]) / 2) / self.ms.abf_sampling_rate
            y_pos = max_piezo_value * 1.02
            n_ms = int(((closet_mvt_after_pos - period_times[1]) / self.ms.abf_sampling_rate) * 1000)
            ax.text(x=x_pos, y=y_pos,
                    s=f"{n_ms} ms", color="dimgrey", zorder=22,
                    ha='center', va="center", fontsize=6, fontweight='bold')

        plt.title(f"piezo {self.ms.description} mvt {self.mvt_index_to_display}/{self.n_mvt_periods}")

        legend_elements = []

        for category in self.categories_name.keys():
            count = self.mvts_count[category]
            legend_elements.append(Patch(facecolor=self.categories_color[category],
                                         edgecolor='black',
                                         label=f'{self.keyboard_code[category]}: {self.categories_name[category]} (x{count})'))

        legend_elements.append(Patch(facecolor="red",
                                     edgecolor='black', label="complex mvt"))
        legend_elements.append(Patch(facecolor="green",
                                     edgecolor='black', label="intermediate behavourial events"))

        # legend_elements.append(Line2D([0], [0], marker="o", color="w", lw=0, label="twitches",
        #                               markerfacecolor='blue', markersize=10))

        ax.legend(handles=legend_elements, loc='lower left')

        if first_time is False:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        if first_time:
            plt.show()

    # def plot_graph(self, first_time=False):
    #     pass

    def change_actual_mvt(self, code_name):
        period_times = self.original_mvt_categories_onsets[self.onset_mvts[self.mvt_index_to_display]]
        self.mvts_count[self.categories_code[code_name]] += 1
        if period_times in self.mvt_categories:
            # means it's been modifier already
            self.mvts_count[self.mvt_categories[period_times]] -= 1
        else:
            self.mvts_count[self.original_mvt_categories[period_times]] -= 1
        self.mvt_categories[period_times] = self.categories_code[code_name]
        # self.original_mvt_categories[period_times] = self.categories_code[code_name]

        self.go_to_next_mvt()

    def save_new_data_and_quit(self):
        pass

    def go_to_next_mvt(self):
        period_times = self.original_mvt_categories_onsets[self.onset_mvts[self.mvt_index_to_display]]
        if period_times not in self.mvt_categories:
            # just keeping the same category
            self.mvt_categories[period_times] = self.original_mvt_categories[period_times]

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
        # print(f"event.key {event.key}")
        if event.key in ["t", "T"]:
            self.change_actual_mvt(code_name="twitches")
        elif event.key in ["m", "M"]:
            self.change_actual_mvt(code_name="short lasting mvt")
        elif event.key in ["n", "N"]:
            self.change_actual_mvt(code_name="noise")
        elif event.key in ["b", "B"]:
            self.change_actual_mvt(code_name="behavourial events")
        if event.key == 'right':
            self.go_to_next_mvt()
        elif event.key == 'left':
            self.go_to_previous_mvt()
