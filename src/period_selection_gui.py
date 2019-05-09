# Ignore warnings
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import sys
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
import hdf5storage
from matplotlib.patches import Patch

import matplotlib
matplotlib.use("TkAgg")
from sys import platform
import time
import pattern_discovery.tools.param as p_disc_tools_param
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.gridspec as gridspec
import os
from pattern_discovery.tools.misc import get_continous_time_periods
import math

if sys.version_info[0] < 3:
    import Tkinter as tk
    from Tkinter import *
else:
    import tkinter as tk
    import tkinter.filedialog as filedialog
    from tkinter import *


# ---------- code for function: event_lambda (begin) --------
def event_lambda(f, *args, **kwds):
    return lambda f=f, args=args, kwds=kwds: f(*args, **kwds)


# TODO: replace it with function from os.path
def get_file_name_and_path(path_file):
    # to get real index, remove 1
    last_slash_index = len(path_file) - path_file[::-1].find("/")
    if last_slash_index == -1:
        return None, None,

    # return path and file_name
    return path_file[:last_slash_index], path_file[last_slash_index:]


class SelectionFrame(tk.Frame):

    def __init__(self, data_and_param, default_path=None):
        # ------ constants for controlling layout ------
        top_buttons_frame_padx = "2m"
        top_buttons_frame_pady = "2m"
        top_buttons_frame_ipadx = "1m"
        top_buttons_frame_ipady = "1m"
        # -------------- end constants ----------------

        cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],
                   [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143],
                   [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952,
                                                          0.779247619], [0.1252714286, 0.3242428571, 0.8302714286],
                   [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238,
                                                                0.8819571429],
                   [0.0059571429, 0.4086142857, 0.8828428571],
                   [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571,
                                                          0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429],
                   [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667,
                                                                0.8467], [0.0779428571, 0.5039857143, 0.8383714286],
                   [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571,
                                                               0.8262714286],
                   [0.0640571429, 0.5569857143, 0.8239571429],
                   [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524,
                                                                0.819852381], [0.0265, 0.6137, 0.8135],
                   [0.0238904762, 0.6286619048,
                    0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667],
                   [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381,
                                                                0.7607190476],
                   [0.0383714286, 0.6742714286, 0.743552381],
                   [0.0589714286, 0.6837571429, 0.7253857143],
                   [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429],
                   [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429,
                                                                0.6424333333],
                   [0.2178285714, 0.7250428571, 0.6192619048],
                   [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619,
                                                                0.5711857143],
                   [0.3481666667, 0.7424333333, 0.5472666667],
                   [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524,
                                                          0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905],
                   [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476,
                                                                0.4493904762],
                   [0.609852381, 0.7473142857, 0.4336857143],
                   [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333],
                   [0.7184095238, 0.7411333333, 0.3904761905],
                   [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667,
                                                          0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762],
                   [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217],
                   [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857,
                                                                0.2886428571],
                   [0.9738952381, 0.7313952381, 0.266647619],
                   [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857,
                                                               0.2164142857], [0.9955333333, 0.7860571429, 0.196652381],
                   [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],
                   [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309],
                   [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333,
                                                          0.0948380952], [0.9661, 0.9514428571, 0.0755333333],
                   [0.9763, 0.9831, 0.0538]]

        self.parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

        self.robin_mac = False

        self.root = Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.validation_before_closing)
        # self.root.title(f"Session {session_number}")
        tk.Frame.__init__(self, master=self.root)
        self.pack(
            ipadx=top_buttons_frame_ipadx,
            ipady=top_buttons_frame_ipady,
            padx=top_buttons_frame_padx,
            pady=top_buttons_frame_pady,
        )
        # self.pack()

        # ------------ colors  -----------------
        self.color_background = "black"
        self.default_color_trace = "white"
        self.default_color_trace_2 = "red"
        self.color_mark_to_remove = "white"
        # ------------- colors (end) --------

        self.trace = data_and_param.trace
        # normalizing it
        self.trace = (self.trace - np.nanmean(self.trace)) / np.nanstd(self.trace)
        self.y_max_value = np.nanmax(self.trace)
        self.y_min_value = np.nanmin(self.trace)
        # in case there is a 2nd trace to display
        self.trace_2 = data_and_param.trace_2
        self.ratio_traces = None
        if self.trace_2 is not None:
            self.trace_2 = (self.trace_2 - np.mean(self.trace_2)) / np.std(self.trace_2)
            min_trace = np.nanmin(self.trace)
            max_trace_2 = np.nanmax(self.trace_2)
            self.trace_2 = self.trace_2 - np.abs(max_trace_2 - min_trace)
            self.y_max_value = np.nanmax((self.y_max_value, max_trace_2))
            self.y_min_value = np.nanmin((self.y_min_value, np.min(self.trace_2)))
            self.ratio_traces = len(self.trace_2) / len(self.trace)
        self.n_times = self.trace.shape[0]
        # contains boolean array
        self.periods = data_and_param.periods
        self.periods_as_tuples = {}
        self.periods_names = data_and_param.periods_names
        if self.periods is None:
            self.periods = {}
            for periods_name in self.periods_names:
                self.periods[periods_name] = np.zeros(self.n_times, dtype="bool")
        self.update_periods()

        # filename on which to save spikenums, is defined when save as is clicked
        self.save_file_name = None
        # path where to save the file_name
        self.save_path = default_path
        self.display_threshold = False
        self.data_and_param = data_and_param

        self.path_result = self.data_and_param.result_path
        self.path_data = self.data_and_param.path_data

        # to know if the actual displayed is saved
        self.is_saved = True
        # used to known if the mouse has been moved between the click and the release
        self.last_click_position = (-1, -1)
        self.first_click_to_remove = None
        self.click_corr_coord = None
        # not used anymore for the UNDO action
        # self.last_action = None
        # list of the last action, used for the undo method, the last one being the last
        self.last_actions = []
        # list of action that has been undone
        self.undone_actions = []
        self.remove_period_mode = False
        # indicated if one mode of period addition is activated
        self.add_period_mode = dict()
        for periods_name in self.periods_names:
            self.add_period_mode[periods_name] = False

        self.periods_color = {}
        index_period = 0
        for period_name in self.periods.keys():
            self.periods_color[period_name] = cm.nipy_spectral(float(index_period + 1) / (len(self.periods) + 1))
            index_period += 1
        # Three horizontal frames to start
        # -------------- top frame (start) ----------------
        top_frame = Frame(self)
        top_frame.pack(side=TOP, expand=YES, fill=BOTH)

        self.add_period_buttons = {}
        for periods_name in self.periods_names:
            period_button = Button(top_frame)
            period_button["text"] = periods_name
            period_button["fg"] = 'red'
            period_button["command"] = event_lambda(self.add_period_switch_mode, periods_name=periods_name)
            period_button.pack(side=LEFT)
            self.add_period_buttons[periods_name] = period_button

            empty_label = Label(top_frame)
            empty_label["text"] = " " * 1
            empty_label.pack(side=LEFT)

        self.remove_period_button = Button(top_frame)
        self.remove_period_button["text"] = ' DEL PERIOD '
        self.remove_period_button["fg"] = 'red'
        self.remove_period_button["command"] = self.remove_period_switch_mode
        self.remove_period_button.pack(side=LEFT)
        # -------------- top frame (end) ----------------

        ################################################################################
        ################################ Middle frame with plot ################################
        ################################################################################
        canvas_frame = Frame(self)
        canvas_frame.pack(side=TOP, expand=YES, fill=BOTH)

        main_plot_frame = Frame(canvas_frame)
        main_plot_frame.pack(side=LEFT, expand=YES, fill=BOTH)
        self.main_plot_frame = main_plot_frame

        # plt.ion()
        if self.robin_mac:
            self.fig = plt.figure(figsize=(10, 4))
        else:
            self.fig = plt.figure(figsize=(12, 6))
        # self.plot_canvas = MyCanvas(self.fig, canvas_frame, self)
        self.plot_canvas = FigureCanvasTkAgg(self.fig, main_plot_frame)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        # self.fig.canvas.mpl_connect('motion_notify_event', self.motion)

        self.gs = gridspec.GridSpec(1, 1, width_ratios=[1], height_ratios=[1])
        # self.gs.update(hspace=0.05)
        # ax1 = plt.subplot(gs[0])
        # ax2 = plt.subplot(gs[1])
        self.axe_plot = None
        # for the second trace
        self.axe_plot_2 = None
        self.line1 = None
        self.line2 = None
        self.plot_graph(first_time=True)

        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=YES)

        self.toolbar = NavigationToolbar2Tk(self.plot_canvas, main_plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side=TOP, fill=BOTH, expand=YES)

        # self.update_plot(new_neuron=True)

        ################################################################################
        ################################ Bottom frame ################################
        ################################################################################
        bottom_frame = Frame(self)
        bottom_frame.pack(side=TOP, expand=YES, fill=BOTH)

        self.save_as_button = Button(bottom_frame)
        self.save_as_button["text"] = ' SAVE AS '
        self.save_as_button["fg"] = "blue"
        self.save_as_button['state'] = "normal"
        self.save_as_button["command"] = event_lambda(self.save_as)
        self.save_as_button.pack(side=RIGHT)

        empty_label = Label(bottom_frame)
        empty_label["text"] = " " * 2
        empty_label.pack(side=RIGHT)

        self.save_button = Button(bottom_frame)
        self.save_button["text"] = ' SAVE '
        self.save_button["fg"] = "blue"
        self.save_button['state'] = DISABLED  # ''normal
        self.save_button["command"] = event_lambda(self.save)
        self.save_button.pack(side=RIGHT)

        empty_label = Label(bottom_frame)
        empty_label["text"] = " " * 2
        empty_label.pack(side=RIGHT)

        self.redo_button = Button(bottom_frame)
        self.redo_button["text"] = ' REDO '
        self.redo_button["fg"] = "blue"
        self.redo_button['state'] = DISABLED  # ''normal
        self.redo_button["command"] = event_lambda(self.redo_action)
        self.redo_button.pack(side=RIGHT)

        empty_label = Label(bottom_frame)
        empty_label["text"] = " " * 2
        empty_label.pack(side=RIGHT)

        self.undo_button = Button(bottom_frame)
        self.undo_button["text"] = ' UNDO '
        self.undo_button["fg"] = "blue"
        self.undo_button['state'] = DISABLED  # ''normal
        self.undo_button["command"] = event_lambda(self.undo_action)
        self.undo_button.pack(side=RIGHT)

        empty_label = Label(bottom_frame)
        empty_label["text"] = " " * 5
        empty_label.pack(side=RIGHT)

        # used for association of keys
        self.keys_pressed = dict()
        self.root.bind_all("<KeyRelease>", self.key_release_action)
        self.root.bind_all("<KeyPress>", self.key_press_action)

    def redo_action(self):
        last_undone_action = self.undone_actions[-1]
        self.undone_actions = self.undone_actions[:-1]
        last_undone_action.redo()
        self.update_last_action(last_undone_action)

        if last_undone_action.is_saved and (not self.is_saved):
            self.save_button['state'] = DISABLED
            self.is_saved = True
        elif self.is_saved:
            self.unsaved()
            # and we put the rest of the last action as not saved as the plot has been saved meanwhile
            for a in self.undone_actions:
                a.is_saved = False

        self.undo_button['state'] = "normal"

        if len(self.undone_actions) == 0:
            self.redo_button['state'] = DISABLED

        self.update_plot(new_x_limit=last_undone_action.x_limits,
                         new_y_limit=last_undone_action.y_limits)

    def undo_action(self):
        """
        Revoke the last action
        :return:
        """

        last_action = self.last_actions[-1]
        self.last_actions = self.last_actions[:-1]
        last_action.undo()

        # if it was saved before last modification and was not saved since, then it is still saved with uptodate version
        if last_action.is_saved and (not self.is_saved):
            self.save_button['state'] = DISABLED
            self.is_saved = True
        elif self.is_saved:
            self.unsaved()
            # and we put the rest of the last action as not saved as the plot has been saved meanwhile
            for a in self.last_actions:
                a.is_saved = False

        self.undone_actions.append(last_action)
        self.redo_button['state'] = "normal"

        if len(self.last_actions) == 0:
            self.undo_button['state'] = DISABLED

        # if different neuron, display the other neuron
        self.update_plot(new_x_limit=last_action.x_limits, new_y_limit=last_action.y_limits)

    def swith_all_click_actions(self, initiator):
        if (initiator != "remove_period_switch_mode") and self.remove_period_mode:
            self.remove_period_switch_mode(from_remove_period_button=False)
        for periods_name, active_mode in self.add_period_mode.items():
            if (initiator != periods_name) and active_mode:
                self.add_period_switch_mode(periods_name=periods_name, from_add_period_button=False)

    def add_period_switch_mode(self, periods_name, from_add_period_button=True):
        if from_add_period_button and (not self.add_period_mode[periods_name]):
            self.swith_all_click_actions(initiator=periods_name)

        self.add_period_mode[periods_name] = not self.add_period_mode[periods_name]

        if self.add_period_mode[periods_name]:
            self.add_period_buttons[periods_name]["fg"] = 'green'
            # in case one click would have been made when remove onset was activated
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
        else:
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
            self.add_period_buttons[periods_name]["fg"] = 'red'

    def remove_period_switch_mode(self, from_remove_period_button=True):
        if from_remove_period_button and (not self.remove_period_mode):
            self.swith_all_click_actions(initiator="remove_period_switch_mode")
        self.remove_period_mode = not self.remove_period_mode

        if self.remove_period_mode:
            self.remove_period_button["fg"] = 'green'
            # in case one click would have been made when remove onset was activated
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
        else:
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
            self.remove_period_button["fg"] = 'red'

    def remove_period(self, x_from, x_to):
        # taking in consideration the case where the click is outside the graph border
        if x_from < 0:
            x_from = 0
        elif x_from > (self.n_times - 1):
            x_from = (self.n_times - 1)

        if x_to < 0:
            x_to = 0
        elif x_to > (self.n_times - 1):
            x_to = (self.n_times - 1)

        if x_from == x_to:
            return

        self.first_click_to_remove = None

        # in case x_from is after x_to
        min_value = min(x_from, x_to)
        max_value = max(x_from, x_to)
        x_from = min_value
        x_to = max_value
        # if the sum is zero, then we're not removing any onset

        # with key being the period, and value a array representing the times
        removed_times = {}
        for period_name, period in self.periods.items():
            active_times = np.where(period[x_from:x_to])[0]
            period[x_from:x_to] = False
            if len(active_times) > 0:
                removed_times[period_name] = active_times + x_from

        if len(removed_times) > 0:
            self.update_periods()
            left_x_limit, right_x_limit = self.axe_plot.get_xlim()
            bottom_limit, top_limit = self.axe_plot.get_ylim()
            self.last_actions.append(RemovePeriodAction(removed_times=removed_times,
                                                        session_frame=self, is_saved=self.is_saved,
                                                        x_limits=(left_x_limit, right_x_limit),
                                                        y_limits=(bottom_limit, top_limit)))
            # no more undone_actions
            self.undone_actions = []
            self.redo_button['state'] = DISABLED
            self.unsaved()
            self.undo_button['state'] = 'normal'
        # update to remove the cross of the first click at least
        self.update_plot()

    def plot_graph(self, first_time=False):
        """

        :param first_time:
        :return:
        """
        if first_time:
            gs_index = 0
            # self.axe_plot = self.fig.add_subplot(self.gs[gs_index])
            if self.trace_2 is not None:
                self.axe_plot_2 = self.fig.add_subplot(111, label="trace_2", frame_on=True)
            self.axe_plot = self.fig.add_subplot(111, label="trace", frame_on=(self.trace_2 is None))
            gs_index += 1

        self.axe_plot.set_facecolor(self.color_background)
        if self.axe_plot_2 is not None:
            self.axe_plot_2.set_facecolor(self.color_background)

        # #################### SMOOTHED TRACE ####################

        color_trace = self.default_color_trace
        color_trace_2 = self.default_color_trace_2
        self.line1, = self.axe_plot.plot(np.arange(self.n_times), self.trace,
                                         color=color_trace, zorder=10)
        # #################### PERIODS FRAMES ####################

        for period_name, periods_tuples in self.periods_as_tuples.items():
            for period_tuple in periods_tuples:
                # if self.axe_plot_2 is not None:
                #     ax_to_use = self.axe_plot_2
                # else:
                ax_to_use = self.axe_plot
                ax_to_use.axvspan(period_tuple[0], period_tuple[1], ymin=0.8, ymax=1,
                                      alpha=0.8, facecolor=self.periods_color[period_name], zorder=1)

        if self.trace_2 is not None:
            self.line2, = self.axe_plot_2.plot(np.arange(len(self.trace_2)), self.trace_2,
                                         color=color_trace_2, zorder=10)

            interval = 200
            self.axe_plot.vlines(np.arange(interval, self.n_times, interval), self.y_min_value,
                                 math.ceil(self.y_max_value),
                                 color="white", linewidth=0.3,
                                 linestyles="dashed", zorder=9)

        # #################### CLICK SCATTER ####################

        if self.first_click_to_remove is not None:
            self.axe_plot.scatter(self.first_click_to_remove["x"], self.first_click_to_remove["y"], marker='x',
                                  c=self.color_mark_to_remove, s=30, zorder=12)
        if self.click_corr_coord:
            self.axe_plot.scatter(self.click_corr_coord["x"], self.click_corr_coord["y"], marker='x',
                                  c="red", s=30, zorder=12)

        legend_elements = []
        # [Line2D([0], [0], color='b', lw=4, label='Line')
        for period_name, color in self.periods_color.items():
            color = self.periods_color[period_name]
            legend_elements.append(Patch(facecolor=color,
                                         edgecolor='black', label=period_name))

        self.axe_plot.legend(handles=legend_elements)
        # by default the y axis zoom is set to fit the wider amplitude of the current neuron
        self.axe_plot.set_ylim(self.y_min_value,
                               math.ceil(self.y_max_value))
        if self.trace_2 is not None:
            self.axe_plot_2.set_ylim(self.y_min_value,
                                   math.ceil(self.y_max_value))
            self.axe_plot_2.set_xticks([])
            self.axe_plot_2.set_yticks([])

        # removing first x_axis
        axes_to_clean = [self.axe_plot, self.axe_plot_2]
        for ax in axes_to_clean:
            if ax is None:
                continue
            # ax.axes.get_xaxis().set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        self.axe_plot.margins(0)
        if self.axe_plot_2 is not None:
            self.axe_plot_2.margins(0)
        self.fig.tight_layout()

    def update_plot(self, amplitude_zoom_fit=True,
                    new_x_limit=None, new_y_limit=None, new_x_limit_2=None):
        # used to keep the same zoom after updating the plot
        # if we change neuron, then back to no zoom mode
        left_x_limit_1, right_x_limit_1 = self.axe_plot.get_xlim()
        bottom_limit_1, top_limit_1 = self.axe_plot.get_ylim()
        self.axe_plot.clear()
        if self.axe_plot_2 is not None:
            left_x_limit_2, right_x_limit_2 = self.axe_plot_2.get_xlim()
            bottom_limit_2, top_limit_2 = self.axe_plot_2.get_ylim()
            self.axe_plot_2.clear()

        self.plot_graph()

        # to keep the same zoom
        self.axe_plot.set_xlim(left=left_x_limit_1, right=right_x_limit_1, auto=None)
        self.axe_plot.set_ylim(bottom=bottom_limit_1, top=top_limit_1, auto=None)
        if self.axe_plot_2 is not None:
            self.axe_plot_2.set_xlim(left=left_x_limit_2, right=right_x_limit_2, auto=None)
            self.axe_plot_2.set_ylim(bottom=bottom_limit_2, top=top_limit_2, auto=None)
        if new_x_limit is not None:
            self.axe_plot.set_xlim(left=new_x_limit[0], right=new_x_limit[1], auto=None)
        if (new_x_limit_2 is not None) and (self.axe_plot_2 is not None):
            self.axe_plot_2.set_xlim(left=new_x_limit_2[0], right=new_x_limit_2[1], auto=None)
        if (new_y_limit is not None) and (not amplitude_zoom_fit):
            self.axe_plot.set_ylim(new_y_limit[0], new_y_limit[1])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def unsaved(self):
        """
        means a changed has been done, and the actual plot is not saved """

        self.is_saved = False
        self.save_button['state'] = 'normal'

    def key_press_action(self, event):
        # print(f"pressed keysym {event.keysym}, keycode {event.keycode}, keysym_num {event.keysym_num}, "
        #       f"state {event.state}")
        if event.keysym in ["Meta_L", "Control_L"]:
            self.keys_pressed[event.keysym] = 1

    def key_release_action(self, event):
        # "Meta_L" keysym for command key
        # "Control_L" keysym for control key
        if event.keysym in self.keys_pressed:
            del self.keys_pressed[event.keysym]
        # print(f"event.keysym_num {event.keysym_num}")
        # if 0, means the key was not recognized, we empty the dict
        if event.keysym_num == 0:
            self.keys_pressed = dict()
        # print(f"released keysym {event.keysym}, keycode {event.keycode}, keysym_num {event.keysym_num}, "
        #       f"state {event.state}")
        # print(f"event.keysym {event.keysym}")

        elif event.char in ["r", "R", "-"]:
            self.remove_period_switch_mode()

        elif event.char in ["S"]:
            self.save()

        if event.keysym == 'Right':
            # C as cell
            self.move_zoom(to_the_left=False)
            # so the back button will come back to the curren view
            self.toolbar.push_current()
        elif event.keysym == 'Left':
            self.move_zoom(to_the_left=True)
            self.toolbar.push_current()

    def update_periods(self):
        for period_name, period in self.periods.items():
            if np.sum(period) == 0:
                self.periods_as_tuples[period_name] = []
            else:
                self.periods_as_tuples[period_name] = get_continous_time_periods(period.astype("int8"))

    def save(self, and_close=False):
        if self.save_file_name is None:
            self.save_as()
            # check if a file has been selected
            if self.save_file_name is None:
                return

        self.save_button['state'] = DISABLED
        self.is_saved = True

        np.savez(os.path.join(self.save_path, self.save_file_name), **self.periods)

        if and_close:
            self.root.destroy()

    def save_as(self):
        initialdir = "/"
        if self.save_path is not None:
            initialdir = self.save_path
        path_and_file_name = filedialog.asksaveasfilename(initialdir=initialdir,
                                                          title="Select file",
                                                          filetypes=[("Numpy files", "*.npz")])
        if path_and_file_name == "":
            return

        # shouldn't happen, but just in case
        if len(path_and_file_name) <= 4:
            return

        # the opener should add the ".mat" extension anyway
        if path_and_file_name[-4:] != ".npz":
            path_and_file_name += ".npz"

        # to get real index, remove 1
        self.save_path, self.save_file_name = get_file_name_and_path(path_and_file_name)

        if self.save_path is None:
            return

        self.save()

    def onclick(self, event):
        """
        Action when a mouse button is pressed
        :param event:
        :return:
        """
        # if event.inaxes is not None:
        #     ax = event.inaxes
        if event.xdata is not None:
            self.last_click_position = (event.xdata, event.ydata)

    def onrelease(self, event):
        """
                Action when a mouse button is released
                :param event:
                :return:
        """
        active_period_name = None
        if not self.remove_period_mode:
            for period_name, active_period in self.add_period_mode.items():
                if active_period:
                    active_period_name = period_name
                    break
            if active_period_name is None:
                return

        if event.dblclick:
            return

        if event.xdata is None:
            return

        if self.last_click_position[0] != event.xdata:
            # the mouse has been moved between the pressing and the release
            return

        if self.first_click_to_remove is not None:
            if self.remove_period_mode:
                self.remove_period(x_from=self.first_click_to_remove["x"], x_to=int(round(event.xdata)))
            elif active_period_name is not None:
                self.add_period(period_name=active_period_name,
                                x_from=self.first_click_to_remove["x"], x_to=int(round(event.xdata)))
        else:
            self.first_click_to_remove = {"x": int(round(event.xdata)), "y": event.ydata}
            self.update_plot()

    def add_period(self, period_name, x_from, x_to):
        # taking in consideration the case where the click is outside the graph border
        if x_from < 0:
            x_from = 0
        elif x_from > (self.n_times - 1):
            x_from = (self.n_times - 1)

        if x_to < 0:
            x_to = 0
        elif x_to > (self.n_times - 1):
            x_to = (self.n_times - 1)

        if x_from == x_to:
            return

        self.first_click_to_remove = None

        # in case x_from is after x_to
        min_value = min(x_from, x_to)
        max_value = max(x_from, x_to)
        x_from = min_value
        x_to = max_value

        if np.all(self.periods[period_name][x_from:x_to]):
            # to remove the cross
            self.update_plot()
            return

        backup_values = np.copy(self.periods[period_name][x_from:x_to])
        self.periods[period_name][x_from:x_to] = True
        self.update_periods()

        left_x_limit, right_x_limit = self.axe_plot.get_xlim()
        bottom_limit, top_limit = self.axe_plot.get_ylim()

        self.update_last_action(AddPeriodAction(period_name=period_name, x_from=x_from, x_to=x_to, session_frame=self,
                                                backup_values=backup_values,
                                                is_saved=self.is_saved,
                                                x_limits=(left_x_limit, right_x_limit),
                                                y_limits=(bottom_limit, top_limit)))
        # no more undone_actions
        self.undone_actions = []
        self.redo_button['state'] = DISABLED

        self.unsaved()
        self.undo_button['state'] = 'normal'

        self.update_plot()

    def update_last_action(self, new_action):
        """
        Keep the size of the last_actions up to five actions
        :param new_action:
        :return:
        """
        self.last_actions.append(new_action)
        if len(self.last_actions) > 5:
            self.last_actions = self.last_actions[1:]

    def validation_before_closing(self):
        if not self.is_saved:
            self.save_as()
        self.root.destroy()

    def move_zoom(self, to_the_left):
        # the plot is zoom out to the max, so no moving
        left_x_limit_1, right_x_limit_1 = self.axe_plot.get_xlim()
        if self.axe_plot_2 is not None:
            left_x_limit_2, right_x_limit_2 = self.axe_plot_2.get_xlim()
        # print(f"left_x_limit_1 {left_x_limit_1}, right_x_limit_1 {right_x_limit_1}, "
        #       f"self.nb_times_traces {self.nb_times_traces}")
        if (left_x_limit_1 <= 0) and (right_x_limit_1 >= (self.n_times - 1)):
            return

        # moving the windown to the right direction keeping 10% of the window in the new one
        length_window = right_x_limit_1 - left_x_limit_1
        if self.axe_plot_2 is not None:
            length_window_2 = right_x_limit_2 - left_x_limit_2
        if to_the_left:
            if left_x_limit_1 <= 0:
                return
            new_right_x_limit = int(left_x_limit_1 + (0.1 * length_window))
            if self.axe_plot_2 is not None:
                # new_right_x_limit_2 = int(left_x_limit_2 + (0.1 * length_window_2))
                new_right_x_limit_2 = int(new_right_x_limit * self.ratio_traces)
            new_left_x_limit = new_right_x_limit - length_window
            new_left_x_limit = max(new_left_x_limit, 0)
            if self.axe_plot_2 is not None:
                new_left_x_limit_2 = int(new_left_x_limit * self.ratio_traces)
                # new_left_x_limit_2 = new_right_x_limit_2 - length_window_2
                # new_left_x_limit_2 = max(new_left_x_limit_2, 0)
            if new_right_x_limit <= new_left_x_limit:
                return
        else:
            if right_x_limit_1 >= (self.n_times - 1):
                return
            new_left_x_limit = int(right_x_limit_1 - (0.1 * length_window))
            if self.axe_plot_2 is not None:
                new_left_x_limit_2 = int(new_left_x_limit * self.ratio_traces)
                # new_left_x_limit_2 = int(right_x_limit_2 - (0.1 * length_window_2))
            new_right_x_limit = new_left_x_limit + length_window
            new_right_x_limit = min(new_right_x_limit, self.n_times - 1)
            if self.axe_plot_2 is not None:
                new_right_x_limit_2 = int(new_right_x_limit * self.ratio_traces)
                # new_right_x_limit_2 = new_left_x_limit_2 + length_window_2
                # new_right_x_limit_2 = min(new_right_x_limit_2, self.n_times - 1)
            if new_left_x_limit >= new_right_x_limit:
                return

        new_x_limit = (new_left_x_limit, new_right_x_limit)
        if self.axe_plot_2 is None:
            new_x_limit_2 = None
        else:
            new_x_limit_2 = (new_left_x_limit_2, new_right_x_limit_2)
        self.update_plot(new_x_limit=new_x_limit, new_x_limit_2=new_x_limit_2)


class DataAndParam(p_disc_tools_param.Parameters):
    def __init__(self, path_data, result_path):
        self.time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        super().__init__(path_results=result_path, time_str=self.time_str, bin_size=1)
        self.result_path = result_path
        self.path_data = path_data
        self.trace = None
        self.trace_2 = None
        self.periods = None
        self.periods_names = None


class ManualAction:
    def __init__(self, session_frame, is_saved, x_limits, y_limits):
        self.session_frame = session_frame
        self.is_saved = is_saved
        # tuple representing the limit of the plot when the action was done, used to get the same zoom
        # when undo or redo
        self.x_limits = x_limits
        self.y_limits = y_limits

    def undo(self):
        pass

    def redo(self):
        pass


class AddPeriodAction(ManualAction):
    def __init__(self, period_name, x_from, x_to, backup_values, **kwargs):
        super().__init__(**kwargs)
        self.period_name = period_name
        self.x_from = x_from
        self.x_to = x_to
        self.backup_values = backup_values

    def undo(self):
        super().undo()
        self.session_frame.periods[self.period_name][self.x_from:self.x_to] = self.backup_values
        self.session_frame.update_periods()

    def redo(self):
        super().redo()
        self.session_frame.periods[self.period_name][self.x_from:self.x_to] = True
        self.session_frame.update_periods()


class RemovePeriodAction(ManualAction):
    def __init__(self, removed_times, **kwargs):
        super().__init__(**kwargs)
        self.removed_times = removed_times

    def undo(self):
        super().undo()
        for period_name, removed_times in self.removed_times.items():
            self.session_frame.periods[period_name][removed_times] = True
        self.session_frame.update_periods()

    def redo(self):
        super().redo()
        for period_name, removed_times in self.removed_times.items():
            self.session_frame.periods[period_name][removed_times] = False
        self.session_frame.update_periods()


class MySessionButton(Button):

    def __init__(self, master):
        # ------ constants for controlling layout ------
        button_width = 10
        button_height = 10

        button_padx = "10m"  ### (2)
        button_pady = "1m"  ### (2)
        # -------------- end constants ----------------

        Button.__init__(self, master)
        self.configure(
            width=button_width,  ### (1)
            height=button_height,
            padx=button_padx,  ### (2)
            pady=button_pady  ### (2)
        )


class OptionsFrame(tk.Frame):

    def __init__(self, data_and_param, master=None):
        # ------ constants for controlling layout ------
        buttons_frame_padx = "3m"
        buttons_frame_pady = "2m"
        buttons_frame_ipadx = "3m"
        buttons_frame_ipady = "1m"

        # -------------- end constants ----------------
        tk.Frame.__init__(self, master)
        self.pack(
            ipadx=buttons_frame_ipadx,
            ipady=buttons_frame_ipady,
            padx=buttons_frame_padx,
            pady=buttons_frame_pady,
        )

        # self.pack()
        self.data_and_param = data_and_param
        # dict, filled if a periods selection is loaded.
        # the key is a string representing the period id, and the value is a 1-D array of bool, True if the period is
        # active for a given timestamps
        self.loaded_periods = None
        # names of periods, will be use if self.loaded_periods
        # if self.periods_names is None, then the option_menu selection is used and will name periods as period_{n}
        self.periods_names = None
        self.trace = None
        self.trace_2 = None

        # maximum number of type of periods
        self.n_max_periods_type = 5

        self.option_menu_variable = None
        # to avoid garbage collector
        self.file_selection_buttons = dict()
        self.go_button = None
        self.last_path_open = None
        self.create_buttons()

    def go_go(self):
        self.go_button['state'] = DISABLED

        self.data_and_param.trace = self.trace
        self.data_and_param.trace_2 = self.trace_2

        self.data_and_param.periods = self.loaded_periods

        if self.loaded_periods is None:
            if self.periods_names is not None:
                self.data_and_param.periods_names = self.periods_names
            else:
                self.data_and_param.periods_names = [f"period_{n}"
                                                     for n in np.arange(1, int(self.option_menu_variable.get()) + 1)]
        else:
            self.data_and_param.periods_names = list(self.data_and_param.periods.keys())

        # print(f"trace {self.data_and_param.trace.shape}")
        # print(f"periods_names {self.data_and_param.periods_names}")

        f = SelectionFrame(data_and_param=self.data_and_param,
                           default_path=self.last_path_open)
        f.mainloop()

    # open file selection, with memory of the last folder + use last folder as path to save new data with timestr
    def open_new_onset_selection_frame(self, data_to_load_str, open_with_file_selector):
        if open_with_file_selector:
            initial_dir = self.data_and_param.path_data
            if self.last_path_open is not None:
                initial_dir = self.last_path_open

            if data_to_load_str == "params":
                file_types = []  # ("Text files", "*.txt")
            elif data_to_load_str == "periods":
                file_types = (("Numpy files", "*.npy"), ("Numpy files", "*.npz"), ("Matlab files", "*.mat"))
            elif data_to_load_str == "trace 1":
                file_types = (("Numpy files", "*.npy"), ("Matlab files", "*.mat"))
            elif data_to_load_str == "trace 2":
                file_types = (("Numpy files", "*.npy"), ("Matlab files", "*.mat"))
            else:
                print(f"Unknown data_to_load_str {data_to_load_str}")
                return
            file_name = filedialog.askopenfilename(
                initialdir=initial_dir,
                filetypes=file_types,
                title=f"Choose a file to load {data_to_load_str}")
            if file_name == "":
                return
            self.last_path_open, file_name_only = get_file_name_and_path(file_name)

            if data_to_load_str == "params":
                self.periods_names = []
                with open(file_name, "r", encoding='UTF-8') as file:
                    for nb_line, line in enumerate(file):
                        if line[-1] == '\n':
                            line = line[:-1]
                        self.periods_names.append(line)
            elif data_to_load_str == "periods":
                if file_name[-3:] == "mat":
                    data_file = hdf5storage.loadmat(file_name)
                    for key, value in data_file.items():
                        # TODO: need to figure out if other array are saved than the one we want to load
                        print(f"key {key}, value.shape {value.shape}")
                elif file_name[-3:] == "npy":
                    period_array = np.load(file_name)
                    self.loaded_periods = dict()
                    self.loaded_periods["period_1"] = period_array
                elif file_name[-3:] == "npz":
                    npz_file = np.load(file_name)
                    self.loaded_periods = dict()
                    for item, value in npz_file.items():
                        self.loaded_periods[item] = value
            elif data_to_load_str == "trace 1":
                # print(f"file_name[-3:] {file_name[-3:]}")
                if file_name[-3:] == "mat":
                    data_file = hdf5storage.loadmat(file_name)
                    for key, value in data_file.items():
                        # TODO: need to figure out if other array are saved than the one we want to load
                        print(f"key {key}, value.shape {value.shape}")
                elif file_name[-3:] == "npy":
                    self.trace = np.load(file_name)
                self.go_button['state'] = "normal"
            elif data_to_load_str == "trace 2":
                # print(f"file_name[-3:] {file_name[-3:]}")
                if file_name[-3:] == "mat":
                    data_file = hdf5storage.loadmat(file_name)
                    for key, value in data_file.items():
                        # TODO: need to figure out if other array are saved than the one we want to load
                        print(f"key {key}, value.shape {value.shape}")
                elif file_name[-3:] == "npy":
                    self.trace_2 = np.load(file_name)
                self.go_button['state'] = "normal"

            self.file_selection_buttons[data_to_load_str]["fg"] = "grey"

    def create_buttons(self):
        colors = ["blue", "orange", "green", "pink", "brown", "yellow"]
        # for c in colors:
        #     ttk.Style().configure(f'black/{c}.TButton', foreground='black', background=f'{c}')
        data_to_load = ["params", "trace 1", "trace 2", "periods"]

        # create new frames
        menu_frame = Frame(self)
        menu_frame.pack(side=TOP,
                        expand=YES,
                        fill=BOTH)

        self.option_menu_variable = StringVar(self)
        self.option_menu_variable.set("1")  # default value
        w = OptionMenu(self, self.option_menu_variable, *[str(i) for i in (np.arange(self.n_max_periods_type) + 1)])
        w.pack()

        button_frame = Frame(self)
        button_frame.pack(side=TOP,
                          expand=YES,
                          fill=BOTH)  #
        # height=50,

        for i, data_to_load_str in enumerate(data_to_load):
            # create a frame for each mouse, so the button will on the same line
            button = MySessionButton(button_frame)
            button["text"] = f'{data_to_load_str}'
            button["fg"] = colors[i % len(colors)]
            # c = colors[i % len(colors)]
            # button["style"] = f'black/{c}.TButton'
            button.pack(side=LEFT)
            self.file_selection_buttons[data_to_load_str] = button
            # if session_number > 45:
            #     open_with_file_selector = True
            # else:
            #     open_with_file_selector = False
            open_with_file_selector = True
            button["command"] = event_lambda(self.open_new_onset_selection_frame, data_to_load_str=data_to_load_str,
                                             open_with_file_selector=open_with_file_selector)

        self.go_button = MySessionButton(button_frame)
        self.go_button["text"] = f'GO'
        self.go_button["fg"] = "red"
        self.go_button['state'] = DISABLED
        # self.go_button['state'] = "normal"
        # button["style"] = f'black/{c}.TButton'
        self.go_button.pack(side=LEFT)
        self.go_button["command"] = event_lambda(self.go_go)


def main():
    root_path = None
    with open("param_hne.txt", "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split('=')
            root_path = line_list[1]
    if root_path is None:
        raise Exception("Root path is None")

    path_data = root_path + "data/"

    print(f'platform: {platform}')
    root = Tk()
    root.title(f"Options")

    result_path = root_path + "results_hne"

    data_and_param = DataAndParam(path_data=path_data, result_path=result_path)

    app = OptionsFrame(data_and_param=data_and_param,
                       master=root)
    app.mainloop()
    # root.destroy()


main()
