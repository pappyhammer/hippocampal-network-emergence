import numpy as np
import sys
from datetime import datetime
import hdf5storage
import matplotlib
from sys import platform

matplotlib.use("TkAgg")
import scipy.io as sio
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# from PIL import ImageTk, Image
import math
import matplotlib.gridspec as gridspec
# import matplotlib.image as mpimg
from mouse_session_loader import load_mouse_sessions
import pattern_discovery.tools.param as p_disc_tools_param
from matplotlib import patches
import scipy.signal as signal
import matplotlib.image as mpimg
from random import randint
import scipy.ndimage.morphology as morphology
import os
import cv2
from PIL import Image, ImageSequence
import PIL
# from PIL import ImageTk
from itertools import cycle
from matplotlib import animation

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


# ---------- code for function: event_lambda (end) -----------


class ErrorMessageFrame(tk.Frame):

    def __init__(self, error_message):
        self.root = Tk()
        self.root.title(f"Error")
        tk.Frame.__init__(self, master=self.root)
        self.pack()

        new_frame = Frame(self)
        new_frame.pack(side=TOP,
                       expand=YES,
                       fill=BOTH)

        msg_label = Label(new_frame)
        msg_label["text"] = f" {error_message} "
        msg_label.pack(side=LEFT)

        new_frame_ = Frame(self)
        new_frame_.pack(side=TOP,
                        expand=YES,
                        fill=BOTH)

        empty_label = Label(new_frame_)
        empty_label["text"] = " " * 2
        empty_label.pack(side=TOP)

        ok_button = Button(new_frame_)
        ok_button["text"] = ' OK '
        ok_button["fg"] = "red"
        ok_button["command"] = event_lambda(self.root.destroy)
        ok_button.pack(side=TOP)

        empty_label = Label(new_frame_)
        empty_label["text"] = " " * 2
        empty_label.pack(side=TOP)


class DataAndParam(p_disc_tools_param.Parameters):
    def __init__(self, result_path):
        self.time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        super().__init__(path_results=result_path, time_str=self.time_str, bin_size=1)
        self.cell_assemblies_data_path = None
        self.best_order_data_path = None
        self.spike_nums = None
        self.peak_nums = None
        self.traces = None
        self.raw_traces = None
        self.ms = None
        self.result_path = result_path
        self.path_data = result_path
        self.inter_neurons = None
        self.cells_to_remove = None
        self.cells_map_img = None


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


class ChooseSessionFrame(tk.Frame):

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

        self.available_ms_str = ["p6_18_02_07_a001_ms", "p6_18_02_07_a002_ms",
                                 "p7_171012_a000_ms", "p7_18_02_08_a000_ms",
                                 "p7_17_10_18_a002_ms", "p7_17_10_18_a004_ms",
                                 "p7_18_02_08_a001_ms", "p7_18_02_08_a002_ms",
                                 "p7_18_02_08_a003_ms",
                                 "p8_18_02_09_a000_ms", "p8_18_02_09_a001_ms",
                                 "p8_18_10_24_a005_ms", "p8_18_10_17_a001_ms",
                                 "p8_18_10_17_a000_ms",  # new
                                 "p9_17_12_06_a001_ms", "p9_17_12_20_a001_ms",
                                 "p9_18_09_27_a003_ms",  # new
                                 "p10_17_11_16_a003_ms",
                                 "p11_17_11_24_a001_ms", "p11_17_11_24_a000_ms",
                                 "p12_17_11_10_a002_ms", "p12_171110_a000_ms",
                                 "p13_18_10_29_a000_ms",  # new
                                 "p13_18_10_29_a001_ms",
                                 "p14_18_10_23_a000_ms",
                                 "p14_18_10_30_a001_ms",
                                 "p60_arnaud_ms", "p60_a529_2015_02_25_ms"]

        self.option_menu_variable = None
        # to avoid garbage collector
        self.file_selection_buttons = dict()
        self.go_button = None
        self.last_path_open = None
        self.create_buttons()

    def go_go(self):
        self.go_button['state'] = DISABLED
        ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=[self.option_menu_variable.get()],
                                                param=self.data_and_param,
                                                load_traces=True, load_abf=True)
        self.data_and_param.ms = ms_str_to_ms_dict[self.option_menu_variable.get()]
        self.data_and_param.traces = self.data_and_param.ms.traces
        self.data_and_param.raw_traces = self.data_and_param.ms.raw_traces

        n_cells = len(self.data_and_param.traces)
        n_times = len(self.data_and_param.traces[0, :])
        if (self.data_and_param.peak_nums is None) or (self.data_and_param.spike_nums is None):
            # then we do an automatic detection
            self.data_and_param.peak_nums = np.zeros((n_cells, n_times), dtype="uint8")
            for cell in np.arange(n_cells):
                peaks, properties = signal.find_peaks(x=self.data_and_param.traces[cell], distance=2)
                # print(f"peaks {peaks}")
                self.data_and_param.peak_nums[cell, peaks] = 1
            self.data_and_param.spike_nums = np.zeros((n_cells, n_times), dtype="uint8")
            for cell in np.arange(n_cells):
                # first_derivative = np.diff(self.data_and_param.traces[cell]) / np.diff(np.arange(n_times))
                # onsets = np.where(np.abs(first_derivative) < 0.1)[0]
                # print(f"np.min(first_derivative) {np.min(first_derivative)}")
                # fig = plt.figure()
                # plt.plot(first_derivative)
                # plt.show()
                onsets = []
                diff_values = np.diff(self.data_and_param.traces[cell])
                for index, value in enumerate(diff_values):
                    if index == (len(diff_values) - 1):
                        continue
                    if value < 0:
                        if diff_values[index + 1] >= 0:
                            onsets.append(index + 1)
                # print(f"onsets {len(onsets)}")
                self.data_and_param.spike_nums[cell, np.array(onsets)] = 1
        f = ManualOnsetFrame(data_and_param=self.data_and_param,
                             default_path=self.last_path_open)
        f.mainloop()

    # open file selection, with memory of the last folder + use last folder as path to save new data with timestr
    def open_new_onset_selection_frame(self, data_to_load_str, data_and_param, open_with_file_selector):
        if open_with_file_selector:
            initial_dir = self.data_and_param.result_path
            if self.last_path_open is not None:
                initial_dir = self.last_path_open

            # if data_to_load_str == "cells img":
            #     file_name = filedialog.askopenfilename(
            #         initialdir=initial_dir,
            #         filetypes=(("Tiff files", "*.tif"), ("Tiff files", "*.tiff")),
            #         title=f"Choose a file to load {data_to_load_str}")
            #     if file_name == "":
            #         return
            #
            #     self.last_path_open, file_name_only = get_file_name_and_path(file_name)
            #
            #     data_and_param.cells_map_img = mpimg.imread(file_name)
            # else:
            file_name = filedialog.askopenfilename(
                initialdir=initial_dir,
                filetypes=(("Matlab files", "*.mat"), ("Numpy files", "*.npy")),
                title=f"Choose a file to load {data_to_load_str}")
            if file_name == "":
                return
            self.last_path_open, file_name_only = get_file_name_and_path(file_name)

            data_file = hdf5storage.loadmat(file_name)

            if data_to_load_str == "onsets & peaks":
                if "LocPeakMatrix" in data_file:
                    peak_nums_matlab = data_file['LocPeakMatrix'].astype(int)
                    peak_nums = np.zeros((len(peak_nums_matlab), len(peak_nums_matlab[0, :])), dtype="int8")
                    for i in np.arange(len(peak_nums_matlab)):
                        peak_nums[i, np.where(peak_nums_matlab[i, :])[0] - 1] = 1
                    data_and_param.peak_nums = peak_nums
                elif "LocPeakMatrix_Python" in data_file:
                    data_and_param.peak_nums = data_file['LocPeakMatrix_Python'].astype(int)
                else:
                    e = ErrorMessageFrame(error_message="LocPeakMatrix not found")
                    e.mainloop()
                    return
                if "Bin100ms_spikedigital" in data_file:
                    spike_nums_matlab = data_file['Bin100ms_spikedigital'].astype(int)
                    spike_nums = np.zeros((len(spike_nums_matlab), len(spike_nums_matlab[0, :])), dtype="int8")
                    for i in np.arange(len(spike_nums_matlab)):
                        spike_nums[i, np.where(spike_nums_matlab[i, :])[0] - 1] = 1
                    data_and_param.spike_nums = spike_nums
                    # data_and_param.spike_nums = data_file['Bin100ms_spikedigital'].astype(int)
                elif "Bin100ms_spikedigital_Python" in data_file:
                    data_and_param.spike_nums = data_file['Bin100ms_spikedigital_Python'].astype(int)
                else:
                    e = ErrorMessageFrame(error_message="Bin100ms_spikedigital not found")
                    e.mainloop()
                    return
                if "inter_neurons" in data_file:
                    data_and_param.inter_neurons = data_file['inter_neurons'].astype(int)
                if "cells_to_remove" in data_file:
                    data_and_param.cells_to_remove = data_file['cells_to_remove'].astype(int)
                # elif data_to_load_str == "C_df":
                #     if "C_df" in data_file:
                #         data_and_param.traces = data_file['C_df'].astype(float)
                #     else:
                #         e = ErrorMessageFrame(error_message="C_df not found")
                #         e.mainloop()
                #         return
                # elif data_to_load_str == "raw traces":
                #     if "raw_traces" in data_file:
                #         data_and_param.raw_traces = data_file['raw_traces'].astype(float)
                #     else:
                #         e = ErrorMessageFrame(error_message="raw_traces not found")
                #         e.mainloop()
                #         return

            self.file_selection_buttons[data_to_load_str]["fg"] = "grey"

            # if data_and_param.peak_nums is None:
            #     return
            # if data_and_param.spike_nums is None:
            #     return
            # if data_and_param.traces is None:
            #     return
            # if data_and_param.raw_traces is None:
            #     return
            # if data_and_param.cells_map_img is None:
            #     return

            # self.go_button['state'] = "normal"

    def create_buttons(self):
        colors = ["blue", "orange", "green"]
        # for c in colors:
        #     ttk.Style().configure(f'black/{c}.TButton', foreground='black', background=f'{c}')
        data_to_load = ["onsets & peaks"]

        # create new frames
        menu_frame = Frame(self)
        menu_frame.pack(side=TOP,
                        expand=YES,
                        fill=BOTH)

        self.option_menu_variable = StringVar(self)
        self.option_menu_variable.set(self.available_ms_str[0])  # default value
        w = OptionMenu(self, self.option_menu_variable, *self.available_ms_str)
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
            button["fg"] = colors[i]
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
                                             data_and_param=self.data_and_param,
                                             open_with_file_selector=open_with_file_selector)

        self.go_button = MySessionButton(button_frame)
        self.go_button["text"] = f'GO'
        self.go_button["fg"] = "red"
        # self.go_button['state'] = DISABLED
        self.go_button['state'] = "normal"
        # button["style"] = f'black/{c}.TButton'
        self.go_button.pack(side=LEFT)
        self.go_button["command"] = event_lambda(self.go_go)


class ManualAction:
    def __init__(self, session_frame, neuron, is_saved, x_limits, y_limits):
        self.session_frame = session_frame
        self.neuron = neuron
        self.is_saved = is_saved
        # tuple representing the limit of the plot when the action was done, used to get the same zoom
        # when undo or redo
        self.x_limits = x_limits
        self.y_limits = y_limits

    def undo(self):
        pass

    def redo(self):
        pass


class RemoveOnsetAction(ManualAction):
    def __init__(self, removed_times, **kwargs):
        super().__init__(**kwargs)
        self.removed_times = removed_times

    def undo(self):
        super().undo()
        self.session_frame.onset_times[self.neuron, self.removed_times] = 1
        self.session_frame.spike_nums[self.neuron, self.removed_times] = 1

    def redo(self):
        super().redo()

        self.session_frame.onset_times[self.neuron, self.removed_times] = 0
        self.session_frame.spike_nums[self.neuron, self.removed_times] = 0


class RemovePeakAction(ManualAction):
    def __init__(self, removed_times, amplitudes, removed_onset_action=None, **kwargs):
        super().__init__(**kwargs)
        self.removed_times = removed_times
        self.amplitudes = amplitudes
        self.removed_onset_action = removed_onset_action

    def undo(self):
        super().undo()
        self.session_frame.peak_nums[self.neuron, self.removed_times] = self.amplitudes
        if self.removed_onset_action is not None:
            self.removed_onset_action.undo()

    def redo(self):
        super().redo()
        self.session_frame.peak_nums[self.neuron, self.removed_times] = 0
        if self.removed_onset_action is not None:
            self.removed_onset_action.redo()


class AddOnsetAction(ManualAction):
    def __init__(self, added_time, add_peak_action=None, **kwargs):
        """

        :param added_time:
        :param add_peak_action: if not None, means a add peak action has been associated to an add onset
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.added_time = added_time
        self.add_peak_action = add_peak_action

    def undo(self):
        super().undo()
        self.session_frame.onset_times[self.neuron, self.added_time] = 0
        self.session_frame.spike_nums[self.neuron, self.added_time] = 0
        if self.add_peak_action is not None:
            self.add_peak_action.undo()

    def redo(self):
        super().redo()
        self.session_frame.onset_times[self.neuron, self.added_time] = 1
        self.session_frame.spike_nums[self.neuron, self.added_time] = 1
        if self.add_peak_action is not None:
            self.add_peak_action.redo()


class AddPeakAction(ManualAction):
    def __init__(self, added_time, amplitude, **kwargs):
        super().__init__(**kwargs)
        self.added_time = added_time
        self.amplitude = amplitude

    def undo(self):
        super().undo()
        self.session_frame.peak_nums[self.neuron, self.added_time] = 0

    def redo(self):
        super().redo()
        self.session_frame.peak_nums[self.neuron, self.added_time] = 1


def get_file_name_and_path(path_file):
    # to get real index, remove 1
    last_slash_index = len(path_file) - path_file[::-1].find("/")
    if last_slash_index == -1:
        return None, None,

    # return path and file_name
    return path_file[:last_slash_index], path_file[last_slash_index:]


class MyCanvas(FigureCanvasTkAgg):
    def __init__(self, figure, parent_frame, manual_onset_frame):
        FigureCanvasTkAgg.__init__(self, figure, parent_frame)

    def button_press_event(self, event, **args):
        print(f"{event.__dict__}")
        # Event attributes
        # {'serial': 517, 'num': 1, 'height': '??', 'keycode': '??', 'state': 0, 'time': 1091938502, 'width': '??',
        # 'x': 399, 'y': 135, 'char': '??', 'send_event': False, 'keysym': '??', 'keysym_num': '??',
        # 'type': <EventType.ButtonPress: '4'>, 'widget': <tkinter.Canvas object .!manualonsetframe.!frame2.!canvas>,
        # 'x_root': 455, 'y_root': 267, 'delta': 0}
        # num is the button number
        if 'dblclick' not in args:
            dblclick = False
        else:
            dblclick = True
        print(f"event['x']: {event.x}, event['y']: {event.x}, dblclick: {dblclick}")
        super().button_press_event(event, **args)


class ManualOnsetFrame(tk.Frame):

    def __init__(self, data_and_param, default_path=None):
        # ------ constants for controlling layout ------
        top_buttons_frame_padx = "2m"
        top_buttons_frame_pady = "2m"
        top_buttons_frame_ipadx = "1m"
        top_buttons_frame_ipady = "1m"
        # -------------- end constants ----------------

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
        self.color_onset = "dimgrey"
        self.color_trace = "darkblue"
        self.color_early_born = "darkgreen"
        self.color_peak = "yellow"
        self.color_edge_peak = "black"
        self.color_peak_under_threshold = "red"
        self.color_threshold_line = "red"  # "cornflowerblue"
        self.color_mark_to_remove = "black"
        self.color_run_period = "lightcoral"
        self.color_trace_activity = "black"
        # self.color_raw_trace = "darkgoldenrod"
        self.color_raw_trace = "cornflowerblue"
        # ------------- colors (end) --------

        # filename on which to save spikenums, is defined when save as is clicked
        self.save_file_name = None
        # path where to save the file_name
        self.save_path = default_path
        self.display_threshold = False
        # in number of frames, one frame = 100 ms
        self.decay = 10
        # factor to multiply to decay to define delay between onset and peak
        self.decay_factor = 1
        # number of std of trace to add to the threshold
        self.nb_std_thresold = 0.1
        self.data_and_param = data_and_param
        self.path_result = self.data_and_param.result_path
        self.spike_nums = data_and_param.spike_nums
        self.nb_neurons = len(self.spike_nums)
        self.nb_times = len(self.spike_nums[0, :])
        self.traces = data_and_param.traces
        self.nb_times_traces = len(self.traces[0, :])
        # dimension reduction in order to fit to traces times, for onset times
        self.onset_times = np.zeros((self.nb_neurons, self.nb_times), dtype="int8")
        self.onset_numbers_label = None
        self.update_onset_times()
        self.peak_nums = self.data_and_param.peak_nums
        # print(f"len(peak_nums) {len(peak_nums)}")
        self.raw_traces = self.data_and_param.raw_traces
        self.display_mvt = False
        self.mvt_frames_periods = None
        if self.data_and_param.ms.mvt_frames_periods is not None:
            # mouse is running
            self.mvt_frames_periods = self.data_and_param.ms.mvt_frames_periods
        elif self.data_and_param.ms.complex_mvt_frames_periods is not None:
            # print(f"self.data_and_param.ms.complex_mvt_frames_periods")
            self.mvt_frames_periods = []
            self.mvt_frames_periods.extend(self.data_and_param.ms.complex_mvt_frames_periods)
            self.mvt_frames_periods.extend(self.data_and_param.ms.intermediate_behavourial_events_frames_periods)

        self.tiff_movie = None
        self.raw_traces_binned = None
        # self.raw_traces_binned = np.zeros((self.nb_neurons, self.nb_times // 10), dtype="float")
        # # mean by 10 frames +
        # for cell, trace in enumerate(self.raw_traces):
        #     # mean by 10 frames
        #     self.raw_traces_binned[cell] = np.mean(np.split(np.array(trace), (self.nb_times // 10)), axis=1)
        #     # z_score
        #     self.raw_traces_binned[cell] = (self.raw_traces_binned[cell] - np.mean(self.raw_traces_binned[cell])) \
        #                                    / np.std(self.raw_traces_binned[cell])
        #     self.raw_traces_binned[cell] -= 4
        #     # print(f"np.min(self.raw_traces_binned[cell, :]) {np.min(self.raw_traces_binned[cell, :])}")
        # Y alignment
        self.normalize_traces()
        # array of 1 D, representing the number of spikes at each time
        self.activity_count = np.sum(self.spike_nums, axis=0)

        # initializing inter_neurons and cells_to_remove
        self.inter_neurons = np.zeros(self.nb_neurons, dtype="uint8")
        if self.data_and_param.inter_neurons is not None:
            for inter_neuron in self.data_and_param.inter_neurons:
                self.inter_neurons[inter_neuron] = 1
        self.cells_to_remove = np.zeros(self.nb_neurons, dtype="uint8")
        if self.data_and_param.cells_to_remove is not None:
            for cell_to_remove in self.data_and_param.cells_to_remove:
                self.cells_to_remove[cell_to_remove] = 1
        self.display_raw_traces = True
        self.raw_traces_seperate_plot = False
        # neuron's trace displayed
        self.current_neuron = 0
        # indicate if an action might remove or add an onset
        self.add_onset_mode = False
        self.remove_onset_mode = False
        self.add_peak_mode = False
        self.remove_peak_mode = False
        self.remove_all_mode = False
        # to know if the actual displayed is saved
        self.is_saved = True
        # used to known if the mouse has been moved between the click and the release
        self.last_click_position = (-1, -1)
        self.first_click_to_remove = None
        # not used anymore for the UNDO action
        # self.last_action = None
        # list of the last action, used for the undo method, the last one being the last
        self.last_actions = []
        # list of action that has been undone
        self.undone_actions = []

        # Three horizontal frames to start
        # -------------- top frame (start) ----------------
        top_frame = Frame(self)
        top_frame.pack(side=TOP, expand=YES, fill=BOTH)

        # self.spin_box_button = Spinbox(top_frame, from_=0, to=self.nb_neurons - 1, fg="blue", justify=CENTER,
        #                                width=4, state="readonly")
        # self.spin_box_button["command"] = event_lambda(self.spin_box_update)
        # # self.spin_box_button.config(command=event_lambda(self.spin_box_update))
        # self.spin_box_button.pack(side=LEFT)

        # self.neuron_string_var = StringVar()
        self.neuron_entry_widget = Entry(top_frame, fg="blue", justify=CENTER,
                                         width=3)
        self.neuron_entry_widget.insert(0, "0")
        self.neuron_entry_widget.bind("<KeyRelease>", self.go_to_neuron_button_action)
        # self.neuron_string_var.set("0")
        # self.neuron_string_var.trace("w", self.neuron_entry_change)
        # self.neuron_entry_widget.focus_set()
        self.neuron_entry_widget.pack(side=LEFT)

        # empty_label = Label(top_frame)
        # empty_label["text"] = " " * 2
        # empty_label.pack(side=LEFT)

        self.go_button = Button(top_frame)
        self.go_button["text"] = ' GO '
        self.go_button["fg"] = "blue"
        self.go_button["command"] = event_lambda(self.go_to_neuron_button_action)
        self.go_button.pack(side=LEFT)

        empty_label = Label(top_frame)
        empty_label["text"] = " " * 5
        empty_label.pack(side=LEFT)

        self.prev_button = Button(top_frame)
        self.prev_button["text"] = ' previous cell '
        self.prev_button["fg"] = "blue"
        self.prev_button['state'] = DISABLED  # ''normal
        self.prev_button["command"] = event_lambda(self.select_previous_neuron)

        self.prev_button.pack(side=LEFT)

        empty_label = Label(top_frame)
        empty_label["text"] = " " * 2
        empty_label.pack(side=LEFT)

        self.neuron_label = Label(top_frame)
        self.neuron_label["text"] = "0"
        self.neuron_label["fg"] = "red"
        self.neuron_label.pack(side=LEFT)

        empty_label = Label(top_frame)
        empty_label["text"] = " " * 2
        empty_label.pack(side=LEFT)

        self.next_button = Button(top_frame)
        self.next_button["text"] = ' next cell '
        self.next_button["fg"] = 'blue'
        self.next_button["command"] = event_lambda(self.select_next_neuron)
        self.next_button.pack(side=LEFT)

        empty_label = Label(top_frame)
        empty_label["text"] = " " * 3
        empty_label.pack(side=LEFT)

        self.zoom_fit_button = Button(top_frame)
        self.zoom_fit_button["text"] = ' y zoom fit '
        self.zoom_fit_button["fg"] = 'blue'
        self.zoom_fit_button["command"] = event_lambda(self.update_plot, amplitude_zoom_fit=True)
        self.zoom_fit_button.pack(side=LEFT)

        empty_label = Label(top_frame)
        empty_label["text"] = " " * 3
        empty_label.pack(side=LEFT)

        # empty_label = Label(top_frame)
        # empty_label["text"] = " " * 5
        # empty_label.pack(side=LEFT)

        self.add_onset_button = Button(top_frame)
        self.add_onset_button["text"] = ' ADD ONSET OFF '
        self.add_onset_button["fg"] = 'red'
        self.add_onset_button["command"] = self.add_onset_switch_mode
        self.add_onset_button.pack(side=LEFT)

        empty_label = Label(top_frame)
        empty_label["text"] = " " * 1
        empty_label.pack(side=LEFT)

        self.onset_numbers_label = Label(top_frame)
        self.onset_numbers_label["text"] = f"{self.numbers_of_onset()}"
        self.onset_numbers_label.pack(side=LEFT)

        empty_label = Label(top_frame)
        empty_label["text"] = " " * 1
        empty_label.pack(side=LEFT)

        self.remove_onset_button = Button(top_frame)
        self.remove_onset_button["text"] = ' REMOVE ONSET OFF '
        self.remove_onset_button["fg"] = 'red'
        self.remove_onset_button["command"] = self.remove_onset_switch_mode
        self.remove_onset_button.pack(side=LEFT)

        # empty_label = Label(top_frame)
        # empty_label["text"] = " " * 5
        # empty_label.pack(side=LEFT)
        #
        # onset_label = Label(top_frame)
        # onset_label["text"] = f"Numbers of onset: "
        # onset_label.pack(side=LEFT)

        empty_label = Label(top_frame)
        empty_label["text"] = " " * 3
        empty_label.pack(side=LEFT)

        self.add_peak_button = Button(top_frame)
        self.add_peak_button["text"] = ' ADD PEAK OFF '
        self.add_peak_button["fg"] = 'red'
        self.add_peak_button["command"] = self.add_peak_switch_mode
        self.add_peak_button.pack(side=LEFT)

        empty_label = Label(top_frame)
        empty_label["text"] = " " * 1
        empty_label.pack(side=LEFT)

        self.peak_numbers_label = Label(top_frame)
        self.peak_numbers_label["text"] = f"{self.numbers_of_peak()}"
        self.peak_numbers_label.pack(side=LEFT)

        empty_label = Label(top_frame)
        empty_label["text"] = " " * 1
        empty_label.pack(side=LEFT)

        self.remove_peak_button = Button(top_frame)
        self.remove_peak_button["text"] = ' REMOVE PEAK OFF '
        self.remove_peak_button["fg"] = 'red'
        self.remove_peak_button["command"] = self.remove_peak_switch_mode
        self.remove_peak_button.pack(side=LEFT)

        empty_label = Label(top_frame)
        empty_label["text"] = " " * 3
        empty_label.pack(side=LEFT)

        self.remove_all_button = Button(top_frame)
        self.remove_all_button["text"] = ' REMOVE ALL OFF '
        self.remove_all_button["fg"] = 'red'
        self.remove_all_button["command"] = self.remove_all_switch_mode
        self.remove_all_button.pack(side=LEFT)

        # -------------- top frame (end) ----------------

        ################################################################################
        ################################ Middle frame with plot ################################
        ################################################################################
        canvas_frame = Frame(self)
        canvas_frame.pack(side=TOP, expand=YES, fill=BOTH)

        main_plot_frame = Frame(canvas_frame)
        main_plot_frame.pack(side=LEFT, expand=YES, fill=BOTH)

        self.display_michou = False

        # plt.ion()
        self.fig = plt.figure(figsize=(10, 6))
        # self.plot_canvas = MyCanvas(self.fig, canvas_frame, self)
        self.plot_canvas = FigureCanvasTkAgg(self.fig, main_plot_frame)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.fig.canvas.mpl_connect('motion_notify_event', self.motion)
        # self.plot_canvas.
        #     bind("<Button-1>", self.callback_click_fig())
        if self.raw_traces_seperate_plot:
            if self.raw_traces is None:
                self.gs = gridspec.GridSpec(2, 1, width_ratios=[1], height_ratios=[5, 1])
            else:
                self.gs = gridspec.GridSpec(3, 1, width_ratios=[1], height_ratios=[5, 5, 1])
        else:
            self.gs = gridspec.GridSpec(2, 1, width_ratios=[1], height_ratios=[5, 1])
        # self.gs.update(hspace=0.05)
        # ax1 = plt.subplot(gs[0])
        # ax2 = plt.subplot(gs[1])
        self.axe_plot = None
        self.axe2_plot = None
        self.axe_plot_raw = None
        self.ax1_bottom_scatter = None
        self.ax1_top_scatter = None
        self.line1 = None
        self.line2 = None
        self.plot_graph(first_time=True)

        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=YES)

        self.toolbar = NavigationToolbar2Tk(self.plot_canvas, main_plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side=TOP, fill=BOTH, expand=YES)

        side_bar_frame = Frame(canvas_frame)
        side_bar_frame.pack(side=LEFT, expand=YES, fill=BOTH)

        map_frame = Frame(side_bar_frame)
        map_frame.pack(side=TOP, expand=YES, fill=BOTH)

        # tif movie loading
        self.last_img_displayed = None
        self.trace_movie_p1 = None
        self.trace_movie_p2 = None
        self.movie_available = False
        self.play_movie = False
        self.n_frames_movie = 1
        self.first_frame_movie = 0
        self.last_frame_movie = 1
        # to avoid garbage collector
        self.anim_movie = None
        self.last_frame_label = None
        # delay between 2 frames, in ms
        self.movie_delay = 10
        # frames to be played by the movie
        self.movie_frames = None
        if self.data_and_param.ms.tif_movie_file_name is not None:
            self.movie_available = True
            im = PIL.Image.open(self.data_and_param.ms.tif_movie_file_name)
            # im.show()
            # nb of frames should be 12500
            n_frames = 0
            dim_x = None
            dim_y = None
            for i, page in enumerate(ImageSequence.Iterator(im)):
                n_frames += 1
                if dim_x is None:
                    imarray = np.array(page)
                    dim_x, dim_y = imarray.shape[0], imarray.shape[1]
            print(f"n_frames {n_frames}, dim_x {dim_x}, dim_y {dim_y}")
            self.tiff_movie = np.zeros((n_frames, dim_x, dim_y), dtype="int16")
            for frame, page in enumerate(ImageSequence.Iterator(im)):
                self.tiff_movie[frame] = np.array(page)

        self.michou_path = "michou/"
        self.michou_img_file_names = []
        # look for filenames in the fisrst directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(self.data_and_param.path_data + self.michou_path):
            for file_name in local_filenames:
                if file_name.endswith(".png") or file_name.endswith(".jpg"):
                    self.michou_img_file_names.append(file_name)
            break
        self.michou_imgs = []
        for file_name in self.michou_img_file_names:
            # image = cv2.imread(self.data_and_param.path_data + self.michou_path + file_name)
            # # OpenCV represents images in BGR order; however PIL represents
            # # images in RGB order, so we need to swap the channels
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # # convert the images to PIL format...
            # image = Image.fromarray(image)
            # image = ImageTk.PhotoImage(image)
            # self.michou_imgs.append(image)
            self.michou_imgs.append(mpimg.imread(self.data_and_param.path_data + self.michou_path + file_name))

        self.n_michou_img = len(self.michou_img_file_names)
        self.michou_img_to_display = -1

        if self.data_and_param.ms.avg_cell_map_img is not None:
            self.map_img_fig = plt.figure(figsize=(4, 4))
            self.background_map_fig = None
            self.axe_plot_map_img = None
            self.cell_contour = None
            # creating all cell_contours
            self.cell_contours = dict()
            self.cell_in_pixel = np.ones((200, 200), dtype="int16")
            self.cell_in_pixel *= -1
            for cell in np.arange(self.nb_neurons):
                # cell contour
                coord = self.data_and_param.ms.coord_obj.coord[cell]
                coord = coord - 1
                coord = coord.astype(int)
                n_coord = len(coord[0, :])
                xy = np.zeros((n_coord, 2))
                for n in np.arange(n_coord):
                    xy[n, 0] = coord[0, n]
                    xy[n, 1] = coord[1, n]
                self.cell_contours[cell] = patches.Polygon(xy=xy,
                                                           fill=False, linewidth=0, facecolor="red",
                                                           edgecolor="red",
                                                           zorder=15, lw=0.6)
                bw = np.zeros((200, 200), dtype="int8")
                # morphology.binary_fill_holes(input
                # print(f"coord[1, :] {coord[1, :]}")
                bw[coord[1, :], coord[0, :]] = 1

                # used to know which cell has been clicked
                img_filled = np.zeros((200, 200), dtype="int8")
                # specifying output, otherwise binary_fill_holes return a boolean array
                morphology.binary_fill_holes(bw, output=img_filled)
                for pixel in np.arange(200):
                    self.cell_in_pixel[pixel, np.where(img_filled[pixel, :])[0]] = cell

            #
            # x, y = np.meshgrid(np.arange(200), np.arange(200))  # make a canvas with coordinates
            # x, y = x.flatten(), y.flatten()
            # points = np.vstack((x, y)).T
            #
            # for cell, cell_contour in self.cell_contours.items():
            #     grid = cell_contour.contains_points(points)
            #     cell_contour.set_fill(False)
            #     mask = grid.reshape(200, 200)
            #     print(f"mask0 {len(np.where(mask[0])[0])} mask1 {len(np.where(mask[1])[0])}")
            #     # x, y = np.meshgrid(np.arange(200), np.arange(200))
            #     # x = x[mask[0]]
            #     # y = y[mask[1]]
            #     # x, y = x.flatten(), y.flatten()
            #     # new_points = np.vstack((x, y)).T
            # raise Exception("mask")
            # print(f"cell {cell}, points: {new_points}")

            # for x in np.arange(200):
            #     for y in np.arange(200):
            #         cell_found = False
            #         for cell, cell_contour in self.cell_contours.items():
            #             if cell_contour.contains_point((x, y)):
            #                 self.cell_in_pixel[x, y] = cell
            # cell_found = True
            #                 print(f"cell {cell}, x {x}, y {y}")
            #                 break
            #         if cell_found:
            #             continue
            # for pixel in np.arange(200):
            #     print(f"pixel_row {pixel} {len(np.where(self.cell_in_pixel[pixel, :]>=0)[0])}")
            #
            # self.plot_canvas = MyCanvas(self.fig, canvas_frame, self)
            self.map_img_canvas = FigureCanvasTkAgg(self.map_img_fig, map_frame)
            self.map_img_fig.canvas.mpl_connect('button_release_event', self.onrelease_map)
            self.plot_map_img(first_time=True)

            self.map_img_canvas.draw()
            self.map_img_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=YES)

        self.magnifier_frame = Frame(side_bar_frame)
        self.magnifier_frame.pack(side=TOP, expand=YES, fill=BOTH)
        self.magnifier_fig = plt.figure(figsize=(4, 4))
        self.axe_plot_magnifier = None
        # represent the x_value on which the magnifier is centered
        self.x_center_magnified = None
        # how many frames before and after the center
        self.magnifier_range = 50
        # self.plot_canvas = MyCanvas(self.fig, canvas_frame, self)
        self.magnifier_canvas = FigureCanvasTkAgg(self.magnifier_fig, self.magnifier_frame)
        # used to update the plot
        self.magnifier_marker = None
        self.magnifier_line = None
        self.plot_magnifier(first_time=True)

        self.magnifier_canvas.draw()
        self.magnifier_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=YES)

        ################################################################################
        ################################ Bottom frame ################################
        ################################################################################
        bottom_frame = Frame(self)
        bottom_frame.pack(side=TOP, expand=YES, fill=BOTH)

        # end_button = Button(bottom_frame)
        # end_button["text"] = ' FINISH '
        # end_button["fg"] = "red"
        # end_button["command"] = event_lambda(self.save_spike_nums, and_close=False)
        # end_button.pack(side=RIGHT)
        #
        # empty_label = Label(bottom_frame)
        # empty_label["text"] = " " * 5
        # empty_label.pack(side=RIGHT)

        self.save_as_button = Button(bottom_frame)
        self.save_as_button["text"] = ' SAVE AS '
        self.save_as_button["fg"] = "blue"
        self.save_as_button['state'] = "normal"
        self.save_as_button["command"] = event_lambda(self.save_as_spike_nums)
        self.save_as_button.pack(side=RIGHT)

        empty_label = Label(bottom_frame)
        empty_label["text"] = " " * 2
        empty_label.pack(side=RIGHT)

        self.save_button = Button(bottom_frame)
        self.save_button["text"] = ' SAVE '
        self.save_button["fg"] = "blue"
        self.save_button['state'] = DISABLED  # ''normal
        self.save_button["command"] = event_lambda(self.save_spike_nums)
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

        self.remove_peaks_under_threshold_button = Button(bottom_frame)
        self.remove_peaks_under_threshold_button["text"] = ' REMOVE PEAKS '
        self.remove_peaks_under_threshold_button["fg"] = "red"
        self.remove_peaks_under_threshold_button['state'] = DISABLED  # ''normal
        self.remove_peaks_under_threshold_button["command"] = event_lambda(self.remove_peaks_under_threshold)
        self.remove_peaks_under_threshold_button.pack(side=RIGHT)

        empty_label = Label(bottom_frame)
        empty_label["text"] = " " * 2
        empty_label.pack(side=RIGHT)
        # from_=1, to=3
        self.spin_box_threshold = Spinbox(bottom_frame, values=list(np.arange(0.1, 10, 0.2)), fg="blue", justify=CENTER,
                                          width=3, state="readonly")
        self.spin_box_threshold["command"] = event_lambda(self.spin_box_threshold_update)
        # self.spin_box_button.config(command=event_lambda(self.spin_box_update))
        self.spin_box_threshold.pack(side=RIGHT)

        empty_label = Label(bottom_frame)
        empty_label["text"] = " " * 2
        empty_label.pack(side=RIGHT)

        self.treshold_var = IntVar()
        threshold_check_box = Checkbutton(bottom_frame, text="Threshold", variable=self.treshold_var, onvalue=1,
                                          offvalue=0, fg=self.color_threshold_line)
        threshold_check_box["command"] = event_lambda(self.threshold_check_box_action)
        threshold_check_box.pack(side=RIGHT)

        if self.raw_traces is not None:
            empty_label = Label(bottom_frame)
            empty_label["text"] = " " * 2
            empty_label.pack(side=RIGHT)

            self.raw_trace_var = IntVar()
            self.raw_trace_check_box = Checkbutton(bottom_frame, text="Raw trace", variable=self.raw_trace_var,
                                                   onvalue=1,
                                                   offvalue=0, fg=self.color_raw_trace)
            if self.display_raw_traces:
                self.raw_trace_check_box.select()
            self.raw_trace_check_box["command"] = event_lambda(self.display_raw_tracecheck_box_action)
            self.raw_trace_check_box.pack(side=RIGHT)

        empty_label = Label(bottom_frame)
        empty_label["text"] = " " * 2
        empty_label.pack(side=RIGHT)

        self.inter_neuron_button = Button(bottom_frame)
        if self.inter_neurons[self.current_neuron] == 0:
            self.inter_neuron_button["text"] = ' not IN '
            self.inter_neuron_button["fg"] = "black"
        else:

            self.inter_neuron_button["text"] = ' IN '
            self.inter_neuron_button["fg"] = "red"

        self.inter_neuron_button["command"] = event_lambda(self.set_inter_neuron)
        self.inter_neuron_button.pack(side=RIGHT)

        empty_label = Label(bottom_frame)
        empty_label["text"] = " " * 1
        empty_label.pack(side=RIGHT)

        self.remove_cell_button = Button(bottom_frame)

        if self.cells_to_remove[self.current_neuron] == 0:
            self.remove_cell_button["text"] = ' not removed '
            self.remove_cell_button["fg"] = "black"
        else:
            self.remove_cell_button["text"] = ' removed '
            self.remove_cell_button["fg"] = "red"

        self.remove_cell_button["command"] = event_lambda(self.remove_cell)
        self.remove_cell_button.pack(side=RIGHT)

        empty_label = Label(bottom_frame)
        empty_label["text"] = " " * 1
        empty_label.pack(side=RIGHT)

        self.magnifier_button = Button(bottom_frame)

        self.magnifier_button["text"] = ' magnified OFF '
        self.magnifier_button["fg"] = "black"
        self.magnifier_mode = False

        self.magnifier_button["command"] = event_lambda(self.switch_magnifier)
        self.magnifier_button.pack(side=RIGHT)

        if self.mvt_frames_periods is not None:
            empty_label = Label(bottom_frame)
            empty_label["text"] = " " * 1
            empty_label.pack(side=RIGHT)

            self.display_mvt_button = Button(bottom_frame)

            self.display_mvt_button["text"] = ' mvt OFF '
            self.display_mvt_button["fg"] = "black"

            self.display_mvt_button["command"] = event_lambda(self.switch_mvt_display)
            self.display_mvt_button.pack(side=RIGHT)

        if self.tiff_movie is not None:
            empty_label = Label(bottom_frame)
            empty_label["text"] = " " * 1
            empty_label.pack(side=RIGHT)

            self.movie_button = Button(bottom_frame)

            self.movie_button["text"] = ' movie OFF '
            self.movie_button["fg"] = "black"
            self.movie_mode = False

            self.movie_button["command"] = event_lambda(self.switch_movie_mode)
            self.movie_button.pack(side=RIGHT)

        # self.avg_cell_map_img = None
        # if self.data_and_param.ms.avg_cell_map_img_file_name is not None:
        #     self.avg_cell_map_img = cv2.imread(self.data_and_param.ms.avg_cell_map_img_file_name)
        #     # OpenCV represents images in BGR order; however PIL represents
        #     # images in RGB order, so we need to swap the channels
        #     self.avg_cell_map_img = cv2.cvtColor(self.avg_cell_map_img, cv2.COLOR_BGR2RGB)
        #     # convert the images to PIL format...
        #     self.avg_cell_map_img = Image.fromarray(self.avg_cell_map_img)
        #     self.avg_cell_map_img = ImageTk.PhotoImage(self.avg_cell_map_img)

        # used for association of keys
        self.keys_pressed = dict()
        self.root.bind_all("<KeyRelease>", self.key_release_action)
        self.root.bind_all("<KeyPress>", self.key_press_action)

    def neuron_entry_change(self, *args):
        print(f"Neuron: {self.neuron_string_var.get()}")

    def switch_michou(self):
        if self.display_michou:
            self.display_michou = False
            self.update_plot_map_img(after_michou=True)
        else:
            self.display_michou = True
            self.michou_img_to_display = randint(0, self.n_michou_img - 1)
            self.update_plot_map_img()

    def switch_movie_mode(self, from_movie_button=True):
        if from_movie_button and (not self.movie_mode):
            if self.remove_peak_mode:
                self.remove_peak_switch_mode(from_remove_peak_button=False)
            if self.remove_onset_mode:
                self.remove_onset_switch_mode(from_remove_onset_button=False)
            if self.add_peak_mode:
                self.add_peak_switch_mode(from_add_peak_button=False)
            if self.add_onset_mode:
                self.add_onset_switch_mode(from_add_onset_button=False)
            if self.remove_all_mode:
                self.remove_all_switch_mode(from_remove_all_button=False)

        if self.movie_mode:
            self.movie_button["text"] = ' movie OFF '
            self.movie_button["fg"] = "black"
            self.movie_mode = False
            if self.play_movie:
                self.play_movie = False
                if self.anim_movie is not None:
                    self.anim_movie.event_source.stop()
                    self.update_contour_for_cell(cell=self.current_neuron)
                    self.update_plot_map_img(after_michou=True)
        else:
            self.movie_button["text"] = ' movie ON '
            self.movie_button["fg"] = "red"
            self.movie_mode = True

        if self.first_click_to_remove is not None:
            self.first_click_to_remove = None
            self.update_plot()

    def switch_mvt_display(self):
        if self.display_mvt:
            self.display_mvt_button["text"] = ' mvt OFF '
            self.display_mvt_button["fg"] = "black"
            self.display_mvt = False
        else:
            self.display_mvt_button["text"] = ' mvt ON '
            self.display_mvt_button["fg"] = "red"
            self.display_mvt = True
        self.update_plot()

    def switch_magnifier(self):
        if self.magnifier_mode:
            self.magnifier_button["text"] = ' magnified OFF '
            self.magnifier_button["fg"] = "black"
            self.magnifier_mode = False
        else:
            self.magnifier_button["text"] = ' magnified ON '
            self.magnifier_button["fg"] = "red"
            self.magnifier_mode = True

    def set_inter_neuron(self):
        if self.inter_neurons[self.current_neuron] == 0:
            self.inter_neuron_button["text"] = " IN "
            self.inter_neuron_button["fg"] = "red"
            self.inter_neurons[self.current_neuron] = 1
        else:
            self.inter_neuron_button["text"] = " not IN "
            self.inter_neuron_button["fg"] = "black"
            self.inter_neurons[self.current_neuron] = 0

    def remove_cell(self):
        if self.cells_to_remove[self.current_neuron] == 0:
            self.remove_cell_button["text"] = " removed "
            self.remove_cell_button["fg"] = "red"
            self.cells_to_remove[self.current_neuron] = 1
        else:
            self.remove_cell_button["text"] = " not removed "
            self.remove_cell_button["fg"] = "black"
            self.cells_to_remove[self.current_neuron] = 0

    def clear_and_update_entry_neuron_widget(self):
        self.neuron_entry_widget.delete(first=0, last=END)
        self.neuron_entry_widget.insert(0, f"{self.current_neuron}")

    def go_to_neuron_button_action(self, event=None):
        # print("go_to_neuron_button_action")
        neuron_selected = self.neuron_entry_widget.get()
        try:
            neuron_selected = int(neuron_selected)
        except (ValueError, TypeError) as e:
            # error if a value that is not an int is selected
            # print("clear entry")
            self.clear_and_update_entry_neuron_widget()
            return
        if neuron_selected == self.current_neuron:
            return

        if (neuron_selected < 0) or (neuron_selected > (self.nb_neurons - 1)):
            self.clear_and_update_entry_neuron_widget()
            return

        if (event is not None) and (event.keysym == 'Return'):
            self.update_neuron(new_neuron=neuron_selected)
        if event is None:
            self.update_neuron(new_neuron=neuron_selected)

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
        # if 0, means the key was not recognized, we empty the dict
        if event.keysym_num == 0:
            self.keys_pressed = dict()
        # print(f"released keysym {event.keysym}, keycode {event.keycode}, keysym_num {event.keysym_num}, "
        #       f"state {event.state}")
        if event.char in ["a", "A", "+"]:
            self.add_onset_switch_mode()
        elif event.char in ["r", "R", "-"]:
            self.remove_all_switch_mode()
        elif event.char in ["m", "M"]:
            self.switch_magnifier()
        elif event.char in ["p", "P"]:
            if self.n_michou_img > 0:
                self.switch_michou()
        elif (event.keysym == "space") and (self.tiff_movie is not None):
            self.switch_movie_mode()
        if event.keysym == 'Right':
            # C as cell
            if ("Meta_L" in self.keys_pressed) or ("Control_L" in self.keys_pressed):
                # ctrl-right, we move to next neuron
                self.select_next_neuron()
            else:
                self.move_zoom(to_the_left=False)
        elif event.keysym == 'Left':
            if ("Meta_L" in self.keys_pressed) or ("Control_L" in self.keys_pressed):
                self.select_previous_neuron()
            else:
                self.move_zoom(to_the_left=True)

    def detect_peaks(self):
        """
        Compute peak_nums, with nb of times equal to traces's one
        :return:
        """
        peak_nums = np.zeros((self.nb_neurons, self.nb_times_traces), dtype="float")

        for neuron, o_t in enumerate(self.onset_times):
            # neuron by neuron
            onset_index = np.where(o_t > 0)[0]
            for index in onset_index:
                # looking for the peak following the onset
                limit_index_search = min((index + (self.decay * self.decay_factor)), self.nb_times_traces - 1)
                if index == limit_index_search:
                    continue
                max_index = np.argmax(self.traces[neuron, index:limit_index_search])
                max_index += index
                if np.size(max_index) == 1:  # or isinstance(max_index, int)
                    peak_nums[neuron, max_index] = self.traces[neuron, max_index]
                else:
                    peak_nums[neuron, max_index] = self.traces[neuron, max_index[0]]

        return peak_nums

    def detect_onset_associated_to_peak(self, peak_times):
        """
        Return an array with the onset times (from trace time) associated to the peak_times.
        We look before each peak_time (1sec before), for current_neurons
        :param peak_times:
        :return:
        """
        onsets_detected = []
        for peak_time in peak_times:
            # looking for the peak preceding the onset
            limit_index_search = max((peak_time - (self.decay * self.decay_factor)), 0)
            if peak_time == limit_index_search:
                continue
            onset_times = np.where(self.onset_times[self.current_neuron, limit_index_search:(peak_time + 1)] > 0)[0]
            onset_times += limit_index_search
            onsets_detected.extend(list(onset_times))

        return np.array(onsets_detected)

    def update_onset_times(self):
        # onset_times as values up to 2
        for n, neuron in enumerate(self.spike_nums):
            self.onset_times[n, :] = neuron
        if self.onset_numbers_label is not None:
            self.onset_numbers_label["text"] = f"{self.numbers_of_onset()}"

    def numbers_of_onset(self):
        return len(np.where(self.onset_times[self.current_neuron, :] > 0)[0])

    def numbers_of_peak(self):
        return len(np.where(self.peak_nums[self.current_neuron, :] > 0)[0])

    def add_onset_switch_mode(self, from_add_onset_button=True):
        # if it was called due to the action of pressing the remove button, we're not calling the remove switch mode
        if from_add_onset_button and (not self.add_onset_mode):
            if self.remove_onset_mode:
                self.remove_onset_switch_mode(from_remove_onset_button=False)
            if self.remove_peak_mode:
                self.remove_peak_switch_mode(from_remove_peak_button=False)
            if self.add_peak_mode:
                self.add_peak_switch_mode(from_add_peak_button=False)
            if self.remove_all_mode:
                self.remove_all_switch_mode(from_remove_all_button=False)
            if self.movie_mode:
                self.switch_movie_mode(from_movie_button=False)
        self.add_onset_mode = not self.add_onset_mode
        if self.add_onset_mode:
            self.add_onset_button["fg"] = 'green'
            self.add_onset_button["text"] = ' ADD ONSET ON '
        else:
            self.add_onset_button["fg"] = 'red'
            self.add_onset_button["text"] = ' ADD ONSET OFF '

    def remove_onset_switch_mode(self, from_remove_onset_button=True):
        # deactivating other button
        if from_remove_onset_button and (not self.remove_onset_mode):
            if self.add_onset_mode:
                self.add_onset_switch_mode(from_add_onset_button=False)
            if self.add_peak_mode:
                self.add_peak_switch_mode(from_add_peak_button=False)
            if self.remove_peak_mode:
                self.remove_peak_switch_mode(from_remove_peak_button=False)
            if self.remove_all_mode:
                self.remove_all_switch_mode(from_remove_all_button=False)
            if self.movie_mode:
                self.switch_movie_mode(from_movie_button=False)
        self.remove_onset_mode = not self.remove_onset_mode

        if self.remove_onset_mode:
            self.remove_onset_button["fg"] = 'green'
            self.remove_onset_button["text"] = ' REMOVE ONSET ON '
            self.first_click_to_remove = None
        else:
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
            self.remove_onset_button["fg"] = 'red'
            self.remove_onset_button["text"] = ' REMOVE ONSET OFF '

    def update_contour_for_cell(self, cell):
        # used in order to have access to contour after animation

        # cell contour
        coord = self.data_and_param.ms.coord_obj.coord[cell]
        coord = coord - 1
        coord = coord.astype(int)
        n_coord = len(coord[0, :])
        xy = np.zeros((n_coord, 2))
        for n in np.arange(n_coord):
            xy[n, 0] = coord[0, n]
            xy[n, 1] = coord[1, n]
        self.cell_contours[cell] = patches.Polygon(xy=xy,
                                                   fill=False, linewidth=0, facecolor="red",
                                                   edgecolor="red",
                                                   zorder=15, lw=0.6)

    def remove_peak_switch_mode(self, from_remove_peak_button=True):
        if from_remove_peak_button and (not self.remove_peak_mode):
            if self.add_peak_mode:
                self.add_peak_switch_mode(from_add_peak_button=False)
            if self.remove_onset_mode:
                self.remove_onset_switch_mode(from_remove_onset_button=False)
            if self.add_onset_mode:
                self.add_onset_switch_mode(from_add_onset_button=False)
            if self.remove_all_mode:
                self.remove_all_switch_mode(from_remove_all_button=False)
            if self.movie_mode:
                self.switch_movie_mode(from_movie_button=False)
        self.remove_peak_mode = not self.remove_peak_mode

        if self.remove_peak_mode:
            self.remove_peak_button["fg"] = 'green'
            self.remove_peak_button["text"] = ' REMOVE PEAK ON '
            # in case one click would have been made when remove onset was activated
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
        else:
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
            self.remove_peak_button["fg"] = 'red'
            self.remove_peak_button["text"] = ' REMOVE PEAK OFF '

    def remove_all_switch_mode(self, from_remove_all_button=True):
        if from_remove_all_button and (not self.remove_all_mode):
            if self.add_peak_mode:
                self.add_peak_switch_mode(from_add_peak_button=False)
            if self.remove_peak_mode:
                self.remove_peak_switch_mode(from_remove_peak_button=False)
            if self.remove_onset_mode:
                self.remove_onset_switch_mode(from_remove_onset_button=False)
            if self.add_onset_mode:
                self.add_onset_switch_mode(from_add_onset_button=False)
            if self.movie_mode:
                self.switch_movie_mode(from_movie_button=False)
        self.remove_all_mode = not self.remove_all_mode

        if self.remove_all_mode:
            self.remove_all_button["fg"] = 'green'
            self.remove_all_button["text"] = ' REMOVE ALL ON '
            # in case one click would have been made when remove onset was activated
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
        else:
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
            self.remove_all_button["fg"] = 'red'
            self.remove_all_button["text"] = ' REMOVE ALL OFF '

    def onrelease_map(self, event):
        """
                        Action when a mouse button is released on cell map
                        :param event:
                        :return:
        """
        if event.dblclick:
            return

        if event.xdata is None:
            return

        # print(f"event.xdata {event.xdata}, event.ydata {event.ydata}")
        x = int(event.xdata)
        y = int(event.ydata)

        new_neuron = self.cell_in_pixel[y, x]
        if new_neuron >= 0:
            if new_neuron != self.current_neuron:
                self.update_neuron(new_neuron=new_neuron)
        # print(f"cell: {self.cell_in_pixel[y, x]}")

    def onrelease(self, event):
        """
                Action when a mouse button is released
                :param event:
                :return:
        """
        if (not self.remove_onset_mode) and (not self.add_onset_mode) and (not self.remove_peak_mode) \
                and (not self.add_peak_mode) \
                and (not self.remove_all_mode) and (not self.movie_mode):
            return

        if event.dblclick:
            return

        if event.xdata is None:
            return

        if self.last_click_position[0] != event.xdata:
            # the mouse has been moved between the pressing and the release
            return

        if self.add_onset_mode or self.add_peak_mode:
            if (event.xdata < 0) or (event.xdata > (self.nb_times_traces - 1)):
                return
            if self.add_onset_mode:
                self.add_onset(int(round(event.xdata)))
            else:
                self.add_peak(at_time=int(round(event.xdata)), amplitude=event.ydata)
            return

        if self.remove_onset_mode or self.remove_peak_mode or \
                self.remove_all_mode or self.movie_mode:
            if self.first_click_to_remove is not None:
                if self.remove_onset_mode:
                    self.remove_onset(x_from=self.first_click_to_remove["x"], x_to=int(round(event.xdata)))
                elif self.remove_peak_mode:
                    self.remove_peak(x_from=self.first_click_to_remove["x"], x_to=int(round(event.xdata)))
                elif self.movie_mode:
                    self.start_playing_movie(x_from=self.first_click_to_remove["x"], x_to=int(round(event.xdata)))
                else:
                    self.remove_all(x_from=self.first_click_to_remove["x"], x_to=int(round(event.xdata)))
            else:
                self.first_click_to_remove = {"x": int(round(event.xdata)), "y": event.ydata}
                self.update_plot()

    def motion(self, event):
        """
        Action when the mouse is moved
        :param event:
        :return:
        """
        if not self.magnifier_mode:
            return

        if (not self.add_onset_mode) and (not self.remove_onset_mode) \
                and (not self.add_peak_mode) and (not self.remove_peak_mode) \
                and (not self.remove_all_mode):
            return

        if event.xdata is None:
            return

        if (event.xdata < 0) or (event.xdata > (self.nb_times_traces - 1)):
            return

        change_frame_ref = False
        if self.x_center_magnified is None:
            self.x_center_magnified = event.xdata
            change_frame_ref = True
        else:
            # changing the window only when there is a change > 20% of the range
            if np.abs(event.xdata - self.x_center_magnified) >= (self.magnifier_range * 0.5):
                self.x_center_magnified = event.xdata
                change_frame_ref = True

        self.update_plot_magnifier(mouse_x_position=event.xdata, mouse_y_position=event.ydata,
                                   change_frame_ref=change_frame_ref)

        # print(f"Mouse position: {event.xdata} { event.ydata}")

    def onclick(self, event):
        """
        Action when a mouse button is pressed
        :param event:
        :return:
        """
        if event.xdata is not None:
            self.last_click_position = (event.xdata, event.ydata)

    def remove_onset(self, x_from, x_to):
        # taking in consideration the case where the click is outside the graph border
        if x_from < 0:
            x_from = 0
        elif x_from > (self.nb_times_traces - 1):
            x_from = (self.nb_times_traces - 1)

        if x_to < 0:
            x_to = 0
        elif x_to > (self.nb_times_traces - 1):
            x_to = (self.nb_times_traces - 1)

        if x_from == x_to:
            return

        self.first_click_to_remove = None

        # in case x_from is after x_to
        min_value = min(x_from, x_to)
        max_value = max(x_from, x_to)
        x_from = min_value
        x_to = max_value
        # if the sum is zero, then we're not removing any onset
        modification_done = (np.sum(self.onset_times[self.current_neuron, x_from:x_to]) > 0)
        if modification_done:
            removed_times = np.where(self.onset_times[self.current_neuron, x_from:x_to] > 0)[0] + x_from
            self.onset_times[self.current_neuron, x_from:x_to] = 0
            self.spike_nums[self.current_neuron, x_from:x_to] = 0
            left_x_limit, right_x_limit = self.axe_plot.get_xlim()
            bottom_limit, top_limit = self.axe_plot.get_ylim()
            self.last_actions.append(RemoveOnsetAction(removed_times=removed_times, session_frame=self,
                                                       neuron=self.current_neuron, is_saved=self.is_saved,
                                                       x_limits=(left_x_limit, right_x_limit),
                                                       y_limits=(bottom_limit, top_limit)))
            # no more undone_actions
            self.undone_actions = []
            self.redo_button['state'] = DISABLED
            self.unsaved()
            self.undo_button['state'] = 'normal'
        # update to remove the cross of the first click
        self.update_after_onset_change()

    def remove_all(self, x_from, x_to):
        # remove onset and peaks
        # taking in consideration the case where the click is outside the graph border
        if x_from < 0:
            x_from = 0
        elif x_from > (self.nb_times_traces - 1):
            x_from = (self.nb_times_traces - 1)

        if x_to < 0:
            x_to = 0
        elif x_to > (self.nb_times_traces - 1):
            x_to = (self.nb_times_traces - 1)

        if x_from == x_to:
            return

        self.first_click_to_remove = None

        # in case x_from is after x_to
        min_value = min(x_from, x_to)
        max_value = max(x_from, x_to)
        x_from = min_value
        x_to = max_value
        # if the sum is zero, then we're not removing any onset
        modification_done = (np.sum(self.onset_times[self.current_neuron, x_from:x_to]) > 0) or \
                            (np.sum(self.peak_nums[self.current_neuron, x_from:x_to]) > 0)
        if modification_done:
            removed_times = np.where(self.onset_times[self.current_neuron, x_from:x_to] > 0)[0] + x_from
            self.onset_times[self.current_neuron, x_from:x_to] = 0
            self.spike_nums[self.current_neuron, x_from:x_to] = 0
            left_x_limit, right_x_limit = self.axe_plot.get_xlim()
            bottom_limit, top_limit = self.axe_plot.get_ylim()
            removed_onset_action = RemoveOnsetAction(removed_times=removed_times, session_frame=self,
                                                     neuron=self.current_neuron, is_saved=self.is_saved,
                                                     x_limits=(left_x_limit, right_x_limit),
                                                     y_limits=(bottom_limit, top_limit))
            removed_times = np.where(self.peak_nums[self.current_neuron, x_from:x_to] > 0)[0] + x_from
            amplitudes = self.peak_nums[self.current_neuron, removed_times]
            self.peak_nums[self.current_neuron, x_from:x_to] = 0
            left_x_limit, right_x_limit = self.axe_plot.get_xlim()
            bottom_limit, top_limit = self.axe_plot.get_ylim()
            self.last_actions.append(RemovePeakAction(removed_times=removed_times, amplitudes=amplitudes,
                                                      session_frame=self, removed_onset_action=removed_onset_action,
                                                      neuron=self.current_neuron, is_saved=self.is_saved,
                                                      x_limits=(left_x_limit, right_x_limit),
                                                      y_limits=(bottom_limit, top_limit)))
            # no more undone_actions
            self.undone_actions = []
            self.redo_button['state'] = DISABLED
            self.unsaved()
            self.undo_button['state'] = 'normal'
        # update to remove the cross of the first click
        self.update_after_onset_change()

    def remove_peaks_under_threshold(self):
        left_x_limit, right_x_limit = self.axe_plot.get_xlim()
        left_x_limit = int(max(left_x_limit, 0))
        right_x_limit = int(min(right_x_limit, self.nb_times_traces - 1))
        peaks = np.where(self.peak_nums[self.current_neuron, left_x_limit:right_x_limit] > 0)[0]
        peaks += left_x_limit
        threshold = self.get_threshold()
        peaks_under_threshold = np.where(self.traces[self.current_neuron, peaks] < threshold)[0]
        if len(peaks_under_threshold) == 0:
            return
        removed_times = peaks[peaks_under_threshold]
        amplitudes = self.peak_nums[self.current_neuron, peaks][peaks_under_threshold]
        self.peak_nums[self.current_neuron, removed_times] = 0
        left_x_limit, right_x_limit = self.axe_plot.get_xlim()
        bottom_limit, top_limit = self.axe_plot.get_ylim()
        onset_times_to_remove = self.detect_onset_associated_to_peak(removed_times)
        removed_onset_action = None
        if len(onset_times_to_remove) > 0:
            self.onset_times[self.current_neuron, onset_times_to_remove] = 0
            self.spike_nums[self.current_neuron, onset_times_to_remove] = 0
            removed_onset_action = RemoveOnsetAction(removed_times=onset_times_to_remove,
                                                     session_frame=self,
                                                     neuron=self.current_neuron, is_saved=self.is_saved,
                                                     x_limits=(left_x_limit, right_x_limit),
                                                     y_limits=(bottom_limit, top_limit))
        # TODO: remove onset also when removing peak manually
        self.last_actions.append(RemovePeakAction(removed_times=removed_times, amplitudes=amplitudes,
                                                  session_frame=self,
                                                  removed_onset_action=removed_onset_action,
                                                  neuron=self.current_neuron, is_saved=self.is_saved,
                                                  x_limits=(left_x_limit, right_x_limit),
                                                  y_limits=(bottom_limit, top_limit)))
        # no more undone_actions
        self.undone_actions = []
        self.redo_button['state'] = DISABLED
        self.unsaved()
        self.undo_button['state'] = 'normal'

        self.update_after_onset_change()

    def remove_peak(self, x_from, x_to):
        # taking in consideration the case where the click is outside the graph border
        if x_from < 0:
            x_from = 0
        elif x_from > (self.nb_times_traces - 1):
            x_from = (self.nb_times_traces - 1)

        if x_to < 0:
            x_to = 0
        elif x_to > (self.nb_times_traces - 1):
            x_to = (self.nb_times_traces - 1)

        if x_from == x_to:
            return

        self.first_click_to_remove = None

        # in case x_from is after x_to
        min_value = min(x_from, x_to)
        max_value = max(x_from, x_to)
        x_from = min_value
        x_to = max_value
        # if the sum is zero, then we're not removing any onset
        modification_done = (np.sum(self.peak_nums[self.current_neuron, x_from:x_to]) > 0)
        if modification_done:
            removed_times = np.where(self.peak_nums[self.current_neuron, x_from:x_to] > 0)[0] + x_from
            amplitudes = self.peak_nums[self.current_neuron, removed_times]
            self.peak_nums[self.current_neuron, x_from:x_to] = 0
            left_x_limit, right_x_limit = self.axe_plot.get_xlim()
            bottom_limit, top_limit = self.axe_plot.get_ylim()
            self.last_actions.append(RemovePeakAction(removed_times=removed_times, amplitudes=amplitudes,
                                                      session_frame=self,
                                                      neuron=self.current_neuron, is_saved=self.is_saved,
                                                      x_limits=(left_x_limit, right_x_limit),
                                                      y_limits=(bottom_limit, top_limit)))
            # no more undone_actions
            self.undone_actions = []
            self.redo_button['state'] = DISABLED
            self.unsaved()
            self.undo_button['state'] = 'normal'
        # update to remove the cross of the first click at least
        self.update_after_onset_change()

    def display_raw_tracecheck_box_action(self):
        self.display_raw_traces = not self.display_raw_traces

        self.update_plot(raw_trace_display_action=True)

    def threshold_check_box_action(self):
        self.display_threshold = not self.display_threshold
        if self.display_threshold:
            self.remove_peaks_under_threshold_button['state'] = 'normal'
        else:
            self.remove_peaks_under_threshold_button['state'] = DISABLED

        self.update_plot()

    def spin_box_threshold_update(self):
        self.nb_std_thresold = float(self.spin_box_threshold.get())
        if self.display_threshold:
            self.update_plot()

    def unsaved(self):
        """
        means a changed has been done, and the actual plot is not saved """

        self.is_saved = False
        self.save_button['state'] = 'normal'

    def normalize_traces(self):
        if self.raw_traces is None:
            return

        # z_score traces
        for i in np.arange(self.nb_neurons):
            self.traces[i, :] = (self.traces[i, :] - np.mean(self.traces[i, :])) / np.std(self.traces[i, :])
            self.raw_traces[i, :] = (self.raw_traces[i, :] - np.mean(self.raw_traces[i, :])) \
                                    / np.std(self.raw_traces[i, :])
        self.raw_traces -= 2
        # for neuron, trace in enumerate(self.traces):
        #     mean_trace = np.mean(trace)
        #     mean_raw_trace = np.mean(self.raw_traces[neuron, :])
        #     dif = mean_trace - mean_raw_trace
        #     self.raw_traces[neuron, :] += dif

    def add_onset(self, at_time):
        if self.onset_times[self.current_neuron, at_time] > 0:
            return

        self.onset_times[self.current_neuron, at_time] = 1
        self.spike_nums[self.current_neuron, at_time] = 1

        # Detecting peak
        add_peak_index = -1
        lets_not_add_peaks = True
        if not lets_not_add_peaks:
            limit_index_search = min((at_time + (self.decay * self.decay_factor)), self.nb_times_traces - 1)
            if at_time != limit_index_search:
                max_index = np.argmax(self.traces[self.current_neuron, at_time:limit_index_search])
                max_index += at_time
                if np.size(max_index) == 1:  # or isinstance(max_index, int)
                    add_peak_index = max_index
                    # self.peak_nums[self.current_neuron, max_index] = self.traces[self.current_neuron, max_index]
                else:
                    add_peak_index = max_index[0]
                # adding a peak only if there is no peak already detected
                if self.peak_nums[self.current_neuron, add_peak_index] > 0:
                    add_peak_index = -1

        left_x_limit, right_x_limit = self.axe_plot.get_xlim()
        bottom_limit, top_limit = self.axe_plot.get_ylim()

        add_peak_action = None
        if add_peak_index > 0:
            self.peak_nums[self.current_neuron, add_peak_index] = self.traces[self.current_neuron, add_peak_index]
            add_peak_action = AddPeakAction(added_time=add_peak_index,
                                            amplitude=self.traces[self.current_neuron, add_peak_index],
                                            session_frame=self,
                                            neuron=self.current_neuron, is_saved=self.is_saved,
                                            x_limits=(left_x_limit, right_x_limit),
                                            y_limits=(bottom_limit, top_limit))

        self.last_actions.append(AddOnsetAction(added_time=at_time, session_frame=self,
                                                add_peak_action=add_peak_action,
                                                neuron=self.current_neuron, is_saved=self.is_saved,
                                                x_limits=(left_x_limit, right_x_limit),
                                                y_limits=(bottom_limit, top_limit)))
        # no more undone_actions
        self.undone_actions = []
        self.redo_button['state'] = DISABLED

        self.unsaved()
        self.undo_button['state'] = 'normal'

        self.update_after_onset_change()

    def add_peak(self, at_time, amplitude=0):
        # print("add_peak")
        if self.peak_nums[self.current_neuron, at_time] > 0:
            return

        # print(f"add_peak {at_time}")
        # using the amplitude from self.traces, the amplitude as argument is where the click was made
        self.peak_nums[self.current_neuron, at_time] = 1

        left_x_limit, right_x_limit = self.axe_plot.get_xlim()
        bottom_limit, top_limit = self.axe_plot.get_ylim()
        self.last_actions.append(AddPeakAction(added_time=at_time,
                                               amplitude=self.traces[self.current_neuron, at_time],
                                               session_frame=self,
                                               neuron=self.current_neuron, is_saved=self.is_saved,
                                               x_limits=(left_x_limit, right_x_limit),
                                               y_limits=(bottom_limit, top_limit)))
        # no more undone_actions
        self.undone_actions = []
        self.redo_button['state'] = DISABLED

        self.unsaved()
        self.undo_button['state'] = 'normal'

        self.update_after_onset_change()

    def add_peak_switch_mode(self, from_add_peak_button=True):
        """

        :param from_add_peak_button: indicate the user click on the add_peak button, otherwise it means the
        function has been called after another button has been clicked
        :return:
        """
        if from_add_peak_button and (not self.add_peak_mode):
            if self.remove_peak_mode:
                self.remove_peak_switch_mode(from_remove_peak_button=False)
            if self.remove_onset_mode:
                self.remove_onset_switch_mode(from_remove_onset_button=False)
            if self.add_onset_mode:
                self.add_onset_switch_mode(from_add_onset_button=False)
            if self.remove_all_mode:
                self.remove_all_switch_mode(from_remove_all_button=False)
            if self.movie_mode:
                self.switch_movie_mode(from_movie_button=False)
        self.add_peak_mode = not self.add_peak_mode
        if self.add_peak_mode:
            self.add_peak_button["fg"] = 'green'
            self.add_peak_button["text"] = ' ADD PEAK ON '
        else:
            self.add_peak_button["fg"] = 'red'
            self.add_peak_button["text"] = ' ADD PEAK OFF '

    def validation_before_closing(self):
        if not self.is_saved:
            self.save_as_spike_nums()
        self.root.destroy()

    def redo_action(self):
        last_undone_action = self.undone_actions[-1]
        self.undone_actions = self.undone_actions[:-1]
        last_undone_action.redo()
        self.last_actions.append(last_undone_action)

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

        # if different neuron, display the other neuron
        if last_undone_action.neuron == self.current_neuron:
            self.update_after_onset_change(new_x_limit=last_undone_action.x_limits,
                                           new_y_limit=last_undone_action.y_limits)
        else:
            self.update_after_onset_change(new_neuron=last_undone_action.neuron,
                                           new_x_limit=last_undone_action.x_limits,
                                           new_y_limit=last_undone_action.y_limits)

    def undo_action(self):
        """
        Revoke the last action
        :return:
        """

        last_action = self.last_actions[-1]
        self.last_actions = self.last_actions[:-1]
        last_action.undo()

        # if self.last_action.onset_added:
        #     # an onset was added
        #     self.onset_times[self.last_action.neuron, self.last_action.added_time] = 0
        #     self.spike_nums[self.last_action.neuron, self.last_action.added_time*2] = 0
        # else:
        #     # an or several onsets were removed
        #     self.onset_times[self.last_action.neuron, self.last_action.removed_times] = 1
        #     self.spike_nums[self.last_action.neuron, self.last_action.removed_times * 2] = 1

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
        if last_action.neuron == self.current_neuron:
            self.update_after_onset_change(new_x_limit=last_action.x_limits, new_y_limit=last_action.y_limits)
        else:
            self.update_after_onset_change(new_neuron=last_action.neuron,
                                           new_x_limit=last_action.x_limits, new_y_limit=last_action.y_limits)

        # self.last_action = None

    def save_as_spike_nums(self):
        initialdir = "/"
        if self.save_path is not None:
            initialdir = self.save_path
        path_and_file_name = filedialog.asksaveasfilename(initialdir=initialdir,
                                                          title="Select file",
                                                          filetypes=[("Matlab files", "*.mat")])
        if path_and_file_name == "":
            return

        # shouldn't happen, but just in case
        if len(path_and_file_name) <= 4:
            return

        # the opener should add the ".mat" extension anyway
        if path_and_file_name[-4:] != ".mat":
            path_and_file_name += ".mat"

        # to get real index, remove 1
        self.save_path, self.save_file_name = get_file_name_and_path(path_and_file_name)

        if self.save_path is None:
            return

        self.save_spike_nums()

    def save_spike_nums(self, and_close=False):
        if self.save_file_name is None:
            self.save_as_spike_nums()
            # check if a file has been selected
            if self.save_file_name is None:
                return

        self.save_button['state'] = DISABLED
        self.is_saved = True

        # self.path_result
        # python format
        # file_name = f"spikenums_session_{self.session_number}_{self.data_and_param.time_str}"
        # np.save(self.save_path + self.save_file_name + ".npy", self.spike_nums)
        # matlab format
        cells_to_remove = np.where(self.cells_to_remove)[0]
        inter_neurons = np.where(self.inter_neurons)[0]
        sio.savemat(self.save_path + self.save_file_name, {'Bin100ms_spikedigital_Python': self.spike_nums,
                                                           'LocPeakMatrix_Python': self.peak_nums,
                                                           'C_df': self.traces, 'raw_traces': self.raw_traces,
                                                           'cells_to_remove': cells_to_remove,
                                                           'inter_neurons': inter_neurons})

        if and_close:
            self.root.destroy()

    def get_threshold(self):
        trace = self.traces[self.current_neuron, :]
        threshold = (self.nb_std_thresold * np.std(trace)) + np.min(self.traces[self.current_neuron, :])
        return threshold

    def plot_magnifier(self, first_time=False, mouse_x_position=None, mouse_y_position=None):
        if first_time:
            self.axe_plot_magnifier = self.magnifier_fig.add_subplot(111)

        if self.x_center_magnified is not None:
            pos_beg_x = int(np.max((0, self.x_center_magnified - self.magnifier_range)))
            pos_end_x = int(np.min((self.nb_times, self.x_center_magnified + self.magnifier_range + 1)))

            max_value = max(np.max(self.traces[self.current_neuron, pos_beg_x:pos_end_x]),
                            np.max(self.raw_traces[self.current_neuron, pos_beg_x:pos_end_x]))
            min_value = min(np.min(self.traces[self.current_neuron, pos_beg_x:pos_end_x]),
                            np.min(self.raw_traces[self.current_neuron, pos_beg_x:pos_end_x]))

            nb_times_to_display = pos_end_x - pos_beg_x

            color_trace = self.color_trace
            self.line1, = self.axe_plot_magnifier.plot(np.arange(nb_times_to_display),
                                                       self.traces[self.current_neuron, pos_beg_x:pos_end_x],
                                                       color=color_trace, zorder=8)

            if (self.raw_traces is not None) and self.display_raw_traces:
                self.axe_plot_magnifier.plot(np.arange(nb_times_to_display),
                                             self.raw_traces[self.current_neuron, pos_beg_x:pos_end_x],
                                             color=self.color_raw_trace, alpha=0.6, zorder=9)

            onsets = np.where(self.onset_times[self.current_neuron, pos_beg_x:pos_end_x] > 0)[0]
            # plotting onsets
            self.axe_plot_magnifier.vlines(onsets, min_value, max_value,
                                           color=self.color_onset, linewidth=1,
                                           linestyles="dashed")

            peaks = np.where(self.peak_nums[self.current_neuron, pos_beg_x:pos_end_x] > 0)[0]
            if len(peaks) > 0:
                reduced_traces = self.traces[self.current_neuron, pos_beg_x:pos_end_x]
                y_peaks = reduced_traces[peaks]
                self.axe_plot_magnifier.scatter(peaks, y_peaks,
                                                marker='o', c=self.color_peak,
                                                edgecolors=self.color_edge_peak, s=30,
                                                zorder=10)
            self.draw_magnifier_marker(mouse_x_position=mouse_x_position, mouse_y_position=mouse_y_position)
            self.axe_plot_magnifier.set_ylim(min_value, max_value + 1)

        if self.x_center_magnified is None:
            main_plot_bottom_y, main_plot_top_y = self.axe_plot.get_ylim()
            self.axe_plot_magnifier.set_ylim(main_plot_bottom_y, main_plot_top_y)

        self.axe_plot_magnifier.spines['right'].set_visible(False)
        self.axe_plot_magnifier.spines['top'].set_visible(False)
        if first_time:
            self.magnifier_fig.tight_layout()

    def draw_magnifier_marker(self, mouse_x_position=None, mouse_y_position=None):
        if (mouse_x_position is None) or (mouse_y_position is None):
            return

        pos_beg_x = int(np.max((0, self.x_center_magnified - self.magnifier_range)))
        pos_end_x = int(np.min((self.nb_times, self.x_center_magnified + self.magnifier_range + 1)))

        max_value = max(np.max(self.traces[self.current_neuron, pos_beg_x:pos_end_x]),
                        np.max(self.raw_traces[self.current_neuron, pos_beg_x:pos_end_x]))
        min_value = min(np.min(self.traces[self.current_neuron, pos_beg_x:pos_end_x]),
                        np.min(self.raw_traces[self.current_neuron, pos_beg_x:pos_end_x]))

        corrected_mouse_y_position = np.min((mouse_y_position, max_value))

        corrected_mouse_x_position = None
        corrected_mouse_y_position = None

        corrected_mouse_x_position = mouse_x_position - pos_beg_x

        if self.magnifier_marker is not None:
            self.magnifier_marker.set_visible(False)

        if self.magnifier_line is not None:
            self.magnifier_line.set_visible(False)

        self.magnifier_marker = self.axe_plot_magnifier.scatter(corrected_mouse_x_position,
                                                                corrected_mouse_y_position,
                                                                marker='x',
                                                                c="black", s=20)
        self.magnifier_line = self.axe_plot_magnifier.vlines(corrected_mouse_x_position, min_value, max_value,
                                                             color="red",
                                                             linewidth=1,
                                                             linestyles=":")

    def update_plot_magnifier(self, mouse_x_position, mouse_y_position, change_frame_ref):
        if change_frame_ref:
            self.magnifier_fig.clear()
            plt.close(self.magnifier_fig)
            self.magnifier_canvas.get_tk_widget().destroy()
            self.magnifier_fig = plt.figure(figsize=(4, 4))
            self.magnifier_canvas = FigureCanvasTkAgg(self.magnifier_fig, self.magnifier_frame)

            self.plot_magnifier(first_time=True, mouse_x_position=mouse_x_position,
                                mouse_y_position=mouse_y_position)

            self.magnifier_canvas.draw()
            self.magnifier_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=YES)
            # self.axe_plot_magnifier.clear()
            # self.plot_magnifier(first_time=False, mouse_x_position=mouse_x_position,
            #                     mouse_y_position=mouse_y_position)
        else:
            self.draw_magnifier_marker(mouse_x_position=mouse_x_position, mouse_y_position=mouse_y_position)

        self.magnifier_fig.canvas.draw()
        self.magnifier_fig.canvas.flush_events()

    def start_playing_movie(self, x_from, x_to):
        self.play_movie = True
        self.first_frame_movie = x_from
        self.last_frame_movie = x_to
        self.n_frames_movie = x_to - x_from
        self.movie_frames = cycle((frame_tiff, frame_index + x_from)
                                  for frame_index, frame_tiff in enumerate(self.tiff_movie[x_from:x_to]))
        self.update_plot_map_img()

    def animate_movie(self, i):
        if not self.play_movie:
            return []
        frame_tiff, frame_index = next(self.movie_frames)
        self.last_img_displayed.set_array(frame_tiff)
        if self.last_frame_label is not None:
            self.last_frame_label.set_visible(False)
        self.last_frame_label = self.axe_plot_map_img.text(x=10, y=10,
                                                           s=f"{frame_index}", color="red", zorder=20,
                                                           ha='center', va="center", fontsize=10, fontweight='bold')
        if self.trace_movie_p1 is not None:
            self.trace_movie_p1[0].set_visible(False)
        if self.trace_movie_p2 is not None:
            self.trace_movie_p2[0].set_visible(False)

        first_frame = self.first_frame_movie
        last_frame = self.last_frame_movie
        len_x = frame_tiff.shape[1]
        if self.n_frames_movie > len_x:
            new_x_values = np.linspace(0, len_x - 1, self.n_frames_movie)
        else:
            new_x_values = np.arange(self.n_frames_movie) + ((len_x - self.n_frames_movie) / 2)
        # back to zero with +2 then inceasing the amplitude
        raw_traces = (self.raw_traces + 2) * 5
        # y-axis is reverse, so we need to inverse the trace
        raw_traces = (frame_tiff.shape[1] * 0.9) - raw_traces
        # need to match the length of our trace to the len
        if first_frame < frame_index:
            self.trace_movie_p1 = self.axe_plot_map_img.plot(new_x_values[:frame_index - first_frame],
                                                             raw_traces[self.current_neuron, first_frame:frame_index],
                                                             color="red", alpha=1, zorder=10, lw=1)
        if last_frame > frame_index:
            self.trace_movie_p2 = self.axe_plot_map_img.plot(new_x_values[frame_index - first_frame:],
                                                             raw_traces[self.current_neuron, frame_index:last_frame],
                                                             color="white", alpha=1, zorder=10, lw=1)

        self.draw_cell_contour()
        artists = [self.last_img_displayed, self.last_frame_label, self.cell_contour]
        if self.trace_movie_p1 is not None:
            artists.append(self.trace_movie_p1[0])
        if self.trace_movie_p2 is not None:
            artists.append(self.trace_movie_p2[0])

        return artists

    def plot_map_img(self, first_time=True, do_blit=False):
        if (self.data_and_param.ms.avg_cell_map_img is None) and (not self.display_michou):
            return

        if first_time:
            self.axe_plot_map_img = self.map_img_fig.add_subplot(111)
            if do_blit:
                self.background_map_fig = self.map_img_fig.canvas.copy_from_bbox(self.axe_plot_map_img.bbox)
        # if self.last_img_displayed is not None:
        #     self.last_img_displayed.set_visible(False)

        if (not first_time) and do_blit:
            # restore background
            self.map_img_fig.canvas.restore_region(self.background_map_fig)

        if self.display_michou:
            if first_time:
                self.last_img_displayed = self.axe_plot_map_img.imshow(self.michou_imgs[self.michou_img_to_display])
            else:
                self.last_img_displayed.set_array(self.michou_imgs[self.michou_img_to_display])
        else:
            if self.play_movie:
                # frame_tiff is numpy array of 2D
                frame_tiff, frame_index = next(self.movie_frames)
                self.last_img_displayed = self.axe_plot_map_img.imshow(frame_tiff, cmap=plt.get_cmap('gray'))
                if self.last_frame_label is not None:
                    self.last_frame_label.set_visible(False)
                self.last_frame_label = self.axe_plot_map_img.text(x=10, y=10,
                                                                   s=f"{frame_index}", color="red", zorder=20,
                                                                   ha='center', va="center", fontsize=10,
                                                                   fontweight='bold')
            else:
                if self.last_frame_label is not None:
                    self.last_frame_label.set_visible(False)
                    self.last_frame_label = None
                if first_time:
                    self.last_img_displayed = self.axe_plot_map_img.imshow(self.data_and_param.ms.avg_cell_map_img,
                                                                           cmap=plt.get_cmap('gray'))
                else:
                    self.last_img_displayed.set_array(self.data_and_param.ms.avg_cell_map_img)
                self.last_img_displayed.set_zorder(1)
            # self.last_img_displayed.set_visible(True)
            self.draw_cell_contour()

        if (not first_time) and do_blit:
            # fill in the axes rectangle
            self.map_img_fig.canvas.blit(self.axe_plot_map_img.bbox)

        if first_time:
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)

            self.map_img_fig.tight_layout()

    def draw_cell_contour(self):
        if self.cell_contour is not None:
            self.cell_contour.set_visible(False)
        self.cell_contour = self.cell_contours[self.current_neuron]
        self.cell_contour.set_visible(True)
        self.axe_plot_map_img.add_patch(self.cell_contour)
        # # cell contour
        # coord = self.data_and_param.ms.coord_obj.coord[self.current_neuron]
        # coord = coord - 1
        # # c_filtered = c.astype(int)
        # n_coord = len(coord[0, :])
        # xy = np.zeros((n_coord, 2))
        # for n in np.arange(n_coord):
        #     xy[n, 0] = coord[0, n]
        #     xy[n, 1] = coord[1, n]
        # self.cell_contour = patches.Polygon(xy=xy,
        #                                     fill=False, linewidth=0, facecolor="red",
        #                                     edgecolor="red",
        #                                     zorder=15, lw=2)
        # self.axe_plot_map_img.add_patch(self.cell_contour)

    def update_plot_map_img(self, after_michou=False, do_blit=False):
        if self.display_michou and (not self.play_movie):
            self.axe_plot_map_img.clear()
            self.plot_map_img(do_blit=do_blit)
        elif self.play_movie:
            # self.axe_plot_map_img.clear()
            self.plot_map_img(do_blit=do_blit)
            self.map_img_fig.canvas.draw()
            self.map_img_fig.canvas.flush_events()
            self.anim_movie = animation.FuncAnimation(self.map_img_fig, func=self.animate_movie,
                                                      frames=self.n_frames_movie,
                                                      blit=True, interval=50, repeat=True)  # repeat=True,
            return
            # self.after(self.movie_delay, self.update_plot_map_img)
        else:
            if after_michou:
                self.axe_plot_map_img.clear()
                self.plot_map_img(do_blit=do_blit)
            else:
                self.draw_cell_contour()

        if not do_blit:
            self.map_img_fig.canvas.draw()
            self.map_img_fig.canvas.flush_events()

    def plot_graph(self, y_max_lim=None, first_time=False):
        """

        :param y_max_lim: used for axvspan in order to fit it to max amplitude according to y_max_lim
        :param first_time:
        :return:
        """

        max_amplitude = np.max(self.traces[self.current_neuron, :])

        if first_time:
            gs_index = 0
            self.axe_plot = self.fig.add_subplot(self.gs[gs_index])
            gs_index += 1
            y_max_lim = math.ceil(np.max(self.traces))
        color_trace = self.color_trace
        self.line1, = self.axe_plot.plot(np.arange(self.nb_times_traces), self.traces[self.current_neuron, :],
                                         color=color_trace, zorder=10)
        if not self.raw_traces_seperate_plot:
            if (self.raw_traces is not None) and self.display_raw_traces:
                self.axe_plot.plot(np.arange(self.nb_times_traces),
                                   self.raw_traces[self.current_neuron, :],
                                   color=self.color_raw_trace, alpha=0.8, zorder=9)
            if self.raw_traces_binned is not None:
                self.axe_plot.plot(np.arange(0, self.nb_times_traces, 10),
                                   self.raw_traces_binned[self.current_neuron, :],
                                   color="green", alpha=0.9, zorder=8)
                # O y-axis line
                # self.axe_plot.hlines(0, 0, self.nb_times_traces - 1, color="black", linewidth=1)
        onsets = np.where(self.onset_times[self.current_neuron, :] > 0)[0]
        max_value = max(np.max(self.traces[self.current_neuron, :]), np.max(self.raw_traces[self.current_neuron, :]))
        min_value = min(np.min(self.traces[self.current_neuron, :]), np.min(self.raw_traces[self.current_neuron, :]))
        if self.raw_traces_binned is not None:
            min_value = min(min_value, np.min(self.raw_traces_binned[self.current_neuron, :]))
        # plotting onsets
        # self.ax1_bottom_scatter = self.axe_plot.scatter(onsets, [0.1] * len(onsets), marker='*', c=self.color_onset, s=20)
        self.axe_plot.vlines(onsets, min_value, max_value, color=self.color_onset, linewidth=1,
                             linestyles="dashed")

        peaks = np.where(self.peak_nums[self.current_neuron, :] > 0)[0]
        if self.display_threshold:
            threshold = self.get_threshold()
            peaks_under_threshold = np.where(self.traces[self.current_neuron, peaks] < threshold)[0]
            peaks_under_threshold_index = peaks[peaks_under_threshold]
            peaks_under_threshold_value = self.traces[self.current_neuron, peaks][peaks_under_threshold]
            # peaks_under_threshold_value_raw = self.raw_traces[self.current_neuron, peaks][peaks_under_threshold]
            peaks_over_threshold = np.where(self.traces[self.current_neuron, peaks] >= threshold)[0]
            peaks_over_threshold_index = peaks[peaks_over_threshold]
            peaks_over_threshold_value = self.traces[self.current_neuron, peaks][peaks_over_threshold]
            # peaks_over_threshold_value_raw = self.raw_traces[self.current_neuron, peaks][peaks_over_threshold]
            # plotting peaks
            # z_order=10 indicate that the scatter will be on top
            self.ax1_bottom_scatter = self.axe_plot.scatter(peaks_over_threshold_index, peaks_over_threshold_value,
                                                            marker='o', c=self.color_peak,
                                                            edgecolors=self.color_edge_peak, s=30, zorder=10)
            self.ax1_bottom_scatter = self.axe_plot.scatter(peaks_under_threshold_index, peaks_under_threshold_value,
                                                            marker='o', c=self.color_peak_under_threshold,
                                                            edgecolors=self.color_edge_peak, s=30, zorder=10)
            # self.ax1_bottom_scatter = self.axe_plot.scatter(peaks_over_threshold_index, peaks_over_threshold_value_raw,
            #                                                 marker='o', c=self.color_peak,
            #                                                 edgecolors=self.color_edge_peak, s=30, zorder=10)
            # self.ax1_bottom_scatter = self.axe_plot.scatter(peaks_under_threshold_index,
            #                                                 peaks_under_threshold_value_raw,
            #                                                 marker='o', c=self.color_peak_under_threshold,
            #                                                 edgecolors=self.color_edge_peak, s=30, zorder=10)

            self.axe_plot.hlines(threshold, 0, self.nb_times_traces - 1, color=self.color_threshold_line, linewidth=1,
                                 linestyles="dashed")
        else:
            # plotting peaks
            # z_order=10 indicate that the scatter will be on top
            self.ax1_bottom_scatter = self.axe_plot.scatter(peaks, self.traces[self.current_neuron, peaks],
                                                            marker='o', c=self.color_peak,
                                                            edgecolors=self.color_edge_peak, s=30, zorder=10)
            # self.ax1_bottom_scatter = self.axe_plot.scatter(peaks, self.raw_traces[self.current_neuron, peaks],
            #                                                 marker='o', c=self.color_peak,
            #                                                 edgecolors=self.color_edge_peak, s=30, zorder=10)
        # not plotting top scatter of onsets
        # self.ax1_top_scatter = self.axe_plot.scatter(onsets, [max_value] * len(onsets),
        # marker='*', c=self.color_onset, s=40)
        if self.first_click_to_remove is not None:
            self.axe_plot.scatter(self.first_click_to_remove["x"], self.first_click_to_remove["y"], marker='x',
                                  c=self.color_mark_to_remove, s=20)

        if (self.mvt_frames_periods is not None) and self.display_mvt:
            for mvt_frames_period in self.mvt_frames_periods:
                # print(f"mvt_frames_period[0], mvt_frames_period[1] {mvt_frames_period[0]} {mvt_frames_period[1]}")
                self.axe_plot.axvspan(mvt_frames_period[0], mvt_frames_period[1], ymax=1,
                                      alpha=0.8, facecolor="red", zorder=1)

        # by default the y axis zoom is set to fit the wider amplitude of the current neuron
        fit_plot_to_all_max = False
        if fit_plot_to_all_max:
            if self.display_raw_traces and (not self.raw_traces_seperate_plot):
                min_value = min(0, np.min(self.raw_traces[self.current_neuron, :]))
                self.axe_plot.set_ylim(min_value, math.ceil(np.max(self.traces)))
            else:
                self.axe_plot.set_ylim(0, math.ceil(np.max(self.traces)))
        else:
            max_value = max(np.max(self.raw_traces[self.current_neuron, :]),
                            np.max(self.traces[self.current_neuron, :]))
            min_value = min(np.min(self.raw_traces[self.current_neuron, :]),
                            np.min(self.traces[self.current_neuron, :]))
            if self.raw_traces_binned is not None:
                min_value = min(min_value, np.min(self.raw_traces_binned[self.current_neuron, :]))

            self.axe_plot.set_ylim(min_value,
                                   math.ceil(max_value))

        # ------------ RAW TRACE own plot----------------------
        if self.raw_traces_seperate_plot and (self.raw_traces is not None):
            if first_time:
                self.axe_plot_raw = self.fig.add_subplot(self.gs[gs_index], sharex=self.axe_plot)
                gs_index += 1

            self.axe_plot_raw.plot(np.arange(self.nb_times_traces),
                                   self.raw_traces[self.current_neuron, :],
                                   color=self.color_raw_trace)
            max_value = np.max(self.raw_traces[self.current_neuron, :])
            min_value = np.min(self.raw_traces[self.current_neuron, :])
            self.axe_plot_raw.vlines(onsets, min_value, max_value, color=self.color_onset, linewidth=1,
                                     linestyles="dashed")

            self.axe_plot.set_ylim(np.min(self.raw_traces[self.current_neuron, :]),
                                   math.ceil(np.max(self.raw_traces[self.current_neuron, :])))

        if first_time:
            self.axe2_plot = self.fig.add_subplot(self.gs[gs_index], sharex=self.axe_plot)
        self.line2, = self.axe2_plot.plot(np.arange(self.nb_times_traces), self.activity_count,
                                          color=self.color_trace_activity)
        max_value = np.max(self.activity_count)
        self.axe2_plot.vlines(onsets, 0, max_value, color=self.color_onset, linewidth=1,
                              linestyles="dashed")

        # removing first x_axis
        axes_to_clean = [self.axe_plot]
        if self.raw_traces_seperate_plot and (self.raw_traces is not None):
            axes_to_clean.append(self.axe_plot_raw)
        for ax in axes_to_clean:
            ax.axes.get_xaxis().set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        self.axe2_plot.spines['right'].set_visible(False)
        self.axe2_plot.spines['top'].set_visible(False)
        self.fig.tight_layout()

    # not used
    # def update_plot_wrong(self):
    #     self.line1.set_data(np.arange(self.nb_times_traces), self.traces[self.current_neuron, :])
    #     self.line2.set_data(np.arange(self.nb_times_traces), self.activity_count)
    #     onsets = np.where(self.onset_times[self.current_neuron, :] > 0)[0]
    #     max_value = np.max(self.traces[self.current_neuron, :])
    #     data = np.hstack((onsets, [0.1] * len(onsets)))
    #     self.ax1_bottom_scatter.set_offsets(data)
    #     data2 = np.hstack((onsets, [max_value] * len(onsets)))
    #     self.ax1_top_scatter.set_offsets(data2)
    #     self.fig.canvas.flush_events()

    def current_max_amplitude(self):
        """
        Ceiling value
        :return:
        """
        return math.ceil(np.max(self.traces[self.current_neuron, :]))

    def move_zoom(self, to_the_left):
        # the plot is zoom out to the max, so no moving
        left_x_limit_1, right_x_limit_1 = self.axe_plot.get_xlim()
        # print(f"left_x_limit_1 {left_x_limit_1}, right_x_limit_1 {right_x_limit_1}, "
        #       f"self.nb_times_traces {self.nb_times_traces}")
        if (left_x_limit_1 <= 0) and (right_x_limit_1 >= (self.nb_times_traces - 1)):
            return

        # moving the windown to the right direction keeping 10% of the window in the new one
        length_window = right_x_limit_1 - left_x_limit_1
        if to_the_left:
            if left_x_limit_1 <= 0:
                return
            new_right_x_limit = int(left_x_limit_1 + (0.1 * length_window))
            new_left_x_limit = new_right_x_limit - length_window
            new_left_x_limit = max(new_left_x_limit, 0)
            if new_right_x_limit <= new_left_x_limit:
                return
        else:
            if right_x_limit_1 >= (self.nb_times_traces - 1):
                return
            new_left_x_limit = int(right_x_limit_1 - (0.1 * length_window))
            new_right_x_limit = new_left_x_limit + length_window
            new_right_x_limit = min(new_right_x_limit, self.nb_times_traces - 1)
            if new_left_x_limit >= new_right_x_limit:
                return

        new_x_limit = (new_left_x_limit, new_right_x_limit)
        self.update_plot(new_x_limit=new_x_limit)

    def update_plot(self, new_neuron=False, amplitude_zoom_fit=True,
                    new_x_limit=None, new_y_limit=None,
                    raw_trace_display_action=False):

        # used to keep the same zoom after updating the plot
        # if we change neuron, then back to no zoom mode
        left_x_limit_1, right_x_limit_1 = self.axe_plot.get_xlim()
        left_x_limit_2, right_x_limit_2 = self.axe2_plot.get_xlim()
        bottom_limit_1, top_limit_1 = self.axe_plot.get_ylim()
        bottom_limit_2, top_limit_2 = self.axe2_plot.get_ylim()
        if self.raw_traces_seperate_plot and (self.raw_traces is not None):
            bottom_limit_raw, top_limit_raw = self.axe_plot_raw.get_ylim()
        self.axe_plot.clear()
        self.axe2_plot.clear()
        if self.raw_traces_seperate_plot and (self.raw_traces is not None):
            self.axe_plot_raw.clear()
        # self.line1.set_ydata(self.traces[self.current_neuron, :])
        if amplitude_zoom_fit:
            y_max_lim = self.current_max_amplitude()
        elif new_neuron:
            y_max_lim = math.ceil(np.max(self.traces))
        else:
            y_max_lim = top_limit_1

        self.plot_graph(y_max_lim)

        # to keep the same zoom
        if not new_neuron:
            self.axe_plot.set_xlim(left=left_x_limit_1, right=right_x_limit_1, auto=None)
            self.axe2_plot.set_xlim(left=left_x_limit_2, right=right_x_limit_2, auto=None)
            self.axe_plot.set_ylim(bottom=bottom_limit_1, top=top_limit_1, auto=None)
            self.axe2_plot.set_ylim(bottom=bottom_limit_2, top=top_limit_2, auto=None)
            # if amplitude_zoom_fit:
            #     if self.display_raw_traces and (self.raw_traces is not None):
            #         self.axe_plot.set_ylim(min(0, np.min(self.raw_traces[self.current_neuron, :])),
            #                                self.current_max_amplitude())
            #     else:
            #         self.axe_plot.set_ylim(0, self.current_max_amplitude())
            # else:
            #     if not raw_trace_display_action:
            #         self.axe_plot.set_ylim(bottom=bottom_limit_1, top=top_limit_1, auto=None)
            #     self.axe2_plot.set_ylim(bottom=bottom_limit_2, top=top_limit_2, auto=None)
            #     if self.raw_traces_seperate_plot and (self.raw_traces is not None):
            #         self.axe_plot_raw.set_ylim(bottom=bottom_limit_raw, top=top_limit_raw,
            #                                    auto=None)
        if new_x_limit is not None:
            self.axe_plot.set_xlim(left=new_x_limit[0], right=new_x_limit[1], auto=None)
            self.axe2_plot.set_xlim(left=new_x_limit[0], right=new_x_limit[1], auto=None)
        if (new_y_limit is not None) and (not amplitude_zoom_fit):
            self.axe_plot.set_ylim(new_y_limit[0], new_y_limit[1])
        # self.line1.set_ydata(self.traces[self.current_neuron, :])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # def spin_box_update(self):
    #     content = int(self.spin_box_button.get())
    #     if content != self.current_neuron:
    #         self.current_neuron = content
    #         self.update_plot()

    # if an onset has been removed or added to traces and spike_nums for current_neuron
    def update_after_onset_change(self, new_neuron=-1,
                                  new_x_limit=None, new_y_limit=None):
        """
        Update the frame is an onset change has been made
        :param new_neuron: if -1, then the neuron hasn't changed, neuron might change if undo or redo are done.
        :return:
        """
        if new_neuron > -1:
            self.current_neuron = new_neuron
        self.onset_numbers_label["text"] = f"{self.numbers_of_onset()}"
        self.peak_numbers_label["text"] = f"{self.numbers_of_peak()}"
        # array of 1 D, representing the number of spikes at each time
        tmp_activity_count = np.sum(self.spike_nums, axis=0)
        # dimension reduction in order to fit to traces times for activity count (spikes count)
        split_result = np.split(tmp_activity_count, self.nb_times_traces)
        self.activity_count = np.sum(split_result, axis=1)
        if new_neuron > -1:
            self.update_neuron(new_neuron=new_neuron,
                               new_x_limit=new_x_limit, new_y_limit=new_y_limit)
        else:
            self.update_plot(new_x_limit=new_x_limit, new_y_limit=new_y_limit)

    def select_previous_neuron(self):
        if self.current_neuron == 0:
            return

        if (self.current_neuron - 1) == 0:
            self.prev_button['state'] = DISABLED

        self.update_neuron(new_neuron=(self.current_neuron - 1))

        self.next_button['state'] = 'normal'

        # self.spin_box_button.invoke("buttondown")

    def select_next_neuron(self):
        if self.current_neuron == (self.nb_neurons - 1):
            return
        self.prev_button['state'] = 'normal'
        if (self.current_neuron + 1) == (self.nb_neurons - 1):
            self.next_button['state'] = DISABLED
        self.update_neuron(new_neuron=(self.current_neuron + 1))

        # self.spin_box_button.invoke("buttonup")

    # TODO: add the neuron number, and do all the changes here, including spinbox, disable next and previous button
    def update_neuron(self, new_neuron,
                      new_x_limit=None, new_y_limit=None):
        """
        Call when the neuron number has changed
        :return:
        """
        self.current_neuron = new_neuron
        self.neuron_label["text"] = f"{self.current_neuron}"
        # self.spin_box_button.icursor(new_neuron)
        self.clear_and_update_entry_neuron_widget()
        self.onset_numbers_label["text"] = f"{self.numbers_of_onset()}"
        self.peak_numbers_label["text"] = f"{self.numbers_of_peak()}"

        if (self.current_neuron + 1) == self.nb_neurons:
            self.next_button['state'] = DISABLED
        else:
            self.next_button['state'] = "normal"

        if self.current_neuron == 0:
            self.prev_button['state'] = DISABLED
        else:
            self.prev_button['state'] = "normal"

        if self.inter_neurons[self.current_neuron] == 0:
            self.inter_neuron_button["text"] = ' not IN '
            self.inter_neuron_button["fg"] = "black"
        else:

            self.inter_neuron_button["text"] = ' IN '
            self.inter_neuron_button["fg"] = "red"

        if self.cells_to_remove[self.current_neuron] == 0:
            self.remove_cell_button["text"] = ' not removed '
            self.remove_cell_button["fg"] = "black"
        else:
            self.remove_cell_button["text"] = ' removed '
            self.remove_cell_button["fg"] = "red"

        self.update_plot(new_neuron=True,
                         new_x_limit=new_x_limit, new_y_limit=new_y_limit)
        self.update_plot_map_img()


def print_save(text, file, to_write, no_print=False):
    if not no_print:
        print(text)
    if to_write:
        file.write(text + '\n')


def main_manual():
    print(f'platform: {platform}')
    root = Tk()
    root.title(f"Session selection")

    root_path = None
    with open("param_hne.txt", "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split('=')
            root_path = line_list[1]
    if root_path is None:
        raise Exception("Root path is None")

    path_data = root_path + "data/"

    data_and_param = DataAndParam(result_path=path_data)

    app = ChooseSessionFrame(data_and_param=data_and_param,
                             master=root)
    app.mainloop()
    # root.destroy()


main_manual()
