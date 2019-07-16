# Ignore warnings
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import sys
from datetime import datetime
import hdf5storage
import matplotlib
from sys import platform
# import cv2
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats
import time
from matplotlib.figure import SubplotParams

matplotlib.use("TkAgg")
import scipy.io as sio
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# from PIL import ImageTk, Image
import math
import matplotlib.gridspec as gridspec
# import matplotlib.image as mpimg
from mouse_session_loader import load_mouse_sessions
import pattern_discovery.tools.param as p_disc_tools_param

from pattern_discovery.display.misc import plot_hist_with_first_perc_and_eb
from pattern_discovery.tools.signal import smooth_convolve
from matplotlib import patches
import scipy.signal as signal
import matplotlib.image as mpimg
from random import randint
import scipy.ndimage.morphology as morphology
import os
from PIL import ImageDraw
import PIL
from shapely import geometry
# from PIL import ImageTk
from itertools import cycle
from matplotlib import animation
import matplotlib.gridspec as gridspec

from cell_classifier import predict_cell_from_saved_model
from transient_classifier import predict_transient_from_saved_model
from pattern_discovery.tools.misc import get_continous_time_periods

import classification_stat

if sys.version_info[0] < 3:
    import Tkinter as tk
    from Tkinter import *
    from Tkinter import ttk
else:
    import tkinter as tk
    import tkinter.filedialog as filedialog
    from tkinter import *
    from tkinter import ttk


# ---------- code for function: event_lambda (begin) --------
def event_lambda(f, *args, **kwds):
    return lambda f=f, args=args, kwds=kwds: f(*args, **kwds)


# ---------- code for function: event_lambda (end) -----------

# see to replace it by messagebox.showerror("Error", "Error message")
# from tkinter import messagebox
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
    def __init__(self, path_data, result_path):
        self.time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        super().__init__(path_results=result_path, time_str=self.time_str, bin_size=1)
        self.cell_assemblies_data_path = None
        self.best_order_data_path = None
        self.spike_nums = None
        self.peak_nums = None
        self.doubtful_frames_nums = None
        self.mvt_frames_nums = None
        self.to_agree_peak_nums = None
        self.to_agree_spike_nums = None
        self.traces = None
        self.raw_traces = None
        self.ms = None
        self.result_path = result_path
        self.path_data = path_data
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

        self.available_ms_str = ["artificial_ms",
                                 "p5_19_03_25_a000_ms", "p5_19_03_25_a001_ms",
                                 "p6_18_02_07_a001_ms", "p6_18_02_07_a002_ms",
                                 "p6_19_02_18_a000_ms",
                                 "p7_171012_a000_ms",
                                 "p7_17_10_18_a002_ms", "p7_17_10_18_a004_ms",
                                 "p7_18_02_08_a000_ms", "p7_18_02_08_a001_ms",
                                 "p7_18_02_08_a002_ms", "p7_18_02_08_a003_ms",
                                 "p7_19_03_05_a000_ms",
                                 "p7_19_03_27_a000_ms", "p7_19_03_27_a001_ms",
                                 "p7_19_03_27_a002_ms",
                                 "p8_18_02_09_a000_ms", "p8_18_02_09_a001_ms",
                                 "p8_18_10_17_a000_ms", "p8_18_10_17_a001_ms",
                                 "p8_18_10_24_a005_ms", "p8_18_10_24_a006_ms",
                                 "p8_19_03_19_a000_ms",
                                 "p9_17_12_06_a001_ms", "p9_17_12_20_a001_ms",
                                 "p9_18_09_27_a003_ms", "p9_19_02_20_a000_ms",
                                 "p9_19_02_20_a001_ms", "p9_19_02_20_a002_ms",
                                 "p9_19_02_20_a003_ms", "p9_19_03_14_a000_ms",
                                 "p9_19_03_14_a001_ms", "p9_19_03_22_a000_ms",
                                 "p9_19_03_22_a001_ms",
                                 "p10_17_11_16_a003_ms", "p10_19_02_21_a002_ms",
                                 "p10_19_02_21_a003_ms", "p10_19_02_21_a005_ms",
                                 "p10_19_03_08_a000_ms", "p10_19_03_08_a001_ms",
                                 "p11_17_11_24_a000_ms", "p11_17_11_24_a001_ms",
                                 "p11_19_02_15_a000_ms", "p11_19_02_22_a000_ms",
                                 "p11_19_04_30_a001_ms",
                                 "p12_17_11_10_a002_ms", "p12_171110_a000_ms",
                                 "p13_18_10_29_a000_ms", "p13_18_10_29_a001_ms",
                                 "p13_19_03_11_a000_ms",
                                 "p14_18_10_23_a000_ms", "p14_18_10_30_a001_ms",
                                 "p16_18_11_01_a002_ms",
                                 "p19_19_04_08_a000_ms", "p19_19_04_08_a001_ms",
                                 "p21_19_04_10_a000_ms", "p21_19_04_10_a001_ms",
                                 "p21_19_04_10_a000_j3_ms",
                                 "p41_19_04_30_a000_ms",
                                 "richard_015_D74_P2_ms", "richard_028_D2_P1_ms",
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
                                                param=self.data_and_param, for_cell_classifier=False,
                                                load_traces=True, load_abf=False)
        self.data_and_param.ms = ms_str_to_ms_dict[self.option_menu_variable.get()]
        self.data_and_param.ms.load_tiff_movie_in_memory()

        if self.data_and_param.ms.raw_traces is not None:
            self.data_and_param.raw_traces = self.data_and_param.ms.raw_traces
        else:
            self.data_and_param.raw_traces = self.data_and_param.ms.build_raw_traces_from_movie()
            print(f"raw_traces.shape {self.data_and_param.raw_traces.shape}")
        if self.data_and_param.ms.traces is not None:
            self.data_and_param.traces = self.data_and_param.ms.traces
        else:
            traces = np.copy(self.data_and_param.raw_traces)
            do_traces_smoothing(traces)
            self.data_and_param.traces = traces

        n_cells = len(self.data_and_param.traces)
        n_times = len(self.data_and_param.traces[0, :])

        keep_onsets_and_peaks_empty = False
        if keep_onsets_and_peaks_empty and (self.data_and_param.peak_nums is None):
            self.data_and_param.peak_nums = np.zeros((n_cells, n_times), dtype="int8")
            self.data_and_param.spike_nums = np.zeros((n_cells, n_times), dtype="int8")
        elif (self.data_and_param.peak_nums is None) or (self.data_and_param.spike_nums is None):
            using_existing_spike_nums_dur = False
            # look if spike_nums_dur exists
            if using_existing_spike_nums_dur and (self.data_and_param.ms.spike_struct.spike_nums_dur is not None):
                self.data_and_param.ms.spike_struct.build_spike_nums_and_peak_nums()
                self.data_and_param.peak_nums = self.data_and_param.ms.spike_struct.peak_nums
                self.data_and_param.spike_nums = self.data_and_param.ms.spike_struct.spike_nums
            else:
                self.data_and_param.peak_nums = np.zeros((n_cells, n_times), dtype="int8")
                self.data_and_param.spike_nums = np.zeros((n_cells, n_times), dtype="int8")
                # then we do an automatic detection
                for cell in np.arange(n_cells):
                    peaks, properties = signal.find_peaks(x=self.data_and_param.traces[cell], distance=2)
                    # print(f"peaks {peaks}")
                    self.data_and_param.peak_nums[cell, peaks] = 1

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
            initial_dir = self.data_and_param.path_data
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
                if "doubtful_frames_nums" in data_file:
                    data_and_param.doubtful_frames_nums = data_file['doubtful_frames_nums'].astype(int)
                if "mvt_frames_nums" in data_file:
                    data_and_param.mvt_frames_nums = data_file['mvt_frames_nums'].astype(int)
                if "inter_neurons" in data_file:
                    data_and_param.inter_neurons = data_file['inter_neurons'].astype(int)
                if "cells_to_remove" in data_file:
                    data_and_param.cells_to_remove = data_file['cells_to_remove'].astype(int)
                if "to_agree_peak_nums" in data_file:
                    data_and_param.to_agree_peak_nums = data_file['to_agree_peak_nums'].astype(int)
                if "to_agree_spike_nums" in data_file:
                    data_and_param.to_agree_spike_nums = data_file['to_agree_spike_nums'].astype(int)
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


class AgreePeakAction(ManualAction):
    def __init__(self, agreed_peaks_index, agree_onset_action, **kwargs):
        super().__init__(**kwargs)
        self.agreed_peaks_index = agreed_peaks_index
        self.agree_onset_action = agree_onset_action

    def undo(self):
        super().undo()
        self.session_frame.to_agree_peak_nums[self.neuron, self.agreed_peaks_index] = 1
        self.session_frame.peak_nums[self.neuron, self.agreed_peaks_index] = 0
        if self.agree_onset_action is not None:
            self.agree_onset_action.undo()
        else:
            self.session_frame.update_to_agree_label()

    def redo(self):
        super().redo()
        self.session_frame.to_agree_peak_nums[self.neuron, self.agreed_peaks_index] = 0
        self.session_frame.peak_nums[self.neuron, self.agreed_peaks_index] = 1
        if self.agree_onset_action is not None:
            self.agree_onset_action.redo()
        else:
            self.session_frame.update_to_agree_label()


class DontAgreePeakAction(ManualAction):
    def __init__(self, not_agreed_peaks_index, dont_agree_onset_action, **kwargs):
        super().__init__(**kwargs)
        self.not_agreed_peaks_index = not_agreed_peaks_index
        self.dont_agree_onset_action = dont_agree_onset_action

    def undo(self):
        super().undo()
        self.session_frame.to_agree_peak_nums[self.neuron, self.not_agreed_peaks_index] = 1
        if self.dont_agree_onset_action is not None:
            self.dont_agree_onset_action.undo()
        else:
            self.session_frame.update_to_agree_label()

    def redo(self):
        super().redo()
        self.session_frame.to_agree_peak_nums[self.neuron, self.not_agreed_peaks_index] = 0
        if self.dont_agree_onset_action is not None:
            self.dont_agree_onset_action.redo()
        else:
            self.session_frame.update_to_agree_label()


class DontAgreeOnsetAction(ManualAction):
    def __init__(self, not_agreed_onsets_index, **kwargs):
        super().__init__(**kwargs)
        self.not_agreed_onsets_index = not_agreed_onsets_index

    def undo(self):
        super().undo()
        self.session_frame.to_agree_spike_nums[self.neuron, self.not_agreed_onsets_index] = 1
        self.session_frame.update_to_agree_label()

    def redo(self):
        super().redo()
        self.session_frame.to_agree_spike_nums[self.neuron, self.not_agreed_onsets_index] = 0
        self.session_frame.update_to_agree_label()


class AgreeOnsetAction(ManualAction):
    def __init__(self, agreed_onsets_index, **kwargs):
        super().__init__(**kwargs)
        self.agreed_onsets_index = agreed_onsets_index

    def undo(self):
        super().undo()
        self.session_frame.to_agree_spike_nums[self.neuron, self.agreed_onsets_index] = 1
        self.session_frame.onset_times[self.neuron, self.agreed_onsets_index] = 0
        self.session_frame.spike_nums[self.neuron, self.agreed_onsets_index] = 0
        self.session_frame.update_to_agree_label()

    def redo(self):
        super().redo()
        self.session_frame.to_agree_spike_nums[self.neuron, self.agreed_onsets_index] = 0
        self.session_frame.onset_times[self.neuron, self.agreed_onsets_index] = 1
        self.session_frame.spike_nums[self.neuron, self.agreed_onsets_index] = 1
        self.session_frame.update_to_agree_label()


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


class AddDoubtfulFramesAction(ManualAction):
    def __init__(self, x_from, x_to, backup_values, **kwargs):
        super().__init__(**kwargs)
        self.x_from = x_from
        self.x_to = x_to
        self.backup_values = backup_values

    def undo(self):
        super().undo()
        self.session_frame.doubtful_frames_nums[self.neuron, self.x_from:self.x_to] = self.backup_values
        self.session_frame.update_doubtful_frames_periods(cell=self.neuron)

    def redo(self):
        super().redo()
        self.session_frame.doubtful_frames_nums[self.neuron, self.x_from:self.x_to] = 1
        self.session_frame.update_doubtful_frames_periods(cell=self.neuron)


class RemoveDoubtfulFramesAction(ManualAction):
    def __init__(self, removed_times, **kwargs):
        super().__init__(**kwargs)
        self.removed_times = removed_times

    def undo(self):
        super().undo()
        self.session_frame.doubtful_frames_nums[self.neuron, self.removed_times] = 1
        self.session_frame.update_doubtful_frames_periods(cell=self.neuron)

    def redo(self):
        super().redo()
        self.session_frame.doubtful_frames_nums[self.neuron, self.removed_times] = 0
        self.session_frame.update_doubtful_frames_periods(cell=self.neuron)


class AddMvtFramesAction(ManualAction):
    def __init__(self, x_from, x_to, backup_values, **kwargs):
        super().__init__(**kwargs)
        self.x_from = x_from
        self.x_to = x_to
        self.backup_values = backup_values

    def undo(self):
        super().undo()
        self.session_frame.mvt_frames_nums[self.neuron, self.x_from:self.x_to] = self.backup_values
        self.session_frame.update_mvt_frames_periods(cell=self.neuron)

    def redo(self):
        super().redo()
        self.session_frame.mvt_frames_nums[self.neuron, self.x_from:self.x_to] = 1
        self.session_frame.update_mvt_frames_periods(cell=self.neuron)


class RemoveMvtFramesAction(ManualAction):
    def __init__(self, removed_times, **kwargs):
        super().__init__(**kwargs)
        self.removed_times = removed_times

    def undo(self):
        super().undo()
        self.session_frame.mvt_frames_nums[self.neuron, self.removed_times] = 1
        self.session_frame.update_mvt_frames_periods(cell=self.neuron)

    def redo(self):
        super().redo()
        self.session_frame.mvt_frames_nums[self.neuron, self.removed_times] = 0
        self.session_frame.update_mvt_frames_periods(cell=self.neuron)


def get_file_name_and_path(path_file):
    # to get real index, remove 1
    last_slash_index = len(path_file) - path_file[::-1].find("/")
    if last_slash_index == -1:
        return None, None,

    # return path and file_name
    return path_file[:last_slash_index], path_file[last_slash_index:]


def do_traces_smoothing(traces):
    # TODO: try butternot filter for a high band filter and then use signal.filtfilt for shifting the smooth signal.
    # smoothing the trace
    windows = ['hanning', 'hamming', 'bartlett', 'blackman']
    i_w = 1
    window_length = 7  # 11
    for i in np.arange(traces.shape[0]):
        smooth_signal = smooth_convolve(x=traces[i], window_len=window_length,
                                        window=windows[i_w])
        beg = (window_length - 1) // 2
        traces[i] = smooth_signal[beg:-beg]


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
        # TODO: put option for different color set (skins), use http://colorbrewer2.org
        self.color_onset = "#225ea8"  # "darkblue" #"darkgoldenrod" #"dimgrey"
        self.color_trace = "#ffffcc"  # "white"
        self.color_early_born = "darkgreen"
        self.color_peak = "#41b6c4"  # "blue"
        self.color_edge_peak = "#a1dab4"  # "white"
        self.color_peak_under_threshold = "red"
        self.color_threshold_line = "red"  # "cornflowerblue"
        self.color_mark_to_remove = "white"
        self.color_run_period = "lightcoral"
        # self.color_raw_trace = "darkgoldenrod"
        self.color_raw_trace = "#a1dab4"  # "cornflowerblue"
        self.classifier_filling_color = "#2c7fb8"
        self.color_transient_predicted_to_look_at = "#ffffcc"
        self.color_bg_predicted_transient_list_box = "#41b6c4"
        self.color_text_gui_default = "#225ea8"
        # ------------- colors (end) --------

        # filename on which to save spikenums, is defined when save as is clicked
        self.save_file_name = None
        # path where to save the file_name
        self.save_path = default_path
        self.save_checked_predictions_dir = default_path
        self.display_threshold = False
        # to display a color code for the peaks depending on the correlation between the source and transient profile
        # changed when clicking on the checkbox
        self.display_correlations = False
        # if False, remove this option, compute faster when loading a cell
        self.correlation_for_each_peak_option = True
        # in number of frames, one frame = 100 ms
        self.decay = 10
        # factor to multiply to decay to define delay between onset and peak
        self.decay_factor = 1
        # number of std of trace to add to the threshold
        self.nb_std_thresold = 0.1
        self.correlation_thresold = 0.5
        self.data_and_param = data_and_param

        # will be a 2D array of len n_cells * n_frames and the value
        # will correspond to the correlation of the peak transient
        # at that frame (if identified, value will be -2 for non identified peak)
        self.peaks_correlation = None
        # dict with key an int representing the cell, and as value a set of int (cells)
        self.overlapping_cells = self.data_and_param.ms.coord_obj.intersect_cells

        self.center_coord = self.data_and_param.ms.coord_obj.center_coord
        self.path_result = self.data_and_param.result_path
        self.path_data = self.data_and_param.path_data
        self.spike_nums = data_and_param.spike_nums
        self.nb_neurons = len(self.spike_nums)
        self.nb_times = len(self.spike_nums[0, :])

        self.raw_traces = self.data_and_param.raw_traces
        use_caiman_demix_trace = True
        if use_caiman_demix_trace:
            self.traces = data_and_param.traces
        else:
            self.traces = np.copy(self.raw_traces)
            # smoothing the trace, but no demixing
            do_traces_smoothing(self.traces)

        # windows = ['hanning', 'hamming', 'bartlett', 'blackman']
        # i_w = 1
        # window_length = 11
        # for i in np.arange(self.nb_neurons):
        #     smooth_signal = smooth_convolve(x=self.traces[i], window_len=window_length,
        #                                     window=windows[i_w])
        #     beg = (window_length - 1) // 2
        #     self.traces[i] = smooth_signal[beg:-beg]

        # will be initialize in the function normalize_traces
        self.ratio_traces = None
        # we plot a trace that show the ratio between traces and raw_traces
        # it could be useful if the traces is partially demixed
        self.use_ratio_traces = False
        self.nb_times_traces = len(self.traces[0, :])
        # dimension reduction in order to fit to traces times, for onset times
        self.onset_times = np.zeros((self.nb_neurons, self.nb_times), dtype="int8")
        self.onset_numbers_label = None
        self.update_onset_times()
        self.peak_nums = self.data_and_param.peak_nums
        self.to_agree_peak_nums = self.data_and_param.to_agree_peak_nums
        self.to_agree_spike_nums = self.data_and_param.to_agree_spike_nums
        # print(f"len(peak_nums) {len(peak_nums)}")
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
        self.source_profile_dict = dict()
        self.source_profile_dict_for_map_of_all_cells = dict()
        # key is int (cell), value is a list with the source profile used for correlation, and the mask
        self.source_profile_correlation_dict = dict()
        # key is the cell and then value is a dict with key the tuple transient (onset, peak)
        # self.corr_source_transient = dict()
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

        # initializing inter_neurons and cells_to_remove
        self.inter_neurons = np.zeros(self.nb_neurons, dtype="uint8")
        if self.data_and_param.inter_neurons is not None:
            for inter_neuron in self.data_and_param.inter_neurons:
                self.inter_neurons[inter_neuron] = 1
        self.cells_to_remove = np.zeros(self.nb_neurons, dtype="uint8")
        if self.data_and_param.cells_to_remove is not None:
            # self.data_and_param.cells_to_remove is actually a list of one array with all cells to remove as an int
            for cell_to_remove in self.data_and_param.cells_to_remove:
                self.cells_to_remove[cell_to_remove] = 1

        # Key cell,
        # value: list of tuple of 2 int indicating the beginning and end of each corrupt frames period
        self.doubtful_frames_periods = dict()
        if self.data_and_param.doubtful_frames_nums is not None:
            self.doubtful_frames_nums = self.data_and_param.doubtful_frames_nums
            for cell in np.arange(self.nb_neurons):
                self.update_doubtful_frames_periods(cell=cell)
        else:
            self.doubtful_frames_nums = np.zeros((self.nb_neurons, self.nb_times_traces), dtype="int8")

        # Key cell,
        # value: list of tuple of 2 int indicating the beginning and end of each mvt frames period
        self.mvt_frames_periods = dict()
        if self.data_and_param.mvt_frames_nums is not None:
            self.mvt_frames_nums = self.data_and_param.mvt_frames_nums
            for cell in np.arange(self.nb_neurons):
                self.update_mvt_frames_periods(cell=cell)
        else:
            self.mvt_frames_nums = np.zeros((self.nb_neurons, self.nb_times_traces), dtype="int8")

        self.display_raw_traces = self.raw_traces is not None
        # neuron's trace displayed
        self.current_neuron = 0
        # indicate if an action might remove or add an onset
        self.add_onset_mode = False
        self.remove_onset_mode = False
        self.add_peak_mode = False
        self.remove_peak_mode = False
        self.remove_all_mode = False
        self.add_doubtful_frames_mode = False
        self.remove_doubtful_frames_mode = False
        self.add_mvt_frames_mode = False
        self.remove_mvt_frames_mode = False
        self.dont_agree_mode = False
        self.agree_mode = False
        # used to remove under the threshold (std or correlation value)
        self.peaks_under_threshold_index = None
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

        # check if a transient classifier model is available
        path_to_tc_model = self.path_data + "transient_classifier_model/"
        self.transient_classifier_json_file = None
        self.transient_classifier_weights_file = None
        # checking if the path exists
        if os.path.isdir(path_to_tc_model):
            # then we look for the json file (representing the model architecture) and the weights file
            # we will assume there is only one file of each in this directory
            # look for filenames in the first directory, if we don't break, it will go through all directories
            for (dirpath, dirnames, local_filenames) in os.walk(path_to_tc_model):
                for file_name in local_filenames:
                    if file_name.endswith(".json"):
                        self.transient_classifier_json_file = path_to_tc_model + file_name
                    if "weights" in file_name:
                        self.transient_classifier_weights_file = path_to_tc_model + file_name
                # looking only in the top directory
                break
        self.show_transient_classifier = False
        self.transient_classifier_threshold = 0.5
        # key is int representing the cell number, and value will be an array of 2D with line representing the frame index
        # and the colums being a float reprenseing for each frame
        # the probability for the cell to be active for each class
        self.transient_prediction = dict()
        # first key is an int (cell), value is a dict
        # the second key is a float representing a threshold, and the value is a list of tuple
        self.transient_prediction_periods = dict()
        # the cell and frame (tuple) in which a vertical line
        # will be draw to indicate at which transient we're looking at
        self.last_predicted_transient_selected = None
        # checking if a prediction results are already loaded in mouse_sessions
        if self.data_and_param.ms.rnn_transients_predictions is not None:
            # print(f"len(self.data_and_param.ms.rnn_transients_predictions) "
            #       f"{len(self.data_and_param.ms.rnn_transients_predictions)} {self.nb_neurons}")
            for cell in np.arange(self.nb_neurons):
                self.transient_prediction_periods[cell] = dict()
                cell_predictions = self.data_and_param.ms.rnn_transients_predictions[cell]
                cell_predictions = cell_predictions.reshape((len(cell_predictions), 1))
                self.transient_prediction[cell] = cell_predictions
        # each key is a cell, the value is a binary array with 1 between onsets and peaks
        # used for predictions
        self.raster_dur_for_a_cell = dict()
        # the prediction values border for which we would like the check the transients
        self.uncertain_prediction_values = (0.25, 0.65)
        # list of tuple (cell, first frame, last frame (included), max prediction)
        self.transient_prediction_periods_to_check = []
        # set of tuple (cell, first frame, last frame (included), max prediction)
        self.transient_prediction_periods_checked = []

        # to print transient predictions stat only once
        self.print_transients_predictions_stat = False
        self.caiman_active_periods = None
        self.caiman_spike_nums = None
        if self.data_and_param.ms.caiman_active_periods is not None:
            self.caiman_active_periods = self.data_and_param.ms.caiman_active_periods
            if self.data_and_param.ms.caiman_spike_nums_dur is not None:
                self.caiman_spike_nums = self.data_and_param.ms.caiman_spike_nums_dur
            else:
                self.caiman_spike_nums = self.data_and_param.ms.caiman_spike_nums
            # for cell in np.arange(self.nb_neurons):
            #     self.caiman_active_periods[cell] = \
            #         get_continous_time_periods(self.data_and_param.ms.caiman_spike_nums_dur[cell, :])

        # Three Vertical frames to start
        # -------------- left frame (start) ----------------
        left_side_frame = Frame(self)
        left_side_frame.pack(side=LEFT, expand=YES, fill=BOTH)

        # self.spin_box_button = Spinbox(left_side_frame, from_=0, to=self.nb_neurons - 1, fg="blue", justify=CENTER,
        #                                width=4, state="readonly")
        # self.spin_box_button["command"] = event_lambda(self.spin_box_update)
        # # self.spin_box_button.config(command=event_lambda(self.spin_box_update))
        # self.spin_box_button.pack(side=LEFT)

        empty_label = Label(left_side_frame)
        empty_label["text"] = ""
        empty_label.pack(side=TOP, expand=YES, fill=BOTH)

        # self.neuron_string_var = StringVar()
        entry_neuron_frame = Frame(left_side_frame)
        entry_neuron_frame.pack(side=TOP, expand=NO, fill="x")
        self.neuron_entry_widget = Entry(entry_neuron_frame, fg=self.color_text_gui_default, justify=CENTER,
                                         width=3)
        self.neuron_entry_widget.insert(0, "0")
        self.neuron_entry_widget.bind("<KeyRelease>", self.go_to_neuron_button_action)
        # self.neuron_string_var.set("0")
        # self.neuron_string_var.trace("w", self.neuron_entry_change)
        # self.neuron_entry_widget.focus_set()
        self.neuron_entry_widget.pack(side=LEFT, expand=YES, fill="x")

        self.go_button = Button(entry_neuron_frame)
        self.go_button["text"] = ' GO '
        self.go_button["fg"] = self.color_text_gui_default
        self.go_button["command"] = event_lambda(self.go_to_neuron_button_action)
        self.go_button.pack(side=LEFT, expand=YES, fill="x")

        change_neuron_frame = Frame(left_side_frame)
        change_neuron_frame.pack(side=TOP, expand=NO, fill="x")

        self.prev_button = Button(change_neuron_frame)
        self.prev_button["text"] = ' <- '
        self.prev_button["fg"] = self.color_text_gui_default
        self.prev_button['state'] = DISABLED  # ''normal
        self.prev_button["command"] = event_lambda(self.select_previous_neuron)
        self.prev_button.pack(side=LEFT, expand=YES, fill="x")

        # empty_label = Label(change_neuron_frame)
        # empty_label["text"] = ""
        # empty_label.pack(side=LEFT)

        self.neuron_label = Label(change_neuron_frame)
        self.neuron_label["text"] = "0"
        self.neuron_label["fg"] = "red"
        self.neuron_label.pack(side=LEFT, expand=YES, fill="x")

        # empty_label = Label(change_neuron_frame)
        # empty_label["text"] = ""
        # empty_label.pack(side=LEFT)

        self.next_button = Button(change_neuron_frame)
        self.next_button["text"] = ' -> '
        self.next_button["fg"] = self.color_text_gui_default
        self.next_button["command"] = event_lambda(self.select_next_neuron)
        self.next_button.pack(side=LEFT, expand=YES, fill="x")

        # empty_label = Label(left_side_frame)
        # empty_label["text"] = " " * 2
        # empty_label.pack(side=LEFT)
        sep = ttk.Separator(left_side_frame)
        sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

        self.add_doubtful_frames_mode_button = Button(left_side_frame)
        self.add_doubtful_frames_mode_button["text"] = ' + DOUBT OFF '
        self.add_doubtful_frames_mode_button["fg"] = 'red'
        self.add_doubtful_frames_mode_button["command"] = self.add_doubtful_frames_switch_mode
        self.add_doubtful_frames_mode_button.pack(side=TOP)
        # expand=NO, fill="x"
        # empty_label = Label(left_side_frame)
        # empty_label["text"] = " " * 1
        # empty_label.pack(side=LEFT)

        self.remove_doubtful_frames_button = Button(left_side_frame)
        self.remove_doubtful_frames_button["text"] = ' - DOUBT OFF '
        self.remove_doubtful_frames_button["fg"] = 'red'
        self.remove_doubtful_frames_button["command"] = self.remove_doubtful_frames_switch_mode
        self.remove_doubtful_frames_button.pack(side=TOP)

        sep = ttk.Separator(left_side_frame)
        sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

        self.add_mvt_frames_mode_button = Button(left_side_frame)
        self.add_mvt_frames_mode_button["text"] = ' + MVT OFF '
        self.add_mvt_frames_mode_button["fg"] = 'red'
        self.add_mvt_frames_mode_button["command"] = self.add_mvt_frames_switch_mode
        self.add_mvt_frames_mode_button.pack(side=TOP)

        self.remove_mvt_frames_button = Button(left_side_frame)
        self.remove_mvt_frames_button["text"] = ' - MVT OFF '
        self.remove_mvt_frames_button["fg"] = 'red'
        self.remove_mvt_frames_button["command"] = self.remove_mvt_frames_switch_mode
        self.remove_mvt_frames_button.pack(side=TOP)

        sep = ttk.Separator(left_side_frame)
        sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

        self.add_onset_button = Button(left_side_frame)
        self.add_onset_button["text"] = ' + ONSET OFF '
        self.add_onset_button["fg"] = 'red'
        self.add_onset_button["command"] = self.add_onset_switch_mode
        self.add_onset_button.pack(side=TOP)

        # empty_label = Label(left_side_frame)
        # empty_label["text"] = "" * 1
        # empty_label.pack(side=TOP)

        self.onset_numbers_label = Label(left_side_frame)
        self.onset_numbers_label["text"] = f"{self.numbers_of_onset()}"
        self.onset_numbers_label.pack(side=TOP)

        self.remove_onset_button = Button(left_side_frame)
        self.remove_onset_button["text"] = ' - ONSET OFF '
        self.remove_onset_button["fg"] = 'red'
        self.remove_onset_button["command"] = self.remove_onset_switch_mode
        self.remove_onset_button.pack(side=TOP)

        sep = ttk.Separator(left_side_frame)
        sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

        self.add_peak_button = Button(left_side_frame)
        self.add_peak_button["text"] = ' + PEAK OFF '
        self.add_peak_button["fg"] = 'red'
        self.add_peak_button["command"] = self.add_peak_switch_mode
        self.add_peak_button.pack(side=TOP)

        self.peak_numbers_label = Label(left_side_frame)
        self.peak_numbers_label["text"] = f"{self.numbers_of_peak()}"
        self.peak_numbers_label.pack(side=TOP)

        self.remove_peak_button = Button(left_side_frame)
        self.remove_peak_button["text"] = ' - PEAK OFF '
        self.remove_peak_button["fg"] = 'red'
        self.remove_peak_button["command"] = self.remove_peak_switch_mode
        self.remove_peak_button.pack(side=TOP)

        sep = ttk.Separator(left_side_frame)
        sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

        # empty_label = Label(left_side_frame)
        # empty_label["text"] = " " * 1
        # empty_label.pack(side=TOP)

        self.remove_all_button = Button(left_side_frame)
        self.remove_all_button["text"] = ' - ALL OFF '
        self.remove_all_button["fg"] = 'red'
        self.remove_all_button["command"] = self.remove_all_switch_mode
        self.remove_all_button.pack(side=TOP)

        self.agree_button = None
        self.dont_agree_button = None
        if (self.to_agree_spike_nums is not None) and (self.to_agree_peak_nums is not None):
            if (np.sum(self.to_agree_spike_nums) > 0) or (np.sum(self.to_agree_peak_nums) > 0):
                sep = ttk.Separator(left_side_frame)
                sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

                # deal with fusion over onsets & peaks over 2 different gui selections
                self.agree_button = Button(left_side_frame)
                self.agree_button["text"] = 'Y'
                self.agree_button["fg"] = 'red'
                self.agree_button["command"] = self.agree_switch_mode
                self.agree_button.pack(side=TOP)

                # empty_label = Label(left_side_frame)
                # empty_label["text"] = "" * 1
                # empty_label.pack(side=TOP)

                self.to_agree_label = Label(left_side_frame)
                self.to_agree_label["text"] = f"{self.numbers_of_onset_to_agree()}/" \
                    f"{self.numbers_of_peak_to_agree()}"
                self.to_agree_label.pack(side=TOP)

                # empty_label = Label(left_side_frame)
                # empty_label["text"] = "" * 1
                # empty_label.pack(side=TOP)

                self.dont_agree_button = Button(left_side_frame)
                self.dont_agree_button["text"] = 'N'
                self.dont_agree_button["fg"] = 'red'
                self.dont_agree_button["command"] = self.dont_agree_switch_mode
                self.dont_agree_button.pack(side=TOP)
            else:
                self.to_agree_label = None
        else:
            self.to_agree_label = None

        if ((self.transient_classifier_weights_file is not None) and (self.transient_classifier_json_file is not None)) \
                or (self.data_and_param.ms.rnn_transients_predictions is not None):
            sep = ttk.Separator(left_side_frame)
            sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

            pred_values_frame = Frame(left_side_frame)
            pred_values_frame.pack(side=TOP, expand=NO, fill="x")
            empty_label = Label(pred_values_frame)
            empty_label["text"] = ""
            empty_label.pack(side=LEFT, expand=YES, fill=BOTH)

            self.pred_value_1_entry_widget = Entry(pred_values_frame, fg=self.color_text_gui_default, justify=CENTER,
                                                   width=3)
            self.pred_value_1_entry_widget.insert(0, f"{self.uncertain_prediction_values[0]}")
            self.pred_value_1_entry_widget.pack(side=LEFT)
            empty_label = Label(pred_values_frame)
            empty_label["text"] = " - "
            empty_label.pack(side=LEFT)
            self.pred_value_2_entry_widget = Entry(pred_values_frame, fg=self.color_text_gui_default, justify=CENTER,
                                                   width=3)
            self.pred_value_2_entry_widget.insert(0, f"{self.uncertain_prediction_values[1]}")
            self.pred_value_2_entry_widget.pack(side=LEFT)

            self.ok_pred_button = Button(pred_values_frame)
            self.ok_pred_button["text"] = ' OK '
            self.ok_pred_button["fg"] = self.color_text_gui_default
            self.ok_pred_button["command"] = self.update_uncertain_prediction_values
            self.ok_pred_button.pack(side=LEFT)

            empty_label = Label(pred_values_frame)
            empty_label["text"] = ""
            empty_label.pack(side=LEFT, expand=YES, fill=BOTH)

            # if predictions are available, then we display a listbox that will allow to visualize predictions
            # that are in a certain range in order to confirm their accuracy
            predictions_list_frame = Frame(left_side_frame)
            predictions_list_frame.pack(side=TOP, expand=NO, fill="x")
            scrollbar = Scrollbar(predictions_list_frame)
            scrollbar.pack(side=RIGHT, fill=Y)

            self.predictions_list_box = Listbox(predictions_list_frame, bd=0, selectmode=BROWSE,
                                                height=15, highlightbackground="red",
                                                font=("Arial", 10), yscrollcommand=scrollbar.set)
            # default height is 10, selectmode=SINGLE
            self.predictions_list_box.bind("<Double-Button-1>", self.predictions_list_box_double_click)

            self.predictions_list_box.bind('<<ListboxSelect>>', self.predictions_list_box_click)
            # width=20
            self.predictions_list_box.pack()

            self.update_transient_prediction_periods_to_check()

            # attach listbox to scrollbar
            scrollbar.config(command=self.predictions_list_box.yview)

        empty_label = Label(left_side_frame)
        empty_label["text"] = ""
        empty_label.pack(side=TOP, expand=YES, fill=BOTH)
        # -------------- left side frame (end) ----------------

        ################################################################################
        ################################ Middle frame with plot ################################
        ################################################################################
        canvas_frame = Frame(self, padx=0, pady=0, bg="black")
        canvas_frame.pack(side=LEFT, expand=YES, fill=BOTH)

        top_bar_canvas_frame = Frame(canvas_frame, padx=0, pady=0, bg="black")
        top_bar_canvas_frame.pack(side=TOP, expand=YES, fill=BOTH)

        main_plot_frame = Frame(canvas_frame, padx=0, pady=0, bg="black")
        main_plot_frame.pack(side=TOP, expand=YES, fill=BOTH)
        self.main_plot_frame = main_plot_frame

        self.display_michou = False

        # plt.ion()
        if self.robin_mac:
            self.fig = plt.figure(figsize=(8, 3))
        else:
            self.fig = plt.figure(figsize=(16, 8))  # 10, 4
        self.fig.patch.set_facecolor('black')
        # self.plot_canvas = MyCanvas(self.fig, canvas_frame, self)
        self.plot_canvas = FigureCanvasTkAgg(self.fig, main_plot_frame)
        # self.plot_canvas.get_tk_widget().configure(bg="black")
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.fig.canvas.mpl_connect('motion_notify_event', self.motion)
        # self.plot_canvas.
        #     bind("<Button-1>", self.callback_click_fig())

        self.raw_traces_median = None
        self.display_raw_traces_median = False
        # self.gs = gridspec.GridSpec(2, 1, width_ratios=[1], height_ratios=[5, 1])
        # using gridspec even for one plot, but could be useful if we want to add plot.
        self.gs = gridspec.GridSpec(1, 1, width_ratios=[1], height_ratios=[1])
        # self.gs.update(hspace=0.05)
        # ax1 = plt.subplot(gs[0])
        # ax2 = plt.subplot(gs[1])
        self.axe_plot = None
        self.ax1_bottom_scatter = None
        self.ax1_top_scatter = None
        self.line1 = None
        self.line2 = None
        self.plot_graph(first_time=True)

        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=YES)

        self.toolbar = NavigationToolbar2Tk(self.plot_canvas, main_plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side=TOP, fill=BOTH, expand=NO)

        self.map_frame = Frame(top_bar_canvas_frame)
        self.map_frame.pack(side=LEFT, expand=NO, fill="x")

        # tif movie loading
        self.last_img_displayed = None
        self.trace_movie_p1 = None
        self.trace_movie_p2 = None
        self.movie_available = False
        self.play_movie = False
        # coordinate for the zoom mode
        self.x_beg_movie = 0
        self.x_end_movie = 0
        self.y_beg_movie = 0
        self.y_end_movie = 0
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
            if self.data_and_param.ms.tiff_movie is None:
                self.data_and_param.ms.load_tiff_movie_in_memory()
            self.tiff_movie = self.data_and_param.ms.tiff_movie
            # will be useful for transient classifier prediction
            self.data_and_param.ms.avg_cell_map_img = np.mean(self.tiff_movie, axis=0)
            self.data_and_param.ms.normalize_movie()
            self.tiff_movie = self.data_and_param.ms.tiff_movie_normalized
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
            if not file_name.startswith("."):
                self.michou_imgs.append(mpimg.imread(self.data_and_param.path_data + self.michou_path + file_name, 0))

        self.n_michou_img = len(self.michou_imgs)
        self.michou_img_to_display = -1

        # first key is the cell, value a dict
        # second key is the frame, value is a list of int representing the value of each pixel
        self.pixels_value_by_cell_and_frame = dict()

        if (self.data_and_param.ms.avg_cell_map_img is not None) or (self.tiff_movie is not None):
            if self.tiff_movie is not None and self.display_raw_traces_median and (self.raw_traces is not None):
                self.raw_traces_median = np.zeros(self.traces.shape)
            if self.robin_mac:
                self.map_img_fig = plt.figure(figsize=(3, 3))
            else:
                self.map_img_fig = plt.figure(figsize=(4, 4))
            self.map_img_fig.patch.set_facecolor('black')

            self.background_map_fig = None
            self.axe_plot_map_img = None
            self.cell_contour = None
            self.cell_contour_movie = None
            # creating all cell_contours
            self.cell_contours = dict()
            n_pixels_x = 200
            n_pixels_y = 200
            if self.tiff_movie is not None:
                n_pixels_x = self.tiff_movie.shape[1]
                n_pixels_y = self.tiff_movie.shape[2]
            self.cell_in_pixel = np.ones((n_pixels_x, n_pixels_y), dtype="int16")
            self.cell_in_pixel *= -1
            for cell in np.arange(self.nb_neurons):
                # cell contour
                coord = self.data_and_param.ms.coord_obj.coord[cell]
                xy = coord.transpose()
                self.cell_contours[cell] = patches.Polygon(xy=xy,
                                                           fill=False, linewidth=0, facecolor="red",
                                                           edgecolor="red",
                                                           zorder=15, lw=0.6)
                # list(polygon.exterior.coords)
                # [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]

                if self.tiff_movie is not None:
                    # the coordinates of the are set to True
                    mask_img = np.zeros((self.tiff_movie.shape[1], self.tiff_movie.shape[2]), dtype="bool")
                bw = np.zeros((n_pixels_x, n_pixels_y), dtype="int8")
                # morphology.binary_fill_holes(input
                # print(f"coord[1, :] {coord[1, :]}")
                bw[coord[1, :], coord[0, :]] = 1

                # used to know which cell has been clicked
                img_filled = np.zeros((n_pixels_x, n_pixels_y), dtype="int8")
                # specifying output, otherwise binary_fill_holes return a boolean array
                morphology.binary_fill_holes(bw, output=img_filled)
                for pixel in np.arange(n_pixels_x):
                    y_coords = np.where(img_filled[pixel, :])[0]
                    if len(y_coords) > 0:
                        self.cell_in_pixel[pixel, y_coords] = cell
                        if self.tiff_movie is not None:
                            mask_img[pixel, y_coords] = True

                if self.tiff_movie is not None and self.display_raw_traces_median and (self.raw_traces is not None):
                    self.raw_traces_median[cell, :] = np.median(self.tiff_movie[:, mask_img], axis=1)

            self.map_img_canvas = FigureCanvasTkAgg(self.map_img_fig, self.map_frame)
            # self.map_img_canvas.get_tk_widget().configure(bg="blue")
            self.map_img_fig.canvas.mpl_connect('button_release_event', self.onrelease_map)
            self.plot_map_img(first_time=True)

            self.map_img_canvas.draw()
            self.map_img_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=YES)

        # Y alignment
        # self.raw_traces_median = None
        self.normalize_traces()
        self.update_plot(new_neuron=True)

        self.magnifier_frame = Frame(top_bar_canvas_frame)
        self.magnifier_frame.pack(side=LEFT, expand=YES, fill="x")
        if self.robin_mac:
            self.magnifier_fig = plt.figure(figsize=(3, 3))
        else:
            self.magnifier_fig = plt.figure(figsize=(4, 4))
        self.magnifier_fig.patch.set_facecolor('black')
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
        ################################ right side frame ################################
        ################################################################################
        right_side_frame = Frame(self)
        right_side_frame.pack(side=LEFT, expand=YES, fill=BOTH)

        # tells if the movie is active
        self.movie_mode = False
        # Then we zoom on the active cell
        self.movie_zoom_mode = True
        # show the source profile of the active cell associate to the transient profile of the selected transient
        # as well as source profile of overlapping cells, and then display the correlation between each source profile
        # and their corresponding transient
        self.source_mode = False

        # to vertically center the buttons
        empty_label = Label(right_side_frame)
        empty_label["text"] = ""
        empty_label.pack(side=TOP, expand=YES, fill=BOTH)

        # to display the source profile of a transient
        self.source_var = IntVar()
        self.source_check_box = Checkbutton(right_side_frame, text="source", variable=self.source_var,
                                            onvalue=1,
                                            offvalue=0)
        # zoom on by default
        if self.source_mode:
            self.source_check_box.select()
        self.source_check_box["command"] = event_lambda(self.switch_source_profile_mode)
        self.source_check_box.pack(side=TOP)

        sep = ttk.Separator(right_side_frame)
        sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

        self.zoom_movie_var = IntVar()
        self.zoom_movie_check_box = Checkbutton(right_side_frame, text="zoom", variable=self.zoom_movie_var,
                                                onvalue=1,
                                                offvalue=0)
        # zoom on by default
        if self.movie_zoom_mode:
            self.zoom_movie_check_box.select()
        self.zoom_movie_check_box["command"] = event_lambda(self.activate_movie_zoom)
        self.zoom_movie_check_box.pack(side=TOP)

        sep = ttk.Separator(right_side_frame)
        sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

        if self.tiff_movie is not None:
            # empty_label = Label(right_side_frame)
            # empty_label["text"] = " " * 1
            # empty_label.pack(side=TOP)

            self.movie_button = Button(right_side_frame)
            self.movie_button["text"] = ' movie OFF '
            self.movie_button["fg"] = "black"
            self.movie_button["command"] = event_lambda(self.switch_movie_mode)
            self.movie_button.pack(side=TOP)

        sep = ttk.Separator(right_side_frame)
        sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

        if self.mvt_frames_periods is not None:
            # empty_label = Label(right_side_frame)
            # empty_label["text"] = " " * 1
            # empty_label.pack(side=TOP)

            self.display_mvt_button = Button(right_side_frame)

            self.display_mvt_button["text"] = ' mvt OFF '
            self.display_mvt_button["fg"] = "black"

            self.display_mvt_button["command"] = event_lambda(self.switch_mvt_display)
            self.display_mvt_button.pack(side=TOP)

        sep = ttk.Separator(right_side_frame)
        sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

        self.magnifier_button = Button(right_side_frame)
        self.magnifier_button["text"] = ' magnified OFF '
        self.magnifier_button["fg"] = "black"
        self.magnifier_mode = False
        self.magnifier_button["command"] = event_lambda(self.switch_magnifier)
        self.magnifier_button.pack(side=TOP)

        sep = ttk.Separator(right_side_frame)
        sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

        if ((self.transient_classifier_weights_file is not None) and (self.transient_classifier_json_file is not None)) \
                or (self.data_and_param.ms.rnn_transients_predictions is not None):
            transient_classifier_frame = Frame(right_side_frame)
            transient_classifier_frame.pack(side=TOP, expand=NO, fill=BOTH)

            empty_label = Label(transient_classifier_frame)
            empty_label["text"] = ""
            empty_label.pack(side=LEFT, expand=YES, fill=BOTH)

            self.transient_classifier_var = IntVar()
            self.transient_classifier_check_box = Checkbutton(transient_classifier_frame, text="tc",
                                                              variable=self.transient_classifier_var, onvalue=1,
                                                              offvalue=0, fg=self.color_threshold_line)
            self.transient_classifier_check_box["command"] = event_lambda(self.transient_classifier_check_box_action)
            self.transient_classifier_check_box.pack(side=LEFT)

            # empty_label = Label(right_side_frame)
            # empty_label["text"] = " " * 1
            # empty_label.pack(side=TOP)
            # from_=1, to=3
            # self.var_spin_box_threshold = StringVar(right_side_frame)
            var = StringVar(transient_classifier_frame)
            self.spin_box_transient_classifier = Spinbox(transient_classifier_frame,
                                                         values=list(np.arange(0.05, 1, 0.05)),
                                                         fg=self.color_text_gui_default, justify=CENTER,
                                                         width=3, textvariable=var,
                                                         state="readonly")  # , textvariable=self.var_spin_box_threshold)
            var.set("0.5")
            # self.var_spin_box_threshold.set(0.5)
            self.spin_box_transient_classifier["command"] = event_lambda(self.spin_box_transient_classifier_update)
            # self.spin_box_button.config(command=event_lambda(self.spin_box_update))
            self.spin_box_transient_classifier.pack(side=LEFT)

            empty_label = Label(transient_classifier_frame)
            empty_label["text"] = ""
            empty_label.pack(side=LEFT, expand=YES, fill=BOTH)

        sep = ttk.Separator(right_side_frame)
        sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

        self.remove_cell_button = Button(right_side_frame)
        if self.cells_to_remove[self.current_neuron] == 0:
            self.remove_cell_button["text"] = ' not removed '
            self.remove_cell_button["fg"] = "black"
        else:
            self.remove_cell_button["text"] = ' removed '
            self.remove_cell_button["fg"] = "red"
        self.remove_cell_button["command"] = event_lambda(self.remove_cell)
        self.remove_cell_button.pack(side=TOP)

        sep = ttk.Separator(right_side_frame)
        sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

        # identify an interneuron as such
        # TODO: See to be put instead a field that allow to put a "marker" on a cell
        self.inter_neuron_button = Button(right_side_frame)
        if self.inter_neurons[self.current_neuron] == 0:
            self.inter_neuron_button["text"] = ' not IN '
            self.inter_neuron_button["fg"] = "black"
        else:

            self.inter_neuron_button["text"] = ' IN '
            self.inter_neuron_button["fg"] = "red"
        self.inter_neuron_button["command"] = event_lambda(self.set_inter_neuron)
        self.inter_neuron_button.pack(side=TOP)

        sep = ttk.Separator(right_side_frame)
        sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

        if self.tiff_movie is not None and self.correlation_for_each_peak_option:
            correlation_frame = Frame(right_side_frame)
            correlation_frame.pack(side=TOP, expand=NO, fill=BOTH)

            empty_label = Label(correlation_frame)
            empty_label["text"] = ""
            empty_label.pack(side=LEFT, expand=YES, fill=BOTH)

            self.peaks_correlation = np.ones(self.traces.shape)
            self.peaks_correlation *= -2

            self.correlation_var = IntVar()
            self.correlation_check_box = Checkbutton(correlation_frame, text="corr", variable=self.correlation_var,
                                                     onvalue=1,
                                                     offvalue=0, fg=self.color_threshold_line)
            self.correlation_check_box["command"] = event_lambda(self.correlation_check_box_action)
            # self.correlation_check_box.select()

            self.correlation_check_box.pack(side=LEFT)

            self.spin_box_correlation = Spinbox(correlation_frame, values=list(np.arange(0.5, 1, 0.05)),
                                                fg=self.color_text_gui_default,
                                                justify=CENTER,
                                                width=3,
                                                state="readonly")  # , textvariable=self.var_spin_box_threshold)
            # self.var_spin_box_threshold.set(0.9)
            self.spin_box_correlation["command"] = event_lambda(self.spin_box_correlation_update)
            # self.spin_box_button.config(command=event_lambda(self.spin_box_update))
            self.spin_box_correlation.pack(side=LEFT)

            empty_label = Label(correlation_frame)
            empty_label["text"] = ""
            empty_label.pack(side=LEFT, expand=YES, fill=BOTH)

        sep = ttk.Separator(right_side_frame)
        sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

        threshold_frame = Frame(right_side_frame)
        threshold_frame.pack(side=TOP, expand=NO, fill=BOTH)

        empty_label = Label(threshold_frame)
        empty_label["text"] = ""
        empty_label.pack(side=LEFT, expand=YES, fill=BOTH)

        self.treshold_var = IntVar()
        self.threshold_check_box = Checkbutton(threshold_frame, text="std", variable=self.treshold_var, onvalue=1,
                                               offvalue=0, fg=self.color_threshold_line)
        self.threshold_check_box["command"] = event_lambda(self.threshold_check_box_action)
        self.threshold_check_box.pack(side=LEFT)

        # from_=1, to=3
        # self.var_spin_box_threshold = StringVar(right_side_frame)
        self.spin_box_threshold = Spinbox(threshold_frame, values=list(np.arange(0.1, 5, 0.1)),
                                          fg=self.color_text_gui_default,
                                          justify=CENTER,
                                          width=3, state="readonly")  # , textvariable=self.var_spin_box_threshold)
        # self.var_spin_box_threshold.set(0.9)
        self.spin_box_threshold["command"] = event_lambda(self.spin_box_threshold_update)
        # self.spin_box_button.config(command=event_lambda(self.spin_box_update))
        self.spin_box_threshold.pack(side=LEFT)

        empty_label = Label(threshold_frame)
        empty_label["text"] = ""
        empty_label.pack(side=LEFT, expand=YES, fill=BOTH)

        sep = ttk.Separator(right_side_frame)
        sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

        self.remove_peaks_under_threshold_button = Button(right_side_frame)
        self.remove_peaks_under_threshold_button["text"] = ' DEL PEAKS '
        self.remove_peaks_under_threshold_button["fg"] = "red"
        self.remove_peaks_under_threshold_button['state'] = DISABLED  # ''normal
        self.remove_peaks_under_threshold_button["command"] = event_lambda(self.remove_peaks_under_threshold)
        self.remove_peaks_under_threshold_button.pack(side=TOP)

        sep = ttk.Separator(right_side_frame)
        sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

        undo_frame = Frame(right_side_frame)
        undo_frame.pack(side=TOP, expand=NO, fill=BOTH)

        empty_label = Label(undo_frame)
        empty_label["text"] = ""
        empty_label.pack(side=LEFT, expand=YES, fill=BOTH)

        self.undo_button = Button(undo_frame)
        self.undo_button["text"] = ' UNDO '
        self.undo_button["fg"] = self.color_text_gui_default
        self.undo_button['state'] = DISABLED  # ''normal
        self.undo_button["command"] = event_lambda(self.undo_action)
        self.undo_button.pack(side=LEFT)

        self.redo_button = Button(undo_frame)
        self.redo_button["text"] = ' REDO '
        self.redo_button["fg"] = self.color_text_gui_default
        self.redo_button['state'] = DISABLED  # ''normal
        self.redo_button["command"] = event_lambda(self.redo_action)
        self.redo_button.pack(side=LEFT)

        empty_label = Label(undo_frame)
        empty_label["text"] = ""
        empty_label.pack(side=LEFT, expand=YES, fill=BOTH)

        sep = ttk.Separator(right_side_frame)
        sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

        self.save_button = Button(right_side_frame)
        self.save_button["text"] = ' SAVE '
        self.save_button["fg"] = self.color_text_gui_default
        self.save_button['state'] = DISABLED  # ''normal
        self.save_button["command"] = event_lambda(self.save_spike_nums)
        self.save_button.pack(side=TOP)

        self.save_as_button = Button(right_side_frame)
        self.save_as_button["text"] = ' SAVE AS '
        self.save_as_button["fg"] = self.color_text_gui_default
        self.save_as_button['state'] = "normal"
        self.save_as_button["command"] = event_lambda(self.save_as_spike_nums)
        self.save_as_button.pack(side=TOP)

        if ((self.transient_classifier_weights_file is not None) and (self.transient_classifier_json_file is not None)) \
                or (self.data_and_param.ms.rnn_transients_predictions is not None):
            sep = ttk.Separator(right_side_frame)
            sep.pack(side=TOP, fill=BOTH, padx=0, pady=10)

            # if predictions are available, then we display a listbox that will allow to visualize predictions
            # that are in a certain range in order to confirm their accuracy
            predictions_list_frame = Frame(right_side_frame)
            predictions_list_frame.pack(side=TOP, expand=NO, fill="x")
            scrollbar = Scrollbar(predictions_list_frame)
            scrollbar.pack(side=RIGHT, fill=Y)

            self.checked_predictions_list_box = Listbox(predictions_list_frame, bd=0, selectmode=BROWSE,
                                                        height=10, highlightbackground="red",
                                                        font=("Arial", 10), yscrollcommand=scrollbar.set)
            # default height is 10
            self.checked_predictions_list_box.bind("<Double-Button-1>", self.checked_predictions_list_box_double_click)
            self.checked_predictions_list_box.bind('<<ListboxSelect>>', self.checked_predictions_list_box_click)

            # width=20
            self.checked_predictions_list_box.pack()

            # self.update_checked_predictions_list_box()

            # attach listbox to scrollbar
            scrollbar.config(command=self.predictions_list_box.yview)

            self.save_pred_button = Button(right_side_frame)
            self.save_pred_button["text"] = ' SAVE PRED '
            self.save_pred_button["fg"] = self.color_text_gui_default
            self.save_pred_button['state'] = DISABLED  # ''normal
            self.save_pred_button["command"] = event_lambda(self.save_checked_predictions)
            self.save_pred_button.pack(side=TOP)

            self.load_pred_button = Button(right_side_frame)
            self.load_pred_button["text"] = ' LOAD PRED '
            self.load_pred_button["fg"] = self.color_text_gui_default
            self.load_pred_button['state'] = 'normal'  # ''normal
            self.load_pred_button["command"] = event_lambda(self.load_checked_predictions)
            self.load_pred_button.pack(side=TOP)

        # to vertically center the buttons
        empty_label = Label(right_side_frame)
        empty_label["text"] = ""
        empty_label.pack(side=TOP, expand=YES, fill=BOTH)

        # used for association of keys
        self.keys_pressed = dict()
        self.root.bind_all("<KeyRelease>", self.key_release_action)
        self.root.bind_all("<KeyPress>", self.key_press_action)

    def neuron_entry_change(self, *args):
        print(f"Neuron: {self.neuron_string_var.get()}")

    def switch_michou(self):
        # can't display images while playing the movie
        if (not self.display_michou) and self.play_movie:
            return

        if self.display_michou:
            self.display_michou = False
            if not self.play_movie:
                self.update_plot_map_img(after_michou=True)
        else:
            self.display_michou = True
            self.michou_img_to_display = randint(0, self.n_michou_img - 1)
            self.update_plot_map_img()

    def switch_movie_mode(self, from_movie_button=True):
        if from_movie_button and (not self.movie_mode):
            self.swith_all_click_actions(initiator="switch_movie_mode")

        if self.movie_mode:
            self.movie_button["text"] = ' movie OFF '
            self.movie_button["fg"] = "black"
            self.movie_mode = False
            if self.play_movie:
                self.play_movie = False
                if self.anim_movie is not None:
                    self.anim_movie.event_source.stop()
                    self.update_contour_for_cell(cell=self.current_neuron)
                    self.update_plot_map_img(after_michou=True, after_movie=True)
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
        self.unsaved()

    def remove_cell(self):
        if self.cells_to_remove[self.current_neuron] == 0:
            self.remove_cell_button["text"] = " removed "
            self.remove_cell_button["fg"] = "red"
            self.cells_to_remove[self.current_neuron] = 1
        else:
            self.remove_cell_button["text"] = " not removed "
            self.remove_cell_button["fg"] = "black"
            self.cells_to_remove[self.current_neuron] = 0
        self.update_plot()
        self.unsaved()

    def clear_and_update_entry_neuron_widget(self):
        self.neuron_entry_widget.delete(first=0, last=END)
        self.neuron_entry_widget.insert(0, f"{self.current_neuron}")

    def update_uncertain_prediction_values(self, event=None):
        """
        Update the widget that contain the limit of the prediction we want to look at
        :param event:
        :return:
        """
        pred_value_1 = self.pred_value_1_entry_widget.get()
        pred_value_2 = self.pred_value_2_entry_widget.get()
        error = False
        try:
            pred_value_1 = float(pred_value_1)
            pred_value_2 = float(pred_value_2)
        except (ValueError, TypeError) as e:
            # error if a value that is not an int is selected
            error = True
        print(f"pred_value_1 {pred_value_1}, pred_value_2 {pred_value_2}")
        if not error:
            if pred_value_1 < 0:
                error = True
            if pred_value_2 > 1:
                error = True
            if pred_value_2 <= pred_value_1:
                error = True
        if error:
            self.pred_value_1_entry_widget.delete(first=0, last=END)
            self.pred_value_1_entry_widget.insert(0, f"{self.uncertain_prediction_values[0]}")

            self.pred_value_2_entry_widget.delete(first=0, last=END)
            self.pred_value_2_entry_widget.insert(0, f"{self.uncertain_prediction_values[1]}")
            return

        self.uncertain_prediction_values = (np.round(pred_value_1, 2), np.round(pred_value_2, 2))
        self.update_transient_prediction_periods_to_check()

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
        # print(f"event.keysym_num {event.keysym_num}")
        # if 0, means the key was not recognized, we empty the dict
        if event.keysym_num == 0:
            self.keys_pressed = dict()
        # print(f"released keysym {event.keysym}, keycode {event.keycode}, keysym_num {event.keysym_num}, "
        #       f"state {event.state}")
        # print(f"event.keysym {event.keysym}")
        if event.char in ["o", "O", "+"]:
            self.add_onset_switch_mode()
        elif event.char in ["p", "P", "*"]:
            self.add_peak_switch_mode()
        elif event.char in ["a", "A"]:
            self.move_zoom(to_the_left=False)
            # so the back button will come back to the curren view
            self.toolbar.push_current()
        elif event.char in ["q", "Q"]:
            self.move_zoom(to_the_left=True)
            # so the back button will come back to the curren view
            self.toolbar.push_current()
        elif event.char in ["r", "R", "-"]:
            self.remove_all_switch_mode()
        elif event.char in ["B", "G"]:
            if self.display_michou is False:
                self.display_michou = True
                self.update_plot_map_img()
            start_time = time.time()
            self.save_sources_profile_map(key_cmap=event.char)
            stop_time = time.time()
            print(f"Time for producing source profiles map: "
                  f"{np.round(stop_time - start_time, 3)} s")
            if self.display_michou is True:
                self.display_michou = False
                self.update_plot_map_img(after_michou=True)
        elif event.char in ["i", "I"]:
            self.switch_magnifier()
        elif event.char in ["p", "P"]:
            if self.n_michou_img > 0:
                self.switch_michou()
        elif event.char in ["z", "Z"]:
            self.activate_movie_zoom(from_check_box=False)
        elif event.char in ["S"]:
            self.save_spike_nums()
        elif event.char in ["s"] and (self.tiff_movie is not None):
            self.switch_source_profile_mode(from_key_shortcut=True)
            # print(f"s self.keys_pressed {self.keys_pressed}")
            # if "Control_L" in self.keys_pressed:
            #     print("s with control")
        elif (event.keysym == "space") and (self.tiff_movie is not None):
            self.switch_movie_mode()
        elif (event.keysym in ["y", "Y"]) and (self.agree_button is not None):
            self.agree_switch_mode()
        elif (event.keysym in ["n", "N"]) and (self.dont_agree_button is not None):
            self.dont_agree_switch_mode()

        if event.keysym == 'Right':
            # C as cell
            if ("Meta_L" in self.keys_pressed) or ("Control_L" in self.keys_pressed):
                # ctrl-right, we move to next neuron
                self.select_next_neuron()
            else:
                self.move_zoom(to_the_left=False)
                # so the back button will come back to the curren view
                self.toolbar.push_current()
        elif event.keysym == 'Left':
            if ("Meta_L" in self.keys_pressed) or ("Control_L" in self.keys_pressed):
                self.select_previous_neuron()
            else:
                self.move_zoom(to_the_left=True)
                self.toolbar.push_current()

    #
    # def detect_peaks(self):
    #     """
    #     Compute peak_nums, with nb of times equal to traces's one
    #     :return:
    #     """
    #     peak_nums = np.zeros((self.nb_neurons, self.nb_times_traces), dtype="float")
    #
    #     for neuron, o_t in enumerate(self.onset_times):
    #         # neuron by neuron
    #         onset_index = np.where(o_t > 0)[0]
    #         for index in onset_index:
    #             # looking for the peak following the onset
    #             limit_index_search = min((index + (self.decay * self.decay_factor)), self.nb_times_traces - 1)
    #             if index == limit_index_search:
    #                 continue
    #             max_index = np.argmax(self.traces[neuron, index:limit_index_search])
    #             max_index += index
    #             if np.size(max_index) == 1:  # or isinstance(max_index, int)
    #                 peak_nums[neuron, max_index] = self.traces[neuron, max_index]
    #             else:
    #                 peak_nums[neuron, max_index] = self.traces[neuron, max_index[0]]
    #
    #     return peak_nums

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

    def numbers_of_onset_to_agree(self):
        if self.to_agree_spike_nums is None:
            return 0
        return len(np.where(self.to_agree_spike_nums[self.current_neuron, :] > 0)[0])

    def numbers_of_peak_to_agree(self):
        if self.to_agree_peak_nums is None:
            return 0
        return len(np.where(self.to_agree_peak_nums[self.current_neuron, :] > 0)[0])

    def swith_all_click_actions(self, initiator):
        if (initiator != "remove_onset_switch_mode") and self.remove_onset_mode:
            self.remove_onset_switch_mode(from_remove_onset_button=False)
        if (initiator != "remove_peak_switch_mode") and self.remove_peak_mode:
            self.remove_peak_switch_mode(from_remove_peak_button=False)
        if (initiator != "add_peak_switch_mode") and self.add_peak_mode:
            self.add_peak_switch_mode(from_add_peak_button=False)
        if (initiator != "remove_all_switch_mode") and self.remove_all_mode:
            self.remove_all_switch_mode(from_remove_all_button=False)
        if (initiator != "switch_movie_mode") and self.movie_mode:
            self.switch_movie_mode(from_movie_button=False)
        if (initiator != "add_onset_switch_mode") and self.add_onset_mode:
            self.add_onset_switch_mode(from_add_onset_button=False)
        if (initiator != "switch_source_profile_mode") and self.source_mode:
            self.switch_source_profile_mode(from_check_box=False, from_key_shortcut=True)
        if (initiator != "add_doubtful_frames_switch_mode") and self.add_doubtful_frames_mode:
            self.add_doubtful_frames_switch_mode(from_add_doubtful_frames_button=False)
        if (initiator != "remove_doubtful_frames_switch_mode") and self.remove_doubtful_frames_mode:
            self.remove_doubtful_frames_switch_mode(from_remove_doubtful_frames_button=False)
        if (initiator != "add_mvt_frames_switch_mode") and self.add_mvt_frames_mode:
            self.add_mvt_frames_switch_mode(from_add_mvt_frames_button=False)
        if (initiator != "remove_mvt_frames_switch_mode") and self.remove_mvt_frames_mode:
            self.remove_mvt_frames_switch_mode(from_remove_mvt_frames_button=False)
        if (initiator != "agree_switch_mode") and self.agree_mode:
            self.agree_switch_mode(from_agree_button=False)
        if (initiator != "dont_agree_switch_mode") and self.dont_agree_mode:
            self.dont_agree_switch_mode(from_dont_agree_button=False)

    def add_onset_switch_mode(self, from_add_onset_button=True):
        # if it was called due to the action of pressing the remove button, we're not calling the remove switch mode
        if from_add_onset_button and (not self.add_onset_mode):
            self.swith_all_click_actions(initiator="add_onset_switch_mode")
        self.add_onset_mode = not self.add_onset_mode
        if self.add_onset_mode:
            self.add_onset_button["fg"] = 'green'
            self.add_onset_button["text"] = ' + ONSET ON '
        else:
            self.add_onset_button["fg"] = 'red'
            self.add_onset_button["text"] = ' + ONSET OFF '

    def remove_onset_switch_mode(self, from_remove_onset_button=True):
        # deactivating other button
        if from_remove_onset_button and (not self.remove_onset_mode):
            self.swith_all_click_actions(initiator="remove_onset_switch_mode")
        self.remove_onset_mode = not self.remove_onset_mode

        if self.remove_onset_mode:
            self.remove_onset_button["fg"] = 'green'
            self.remove_onset_button["text"] = ' - ONSET ON '
            self.first_click_to_remove = None
        else:
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
            self.remove_onset_button["fg"] = 'red'
            self.remove_onset_button["text"] = ' - ONSET OFF '

    def add_doubtful_frames_switch_mode(self, from_add_doubtful_frames_button=True):
        if from_add_doubtful_frames_button and (not self.add_doubtful_frames_mode):
            self.swith_all_click_actions(initiator="add_doubtful_frames_switch_mode")

        self.add_doubtful_frames_mode = not self.add_doubtful_frames_mode
        if self.add_doubtful_frames_mode:
            self.add_doubtful_frames_mode_button["fg"] = 'green'
            self.add_doubtful_frames_mode_button["text"] = ' + DOUBT ON '
            self.first_click_to_remove = None
        else:
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
            self.add_doubtful_frames_mode_button["fg"] = 'red'
            self.add_doubtful_frames_mode_button["text"] = ' + DOUBT OFF '

    def remove_doubtful_frames_switch_mode(self, from_remove_doubtful_frames_button=True):
        # deactivating other button
        if from_remove_doubtful_frames_button and (not self.remove_doubtful_frames_mode):
            self.swith_all_click_actions(initiator="remove_doubtful_frames_switch_mode")

        self.remove_doubtful_frames_mode = not self.remove_doubtful_frames_mode
        if self.remove_doubtful_frames_mode:
            self.remove_doubtful_frames_button["fg"] = 'green'
            self.remove_doubtful_frames_button["text"] = ' - DOUBT ON '
            self.first_click_to_remove = None
        else:
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
            self.remove_doubtful_frames_button["fg"] = 'red'
            self.remove_doubtful_frames_button["text"] = ' - DOUBT OFF '

    def add_mvt_frames_switch_mode(self, from_add_mvt_frames_button=True):
        if from_add_mvt_frames_button and (not self.add_mvt_frames_mode):
            self.swith_all_click_actions(initiator="add_mvt_frames_switch_mode")

        self.add_mvt_frames_mode = not self.add_mvt_frames_mode
        if self.add_mvt_frames_mode:
            self.add_mvt_frames_mode_button["fg"] = 'green'
            self.add_mvt_frames_mode_button["text"] = ' + MVT ON '
            self.first_click_to_remove = None
        else:
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
            self.add_mvt_frames_mode_button["fg"] = 'red'
            self.add_mvt_frames_mode_button["text"] = ' + MVT OFF '

    def remove_mvt_frames_switch_mode(self, from_remove_mvt_frames_button=True):
        # deactivating other button
        if from_remove_mvt_frames_button and (not self.remove_mvt_frames_mode):
            self.swith_all_click_actions(initiator="remove_mvt_frames_switch_mode")

        self.remove_mvt_frames_mode = not self.remove_mvt_frames_mode
        if self.remove_mvt_frames_mode:
            self.remove_mvt_frames_button["fg"] = 'green'
            self.remove_mvt_frames_button["text"] = ' - MVT ON '
            self.first_click_to_remove = None
        else:
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
            self.remove_mvt_frames_button["fg"] = 'red'
            self.remove_mvt_frames_button["text"] = ' - MVT OFF '

    def update_transient_prediction_periods_to_check(self):
        # checking if a prediction results are already loaded in mouse_sessions
        # self.transient_prediction contains prediction for each cell
        if len(self.transient_prediction) == 0:
            return
        # first key is an int (cell), and the value is a list of tuple representing the border of the period
        # with prediction being uncertain
        # list of tupe, first int is the cell index, second is the first frame of the beginning of the window, and
        # last frame is the last frame of the window
        self.transient_prediction_periods_to_check = []
        for cell in np.arange(self.nb_neurons):
            if cell not in self.transient_prediction:
                continue
            predictions = self.transient_prediction[cell]
            predicted_raster_dur = np.zeros(predictions.shape[0], dtype="int8")
            if predictions.shape[1] == 1:
                real_transient_frames = predictions[:, 0] >= self.uncertain_prediction_values[0]
            elif predictions.shape[1] == 3:
                # real transient, fake ones, other (neuropil, decay etc...)
                # keeping predictions about real transient when superior
                # to other prediction on the same frame
                max_pred_by_frame = np.max(predictions, axis=1)
                real_transient_frames = np.logical_and((predictions[:, 0] >= self.uncertain_prediction_values[0]),
                                                       (predictions[:, 0] == max_pred_by_frame))
            elif predictions.shape[1] == 2:
                # real transient, fake ones
                # keeping predictions about real transient superior to the threshold
                # and superior to other prediction on the same frame
                max_pred_by_frame = np.max(predictions, axis=1)
                real_transient_frames = np.logical_and((predictions[:, 0] >= self.uncertain_prediction_values[0]),
                                                       (predictions[:, 0] == max_pred_by_frame))
            predicted_raster_dur[real_transient_frames] = 1
            active_periods = get_continous_time_periods(predicted_raster_dur)
            # now we want to remove all period for whch prediction is > at self.uncertain_prediction_values[1]
            # aka the upper threshold
            active_periods_to_keep = []
            # number of frames before and after the middle of the transient
            n_frames_around = 100
            for period in active_periods:
                max_pred = np.max(predictions[period[0]:period[1] + 1, 0])
                if max_pred <= self.uncertain_prediction_values[1]:
                    middle = int((period[0] + period[1]) / 2)
                    first_frame = max(0, middle - n_frames_around)
                    # last frame included
                    last_frame = min(self.nb_times, middle + n_frames_around)
                    self.transient_prediction_periods_to_check.append((cell, first_frame, last_frame,
                                                                       np.round(max_pred, 2), middle))
                    # print(f"{cell}: {period}: len(np.arange(first_frame, last_frame)) "
                    #       f"{len(np.arange(first_frame, last_frame+1))}")
        # then we update the list_box
        self.update_predictions_list_box()

    def update_predictions_list_box(self):
        self.predictions_list_box.delete('0', 'end')
        for period_index, period in enumerate(self.transient_prediction_periods_to_check):
            self.predictions_list_box.insert(END, f"{period[0]} / {period[3]} / {period[1]}-{period[2]}")
            if period in self.transient_prediction_periods_checked:
                self.predictions_list_box.itemconfig(period_index, bg=self.color_bg_predicted_transient_list_box)
            else:
                self.predictions_list_box.itemconfig(period_index, bg="white")
            # self.predictions_list_box.itemconfig(period_index, fg="red")

    def update_checked_predictions_list_box(self):
        self.checked_predictions_list_box.delete('0', 'end')
        for period_index, period in enumerate(self.transient_prediction_periods_checked):
            self.checked_predictions_list_box.insert(END, f"{period[0]} / {period[3]} / {period[1]}-{period[2]}")
        if len(self.transient_prediction_periods_checked) > 0:
            self.save_pred_button['state'] = 'normal'
        else:
            self.save_pred_button['state'] = DISABLED

    def checked_predictions_list_box_double_click(self, evt):
        cur_selection = evt.widget.curselection()
        # print(f"cur_selection {cur_selection}")
        if len(cur_selection) == 0:
            return
        index_cliked = int(cur_selection[0])
        period = self.transient_prediction_periods_checked[index_cliked]
        try:
            index_predictions_list_box = self.transient_prediction_periods_to_check.index(period)
            self.predictions_list_box.itemconfig(index_predictions_list_box, bg="white")
        except ValueError:
            # it might happen if the list of transients has been updated
            pass
        # now we removed this period from the checked periods and update the listbox
        del self.transient_prediction_periods_checked[index_cliked]
        self.update_checked_predictions_list_box()

    def change_prediction_transient_to_look_at(self, period):
        """
        CHange the plot such as the period indicated is displaued
        :param period: a tuple of len 4: cell, x_left, x_right, pred
        :return:
        """
        new_neuron = period[0]
        new_left_x_limit, new_right_x_limit = (period[1], period[2])
        # set the cell and frame in which a vertical line will be draw to indicate at which transient we're looking at
        self.last_predicted_transient_selected = (period[0], period[4])
        frames_around = 0
        new_left_x_limit = max(0, new_left_x_limit - frames_around)
        new_right_x_limit = min(new_right_x_limit + frames_around, self.nb_times_traces - 1)

        new_top_y_limit = np.max(self.traces[new_neuron, new_left_x_limit:new_right_x_limit + 1]) + 0.5
        new_bottom_y_limit = np.min(self.raw_traces[new_neuron, new_left_x_limit:new_right_x_limit + 1]) - 0.5

        new_x_limit = (new_left_x_limit, new_right_x_limit)
        new_y_limit = (new_bottom_y_limit, new_top_y_limit)
        if new_neuron == self.current_neuron:
            self.update_plot(new_x_limit=new_x_limit, new_y_limit=new_y_limit, amplitude_zoom_fit=False)
        else:
            self.update_neuron(new_neuron=new_neuron,
                               new_x_limit=new_x_limit, new_y_limit=new_y_limit, amplitude_zoom_fit=False)

    def predictions_list_box_click(self, evt):
        cur_selection = evt.widget.curselection()
        if len(cur_selection) == 0:
            return
        index_cliked = int(cur_selection[0])
        period = self.transient_prediction_periods_to_check[index_cliked]
        self.change_prediction_transient_to_look_at(period=period)

    def checked_predictions_list_box_click(self, evt):
        cur_selection = evt.widget.curselection()
        if len(cur_selection) == 0:
            return
        index_cliked = int(cur_selection[0])
        period = self.transient_prediction_periods_checked[index_cliked]
        self.change_prediction_transient_to_look_at(period=period)

    def predictions_list_box_double_click(self, evt):
        cur_selection = evt.widget.curselection()
        if len(cur_selection) == 0:
            return
        index_cliked = int(cur_selection[0])
        period = self.transient_prediction_periods_to_check[index_cliked]
        if period in self.transient_prediction_periods_checked:
            # it means it has already been added to checked transients
            return
        self.transient_prediction_periods_checked.append(period)
        self.predictions_list_box.itemconfig(index_cliked, bg=self.color_bg_predicted_transient_list_box)
        self.update_checked_predictions_list_box()

    def update_contour_for_cell(self, cell):
        # used in order to have access to contour after animation
        # cell contour
        coord = self.data_and_param.ms.coord_obj.coord[cell]
        xy = coord.transpose()
        self.cell_contours[cell] = patches.Polygon(xy=xy,
                                                   fill=False, linewidth=0, facecolor="red",
                                                   edgecolor="red",
                                                   zorder=15, lw=0.6)

    def agree_switch_mode(self, from_agree_button=True):
        if from_agree_button and (not self.agree_mode):
            self.swith_all_click_actions(initiator="agree_switch_mode")
        self.agree_mode = not self.agree_mode

        if self.agree_mode:
            self.agree_button["fg"] = 'green'
            # in case one click would have been made when remove onset was activated
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
        else:
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
            self.agree_button["fg"] = 'red'

    def dont_agree_switch_mode(self, from_dont_agree_button=True):
        if from_dont_agree_button and (not self.dont_agree_mode):
            self.swith_all_click_actions(initiator="dont_agree_switch_mode")
        self.dont_agree_mode = not self.dont_agree_mode

        if self.dont_agree_mode:
            self.dont_agree_button["fg"] = 'green'
            # in case one click would have been made when remove onset was activated
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
        else:
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
            self.dont_agree_button["fg"] = 'red'

    def remove_peak_switch_mode(self, from_remove_peak_button=True):
        if from_remove_peak_button and (not self.remove_peak_mode):
            self.swith_all_click_actions(initiator="remove_peak_switch_mode")
        self.remove_peak_mode = not self.remove_peak_mode

        if self.remove_peak_mode:
            self.remove_peak_button["fg"] = 'green'
            self.remove_peak_button["text"] = ' - PEAK ON '
            # in case one click would have been made when remove onset was activated
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
        else:
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
            self.remove_peak_button["fg"] = 'red'
            self.remove_peak_button["text"] = ' - PEAK OFF '

    def remove_all_switch_mode(self, from_remove_all_button=True):
        if from_remove_all_button and (not self.remove_all_mode):
            self.swith_all_click_actions(initiator="remove_all_switch_mode")
        self.remove_all_mode = not self.remove_all_mode

        if self.remove_all_mode:
            self.remove_all_button["fg"] = 'green'
            self.remove_all_button["text"] = ' - ALL ON '
            # in case one click would have been made when remove onset was activated
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
        else:
            if self.first_click_to_remove is not None:
                self.first_click_to_remove = None
                self.update_plot()
            self.remove_all_button["fg"] = 'red'
            self.remove_all_button["text"] = ' - ALL OFF '

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
                and (not self.add_peak_mode) and (not self.add_doubtful_frames_mode) \
                and (not self.remove_doubtful_frames_mode) and (not self.add_mvt_frames_mode) \
                and (not self.remove_mvt_frames_mode) \
                and (not self.remove_all_mode) and (not self.movie_mode) \
                and (not self.source_mode) and (not self.agree_mode) and (not self.dont_agree_mode):
            return

        if event.dblclick:
            return

        if event.xdata is None:
            return

        if self.last_click_position[0] != event.xdata:
            # the mouse has been moved between the pressing and the release
            return

        if self.add_onset_mode or self.add_peak_mode or self.source_mode:
            if (event.xdata < 0) or (event.xdata > (self.nb_times_traces - 1)):
                return
            if self.add_onset_mode:
                self.add_onset(int(round(event.xdata)))
            elif self.add_peak_mode:
                self.add_peak(at_time=int(round(event.xdata)), amplitude=event.ydata)
            elif self.source_mode:
                # we want to display the source and transient profile of the selected transient
                # print(f"self.source_mode click release {event.xdata}")
                transient = None
                # creating temporary variable so we can add to_agree_onsets & peaks in case it exists
                if self.to_agree_spike_nums is not None:
                    tmp_onset_times = np.copy(self.onset_times[self.current_neuron])
                    tmp_onset_times[self.to_agree_spike_nums[self.current_neuron] > 0] = 1
                    tmp_peak_nums = np.copy(self.peak_nums[self.current_neuron])
                    tmp_peak_nums[self.to_agree_peak_nums[self.current_neuron] > 0] = 1
                else:
                    tmp_onset_times = self.onset_times[self.current_neuron]
                    tmp_peak_nums = self.peak_nums[self.current_neuron]
                # we check if we are between an onset and a peak
                onsets_frames = np.where(tmp_onset_times)[0]
                peaks_frames = np.where(tmp_peak_nums)[0]
                # closest onset before the click
                onsets_before_index = np.where(onsets_frames <= event.xdata)[0]
                if len(onsets_before_index) == 0:
                    # print("len(onsets_before_index) == 0")
                    return
                first_onset_before_frame = onsets_frames[onsets_before_index[-1]]
                # print(f"first_onset_before_frame {first_onset_before_frame}")
                # closest peak before onset
                peaks_before_index = np.where(peaks_frames <= event.xdata)[0]
                # onset should be after peak, otherwise it means the click was not between an onset and a peak and
                # so we choose the previous transient
                if len(peaks_before_index) != 0:
                    first_peak_before_frame = peaks_frames[peaks_before_index[-1]]
                    # print(f"first_peak_before_frame {first_peak_before_frame}")
                    if first_peak_before_frame > first_onset_before_frame:
                        transient = (first_onset_before_frame, first_peak_before_frame)

                if transient is None:
                    # closet peak after the click
                    peaks_after_index = np.where(peaks_frames >= event.xdata)[0]
                    # means we are at the end
                    if len(peaks_after_index) == 0:
                        return
                    first_peak_after_frame = peaks_frames[peaks_after_index[0]]
                    # print(f"first_peak_after_frame {first_peak_after_frame}")
                    # closest onset after the click
                    onsets_after_index = np.where(onsets_frames >= event.xdata)[0]
                    # the peak should be before the onset
                    if len(onsets_after_index) != 0:
                        first_onset_after_frame = onsets_frames[onsets_after_index[0]]
                        # print(f"first_onset_after_frame {first_onset_after_frame}")
                        if first_onset_after_frame < first_peak_after_frame:
                            # print(f"first_onset_after_frame < first_peak_after_frame")
                            return
                    transient = (first_onset_before_frame, first_peak_after_frame)
                # print(f"transient {transient}")
                self.click_corr_coord = {"x": int(round(event.xdata)), "y": event.ydata}
                self.update_plot()
                self.plot_source_transient(transient=transient)
            return

        if self.remove_onset_mode or self.remove_peak_mode or \
                self.remove_all_mode or self.movie_mode or \
                self.add_doubtful_frames_mode or self.remove_doubtful_frames_mode or \
                self.add_mvt_frames_mode or self.remove_mvt_frames_mode or self.agree_mode or self.dont_agree_mode:
            if self.first_click_to_remove is not None:
                if self.remove_onset_mode:
                    self.remove_onset(x_from=self.first_click_to_remove["x"], x_to=int(round(event.xdata)))
                elif self.remove_peak_mode:
                    self.remove_peak(x_from=self.first_click_to_remove["x"], x_to=int(round(event.xdata)))
                elif self.movie_mode:
                    self.start_playing_movie(x_from=self.first_click_to_remove["x"], x_to=int(round(event.xdata)))
                elif self.add_doubtful_frames_mode:
                    self.add_doubtful_frames(x_from=self.first_click_to_remove["x"], x_to=int(round(event.xdata)))
                elif self.remove_doubtful_frames_mode:
                    self.remove_doubtful_frames(x_from=self.first_click_to_remove["x"], x_to=int(round(event.xdata)))
                elif self.add_mvt_frames_mode:
                    self.add_mvt_frames(x_from=self.first_click_to_remove["x"], x_to=int(round(event.xdata)))
                elif self.remove_mvt_frames_mode:
                    self.remove_mvt_frames(x_from=self.first_click_to_remove["x"], x_to=int(round(event.xdata)))
                elif self.remove_all_mode:
                    self.remove_all(x_from=self.first_click_to_remove["x"], x_to=int(round(event.xdata)))
                elif self.agree_mode:
                    self.agree_on_fusion(x_from=self.first_click_to_remove["x"], x_to=int(round(event.xdata)))
                elif self.dont_agree_mode:
                    self.dont_agree_on_fusion(x_from=self.first_click_to_remove["x"], x_to=int(round(event.xdata)))
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
            self.update_last_action(RemoveOnsetAction(removed_times=removed_times, session_frame=self,
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

    def dont_agree_on_fusion(self, x_from, x_to):
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
        # if the sum is zero, then we're not removing any onsets or peaks doubtful
        modification_done = (np.sum(self.to_agree_spike_nums[self.current_neuron, x_from:x_to]) > 0) or \
                            (np.sum(self.to_agree_peak_nums[self.current_neuron, x_from:x_to]) > 0)
        if modification_done:
            # for the UNDO button
            left_x_limit, right_x_limit = self.axe_plot.get_xlim()
            bottom_limit, top_limit = self.axe_plot.get_ylim()

            dont_agree_onset_action = None
            # we remove onsets from the to_agree in the interval
            not_agreed_onsets_index = np.where((self.to_agree_spike_nums[self.current_neuron, x_from:x_to]))[0]
            if len(not_agreed_onsets_index) > 0:
                not_agreed_onsets_index += x_from
                self.to_agree_spike_nums[self.current_neuron, not_agreed_onsets_index] = 0
                dont_agree_onset_action = DontAgreeOnsetAction(not_agreed_onsets_index=not_agreed_onsets_index,
                                                               session_frame=self,
                                                               neuron=self.current_neuron, is_saved=self.is_saved,
                                                               x_limits=(left_x_limit, right_x_limit),
                                                               y_limits=(bottom_limit, top_limit))

            # we remove onsets from the to_agree in the interval and add them to onsets_times
            not_agreed_peaks_index = np.where((self.to_agree_peak_nums[self.current_neuron, x_from:x_to]))[0]
            if len(not_agreed_peaks_index) > 0:
                not_agreed_peaks_index += x_from
                self.to_agree_peak_nums[self.current_neuron, not_agreed_peaks_index] = 0

                left_x_limit, right_x_limit = self.axe_plot.get_xlim()
                bottom_limit, top_limit = self.axe_plot.get_ylim()
                self.update_last_action(DontAgreePeakAction(not_agreed_peaks_index=not_agreed_peaks_index,
                                                            session_frame=self,
                                                            dont_agree_onset_action=dont_agree_onset_action,
                                                            neuron=self.current_neuron, is_saved=self.is_saved,
                                                            x_limits=(left_x_limit, right_x_limit),
                                                            y_limits=(bottom_limit, top_limit)))
            elif dont_agree_onset_action is not None:
                self.update_last_action(dont_agree_onset_action)

            # no more undone_actions
            self.undone_actions = []
            self.redo_button['state'] = DISABLED
            self.unsaved()
            self.undo_button['state'] = 'normal'
            self.update_to_agree_label()
        # update to remove the cross of the first click
        self.update_after_onset_change()

    def agree_on_fusion(self, x_from, x_to):
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
        # if the sum is zero, then we're not keeping any onsets or peaks doubtful
        modification_done = (np.sum(self.to_agree_spike_nums[self.current_neuron, x_from:x_to]) > 0) or \
                            (np.sum(self.to_agree_peak_nums[self.current_neuron, x_from:x_to]) > 0)
        if modification_done:
            # for the UNDO button
            left_x_limit, right_x_limit = self.axe_plot.get_xlim()
            bottom_limit, top_limit = self.axe_plot.get_ylim()

            agree_onset_action = None
            # we remove onsets from the to_agree in the interval and add them to onsets_times
            agreed_onsets_index = np.where((self.to_agree_spike_nums[self.current_neuron, x_from:x_to]))[0]
            if len(agreed_onsets_index) > 0:
                agreed_onsets_index += x_from
                self.to_agree_spike_nums[self.current_neuron, agreed_onsets_index] = 0
                self.onset_times[self.current_neuron, agreed_onsets_index] = 1
                self.spike_nums[self.current_neuron, agreed_onsets_index] = 1
                agree_onset_action = AgreeOnsetAction(agreed_onsets_index=agreed_onsets_index, session_frame=self,
                                                      neuron=self.current_neuron, is_saved=self.is_saved,
                                                      x_limits=(left_x_limit, right_x_limit),
                                                      y_limits=(bottom_limit, top_limit))

            # we remove onsets from the to_agree in the interval and add them to onsets_times
            agreed_peaks_index = np.where((self.to_agree_peak_nums[self.current_neuron, x_from:x_to]))[0]
            if len(agreed_peaks_index) > 0:
                agreed_peaks_index += x_from
                self.to_agree_peak_nums[self.current_neuron, agreed_peaks_index] = 0
                self.peak_nums[self.current_neuron, agreed_peaks_index] = 1

                left_x_limit, right_x_limit = self.axe_plot.get_xlim()
                bottom_limit, top_limit = self.axe_plot.get_ylim()
                self.update_last_action(AgreePeakAction(agreed_peaks_index=agreed_peaks_index,
                                                        session_frame=self, agree_onset_action=agree_onset_action,
                                                        neuron=self.current_neuron, is_saved=self.is_saved,
                                                        x_limits=(left_x_limit, right_x_limit),
                                                        y_limits=(bottom_limit, top_limit)))
            elif agree_onset_action is not None:
                self.update_last_action(agree_onset_action)

            # no more undone_actions
            self.undone_actions = []
            self.redo_button['state'] = DISABLED
            self.unsaved()
            self.undo_button['state'] = 'normal'
            self.update_to_agree_label()
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
            self.update_last_action(RemovePeakAction(removed_times=removed_times, amplitudes=amplitudes,
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

        # peaks = np.where(self.peak_nums[self.current_neuron, left_x_limit:right_x_limit] > 0)[0]
        # peaks += left_x_limit
        # threshold = self.get_threshold()
        # peaks_under_threshold = np.where(self.traces[self.current_neuron, peaks] < threshold)[0]
        # if len(peaks_under_threshold) == 0:
        #     return
        # removed_times = peaks[peaks_under_threshold]
        # amplitudes = self.peak_nums[self.current_neuron, peaks][peaks_under_threshold]

        if len(self.peaks_under_threshold_index) == 0:
            return

        peaks_to_remove = np.where(np.logical_and(self.peaks_under_threshold_index >= left_x_limit,
                                                  self.peaks_under_threshold_index <= right_x_limit))[0]
        if len(peaks_to_remove) == 0:
            return

        removed_times = self.peaks_under_threshold_index[peaks_to_remove]

        # should be useless, as we don't keep amplitudes anymore, but too lazy to change the code that follows
        amplitudes = self.peak_nums[self.current_neuron, removed_times]

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
        self.update_last_action(RemovePeakAction(removed_times=removed_times, amplitudes=amplitudes,
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
            self.update_last_action(RemovePeakAction(removed_times=removed_times, amplitudes=amplitudes,
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

    def switch_source_profile_mode(self, from_check_box=True, from_key_shortcut=False):
        # if from_key_shortcut is True, means we need to uncheck the check_box
        if from_check_box and (not self.source_mode):
            self.swith_all_click_actions(initiator="switch_source_profile_mode")
        if not from_key_shortcut:
            self.source_mode = not self.source_mode
        else:
            if not self.source_mode:
                self.source_check_box.select()
            else:
                self.source_check_box.deselect()
            self.source_mode = not self.source_mode
        if not self.source_mode:
            self.first_click_to_remove = None
            self.update_plot()

    def activate_movie_zoom(self, from_check_box=True):
        if from_check_box:
            self.movie_zoom_mode = not self.movie_zoom_mode
        else:
            # from z keyboard
            if self.tiff_movie is None:
                return
            if not self.movie_zoom_mode:
                self.zoom_movie_check_box.select()
            else:
                self.zoom_movie_check_box.deselect()
            self.movie_zoom_mode = not self.movie_zoom_mode

    def display_raw_tracecheck_box_action(self):
        self.display_raw_traces = not self.display_raw_traces

        self.update_plot(raw_trace_display_action=True)

    def set_transient_classifier_prediction_for_cell(self, cell):
        if cell in self.transient_prediction:
            return
        use_data_augmentation = False
        overlap_value = 0
        predictions = predict_transient_from_saved_model(ms=self.data_and_param.ms, cell=cell,
                                                         weights_file=self.transient_classifier_weights_file,
                                                         json_file=self.transient_classifier_json_file,
                                                         overlap_value=overlap_value,
                                                         use_data_augmentation=use_data_augmentation,
                                                         buffer=0)
        # predictions as two dimension, first one represents the frame, the second one the prediction
        # for each class
        # print(f"predictions.shape {predictions.shape}")
        # raise Exception("TOTO TOTO")
        self.transient_prediction[cell] = predictions
        self.transient_prediction_periods[cell] = dict()

    def transient_classifier_check_box_action(self):
        self.show_transient_classifier = not self.show_transient_classifier
        if self.show_transient_classifier:
            self.set_transient_classifier_prediction_for_cell(self.current_neuron)
            self.print_transients_predictions_stat = True
        self.update_plot()

    def correlation_check_box_action(self, from_std_treshold=False):
        if self.display_threshold and not from_std_treshold:
            self.threshold_check_box_action(from_correlation=True)

        self.display_correlations = not self.display_correlations
        if not from_std_treshold:
            if self.display_correlations:
                if self.display_michou is False:
                    self.display_michou = True
                    self.update_plot_map_img()
                start_time = time.time()
                # computing correlations between source and transients profile for this cell and the overlaping ones
                self.compute_source_and_transients_correlation(main_cell=self.current_neuron)
                stop_time = time.time()
                print(f"Time for computing source and transients correlation for cell {self.current_neuron}: "
                      f"{np.round(stop_time - start_time, 3)} s")
                if self.display_michou is True:
                    self.display_michou = False
                    self.update_plot_map_img(after_michou=True)

            if self.display_correlations:
                self.remove_peaks_under_threshold_button['state'] = 'normal'
            else:
                self.remove_peaks_under_threshold_button['state'] = DISABLED

            self.update_plot()
        else:
            self.correlation_check_box.deselect()

    def threshold_check_box_action(self, from_correlation=False):
        if self.display_correlations and not from_correlation:
            self.correlation_check_box_action(from_std_treshold=True)

        self.display_threshold = not self.display_threshold
        if not from_correlation:
            if self.display_threshold:
                self.remove_peaks_under_threshold_button['state'] = 'normal'
            else:
                self.remove_peaks_under_threshold_button['state'] = DISABLED

            self.update_plot()
        else:
            self.threshold_check_box.deselect()

    def spin_box_transient_classifier_update(self):
        self.transient_classifier_threshold = float(self.spin_box_transient_classifier.get())
        self.print_transients_predictions_stat = True
        if self.show_transient_classifier:
            self.update_plot()

    def spin_box_threshold_update(self):
        self.nb_std_thresold = float(self.spin_box_threshold.get())
        if self.display_threshold:
            self.update_plot()

    def spin_box_correlation_update(self):
        self.correlation_thresold = float(self.spin_box_correlation.get())
        if self.display_correlations:
            self.update_plot()

    def unsaved(self):
        """
        means a changed has been done, and the actual plot is not saved """

        self.is_saved = False
        self.save_button['state'] = 'normal'

    def update_doubtful_frames_periods(self, cell):
        if np.sum(self.doubtful_frames_nums[cell]) == 0:
            self.doubtful_frames_periods[cell] = []
            return

        self.doubtful_frames_periods[cell] = get_continous_time_periods(self.doubtful_frames_nums[cell])

    def update_mvt_frames_periods(self, cell):
        if np.sum(self.mvt_frames_nums[cell]) == 0:
            self.mvt_frames_periods[cell] = []
            return

        self.mvt_frames_periods[cell] = get_continous_time_periods(self.mvt_frames_nums[cell])

    def normalize_traces(self):
        self.ratio_traces = np.zeros((self.nb_neurons, self.traces.shape[1]))
        # z_score traces
        for i in np.arange(self.nb_neurons):
            self.traces[i, :] = (self.traces[i, :] - np.mean(self.traces[i, :])) / np.std(self.traces[i, :])
            if self.raw_traces is not None:
                self.raw_traces[i, :] = (self.raw_traces[i, :] - np.mean(self.raw_traces[i, :])) \
                                        / np.std(self.raw_traces[i, :])
                self.ratio_traces[i] = (self.raw_traces[i] - self.traces[i]) - 2
                # smoothing the trace
                windows = ['hanning', 'hamming', 'bartlett', 'blackman']
                i_w = 1
                window_length = 11
                smooth_signal = smooth_convolve(x=self.ratio_traces[i], window_len=window_length,
                                                window=windows[i_w])
                beg = (window_length - 1) // 2
                self.ratio_traces[i] = smooth_signal[beg:-beg]
            if self.raw_traces_median is not None:
                self.raw_traces_median[i, :] = (self.raw_traces_median[i, :] - np.mean(self.raw_traces_median[i, :])) \
                                               / np.std(self.raw_traces_median[i, :])

        ratio_traces = ((self.raw_traces[self.current_neuron]) / self.traces[self.current_neuron]) - 2
        if self.raw_traces is not None:
            if self.raw_traces_median is not None:
                self.raw_traces -= 4
                self.raw_traces_median -= 2
            else:
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

        self.update_last_action(AddOnsetAction(added_time=at_time, session_frame=self,
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
        self.update_last_action(AddPeakAction(added_time=at_time,
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

    def update_last_action(self, new_action):
        """
        Keep the size of the last_actions up to five actions
        :param new_action:
        :return:
        """
        self.last_actions.append(new_action)
        if len(self.last_actions) > 5:
            self.last_actions = self.last_actions[1:]

    def add_peak_switch_mode(self, from_add_peak_button=True):
        """

        :param from_add_peak_button: indicate the user click on the add_peak button, otherwise it means the
        function has been called after another button has been clicked
        :return:
        """
        if from_add_peak_button and (not self.add_peak_mode):
            self.swith_all_click_actions(initiator="add_peak_switch_mode")
        self.add_peak_mode = not self.add_peak_mode
        if self.add_peak_mode:
            self.add_peak_button["fg"] = 'green'
            self.add_peak_button["text"] = ' + PEAK ON '
        else:
            self.add_peak_button["fg"] = 'red'
            self.add_peak_button["text"] = ' + PEAK OFF '

    def remove_doubtful_frames(self, x_from, x_to):
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
        modification_done = (np.sum(self.doubtful_frames_nums[self.current_neuron, x_from:x_to]) > 0)
        if modification_done:
            removed_times = np.where(self.doubtful_frames_nums[self.current_neuron, x_from:x_to] > 0)[0] + x_from
            self.doubtful_frames_nums[self.current_neuron, x_from:x_to] = 0
            self.update_doubtful_frames_periods(cell=self.current_neuron)
            left_x_limit, right_x_limit = self.axe_plot.get_xlim()
            bottom_limit, top_limit = self.axe_plot.get_ylim()
            self.last_actions.append(RemoveDoubtfulFramesAction(removed_times=removed_times,
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

    def add_doubtful_frames(self, x_from, x_to):
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

        backup_values = np.copy(self.doubtful_frames_nums[self.current_neuron, x_from:x_to])
        # print(f"add corrupt frames from {x_from} to {x_to}")
        self.doubtful_frames_nums[self.current_neuron, x_from:(x_to + 1)] = 1
        self.update_doubtful_frames_periods(cell=self.current_neuron)

        left_x_limit, right_x_limit = self.axe_plot.get_xlim()
        bottom_limit, top_limit = self.axe_plot.get_ylim()
        self.last_actions.append(AddDoubtfulFramesAction(x_from=x_from, x_to=x_to,
                                                         backup_values=backup_values,
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

    def remove_mvt_frames(self, x_from, x_to):
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
        modification_done = (np.sum(self.mvt_frames_nums[self.current_neuron, x_from:x_to]) > 0)
        if modification_done:
            removed_times = np.where(self.mvt_frames_nums[self.current_neuron, x_from:x_to] > 0)[0] + x_from
            self.mvt_frames_nums[self.current_neuron, x_from:x_to] = 0
            self.update_mvt_frames_periods(cell=self.current_neuron)
            left_x_limit, right_x_limit = self.axe_plot.get_xlim()
            bottom_limit, top_limit = self.axe_plot.get_ylim()
            self.last_actions.append(RemoveMvtFramesAction(removed_times=removed_times,
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

    def add_mvt_frames(self, x_from, x_to):
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

        backup_values = np.copy(self.mvt_frames_nums[self.current_neuron, x_from:x_to])
        # print(f"add corrupt frames from {x_from} to {x_to}")
        self.mvt_frames_nums[self.current_neuron, x_from:(x_to + 1)] = 1
        self.update_mvt_frames_periods(cell=self.current_neuron)

        left_x_limit, right_x_limit = self.axe_plot.get_xlim()
        bottom_limit, top_limit = self.axe_plot.get_ylim()
        self.last_actions.append(AddMvtFramesAction(x_from=x_from, x_to=x_to,
                                                    backup_values=backup_values,
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

    def validation_before_closing(self):
        if not self.is_saved:
            self.save_as_spike_nums()
        self.root.destroy()

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

    def save_sources_profile_map(self, key_cmap=None):
        # c_map = plt.get_cmap('gray')
        # if key_cmap is not None:
        #     if key_cmap is "P":
        #         c_map = self.parula_map
        #     if key_cmap is "B":
        #         c_map = plt.get_cmap('Blues')
        c_map = self.parula_map

        # TODO: show cells that are removed, and those that the CNN would think as False with the associated score
        # predicting if a cell is a True on or not
        path_to_model = self.path_data + "cell_classifier_model/"
        # predictions will be an array of length n_cells
        predictions = None
        # checking if the path exists
        if os.path.isdir(path_to_model):
            json_file = None
            weights_file = None
            # then we look for the json file (representing the model architecture) and the weights file
            # we will assume there is only one file of each in this directory
            # look for filenames in the fisrst directory, if we don't break, it will go through all directories
            for (dirpath, dirnames, local_filenames) in os.walk(path_to_model):
                for file_name in local_filenames:
                    if file_name.endswith(".json"):
                        json_file = path_to_model + file_name
                    if "weights" in file_name:
                        weights_file = path_to_model + file_name
                # looking only in the top directory
                break
            if (json_file is not None) and (weights_file is not None):
                # first we "load" the movie in ms as well as peaks and onsets such as defined so far
                self.data_and_param.ms.tiff_movie = self.tiff_movie
                self.data_and_param.ms.normalize_movie()
                self.data_and_param.ms.spike_struct.peak_nums = self.peak_nums
                self.data_and_param.ms.spike_struct.spike_nums = self.spike_nums
                self.data_and_param.ms.spike_struct.n_cells = self.spike_nums.shape[0]
                if self.data_and_param.ms.traces is None:
                    self.data_and_param.ms.traces = self.traces
                if self.data_and_param.ms.raw_traces is None:
                    self.data_and_param.ms.raw_traces = self.raw_traces
                predictions = predict_cell_from_saved_model(ms=self.data_and_param.ms,
                                                            weights_file=weights_file, json_file=json_file)
                print_accuracy = False
                if print_accuracy:
                    total_right = 0
                    for cell_index, cell_value in enumerate(self.cells_to_remove):
                        # print(f"cell_index {cell_index}: {cell_value}")
                        if (cell_value == 0) and (predictions[cell_index] >= 0.5):
                            total_right += 1
                        elif (cell_value == 1) and (predictions[cell_index] < 0.5):
                            total_right += 1
                    # print(f"self.nb_neurons {self.nb_neurons}")
                    # for cell in np.where(self.cells_to_remove)[0]:
                    #     print(f"fake {cell}")
                    print(f"Accuracy: {str(np.round(total_right / self.nb_neurons, 2))}")

        show_distribution_prediction = True
        if (predictions is not None) and show_distribution_prediction:
            distribution = predictions
            hist_color = "blue"
            edge_color = "white"
            tight_x_range = False
            if tight_x_range:
                max_range = np.max(distribution)
                min_range = np.min(distribution)
            else:
                max_range = 1
                min_range = 0
            weights = (np.ones_like(distribution) / (len(distribution))) * 100

            fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                    gridspec_kw={'height_ratios': [1]},
                                    figsize=(12, 12))
            ax1.set_facecolor("black")
            bins = int(np.sqrt(len(distribution)))
            twice_more_bins = True
            if twice_more_bins:
                bins *= 2
            hist_plt, edges_plt, patches_plt = plt.hist(distribution, bins=bins, range=(min_range, max_range),
                                                        facecolor=hist_color,
                                                        edgecolor=edge_color,
                                                        weights=weights, log=False)

            if tight_x_range:
                plt.xlim(min_range, max_range)
            else:
                plt.xlim(0, 1)
                xticks = np.arange(0, 1.1, 0.1)

                ax1.set_xticks(xticks)
                # sce clusters labels
                ax1.set_xticklabels(xticks)
            ylabel = None
            if ylabel is None:
                ax1.set_ylabel("Distribution (%)")
            else:
                ax1.set_ylabel(ylabel)
            ax1.set_xlabel("predictions value")

            save_formats = "pdf"
            path_results = self.path_result + "/" + self.data_and_param.time_str
            if not os.path.isdir(path_results):
                os.mkdir(path_results)

            if isinstance(save_formats, str):
                save_formats = [save_formats]
            for save_format in save_formats:
                fig.savefig(f'{path_results}/{self.data_and_param.ms.description}_cells_prediction_'
                            f'distribution_{self.data_and_param.time_str}.{save_format}',
                            format=f"{save_format}")

            plt.close()

        if predictions is not None:
            # writing them in a file
            with open(f"{path_results}/cell_classifier_cnn_results_{self.data_and_param.ms.description}.txt", "w",
                      encoding='UTF-8') as file:
                file.write(f"{' '.join(str(p) for p in predictions)}" + '\n')
        # removed cells in cmap Gray, other cells in Parula
        # decide threshold  and if CNN model available, red border is real cell, green border if false
        # displaying value in the righ bottom border
        n_cells = len(self.traces)
        n_cells_by_row = 20
        n_pixels_by_cell_x = 20
        n_pixels_by_cell_y = 20
        len_x = n_cells_by_row * n_pixels_by_cell_x
        len_y = math.ceil(n_cells / n_cells_by_row) * n_pixels_by_cell_y
        # sources_profile_map = np.zeros((len_y, len_x), dtype="int16")

        sources_profile_fig = plt.figure(figsize=(20, 20),
                                         subplotpars=SubplotParams(hspace=0, wspace=0))
        fig_patch = sources_profile_fig.patch
        rgba = c_map(0)
        fig_patch.set_facecolor(rgba)

        # sources_profile_fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 0.1, 'h_pad': 0.1})

        # looking at how many overlapping cell current_neuron has
        intersect_cells = self.overlapping_cells[self.current_neuron]
        # print(f"len(intersect_cells) {len(intersect_cells)}")
        cells_color = dict()
        for index, cell_inter in enumerate(np.arange(n_cells)):
            cells_color[cell_inter] = cm.nipy_spectral(float(index + 1) / (len(intersect_cells) + 1))

        # now adding as many suplots as need, depending on how many overlap has the cell
        n_columns = n_cells_by_row
        width_ratios = [100 // n_columns] * n_columns
        n_lines = (((n_cells - 1) // n_columns) + 1) * 2
        height_ratios = [100 // n_lines] * n_lines
        grid_spec = gridspec.GridSpec(n_lines, n_columns, width_ratios=width_ratios,
                                      height_ratios=height_ratios,
                                      figure=sources_profile_fig, wspace=0, hspace=0)

        # building the subplots to displays the sources and transients
        ax_source_profile_by_cell = dict()

        max_len_x = None
        max_len_y = None
        for cell_to_display in np.arange(n_cells):
            poly_gon = self.data_and_param.ms.coord_obj.cells_polygon[cell_to_display]

            if max_len_x is None:
                tmp_minx, tmp_miny, tmp_maxx, tmp_maxy = np.array(list(poly_gon.bounds)).astype(int)
                max_len_x = tmp_maxx - tmp_minx
                max_len_y = tmp_maxy - tmp_miny
            else:
                tmp_minx, tmp_miny, tmp_maxx, tmp_maxy = np.array(list(poly_gon.bounds)).astype(int)
                max_len_x = max(max_len_x, tmp_maxx - tmp_minx)
                max_len_y = max(max_len_y, tmp_maxy - tmp_miny)
        bounds = (0, 0, max_len_x, max_len_y)

        for cell_index, cell_to_display in enumerate(np.arange(n_cells)):
            line_gs = (cell_index // n_columns) * 2
            col_gs = cell_index % n_columns

            ax_source_profile_by_cell[cell_to_display] = sources_profile_fig.add_subplot(grid_spec[line_gs, col_gs])
            ax = ax_source_profile_by_cell[cell_to_display]
            # ax_source_profile_by_cell[cell_to_display].set_facecolor("black")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            if predictions is None:
                ax.set_frame_on(False)
            else:
                # 3 range of color
                if predictions[cell_index] > 0.6:
                    frame_color = "green"
                elif predictions[cell_index] < 0.3:
                    frame_color = "red"
                else:
                    frame_color = "black"
                for spine in ax.spines.values():
                    spine.set_edgecolor(frame_color)
                    spine.set_linewidth(1.5)
                # ax.spines['bottom'].set_color(frame_color)
                # ax.spines['top'].set_color(frame_color)
                # ax.spines['right'].set_color(frame_color)
                # ax.spines['left'].set_color(frame_color)
        # saving source profile with full frame for SEUDO in a matlab file
        bounds = None
        seudo_source_profiles = None
        for cell_index, cell_to_display in enumerate(np.arange(n_cells)):
            if cell_to_display not in self.source_profile_dict_for_map_of_all_cells:
                source_profile, minx, miny, mask_source_profile = self.get_source_profile(cell=cell_to_display,
                                                                                          pixels_around=3,
                                                                                          bounds=bounds,
                                                                                          with_full_frame=True)
                source_profile[mask_source_profile] = 0
                if seudo_source_profiles is None:
                    seudo_source_profiles = np.zeros((n_cells, source_profile.shape[0], source_profile.shape[1]))
                seudo_source_profiles[cell_index] = source_profile

        bounds = None
        for cell_index, cell_to_display in enumerate(np.arange(n_cells)):
            if cell_to_display not in self.source_profile_dict_for_map_of_all_cells:
                source_profile, minx, miny, mask_source_profile = self.get_source_profile(cell=cell_to_display,
                                                                                          pixels_around=3,
                                                                                          bounds=bounds)
                xy_source = self.get_cell_new_coord_in_source(cell=cell_to_display, minx=minx, miny=miny)
                self.source_profile_dict_for_map_of_all_cells[cell_to_display] = [source_profile, minx, miny,
                                                                                  mask_source_profile,
                                                                                  xy_source]
            else:
                source_profile, minx, miny, mask_source_profile, xy_source = \
                    self.source_profile_dict_for_map_of_all_cells[cell_to_display]
            if self.cells_to_remove[cell_to_display] == 1:
                c_map = plt.get_cmap('gray')
            else:
                c_map = self.parula_map
            img_src_profile = ax_source_profile_by_cell[cell_to_display].imshow(source_profile,
                                                                                cmap=c_map)
            with_mask = False
            if with_mask:
                source_profile[mask_source_profile] = 0
                img_src_profile.set_array(source_profile)

            lw = 0.2
            contour_cell = patches.Polygon(xy=xy_source,
                                           fill=False,
                                           edgecolor="red",
                                           zorder=15, lw=lw)
            ax_source_profile_by_cell[cell_to_display].add_patch(contour_cell)

            # if key_cmap in ["B", "P"]:
            #     color_text = "red"
            # else:
            #     color_text = "blue"
            if self.cells_to_remove[cell_to_display] == 1:
                color_text = "blue"
            else:
                color_text = "red"
            ax_source_profile_by_cell[cell_to_display].text(x=1.5, y=1,
                                                            s=f"{cell_to_display}", color=color_text, zorder=20,
                                                            ha='center', va="center", fontsize=3, fontweight='bold')

            if predictions is not None:
                predict_value = str(round(predictions[cell_to_display], 2))
                ax_source_profile_by_cell[cell_to_display].text(x=source_profile.shape[1] - 2.7,
                                                                y=source_profile.shape[0] - 2,
                                                                s=f"{predict_value}", color=color_text, zorder=20,
                                                                ha='center', va="center", fontsize=3, fontweight='bold')

        # plt.show()

        save_formats = ["pdf"]
        path_results = self.path_result + "/" + self.data_and_param.time_str
        if not os.path.isdir(path_results):
            os.mkdir(path_results)

        sio.savemat(f"{path_results}/" + f"{self.data_and_param.ms.description}_seudo_source_profiles.mat",
                    {'seudo_source_profiles': seudo_source_profiles})

        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            sources_profile_fig.savefig(f'{path_results}/'
                                        f'{self.data_and_param.ms.description}_source_profiles_map_cmap_{key_cmap}'
                                        f'_{self.data_and_param.time_str}.{save_format}',
                                        format=f"{save_format}",
                                        facecolor=sources_profile_fig.get_facecolor(), edgecolor='none')

    def load_checked_predictions(self):
        options = {}
        options['initialdir'] = self.data_and_param.path_data
        if self.save_checked_predictions_dir is not None:
            options['initialdir'] = self.save_checked_predictions_dir
        options['title'] = "Choose a directory to open files"
        options['mustexist'] = False
        path_for_files = filedialog.askdirectory(**options)
        if path_for_files == "":
            return None
        file_names_to_load = []
        for (dirpath, dirnames, local_filenames) in os.walk(path_for_files):
            for file_name in local_filenames:
                if file_name.endswith(".npy") and file_name.startswith(self.data_and_param.ms.description):
                    file_names_to_load.append(file_name)
            break

        self.save_checked_predictions_dir = path_for_files

        for file_name in file_names_to_load:
            underscores_pos = [pos for pos, char in enumerate(file_name) if char == "_"]
            if len(underscores_pos) < 4:
                continue
            # the last 4 indicates how to get cell number and frames
            middle_frame = int(file_name[underscores_pos[-1] + 1:-4])
            last_frame = int(file_name[underscores_pos[-2] + 1:underscores_pos[-1]])
            first_frame = int(file_name[underscores_pos[-3] + 1:underscores_pos[-2]])
            cell = int(file_name[underscores_pos[-4] + 1:underscores_pos[-3]])
            self.update_transient_prediction_periods_to_check()
            predictions = self.transient_prediction[cell]
            max_pred = np.max(predictions[max(0, middle_frame - 2):min(middle_frame + 2, len(predictions) + 1), 0])
            max_pred = np.round(max_pred, 2)
            new_pred_period = (cell, first_frame, last_frame, max_pred, middle_frame)
            raster_dur = np.load(os.path.join(path_for_files, file_name))
            # print(f"raster_dur.shape {raster_dur.shape}")
            periods = get_continous_time_periods(raster_dur)

            # tabula rasa
            self.onset_times[cell, first_frame:last_frame] = 0
            self.spike_nums[cell, first_frame:last_frame] = 0
            self.peak_nums[cell, first_frame:last_frame] = 0

            for period in periods:
                if period[0] > 0:
                    self.onset_times[cell, first_frame + period[0]] = 1
                    self.spike_nums[cell, first_frame + period[0]] = 1
                if first_frame + period[1] != (last_frame - 1):
                    self.peak_nums[cell, first_frame + period[1]] = 1

            self.transient_prediction_periods_checked.append(new_pred_period)
            # then we change the foreground of the predictions_list_boxs
            if new_pred_period in self.transient_prediction_periods_checked:
                if new_pred_period in self.transient_prediction_periods_to_check:
                    index_predictions_list_box = self.transient_prediction_periods_to_check.index(new_pred_period)
                    self.predictions_list_box.itemconfig(index_predictions_list_box,
                                                         bg=self.color_bg_predicted_transient_list_box)

        self.update_checked_predictions_list_box()
        self.update_after_onset_change()

    def save_checked_predictions(self):
        self.save_pred_button['state'] = DISABLED
        options = {}
        options['initialdir'] = "/"
        if self.save_checked_predictions_dir is not None:
            options['initialdir'] = self.save_checked_predictions_dir
        options['title'] = "Choose a directory to save files"
        options['mustexist'] = False
        dir_name = filedialog.askdirectory(**options)
        if dir_name == "":
            return None
        self.save_checked_predictions_dir = dir_name
        # print(f"dir_name {dir_name}")
        # then we create a npy file for each period, that will contain a binary vector, a raster_dur of 1 dimension
        # with 1 if the cell is active at this cell
        for period in self.transient_prediction_periods_checked:
            cell = period[0]
            first_frame = period[1]
            # last frame not included
            last_frame = period[2]
            middle_frame = period[4]
            file_name = self.data_and_param.ms.description + f"_{cell}_{first_frame}_{last_frame}_{middle_frame}"
            # building the raster_dur for all the frames, to make sure we don't miss a long transient that will
            # start or finish after the period
            # TODO: could be optimize to look for a onsets and peaks just a few hundred frames around the period
            raster_dur = np.zeros(self.nb_times, dtype="int8")
            peaks_index = np.where(self.peak_nums[cell, :] > 0)[0]
            onsets_index = np.where(self.onset_times[cell, :] > 0)[0]
            for onset_index in onsets_index:
                peaks_after = np.where(peaks_index > onset_index)[0]
                if len(peaks_after) == 0:
                    continue
                peaks_after = peaks_index[peaks_after]
                peak_after = peaks_after[0]
                raster_dur[onset_index:peak_after + 1] = 1
            raster_dur = raster_dur[period[1]:period[2]]
            np.save(file=os.path.join(dir_name, file_name), arr=raster_dur)

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
        if self.to_agree_peak_nums is not None:
            sio.savemat(self.save_path + self.save_file_name, {'Bin100ms_spikedigital_Python': self.spike_nums,
                                                               'LocPeakMatrix_Python': self.peak_nums,
                                                               'cells_to_remove': cells_to_remove,
                                                               'inter_neurons': inter_neurons,
                                                               "doubtful_frames_nums": self.doubtful_frames_nums,
                                                               "mvt_frames_nums": self.mvt_frames_nums,
                                                               "to_agree_peak_nums": self.to_agree_peak_nums,
                                                               "to_agree_spike_nums": self.to_agree_spike_nums})
        else:
            sio.savemat(self.save_path + self.save_file_name, {'Bin100ms_spikedigital_Python': self.spike_nums,
                                                               'LocPeakMatrix_Python': self.peak_nums,
                                                               'cells_to_remove': cells_to_remove,
                                                               'inter_neurons': inter_neurons,
                                                               "doubtful_frames_nums": self.doubtful_frames_nums,
                                                               "mvt_frames_nums": self.mvt_frames_nums})
        if and_close:
            self.root.destroy()

    def get_threshold(self):
        trace = self.traces[self.current_neuron, :]
        threshold = (self.nb_std_thresold * np.std(trace)) + np.min(self.traces[self.current_neuron, :])
        # + abs(np.min(self.traces[self.current_neuron, :]))
        return threshold

    def plot_magnifier(self, first_time=False, mouse_x_position=None, mouse_y_position=None):
        """
        Plot the magnifier
        :param first_time: if True, means the function is called for the first time,
        allow sto initialize some variables.
        :param mouse_x_position: indicate the x position of the mouse cursor
        :param mouse_y_position: indicate the y position of the mouse cursor
        :return: None
        """
        if first_time:
            self.axe_plot_magnifier = self.magnifier_fig.add_subplot(111)
            self.axe_plot_magnifier.get_xaxis().set_visible(False)
            self.axe_plot_magnifier.get_yaxis().set_visible(False)
            self.axe_plot_magnifier.set_facecolor("black")

        if self.x_center_magnified is not None:
            pos_beg_x = int(np.max((0, self.x_center_magnified - self.magnifier_range)))
            pos_end_x = int(np.min((self.nb_times, self.x_center_magnified + self.magnifier_range + 1)))

            if self.raw_traces is not None:
                max_value = max(np.max(self.traces[self.current_neuron, pos_beg_x:pos_end_x]),
                                np.max(self.raw_traces[self.current_neuron, pos_beg_x:pos_end_x]))
                min_value = min(np.min(self.traces[self.current_neuron, pos_beg_x:pos_end_x]),
                                np.min(self.raw_traces[self.current_neuron, pos_beg_x:pos_end_x]))
            else:
                max_value = np.max(self.traces[self.current_neuron, pos_beg_x:pos_end_x])
                min_value = np.min(self.traces[self.current_neuron, pos_beg_x:pos_end_x])

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
        # if first_time:
        #     self.magnifier_fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 0.2, 'h_pad': 0.2})
        self.magnifier_fig.tight_layout()

    def draw_magnifier_marker(self, mouse_x_position=None, mouse_y_position=None):
        if (mouse_x_position is None) or (mouse_y_position is None):
            return

        pos_beg_x = int(np.max((0, self.x_center_magnified - self.magnifier_range)))
        pos_end_x = int(np.min((self.nb_times, self.x_center_magnified + self.magnifier_range + 1)))

        if self.raw_traces is not None:
            max_value = max(np.max(self.traces[self.current_neuron, pos_beg_x:pos_end_x]),
                            np.max(self.raw_traces[self.current_neuron, pos_beg_x:pos_end_x]))
            min_value = min(np.min(self.traces[self.current_neuron, pos_beg_x:pos_end_x]),
                            np.min(self.raw_traces[self.current_neuron, pos_beg_x:pos_end_x]))
        else:
            max_value = np.max(self.traces[self.current_neuron, pos_beg_x:pos_end_x])
            min_value = np.min(self.traces[self.current_neuron, pos_beg_x:pos_end_x])

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

    def corr_between_source_and_transient(self, cell, transient, pixels_around=1, redo_computation=False):
        """

        :param cell:
        :param transient:
        :param pixels_around:
        :param redo_computation: if True, means that even if the correlation has been done before for the peak,
        it will be redo (useful if the onset has changed for exemple
        :return:
        """
        # if cell not in self.corr_source_transient:
        #     self.corr_source_transient[cell] = dict()
        #
        # elif transient in self.corr_source_transient[cell]:
        #     return self.corr_source_transient[cell][transient]

        if (redo_computation is False) and (self.peaks_correlation[cell, transient[1]] >= -1):
            return self.peaks_correlation[cell, transient[1]]

        poly_gon = self.data_and_param.ms.coord_obj.cells_polygon[cell]

        # Correlation test
        bounds_corr = np.array(list(poly_gon.bounds)).astype(int)

        # looking if this source has been computed before
        if cell in self.source_profile_correlation_dict:
            source_profile_corr, mask_source_profile = self.source_profile_correlation_dict[cell]
        else:
            source_profile_corr, minx_corr, \
            miny_corr, mask_source_profile = self.get_source_profile(cell=cell,
                                                                     pixels_around=pixels_around,
                                                                     bounds=bounds_corr, buffer=1)
            # normalizing
            source_profile_corr = source_profile_corr - np.mean(source_profile_corr)
            # we want the mask to be at ones over the cell
            mask_source_profile = (1 - mask_source_profile).astype(bool)
            self.source_profile_correlation_dict[cell] = (source_profile_corr, mask_source_profile)

        transient_profile_corr, minx_corr, miny_corr = self.get_transient_profile(cell=cell,
                                                                                  transient=transient,
                                                                                  pixels_around=pixels_around,
                                                                                  bounds=bounds_corr)
        transient_profile_corr = transient_profile_corr - np.mean(transient_profile_corr)

        pearson_corr, pearson_p_value = stats.pearsonr(source_profile_corr[mask_source_profile],
                                                       transient_profile_corr[mask_source_profile])

        self.peaks_correlation[cell, transient[1]] = pearson_corr

        return pearson_corr

    def compute_source_and_transients_correlation(self, main_cell, redo_computation=False, with_overlapping_cells=True):
        """
        Compute the source and transient profiles of a given cell. Should be call for each new neuron displayed
        :param cell:
        :param redo_computation: if True, means that even if the correlation has been done before for this cell,
        it will be redo (useful if the onsets or peaks has changed for exemple)
        :return:
        """
        # the tiff_movie is necessary to compute the source and transient profile
        if self.tiff_movie is None:
            return

        cells = [main_cell]
        if with_overlapping_cells:
            overlapping_cells = self.overlapping_cells[self.current_neuron]
            cells += list(overlapping_cells)

        for cell in cells:
            peaks_frames = np.where(self.peak_nums[self.current_neuron, :] > 0)[0]
            if len(peaks_frames) == 0:
                return
            if redo_computation is False:
                # it means all peaks correlation are knoww
                if np.min(self.peaks_correlation[cell, peaks_frames]) > -2:
                    # means correlation has been computed before
                    continue

            # first computing the list of transients based on peaks and onsets preceeding the

            onsets_frames = np.where(self.onset_times[self.current_neuron, :] > 0)[0]
            for peak_frame in peaks_frames:
                onsets_before_peak = np.where(onsets_frames < peak_frame)[0]
                if len(onsets_before_peak) == 0:
                    continue
                first_onset_before_peak = onsets_frames[onsets_before_peak[-1]]
                transient = (first_onset_before_peak, peak_frame)
                # the correlation will be saved in the array  elf.peaks_correlation
                self.corr_between_source_and_transient(cell=cell, transient=transient)

    def plot_source_transient(self, transient):
        # transient is a tuple of int, reprensenting the frame of the onset and the frame of the peak
        # using the magnifier figure
        self.magnifier_fig.clear()
        plt.close(self.magnifier_fig)
        self.magnifier_canvas.get_tk_widget().destroy()
        if self.robin_mac:
            self.magnifier_fig = plt.figure(figsize=(3, 3))
        else:
            self.magnifier_fig = plt.figure(figsize=(4, 4))
        self.magnifier_fig.patch.set_facecolor('black')
        self.magnifier_canvas = FigureCanvasTkAgg(self.magnifier_fig, self.magnifier_frame)
        # fig = plt.figure(figsize=size_fig)
        self.magnifier_fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 0.1, 'h_pad': 0.1})

        # looking at how many overlapping cell current_neuron has
        intersect_cells = self.overlapping_cells[self.current_neuron]
        # print(f"len(intersect_cells) {len(intersect_cells)}")
        cells_color = dict()
        cells_color[self.current_neuron] = "red"
        cells_to_display = [self.current_neuron]
        for index, cell_inter in enumerate(intersect_cells):
            cells_color[cell_inter] = cm.nipy_spectral(float(index + 1) / (len(intersect_cells) + 1))

        cells_to_display.extend(intersect_cells)
        n_cells_to_display = len(cells_to_display)
        # now adding as many suplots as need, depending on how many overlap has the cell
        # TODO: to adapt depending on the resolution and the size of the main figure
        n_columns = 8
        width_ratios = [100 // n_columns] * n_columns
        n_lines = (((n_cells_to_display - 1) // n_columns) + 1) * 2
        height_ratios = [100 // n_lines] * n_lines
        grid_spec = gridspec.GridSpec(n_lines, n_columns, width_ratios=width_ratios,
                                      height_ratios=height_ratios,
                                      figure=self.magnifier_fig, hspace=0.125, wspace=0.05)

        # building the subplots to displays the sources and transients
        ax_source_profile_by_cell = dict()
        # ax_top_source_profile_by_cell = dict()
        ax_source_transient_by_cell = dict()
        for cell_index, cell_to_display in enumerate(cells_to_display):
            line_gs = (cell_index // n_columns) * 2
            col_gs = cell_index % n_columns
            ax_source_profile_by_cell[cell_to_display] = self.magnifier_fig.add_subplot(grid_spec[line_gs, col_gs])
            ax_source_profile_by_cell[cell_to_display].get_yaxis().set_visible(False)
            ax_source_profile_by_cell[cell_to_display].set_facecolor("black")
            # ax_top_source_profile_by_cell[cell_to_display] = ax_source_profile_by_cell[cell_to_display].twiny()
            for spine in ax_source_profile_by_cell[cell_to_display].spines.values():
                spine.set_edgecolor(cells_color[cell_to_display])
                spine.set_linewidth(2)
            ax_source_transient_by_cell[cell_to_display] = \
                self.magnifier_fig.add_subplot(grid_spec[line_gs + 1, col_gs])
            ax_source_transient_by_cell[cell_to_display].get_xaxis().set_visible(False)
            ax_source_transient_by_cell[cell_to_display].get_yaxis().set_visible(False)
            ax_source_transient_by_cell[cell_to_display].set_facecolor("black")
            for spine in ax_source_transient_by_cell[cell_to_display].spines.values():
                spine.set_edgecolor(cells_color[cell_to_display])
                spine.set_linewidth(2)

        # should be a np.array with x, y len equal
        source_profile_by_cell = dict()
        transient_profile_by_cell = dict()

        size_square = 40
        frame_tiff = self.tiff_movie[transient[-1]]
        len_x = frame_tiff.shape[1]
        len_y = frame_tiff.shape[0]
        # calculating the bound that will surround all the cells
        minx = None
        maxx = None
        miny = None
        maxy = None
        corr_by_cell = dict()
        for cell_to_display in cells_to_display:
            poly_gon = self.data_and_param.ms.coord_obj.cells_polygon[cell_to_display]

            if minx is None:
                minx, miny, maxx, maxy = np.array(list(poly_gon.bounds)).astype(int)
            else:
                tmp_minx, tmp_miny, tmp_maxx, tmp_maxy = np.array(list(poly_gon.bounds)).astype(int)
                minx = min(minx, tmp_minx)
                miny = min(miny, tmp_miny)
                maxx = max(maxx, tmp_maxx)
                maxy = max(maxy, tmp_maxy)
        bounds = (minx, miny, maxx, maxy)
        # show the extents of all the cells that overlap with the main cell
        # print(f"maxx-minx {maxx-minx}, maxy-miny {maxy-miny}")

        for cell_index, cell_to_display in enumerate(cells_to_display):
            # self.x_beg_movie, self.x_end_movie, self.y_beg_movie, self.y_end_movie = \
            #     self.square_coord_around_cell(cell=cell_to_display, size_square=size_square,
            #                                   x_len_max=len_x, y_len_max=len_y)
            # tiff_array = frame_tiff[self.y_beg_movie:self.y_end_movie,
            #              self.x_beg_movie:self.x_end_movie]
            # ax_source_profile_by_cell[cell_to_display].imshow(tiff_array, cmap=plt.get_cmap('gray'))
            first_time = False
            if cell_to_display not in self.source_profile_dict:
                source_profile, minx, miny, mask_source_profile = self.get_source_profile(cell=cell_to_display,
                                                                                          pixels_around=3,
                                                                                          bounds=bounds)
                xy_source = self.get_cell_new_coord_in_source(cell=cell_to_display, minx=minx, miny=miny)
                self.source_profile_dict[cell_to_display] = [source_profile, minx, miny, mask_source_profile,
                                                             xy_source]
            else:
                source_profile, minx, miny, mask_source_profile, xy_source = \
                    self.source_profile_dict[cell_to_display]

            img_src_profile = ax_source_profile_by_cell[cell_to_display].imshow(source_profile,
                                                                                cmap=plt.get_cmap('gray'))
            with_mask = False
            if with_mask:
                source_profile[mask_source_profile] = 0
                img_src_profile.set_array(source_profile)

            lw = 1
            contour_cell = patches.Polygon(xy=xy_source,
                                           fill=False,
                                           edgecolor=cells_color[cell_to_display],
                                           zorder=15, lw=lw)
            ax_source_profile_by_cell[cell_to_display].add_patch(contour_cell)

            pearson_corr = self.corr_between_source_and_transient(cell=cell_to_display,
                                                                  transient=transient,
                                                                  pixels_around=1)

            pearson_corr = np.round(pearson_corr, 2)
            # ax_source_profile_by_cell[cell_to_display].text(x=3, y=3,
            #                                                 s=f"{cell_to_display}", color="blue", zorder=20,
            #                                                 ha='center', va="center", fontsize=7, fontweight='bold')
            # displaying correlation between source and transient profile
            min_x_axis, max_x_axis = ax_source_profile_by_cell[cell_to_display].get_xlim()
            ax_source_profile_by_cell[cell_to_display].set_xticks([max_x_axis / 2])
            # ax_source_profile_by_cell[cell_to_display].set_xticklabels([f"{pearson_corr} / {percentage_high_corr}%"])
            ax_source_profile_by_cell[cell_to_display].set_xticklabels([f"{cell_to_display} -> {pearson_corr}"])
            # if pearson_p_value < 0.05:
            #     label_color = "red"
            # else:
            #     label_color = "black"
            ax_source_profile_by_cell[cell_to_display].xaxis.set_tick_params(labelsize=8, pad=0.1,
                                                                             labelcolor="white")
            ax_source_profile_by_cell[cell_to_display].xaxis.set_ticks_position('none')
            # displaying cell number
            # min_x_axis, max_x_axis = ax_top_source_profile_by_cell[cell_to_display].get_xlim()
            # # ax_top_source_profile_by_cell[cell_to_display].set_xlim(left=min_x_axis, right=max_x_axis, auto=None)
            # ax_top_source_profile_by_cell[cell_to_display].set_xticks([max_x_axis / 2])
            # ax_top_source_profile_by_cell[cell_to_display].set_xticklabels([cell_to_display])
            # ax_top_source_profile_by_cell[cell_to_display].xaxis.set_tick_params(labelsize=8, pad=0.1,
            #                                                                  labelcolor=cells_color[cell_to_display])
            # ax_top_source_profile_by_cell[cell_to_display].xaxis.set_ticks_position('none')

            transient_profile, minx, miny = self.get_transient_profile(cell=cell_to_display, transient=transient,
                                                                       pixels_around=3, bounds=bounds)
            ax_source_transient_by_cell[cell_to_display].imshow(transient_profile, cmap=plt.get_cmap('gray'))
            for cell_to_contour in cells_to_display:
                # the new coordinates of the cell
                xy = self.get_cell_new_coord_in_source(cell=cell_to_contour, minx=minx, miny=miny)
                lw = 0.5
                if cell_to_display == cell_to_contour:
                    lw = 1
                contour_cell = patches.Polygon(xy=xy,
                                               fill=False,
                                               edgecolor=cells_color[cell_to_contour],
                                               zorder=15, lw=lw)
                ax_source_transient_by_cell[cell_to_display].add_patch(contour_cell)

        self.magnifier_canvas.draw()
        self.magnifier_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=YES)

    def get_source_profile(self, cell, pixels_around=0, bounds=None, buffer=None, with_full_frame=False):
        """

        :param cell:
        :param pixels_around:
        :param bounds:
        :param buffer:
        :param with_full_frame:  Average the full frame
        :return:
        """
        # print("get_source_profile")
        len_frame_x = self.tiff_movie[0].shape[1]
        len_frame_y = self.tiff_movie[0].shape[0]

        # determining the size of the square surrounding the cell
        poly_gon = self.data_and_param.ms.coord_obj.cells_polygon[cell]
        if bounds is None:
            minx, miny, maxx, maxy = np.array(list(poly_gon.bounds)).astype(int)
        else:
            minx, miny, maxx, maxy = bounds

        if with_full_frame:
            minx = 0
            miny = 0
            maxx = len_frame_x - 1
            maxy = len_frame_y - 1
        else:
            minx = max(0, minx - pixels_around)
            miny = max(0, miny - pixels_around)
            maxx = min(len_frame_x - 1, maxx + pixels_around)
            maxy = min(len_frame_y - 1, maxy + pixels_around)

        len_x = maxx - minx + 1
        len_y = maxy - miny + 1

        # mask used in order to keep only the cells pixel
        # the mask put all pixels in the polygon, including the pixels on the exterior line to zero
        scaled_poly_gon = self.scale_polygon_to_source(poly_gon=poly_gon, minx=minx, miny=miny)
        img = PIL.Image.new('1', (len_x, len_y), 1)
        if buffer is not None:
            scaled_poly_gon = scaled_poly_gon.buffer(buffer)
        ImageDraw.Draw(img).polygon(list(scaled_poly_gon.exterior.coords), outline=0, fill=0)
        mask = np.array(img)
        # mask = np.ones((len_x, len_y))
        # cv2.fillPoly(mask, scaled_poly_gon, 0)
        # mask = mask.astype(bool)

        source_profile = np.zeros((len_y, len_x))

        # selectionning the best peak to produce the source_profile
        peaks = np.where(self.peak_nums[cell, :] > 0)[0]
        threshold = np.percentile(self.traces[cell, peaks], 95)
        selected_peaks = peaks[np.where(self.traces[cell, peaks] > threshold)[0]]
        # max 10 peaks, min 5 peaks
        if len(selected_peaks) > 10:
            p = 10 / len(peaks)
            threshold = np.percentile(self.traces[cell, peaks], (1 - p) * 100)
            selected_peaks = peaks[np.where(self.traces[cell, peaks] > threshold)[0]]
        elif (len(selected_peaks) < 5) and (len(peaks) > 5):
            p = 5 / len(peaks)
            threshold = np.percentile(self.traces[cell, peaks], (1 - p) * 100)
            selected_peaks = peaks[np.where(self.traces[cell, peaks] > threshold)[0]]

        # print(f"threshold {threshold}")
        # print(f"n peaks: {len(selected_peaks)}")

        onsets_frames = np.where(self.onset_times[cell, :] > 0)[0]
        if self.raw_traces is not None:
            raw_traces = np.copy(self.raw_traces)
            # so the lowest value is zero
            raw_traces += abs(np.min(raw_traces))
        else:
            raw_traces = np.copy(self.traces)
            # so the lowest value is zero
            raw_traces += abs(np.min(raw_traces))
        for peak in selected_peaks:
            tmp_source_profile = np.zeros((len_y, len_x))
            onsets_before_peak = np.where(onsets_frames <= peak)[0]
            if len(onsets_before_peak) == 0:
                # shouldn't arrive
                continue
            onset = onsets_frames[onsets_before_peak[-1]]
            # print(f"onset {onset}, peak {peak}")
            frames_tiff = self.tiff_movie[onset:peak + 1]
            for frame_index, frame_tiff in enumerate(frames_tiff):
                tmp_source_profile += (frame_tiff[miny:maxy + 1, minx:maxx + 1] * raw_traces[cell, onset + frame_index])
            # averaging
            tmp_source_profile = tmp_source_profile / (np.sum(raw_traces[cell, onset:peak + 1]))
            source_profile += tmp_source_profile

        source_profile = source_profile / len(selected_peaks)

        return source_profile, minx, miny, mask

    def get_transient_profile(self, cell, transient, pixels_around=0, bounds=None):
        len_frame_x = self.tiff_movie[0].shape[1]
        len_frame_y = self.tiff_movie[0].shape[0]

        # determining the size of the square surrounding the cell
        if bounds is None:
            poly_gon = self.data_and_param.ms.coord_obj.cells_polygon[cell]
            minx, miny, maxx, maxy = np.array(list(poly_gon.bounds)).astype(int)
        else:
            minx, miny, maxx, maxy = bounds

        minx = max(0, minx - pixels_around)
        miny = max(0, miny - pixels_around)
        maxx = min(len_frame_x - 1, maxx + pixels_around)
        maxy = min(len_frame_y - 1, maxy + pixels_around)

        len_x = maxx - minx + 1
        len_y = maxy - miny + 1

        transient_profile = np.zeros((len_y, len_x))
        frames_tiff = self.tiff_movie[transient[0]:transient[-1] + 1]
        # print(f"transient[0] {transient[0]}, transient[1] {transient[1]}")
        # now we do the weighted average
        if self.raw_traces is not None:
            raw_traces = np.copy(self.raw_traces)
            # so the lowest value is zero
            raw_traces += abs(np.min(raw_traces))
        else:
            raw_traces = np.copy(self.traces)
            # so the lowest value is zero
            raw_traces += abs(np.min(raw_traces))
        for frame_index, frame_tiff in enumerate(frames_tiff):
            # print(f"frame_index {frame_index}")
            transient_profile += (
                    frame_tiff[miny:maxy + 1, minx:maxx + 1] * raw_traces[cell, transient[0] + frame_index])
        # averaging
        transient_profile = transient_profile / (np.sum(raw_traces[cell, transient[0]:transient[-1] + 1]))

        return transient_profile, minx, miny

    def scale_polygon_to_source(self, poly_gon, minx, miny):
        coords = list(poly_gon.exterior.coords)
        scaled_coords = []
        for coord in coords:
            scaled_coords.append((coord[0] - minx, coord[1] - miny))
        # print(f"scaled_coords {scaled_coords}")
        return geometry.Polygon(scaled_coords)

    def get_cell_new_coord_in_source(self, cell, minx, miny):
        coord = self.data_and_param.ms.coord_obj.coord[cell]
        # coord = coord - 1
        coord = coord.astype(int)
        n_coord = len(coord[0, :])
        xy = np.zeros((n_coord, 2))
        for n in np.arange(n_coord):
            # shifting the coordinates in the square size_square+1
            xy[n, 0] = coord[0, n] - minx
            xy[n, 1] = coord[1, n] - miny
        return xy

    def update_plot_magnifier(self, mouse_x_position, mouse_y_position, change_frame_ref):
        if change_frame_ref:
            self.magnifier_fig.clear()
            plt.close(self.magnifier_fig)
            self.magnifier_canvas.get_tk_widget().destroy()
            if self.robin_mac:
                self.magnifier_fig = plt.figure(figsize=(3, 3))
            else:
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
        self.first_frame_movie = min(x_from, x_to)
        self.last_frame_movie = max(x_to, x_from)
        self.n_frames_movie = np.abs(x_to - x_from)
        if self.n_frames_movie < 3:
            self.play_movie = False
            self.update_plot()
            return
        self.cell_contour_movie = None
        self.movie_frames = cycle((frame_tiff, frame_index + self.first_frame_movie)
                                  for frame_index, frame_tiff in
                                  enumerate(self.tiff_movie[self.first_frame_movie:self.last_frame_movie + 1]))

        self.update_plot_map_img()

    def square_coord_around_cell(self, cell, size_square, x_len_max, y_len_max):
        """
        For a given cell, give the coordinates of the square surrounding the cell.

        :param cell:
        :param size_square:
        :param x_len_max:
        :param y_len_max:
        :return: (x_beg, x_end, y_beg, y_end)
        """
        c_x, c_y = self.center_coord[cell]
        # c_y correspond to
        c_y = int(c_y)
        c_x = int(c_x)
        # print(f"len_x {len_x} len_y {len_y}")
        # print(f"c_x {c_x} c_y {c_y}")
        # limit of the new frame, should make a square
        x_beg_movie = max(0, c_x - (size_square // 2))
        x_end_movie = min(x_len_max, c_x + (size_square // 2) + 1)
        # means the cell is near a border
        if (x_end_movie - x_beg_movie) < (size_square + 1):
            if (c_x - x_beg_movie) < (x_end_movie - c_x - 1):
                x_end_movie += ((size_square + 1) - (x_end_movie - x_beg_movie))
            else:
                x_beg_movie -= ((size_square + 1) - (x_end_movie - x_beg_movie))

        y_beg_movie = max(0, c_y - (size_square // 2))
        y_end_movie = min(y_len_max, c_y + (size_square // 2) + 1)
        if (y_end_movie - y_beg_movie) < (size_square + 1):
            if (c_y - y_beg_movie) < (y_end_movie - c_y - 1):
                y_end_movie += ((size_square + 1) - (y_end_movie - y_beg_movie))
            else:
                y_beg_movie -= ((size_square + 1) - (y_end_movie - y_beg_movie))

        return x_beg_movie, x_end_movie, y_beg_movie, y_end_movie

    def animate_movie(self, i):
        zoom_mode = self.movie_zoom_mode
        # if self.current_neuron not in self.center_coord:
        #     zoom_mode = False
        if not self.play_movie:
            return []
        result = next(self.movie_frames, None)
        if result is None:
            return
        frame_tiff, frame_index = result
        # for zoom purpose
        len_x = frame_tiff.shape[1]
        len_y = frame_tiff.shape[0]
        if zoom_mode and ((i == -1) or (self.cell_contour_movie is None)):
            size_square = 80
            # zoom around the cell
            self.x_beg_movie, self.x_end_movie, self.y_beg_movie, self.y_end_movie = \
                self.square_coord_around_cell(cell=self.current_neuron, size_square=size_square,
                                              x_len_max=len_x, y_len_max=len_y)
            # used to change the coord of the polygon
            # x_shift = self.x_beg_movie - c_x
            # y_shift = self.y_beg_movie - c_y

            # cell contour
            coord = self.data_and_param.ms.coord_obj.coord[self.current_neuron]
            # coord = coord - 1
            coord = coord.astype(int)
            n_coord = len(coord[0, :])
            xy = np.zeros((n_coord, 2))
            for n in np.arange(n_coord):
                # shifting the coordinates in the square size_square+1
                xy[n, 0] = coord[0, n] - self.x_beg_movie
                xy[n, 1] = coord[1, n] - self.y_beg_movie
                # then multiplying to fit it to the len of the original image
                xy[n, 0] = (xy[n, 0] * len_x) / (size_square + 1)
                xy[n, 1] = (xy[n, 1] * len_y) / (size_square + 1)
            self.cell_contour_movie = patches.Polygon(xy=xy,
                                                      fill=False, linewidth=0, facecolor="red",
                                                      edgecolor="red",
                                                      zorder=15, lw=0.6)
        # x_beg, x_end, y_beg, y_end
        # print(f"frame_tiff[10, :] {frame_tiff[10, :]}")
        if zoom_mode:
            tiff_array = frame_tiff[self.y_beg_movie:self.y_end_movie,
                         self.x_beg_movie:self.x_end_movie]

            # if we do imshow, then the size of the image will be the one of the square, with no zoom
            if i == -1:
                self.last_img_displayed = self.axe_plot_map_img.imshow(frame_tiff,
                                                                       cmap=plt.get_cmap('gray'))
                self.last_img_displayed.set_array(tiff_array)
            else:
                self.last_img_displayed.set_array(tiff_array)
        else:
            if i == -1:
                self.last_img_displayed = self.axe_plot_map_img.imshow(frame_tiff,
                                                                       cmap=plt.get_cmap('gray'))
            else:
                self.last_img_displayed.set_array(frame_tiff)

        if self.last_frame_label is not None:
            self.last_frame_label.set_visible(False)

        x_text = 20
        y_text = 10
        self.last_frame_label = self.axe_plot_map_img.text(x=x_text, y=y_text,
                                                           s=f"{frame_index}", color="red", zorder=20,
                                                           ha='center', va="center", fontsize=10, fontweight='bold')
        if self.trace_movie_p1 is not None:
            self.trace_movie_p1[0].set_visible(False)
        if self.trace_movie_p2 is not None:
            self.trace_movie_p2[0].set_visible(False)

        first_frame = self.first_frame_movie
        last_frame = self.last_frame_movie
        # if zoom_mode:
        #     len_x = size_square
        # else:

        if self.n_frames_movie > len_x:
            new_x_values = np.linspace(0, len_x - 1, self.n_frames_movie)
        else:
            new_x_values = np.arange(self.n_frames_movie) + ((len_x - self.n_frames_movie) / 2)
        # if zoom_mode:
        #     increase_factor = 1.5
        # else:
        increase_factor = 6
        # back to zero with +2 then inceasing the amplitude
        if self.raw_traces is not None:
            raw_traces = (self.raw_traces + 2) * increase_factor
        else:
            raw_traces = (self.traces + 2) * increase_factor
        # y-axis is reverse, so we need to inverse the trace
        # if zoom_mode:
        #     len_y = size_square
        # else:
        if zoom_mode:
            raw_traces = (len_y * 0.8) - raw_traces
        else:
            raw_traces = (len_y * 0.6) - raw_traces
        # need to match the length of our trace to the len
        # if zoom_mode:
        #     trace_lw = 1
        # else:
        trace_lw = 1
        if first_frame < frame_index:
            self.trace_movie_p1 = self.axe_plot_map_img.plot(new_x_values[:frame_index - first_frame],
                                                             raw_traces[self.current_neuron, first_frame:frame_index],
                                                             color="red", alpha=1, zorder=10, lw=trace_lw)
        if last_frame > frame_index:
            self.trace_movie_p2 = self.axe_plot_map_img.plot(new_x_values[frame_index - first_frame:],
                                                             raw_traces[self.current_neuron, frame_index:last_frame],
                                                             color="white", alpha=1, zorder=10, lw=trace_lw)

        self.draw_cell_contour()
        if zoom_mode:
            artists = [self.last_img_displayed, self.last_frame_label, self.cell_contour_movie]
        else:
            artists = [self.last_img_displayed, self.last_frame_label, self.cell_contour]
        if self.trace_movie_p1 is not None:
            artists.append(self.trace_movie_p1[0])
        if self.trace_movie_p2 is not None:
            artists.append(self.trace_movie_p2[0])
        # print(f"x_lim: {self.axe_plot_map_img.get_xaxis().get_xlim()}, y_lim: "
        #       f"{self.axe_plot_map_img.get_yaxis().get_y_lim()}")
        return artists

    def plot_map_img(self, first_time=True, after_movie=False):
        if (self.data_and_param.ms.avg_cell_map_img is None) and (not self.display_michou):
            return

        if first_time:
            self.axe_plot_map_img = self.map_img_fig.add_subplot(111)
        # if self.last_img_displayed is not None:
        #     self.last_img_displayed.set_visible(False)

        if self.display_michou:
            if first_time:
                self.last_img_displayed = self.axe_plot_map_img.imshow(self.michou_imgs[self.michou_img_to_display])
            else:
                self.last_img_displayed.set_array(self.michou_imgs[self.michou_img_to_display])
        else:
            if self.play_movie:
                self.animate_movie(i=-1)
                # # frame_tiff is numpy array of 2D
                # frame_tiff, frame_index = next(self.movie_frames)
                # self.last_img_displayed = self.axe_plot_map_img.imshow(frame_tiff, cmap=plt.get_cmap('gray'))
                # if self.last_frame_label is not None:
                #     self.last_frame_label.set_visible(False)
                # self.last_frame_label = self.axe_plot_map_img.text(x=10, y=10,
                #                                                    s=f"{frame_index}", color="red", zorder=20,
                #                                                    ha='center', va="center", fontsize=10,
                #                                                    fontweight='bold')
            else:
                if self.last_frame_label is not None:
                    self.last_frame_label.set_visible(False)
                    self.last_frame_label = None
                if first_time or after_movie:
                    # after that the size of the image will stay this one
                    self.last_img_displayed = self.axe_plot_map_img.imshow(self.data_and_param.ms.avg_cell_map_img,
                                                                           cmap=plt.get_cmap('gray'))
                else:
                    self.last_img_displayed.set_array(self.data_and_param.ms.avg_cell_map_img)
                self.last_img_displayed.set_zorder(1)
                # self.last_img_displayed.set_visible(True)
                self.draw_cell_contour()

        if first_time:
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)

            self.map_img_fig.tight_layout()
            # self.map_img_fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 0.2, 'h_pad': 0.2})

    def draw_cell_contour(self):
        if self.cell_contour is not None:
            self.cell_contour.set_visible(False)
        if self.play_movie and (self.cell_contour_movie is not None) and self.movie_zoom_mode:
            self.axe_plot_map_img.add_patch(self.cell_contour_movie)
        else:
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

    def update_plot_map_img(self, after_michou=False, after_movie=False):
        # if self.play_movie or after_movie:
        #     self.map_img_fig.clear()
        #     plt.close(self.map_img_fig)
        #     self.map_img_canvas.get_tk_widget().destroy()
        #     self.map_img_fig = plt.figure(figsize=(4, 4))
        #     self.map_img_canvas = FigureCanvasTkAgg(self.map_img_fig, self.map_frame)
        #     self.map_img_fig.canvas.mpl_connect('button_release_event', self.onrelease_map)
        #
        #     if self.axe_plot_map_img is not None:
        #         self.axe_plot_map_img.clear()
        #     self.plot_map_img(first_time=True)
        #
        #     self.map_img_canvas.draw()
        #     self.map_img_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=YES)
        #     if self.play_movie:
        #         self.anim_movie = animation.FuncAnimation(self.map_img_fig, func=self.animate_movie,
        #                                                   frames=self.n_frames_movie,
        #                                                   blit=True, interval=50, repeat=True)
        # else:
        if self.display_michou and (not self.play_movie):
            self.axe_plot_map_img.clear()
            self.plot_map_img()
        elif self.play_movie:
            # self.axe_plot_map_img.clear()
            self.plot_map_img()
            self.map_img_fig.canvas.draw()
            self.map_img_fig.canvas.flush_events()
            self.anim_movie = animation.FuncAnimation(self.map_img_fig, func=self.animate_movie,
                                                      frames=self.n_frames_movie,
                                                      blit=True, interval=100, repeat=True)  # repeat=True,
            return
            # self.after(self.movie_delay, self.update_plot_map_img)
        else:
            if after_michou or after_movie:
                self.axe_plot_map_img.clear()
                self.plot_map_img(after_movie=after_movie)
            else:
                self.draw_cell_contour()

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

        if self.cells_to_remove[self.current_neuron] == 1:
            self.axe_plot.set_facecolor("gray")
        else:
            self.axe_plot.set_facecolor("black")

        # #################### SMOOTHED TRACE ####################

        color_trace = self.color_trace
        self.line1, = self.axe_plot.plot(np.arange(self.nb_times_traces), self.traces[self.current_neuron, :],
                                         color=color_trace, zorder=10)

        # #################### TRANSIENT CLASSIFIER VALUES ####################

        if self.show_transient_classifier:
            if self.current_neuron in self.transient_prediction:
                # TODO: work on the shape of predictions, should be (n_celles, n_class)
                predictions = self.transient_prediction[self.current_neuron]
                # print(f"predictions > 1: {predictions[predictions > 1]}")
                if len(predictions.shape) == 1:
                    pass
                if (len(predictions.shape) == 1) or predictions.shape[1] == 1:
                    predictions_color = "red"
                else:
                    predictions_color = "dimgrey"
                if len(predictions.shape) == 1:
                    self.axe_plot.plot(np.arange(self.nb_times_traces),
                                       predictions / 2,
                                       color=predictions_color, zorder=20)
                else:
                    for index_pred in np.arange(predictions.shape[1]):
                        self.axe_plot.plot(np.arange(self.nb_times_traces),
                                           (predictions[:, index_pred] / 2) - (0.75 * index_pred),
                                           color=predictions_color, zorder=20)
                threshold_tc = self.transient_classifier_threshold
                self.axe_plot.hlines(threshold_tc / 2, 0, self.nb_times_traces - 1, color="red",
                                     linewidth=0.5,
                                     linestyles="dashed")
                self.axe_plot.hlines(1 / 2, 0, self.nb_times_traces - 1, color="red",
                                     linewidth=0.5)
                if threshold_tc not in self.transient_prediction_periods[self.current_neuron]:
                    if len(predictions.shape) == 1:
                        predicted_raster_dur = np.zeros(len(predictions), dtype="int8")
                    else:
                        predicted_raster_dur = np.zeros(predictions.shape[0], dtype="int8")

                    if len(predictions.shape) == 1:
                        real_transient_frames = predictions >= threshold_tc
                    elif predictions.shape[1] == 1:
                        real_transient_frames = predictions[:, 0] >= threshold_tc
                    elif predictions.shape[1] == 3:
                        # real transient, fake ones, other (neuropil, decay etc...)
                        # keeping predictions about real transient when superior
                        # to other prediction on the same frame
                        max_pred_by_frame = np.max(predictions, axis=1)
                        real_transient_frames = (predictions[:, 0] == max_pred_by_frame)
                        for class_index in np.arange(predictions.shape[1]):
                            frames_index = (predictions[:, class_index] == max_pred_by_frame)
                            tmp_raster_dur = np.zeros(predictions.shape[0], dtype="int8")
                            tmp_raster_dur[frames_index] = 1
                            periods = get_continous_time_periods(tmp_raster_dur)
                            for period in periods:
                                frames = np.arange(period[0], period[1] + 1)
                                self.axe_plot.plot(frames,
                                                   (predictions[frames, class_index] / 2) -
                                                   (0.75 * class_index), lw=2,
                                                   color="red", zorder=21)
                    elif predictions.shape[1] == 2:
                        # real transient, fake ones
                        # keeping predictions about real transient superior to the threshold
                        # and superior to other prediction on the same frame
                        max_pred_by_frame = np.max(predictions, axis=1)
                        real_transient_frames = np.logical_and((predictions[:, 0] >= threshold_tc),
                                                               (predictions[:, 0] == max_pred_by_frame))
                    predicted_raster_dur[real_transient_frames] = 1
                    active_periods = get_continous_time_periods(predicted_raster_dur)
                    self.transient_prediction_periods[self.current_neuron][threshold_tc] = active_periods
                else:
                    active_periods = self.transient_prediction_periods[self.current_neuron][threshold_tc]
                    if (len(predictions.shape) > 1) and (predictions.shape[1] == 3):
                        # real transient, fake ones, other (neuropil, decay etc...)
                        # keeping predictions about real transient when superior
                        # to other prediction on the same frame
                        max_pred_by_frame = np.max(predictions, axis=1)
                        real_transient_frames = (predictions[:, 0] == max_pred_by_frame)
                        for class_index in np.arange(predictions.shape[1]):
                            frames_index = (predictions[:, class_index] == max_pred_by_frame)
                            tmp_raster_dur = np.zeros(predictions.shape[0], dtype="int8")
                            tmp_raster_dur[frames_index] = 1
                            periods = get_continous_time_periods(tmp_raster_dur)
                            for period in periods:
                                frames = np.arange(period[0], period[1] + 1)
                                self.axe_plot.plot(frames,
                                                   (predictions[frames, class_index] / 2) -
                                                   (0.75 * class_index), lw=2,
                                                   color="red", zorder=21)

                for i_ap, active_period in enumerate(active_periods):
                    period = np.arange(active_period[0], active_period[1] + 1)
                    min_traces = np.min(self.traces[self.current_neuron, period])  # - 0.1
                    y2 = np.repeat(min_traces, len(period))
                    self.axe_plot.fill_between(x=period, y1=self.traces[self.current_neuron, period], y2=y2,
                                               color=self.classifier_filling_color)

                # display the stat only when the threshold is changed
                if self.print_transients_predictions_stat:
                    # first we build the raster_dur for this cell
                    if self.current_neuron not in self.raster_dur_for_a_cell:
                        raster_dur = np.zeros(self.nb_times_traces, dtype="int8")
                        onsets_index = np.where(self.onset_times[self.current_neuron, :])[0]
                        peaks_index = np.where(self.peak_nums[self.current_neuron, :])[0]

                        for onset_index in onsets_index:
                            peaks_after = np.where(peaks_index > onset_index)[0]
                            if len(peaks_after) == 0:
                                continue
                            peaks_after = peaks_index[peaks_after]
                            peak_after = peaks_after[0]
                            # print(f"onset_index {onset_index}, peak_after {peak_after}")

                            raster_dur[onset_index:peak_after + 1] = 1
                        self.raster_dur_for_a_cell[self.current_neuron] = raster_dur
                    else:
                        raster_dur = self.raster_dur_for_a_cell[self.current_neuron]

                    predicted_raster_dur = np.zeros(predictions.shape[0], dtype="int8")
                    if len(predictions.shape) == 1:
                        real_transient_frames = predictions >= threshold_tc
                    elif predictions.shape[1] == 1:
                        real_transient_frames = predictions[:, 0] >= threshold_tc
                    elif predictions.shape[1] == 3:
                        # real transient, fake ones, other (neuropil, decay etc...)
                        # keeping predictions about real transient when superior
                        # to other prediction on the same frame
                        max_pred_by_frame = np.max(predictions, axis=1)
                        real_transient_frames = (predictions[:, 0] == max_pred_by_frame)
                    elif predictions.shape[1] == 2:
                        # real transient, fake ones
                        # keeping predictions about real transient superior to the threshold
                        # and superior to other prediction on the same frame
                        max_pred_by_frame = np.max(predictions, axis=1)
                        real_transient_frames = np.logical_and((predictions[:, 0] >= threshold_tc),
                                                               (predictions[:, 0] == max_pred_by_frame))
                    predicted_raster_dur[real_transient_frames] = 1
                    frames_stat, transients_stat = classification_stat.compute_stats(raster_dur, predicted_raster_dur,
                                                                                     self.traces[self.current_neuron])

                    # frames stats
                    print(f"Cell {self.current_neuron} with threshold {threshold_tc}")
                    print(f"Frames stat:")
                    for key, value in frames_stat.items():
                        print(f"{key}: {str(np.round(value, 4))}")

                    print(f"###")
                    print(f"Transients stat:")
                    for key, value in transients_stat.items():
                        print(f"{key}: {str(np.round(value, 4))}")

                    self.print_transients_predictions_stat = False

        if self.raw_traces_median is not None:
            self.axe_plot.plot(np.arange(self.nb_times_traces),
                               self.raw_traces_median[self.current_neuron, :],
                               color="red", alpha=0.8, zorder=9)

        if (self.raw_traces is not None) and self.display_raw_traces:
            self.axe_plot.plot(np.arange(self.nb_times_traces),
                               self.raw_traces[self.current_neuron, :],
                               color=self.color_raw_trace, alpha=0.8, zorder=9)

            if self.use_ratio_traces and (self.ratio_traces is not None):
                # we plot a trace that show the ratio between traces and raw_traces
                # it could be useful if the traces is partially demixed
                self.axe_plot.plot(np.arange(self.nb_times_traces),
                                   self.ratio_traces[self.current_neuron, :],
                                   linewidth=0.5,
                                   color="red", alpha=0.8, zorder=15)
                self.axe_plot.hlines(-2, 0, self.nb_times_traces - 1, color="black",
                                     linewidth=0.5,
                                     linestyles="dashed")
                y2 = np.repeat(-2, self.nb_times_traces)
                self.axe_plot.fill_between(x=np.arange(self.nb_times_traces),
                                           y1=self.ratio_traces[self.current_neuron, :], y2=y2,
                                           color="red", alpha=0.6)

            if self.raw_traces_binned is not None:
                self.axe_plot.plot(np.arange(0, self.nb_times_traces, 10),
                                   self.raw_traces_binned[self.current_neuron, :],
                                   color="green", alpha=0.9, zorder=8)
                # O y-axis line
                # self.axe_plot.hlines(0, 0, self.nb_times_traces - 1, color="black", linewidth=1)

        # #################### MIN & MAX VALUE ####################

        if self.raw_traces is not None:
            max_value = max(np.max(self.traces[self.current_neuron, :]),
                            np.max(self.raw_traces[self.current_neuron, :]))
            min_value = min(np.min(self.traces[self.current_neuron, :]),
                            np.min(self.raw_traces[self.current_neuron, :]))
        else:
            max_value = np.max(self.traces[self.current_neuron, :])
            min_value = np.min(self.traces[self.current_neuron, :])

        if self.raw_traces_binned is not None:
            min_value = min(min_value, np.min(self.raw_traces_binned[self.current_neuron, :]))

        # #################### transient predicted to look at ###########
        # used when we want to look at some specific predicted transients
        if (self.last_predicted_transient_selected is not None) and \
                (self.last_predicted_transient_selected[0] == self.current_neuron):
            self.axe_plot.vlines(self.last_predicted_transient_selected[1], min_value, max_value,
                                 color=self.color_transient_predicted_to_look_at, linewidth=1)

        # #################### ONSETS ####################

        onsets = np.where(self.onset_times[self.current_neuron, :] > 0)[0]
        self.axe_plot.vlines(onsets, min_value, max_value, color=self.color_onset, linewidth=1,
                             linestyles="dashed")

        # #################### TO AGREE ONSETS ####################

        if self.to_agree_spike_nums is not None:
            to_agree_onsets = np.where(self.to_agree_spike_nums[self.current_neuron, :])[0]
            self.axe_plot.vlines(to_agree_onsets, min_value, max_value, color="red", linewidth=1.5,
                                 linestyles="dashed")

        # #################### DOUBTFUL FRAMES ####################

        if (self.current_neuron in self.doubtful_frames_periods) and \
                (len(self.doubtful_frames_periods[self.current_neuron]) > 0):
            for doubtful_frames_period in self.doubtful_frames_periods[self.current_neuron]:
                self.axe_plot.axvspan(doubtful_frames_period[0], doubtful_frames_period[1], ymax=1,
                                      alpha=0.5, facecolor="black", zorder=15)

        # #################### CAIMAN ACTIVE PERIODS ####################

        if self.caiman_active_periods is not None:
            for caiman_period in self.caiman_active_periods[self.current_neuron]:
                self.axe_plot.hlines(min_value + 0.5, caiman_period[0], caiman_period[1],
                                     color="black", linewidth=5, zorder=8)
            self.axe_plot.vlines(np.where(self.caiman_spike_nums[self.current_neuron])[0],
                                 min_value, min_value + 0.5,
                                 color="green", linewidth=2, zorder=7)

        # #################### PEAKS ####################

        size_peak_scatter = 50
        peaks = np.where(self.peak_nums[self.current_neuron, :] > 0)[0]
        if self.display_threshold:
            threshold = self.get_threshold()
            peaks_under_threshold = np.where(self.traces[self.current_neuron, peaks] < threshold)[0]
            if len(peaks_under_threshold) == 0:
                peaks_under_threshold_index = []
                peaks_under_threshold_value = []
                self.peaks_under_threshold_index = []
            else:
                peaks_under_threshold_index = peaks[peaks_under_threshold]
                self.peaks_under_threshold_index = peaks_under_threshold_index
                peaks_under_threshold_value = self.traces[self.current_neuron, peaks][peaks_under_threshold]
            # to display the peaks on the raw trace as well
            # peaks_under_threshold_value_raw = self.raw_traces[self.current_neuron, peaks][peaks_under_threshold]
            peaks_over_threshold = np.where(self.traces[self.current_neuron, peaks] >= threshold)[0]
            if len(peaks_over_threshold) == 0:
                peaks_over_threshold_index = []
                peaks_over_threshold_value = []
            else:
                peaks_over_threshold_index = peaks[peaks_over_threshold]
                peaks_over_threshold_value = self.traces[self.current_neuron, peaks][peaks_over_threshold]

            # plotting peaks
            # z_order=10 indicate that the scatter will be on top
            if len(peaks_over_threshold_index) > 0:
                self.ax1_bottom_scatter = self.axe_plot.scatter(peaks_over_threshold_index, peaks_over_threshold_value,
                                                                marker='o', c=self.color_peak,
                                                                edgecolors=self.color_edge_peak, s=size_peak_scatter,
                                                                zorder=10)
            if len(peaks_over_threshold_index) > 0:
                self.ax1_bottom_scatter = self.axe_plot.scatter(peaks_under_threshold_index,
                                                                peaks_under_threshold_value,
                                                                marker='o', c=self.color_peak_under_threshold,
                                                                edgecolors=self.color_edge_peak, s=size_peak_scatter,
                                                                zorder=10)
            # self.ax1_bottom_scatter = self.axe_plot.scatter(peaks_over_threshold_index, peaks_over_threshold_value_raw,
            #                                                 marker='o', c=self.color_peak,
            #                                                 edgecolors=self.color_edge_peak, s=30, zorder=10)
            # self.ax1_bottom_scatter = self.axe_plot.scatter(peaks_under_threshold_index,
            #                                                 peaks_under_threshold_value_raw,
            #                                                 marker='o', c=self.color_peak_under_threshold,
            #                                                 edgecolors=self.color_edge_peak, s=30, zorder=10)

            self.axe_plot.hlines(threshold, 0, self.nb_times_traces - 1, color=self.color_threshold_line, linewidth=1,
                                 linestyles="dashed")
        elif self.display_correlations:
            if self.correlation_for_each_peak_option and (self.peaks_correlation is not None):
                color_over_threshold = self.color_peak
                color_under_threshold = self.color_peak_under_threshold
                color_undetermined = "cornflowerblue"
                threshold = self.correlation_thresold

                peaks_over_threshold = np.where(self.peaks_correlation[self.current_neuron, peaks] >= threshold)[0]
                peaks_over_threshold_index = peaks[peaks_over_threshold]
                if len(peaks_over_threshold_index) == 0:
                    peaks_over_threshold_value = []
                else:
                    peaks_over_threshold_value = self.traces[self.current_neuron, peaks_over_threshold_index]

                # among peaks under treshold we need to find the ones with not overlaping cell over the tresholds
                # using self.peaks_correlation and the self.overlaping_cells dict
                peaks_left = np.where(self.peaks_correlation[self.current_neuron, peaks] < threshold)[0]
                peaks_left_index = peaks[peaks_left]

                overlapping_cells = self.overlapping_cells[self.current_neuron]
                if len(overlapping_cells) == 0:
                    peaks_under_threshold_index = peaks_left_index
                    peaks_under_threshold_value = self.traces[self.current_neuron, peaks_under_threshold_index]
                    peaks_undetermined_index = []
                    peaks_undetermined_value = []
                else:
                    overlapping_cells = np.array(list(overlapping_cells))
                    peaks_under_threshold_index = []
                    peaks_undetermined_index = []
                    # print(f"overlapping_cells {overlapping_cells}, peaks_left_index {peaks_left_index}")
                    for peak in peaks_left_index:
                        cells_over_threshold = np.where(self.peaks_correlation[overlapping_cells, peak] >= threshold)[0]
                        if len(cells_over_threshold) > 0:
                            # means at least an overlapping cell is correlated to its source for this peak transient
                            peaks_under_threshold_index.append(peak)
                        else:
                            # means the peak is not due to an overlapping cell, movement might have happened
                            peaks_undetermined_index.append(peak)
                    peaks_under_threshold_index = np.array(peaks_under_threshold_index)
                    if len(peaks_under_threshold_index) == 0:
                        peaks_under_threshold_value = []
                    else:
                        peaks_under_threshold_value = self.traces[self.current_neuron, peaks_under_threshold_index]
                    peaks_undetermined_index = np.array(peaks_undetermined_index)
                    if len(peaks_undetermined_index) == 0:
                        peaks_undetermined_value = []
                    else:
                        peaks_undetermined_value = self.traces[self.current_neuron, peaks_undetermined_index]
                # use to remove them
                self.peaks_under_threshold_index = peaks_under_threshold_index

                if len(peaks_over_threshold_index) > 0:
                    self.ax1_bottom_scatter = self.axe_plot.scatter(peaks_over_threshold_index,
                                                                    peaks_over_threshold_value,
                                                                    marker='o', c=color_over_threshold,
                                                                    edgecolors=self.color_edge_peak,
                                                                    s=size_peak_scatter, zorder=10)

                if len(peaks_under_threshold_index) > 0:
                    self.ax1_bottom_scatter = self.axe_plot.scatter(peaks_under_threshold_index,
                                                                    peaks_under_threshold_value,
                                                                    marker='o', c=color_under_threshold,
                                                                    edgecolors=self.color_edge_peak,
                                                                    s=size_peak_scatter, zorder=10)
                if len(peaks_undetermined_index) > 0:
                    self.ax1_bottom_scatter = self.axe_plot.scatter(peaks_undetermined_index,
                                                                    peaks_undetermined_value,
                                                                    marker='o', c=color_undetermined,
                                                                    edgecolors=self.color_edge_peak,
                                                                    s=size_peak_scatter, zorder=10)

        else:
            # plotting peaks
            # z_order=10 indicate that the scatter will be on top
            self.ax1_bottom_scatter = self.axe_plot.scatter(peaks, self.traces[self.current_neuron, peaks],
                                                            marker='o', c=self.color_peak,
                                                            edgecolors=self.color_edge_peak, s=size_peak_scatter,
                                                            zorder=10)
            # self.ax1_bottom_scatter = self.axe_plot.scatter(peaks, self.raw_traces[self.current_neuron, peaks],
            #                                                 marker='o', c=self.color_peak,
            #                                                 edgecolors=self.color_edge_peak, s=30, zorder=10)

        # #################### TO AGREE PEAKS ####################

        if self.to_agree_peak_nums is not None:
            to_agree_peaks = np.where(self.to_agree_peak_nums[self.current_neuron, :])[0]
            peaks_amplitude = self.traces[self.current_neuron, to_agree_peaks]
            self.ax1_bottom_scatter = self.axe_plot.scatter(to_agree_peaks,
                                                            peaks_amplitude,
                                                            marker='o', c="black",
                                                            edgecolors="red",
                                                            s=size_peak_scatter * 1.2, zorder=10)

        # #################### CLICK SCATTER ####################

        if self.first_click_to_remove is not None:
            self.axe_plot.scatter(self.first_click_to_remove["x"], self.first_click_to_remove["y"], marker='x',
                                  c=self.color_mark_to_remove, s=30)
        if self.click_corr_coord:
            self.axe_plot.scatter(self.click_corr_coord["x"], self.click_corr_coord["y"], marker='x',
                                  c="red", s=30)

        if (self.mvt_frames_periods is not None) and self.display_mvt:
            for mvt_frames_period in self.mvt_frames_periods:
                # print(f"mvt_frames_period[0], mvt_frames_period[1] {mvt_frames_period[0]} {mvt_frames_period[1]}")
                self.axe_plot.axvspan(mvt_frames_period[0], mvt_frames_period[1], ymax=1,
                                      alpha=0.8, facecolor="red", zorder=1)

        # by default the y axis zoom is set to fit the wider amplitude of the current neuron
        fit_plot_to_all_max = False
        if fit_plot_to_all_max:
            if self.display_raw_traces and (self.raw_traces is not None):
                min_value = min(0, np.min(self.raw_traces[self.current_neuron, :]))
                self.axe_plot.set_ylim(min_value, math.ceil(np.max(self.traces)))
            else:
                self.axe_plot.set_ylim(0, math.ceil(np.max(self.traces)))
        else:
            if self.raw_traces is not None:
                max_value = max(np.max(self.raw_traces[self.current_neuron, :]),
                                np.max(self.traces[self.current_neuron, :]))
                min_value = min(np.min(self.raw_traces[self.current_neuron, :]),
                                np.min(self.traces[self.current_neuron, :]))
            else:
                max_value = np.max(self.traces[self.current_neuron, :])
                min_value = np.min(self.traces[self.current_neuron, :])
            if self.raw_traces_binned is not None:
                min_value = min(min_value, np.min(self.raw_traces_binned[self.current_neuron, :]))

            self.axe_plot.set_ylim(min_value,
                                   math.ceil(max_value))

        # removing first x_axis
        axes_to_clean = [self.axe_plot]
        for ax in axes_to_clean:
            # ax.axes.get_xaxis().set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        self.axe_plot.yaxis.label.set_color("white")
        self.axe_plot.tick_params(axis='y', colors="white")
        self.axe_plot.xaxis.label.set_color("white")
        self.axe_plot.tick_params(axis='x', colors="white")
        self.axe_plot.margins(0)
        self.fig.tight_layout()
        # tight_layout is posing issue on Ubuntu
        # self.fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 0.1, 'h_pad': 0.1})

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
                    new_x_limit=None, new_y_limit=None, changing_face_color=False,
                    raw_trace_display_action=False):
        # used to keep the same zoom after updating the plot
        # if we change neuron, then back to no zoom mode
        left_x_limit_1, right_x_limit_1 = self.axe_plot.get_xlim()
        bottom_limit_1, top_limit_1 = self.axe_plot.get_ylim()
        self.axe_plot.clear()
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
            self.axe_plot.set_ylim(bottom=bottom_limit_1, top=top_limit_1, auto=None)
        if new_x_limit is not None:
            self.axe_plot.set_xlim(left=new_x_limit[0], right=new_x_limit[1], auto=None)
        if (new_y_limit is not None) and (not amplitude_zoom_fit):
            self.axe_plot.set_ylim(new_y_limit[0], new_y_limit[1])
        # self.line1.set_ydata(self.traces[self.current_neuron, :])
        if new_neuron or changing_face_color:
            self.plot_canvas.draw()
            self.plot_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=YES)
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    # def spin_box_update(self):
    #     content = int(self.spin_box_button.get())
    #     if content != self.current_neuron:
    #         self.current_neuron = content
    #         self.update_plot()

    def update_to_agree_label(self):
        if self.to_agree_label is not None:
            self.to_agree_label["text"] = f"{self.numbers_of_onset_to_agree()}/{self.numbers_of_peak_to_agree()}"

    # if an onset has been removed or added to traces and spike_nums for current_neuron
    def update_after_onset_change(self, new_neuron=-1,
                                  new_x_limit=None, new_y_limit=None):
        """
        Update the frame if an onset change has been made
        :param new_neuron: if -1, then the neuron hasn't changed, neuron might change if undo or redo are done.
        :return:
        """
        if new_neuron > -1:
            self.current_neuron = new_neuron
        self.onset_numbers_label["text"] = f"{self.numbers_of_onset()}"
        self.peak_numbers_label["text"] = f"{self.numbers_of_peak()}"
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

    def update_neuron(self, new_neuron,
                      new_x_limit=None, new_y_limit=None, amplitude_zoom_fit=True):
        """
        Call when the neuron number has changed
        :return:
        """
        self.current_neuron = new_neuron
        if self.show_transient_classifier:
            self.set_transient_classifier_prediction_for_cell(cell=self.current_neuron)
            self.print_transients_predictions_stat = True
        if self.correlation_for_each_peak_option:
            if self.display_correlations:
                self.correlation_check_box_action(from_std_treshold=True)
                # start_time = time.time()
                # self.compute_source_and_transients_correlation(main_cell=self.current_neuron)
                # stop_time = time.time()
                # print(f"Time for computing source and transients correlation for cell {self.current_neuron}: "
                #       f"{np.round(stop_time-start_time, 3)} s")

        self.neuron_label["text"] = f"{self.current_neuron}"
        # self.spin_box_button.icursor(new_neuron)
        self.clear_and_update_entry_neuron_widget()
        self.onset_numbers_label["text"] = f"{self.numbers_of_onset()}"
        self.peak_numbers_label["text"] = f"{self.numbers_of_peak()}"
        self.update_to_agree_label()

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

        self.first_click_to_remove = None
        self.click_corr_coord = None
        self.update_plot(new_neuron=True,
                         new_x_limit=new_x_limit, new_y_limit=new_y_limit,
                         amplitude_zoom_fit=amplitude_zoom_fit)
        self.update_plot_map_img()


def print_save(text, file, to_write, no_print=False):
    if not no_print:
        print(text)
    if to_write:
        file.write(text + '\n')


def merge_close_values(raster, raster_to_fill, cell, merging_threshold):
    """

    :param raster: Raster is a 2d binary array, lines represents cells, columns binary values
    :param cell: which cell to merge
    :param merging_threshold: times separation between two values under which to merge them
    :return:
    """
    values = raster[cell]
    indices = np.where(values)[0]
    peaks_diff = np.diff(indices)
    to_merge_indices = np.where(peaks_diff < merging_threshold)[0]
    for index_to_merge in to_merge_indices:
        value_1 = indices[index_to_merge]
        if raster[cell, value_1] == 0:
            # could be the case if more than 2 peaks were closes, then the first 2 would be already merge
            # then the next loop in n_gound_truther will do the job
            continue
        value_2 = indices[index_to_merge + 1]
        new_peak_index = (value_1 + value_2) // 2
        raster[cell, value_1] = 0
        raster[cell, value_2] = 0
        raster_to_fill[cell, new_peak_index] = 1
        # raster[cell, new_peak_index] = 1


def fusion_gui_selection(path_data):
    rep_fusion = "for_fusion"
    file_names = []
    txt_to_read = None
    # merge close ones
    # if True merge spikes or peaks that are less then merging_threshold, taking the average value
    merge_close_ones = True
    merging_threshold = 7
    # how many people did ground truth, useulf to merge close ones
    # should be superior to 1
    n_ground_truther = 2

    # look for filenames in the fisrst directory, if we don't break, it will go through all directories
    for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(path_data, rep_fusion)):
        file_names.extend(local_filenames)
        break

    if len(file_names) == 0:
        return

    for file_name in file_names:
        if file_name.endswith(".txt"):
            txt_to_read = file_name

    data_files = []
    cells_by_file = []
    n_cells = None
    n_times = None
    with open(os.path.join(path_data, rep_fusion, txt_to_read), "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split(':')
            data_file_name = line_list[0]
            # finding the file
            for file_name in file_names:
                if (data_file_name + ".mat").lower() in file_name.lower():
                    print(f"data_file_name {data_file_name}")
                    data_file = hdf5storage.loadmat(os.path.join(path_data, rep_fusion, file_name))
                    data_files.append(data_file)
                    if n_cells is None:
                        peak_nums = data_file['LocPeakMatrix_Python'].astype(int)
                        n_cells = peak_nums.shape[0]
                        n_times = peak_nums.shape[1]

                    cells_str_split = line_list[1].split()
                    cells = []
                    for cells_str in cells_str_split:
                        if "-" in cells_str:
                            limits = cells_str.split('-')
                            cells.extend(list(range(int(limits[0]), int(limits[1]) + 1)))
                        else:
                            cells.append(int(cells_str))
                    cells_by_file.append(cells)
                    continue

    peak_nums = np.zeros((n_cells, n_times), dtype="int8")
    spike_nums = np.zeros((n_cells, n_times), dtype="int8")
    to_agree_peak_nums = np.zeros((n_cells, n_times), dtype="int8")
    to_agree_spike_nums = np.zeros((n_cells, n_times), dtype="int8")
    doubtful_frames_nums = np.zeros((n_cells, n_times), dtype="int8")
    mvt_frames_nums = np.zeros((n_cells, n_times), dtype="int8")
    inter_neurons = np.zeros(0, dtype="int16")
    cells_to_remove = np.zeros(0, dtype="int16")
    cells_fusioned = []

    # in case if one of the file is a fusion file
    for index_data, data_file in enumerate(data_files):
        if "to_agree_peak_nums" in data_file:
            to_agree_peak_nums = data_file['to_agree_peak_nums'].astype(int)
        if "to_agree_spike_nums" in data_file:
            to_agree_spike_nums = data_file['to_agree_spike_nums'].astype(int)

    for index_data, data_file in enumerate(data_files):
        if "inter_neurons" in data_file:
            inter_neurons_data = data_file['inter_neurons'].astype(int)
            if len(inter_neurons_data) > 0:
                inter_neurons = np.union1d(inter_neurons, inter_neurons_data[0])
        if "cells_to_remove" in data_file:
            cells_to_remove_data = data_file['cells_to_remove'].astype(int)
            if len(cells_to_remove_data) > 0:
                cells_to_remove = np.union1d(cells_to_remove, cells_to_remove_data[0])

        peak_nums_data = data_file['LocPeakMatrix_Python'].astype(int)
        spike_nums_data = data_file['Bin100ms_spikedigital_Python'].astype(int)
        cells_fusioned.extend(list(cells_by_file[index_data]))
        for cell in cells_by_file[index_data]:
            # to_agree_peaks = np.where(to_agree_peak_nums[cell])[0]
            # to_agree_onsets = np.where(to_agree_spike_nums[cell])[0]

            # checking if we have added some peaks before
            if np.sum(peak_nums[cell]) == 0:
                peak_nums[cell] = peak_nums_data[cell]
            else:
                peaks_index_data = np.where(peak_nums_data[cell])[0]
                peaks_index = np.where(peak_nums[cell])[0]
                peaks_index_to_agree = np.setxor1d(peaks_index_data, peaks_index, assume_unique=True)
                peaks_index_agreed = np.intersect1d(peaks_index_data, peaks_index, assume_unique=True)
                to_agree_peak_nums[cell, peaks_index_to_agree] = 1
                peak_nums[cell, peaks_index_to_agree] = 0
                peak_nums[cell, peaks_index_agreed] = 1

            if np.sum(spike_nums[cell]) == 0:
                spike_nums[cell] = spike_nums_data[cell]
            else:
                onsets_index_data = np.where(spike_nums_data[cell])[0]
                onsets_index = np.where(spike_nums[cell])[0]
                onsets_index_to_agree = np.setxor1d(onsets_index_data, onsets_index, assume_unique=True)
                onsets_index_agreed = np.intersect1d(onsets_index_data, onsets_index, assume_unique=True)
                to_agree_spike_nums[cell, onsets_index_to_agree] = 1
                spike_nums[cell, onsets_index_to_agree] = 0
                spike_nums[cell, onsets_index_agreed] = 1

        if "doubtful_frames_nums" in data_file:
            doubtful_frames_nums_data = data_file['doubtful_frames_nums'].astype(int)
            for cell in np.arange(n_cells):
                if np.sum(doubtful_frames_nums_data[cell]) > 0:
                    doubtful_frames_nums[cell, doubtful_frames_nums_data[cell] > 0] = 1

        if "mvt_frames_nums" in data_file:
            mvt_frames_nums_data = data_file['mvt_frames_nums'].astype(int)
            for cell in np.arange(n_cells):
                if np.sum(mvt_frames_nums_data[cell]) > 0:
                    mvt_frames_nums[cell, mvt_frames_nums_data[cell] > 0] = 1

    cells_fusioned = np.unique(cells_fusioned)
    if merge_close_ones:
        for n in np.arange(n_ground_truther - 1):
            for cell in cells_fusioned:
                merge_close_values(raster=to_agree_peak_nums, raster_to_fill=peak_nums,
                                   cell=cell, merging_threshold=merging_threshold)
                merge_close_values(raster=to_agree_spike_nums, raster_to_fill=spike_nums,
                                   cell=cell, merging_threshold=merging_threshold)

    # now we want to fill the cells that didn't have to be fusionned, using one the data file
    cells_to_fill = np.setxor1d(cells_fusioned, np.arange(n_cells), assume_unique=True)
    for cell in cells_to_fill:
        peak_nums_data = data_files[0]['LocPeakMatrix_Python'].astype(int)
        spike_nums_data = data_files[0]['Bin100ms_spikedigital_Python'].astype(int)
        peak_nums[cell] = peak_nums_data[cell]
        spike_nums[cell] = spike_nums_data[cell]

    sio.savemat(os.path.join(path_data, rep_fusion, "fusion.mat"), {'Bin100ms_spikedigital_Python': spike_nums,
                                                                    'LocPeakMatrix_Python': peak_nums,
                                                                    'cells_to_remove': cells_to_remove,
                                                                    'inter_neurons': inter_neurons,
                                                                    "doubtful_frames_nums": doubtful_frames_nums,
                                                                    "mvt_frames_nums": mvt_frames_nums,
                                                                    "to_agree_peak_nums": to_agree_peak_nums,
                                                                    "to_agree_spike_nums": to_agree_spike_nums})


def main_manual():
    root_path = None
    with open("param_hne.txt", "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split('=')
            root_path = line_list[1]
    if root_path is None:
        raise Exception("Root path is None")

    path_data = root_path + "data/"

    do_fusion_gui_selection = False

    if do_fusion_gui_selection:
        fusion_gui_selection(path_data=path_data)
        return

    print(f'platform: {platform}')
    root = Tk()
    root.title(f"Session selection")

    result_path = root_path + "results_hne"

    data_and_param = DataAndParam(path_data=path_data, result_path=result_path)

    app = ChooseSessionFrame(data_and_param=data_and_param,
                             master=root)
    app.mainloop()
    # root.destroy()


main_manual()
