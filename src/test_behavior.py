from datetime import datetime
from pynwb import NWBHDF5IO
from abc import abstractmethod
from cv2 import VideoCapture
import cv2
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from skimage.segmentation import random_walker
import skimage.draw as draw
import sys
from random import randrange
import yaml
from scipy import signal
import time



class DraggableRectangle:
    lock = None  # only one can be animated at a time
    def __init__(self, rect):
        self.rect = rect
        self.press = None
        self.background = None

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes: return
        if DraggableRectangle.lock is not None: return
        contains, attrd = self.rect.contains(event)
        if not contains: return
        # print('event contains', self.rect.xy)
        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata
        DraggableRectangle.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.rect.figure.canvas
        axes = self.rect.axes
        self.rect.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.rect.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.rect)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if DraggableRectangle.lock is not self:
            return
        if event.inaxes != self.rect.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)

        canvas = self.rect.figure.canvas
        axes = self.rect.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.rect)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        'on release we reset the press data'
        if DraggableRectangle.lock is not self:
            return

        self.press = None
        DraggableRectangle.lock = None

        # turn off the rect animation property and reset the background
        self.rect.set_animated(False)
        self.background = None

        # redraw the full figure
        self.rect.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

class ImageAreaSelection:

    def __init__(self, video_reader, initial_frame, n_frames, size_rect, title, binary_thresholds,
                 x_bottom_left, y_bottom_left):
        """
            Build a plot with a image a rectangle of size_rect that can be moved over the image.
            Call bottom_left coin coordinates
        Args:
            video_reader:
            n_frames:
            initial_frame:
            size_rect:
            title:
            binary_thresholds: list of 1 to 2 int, between 0 and 255, in ascending order, representing the different
            level of gray
            x_bottom_left:
            y_bottom_left:
        """
        self.video_reader = video_reader
        self.n_frames = n_frames
        self.initial_frame = initial_frame
        self.binary_thresholds = binary_thresholds
        self.size_rect = list(size_rect)
        self.title = title
        self.fig = None
        self.last_frames = []
        self.imshow_obj = None
        self.current_frame = initial_frame
        self.binary_mode = False
        self.rect = None
        self.step_rect_dim = 5
        # bottom left corner coordinates
        self.x_bottom_left = x_bottom_left
        self.y_bottom_left = y_bottom_left

    def select_area(self):
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.title(self.title)
        ax = self.fig.add_subplot(111)
        img_frame = self.video_reader.get_frame(self.current_frame)
        ax.set_ylim(img_frame.shape[0])
        ax.set_xlim(img_frame.shape[1])
        # ax.imshow(np.fliplr(img), origin='lower')
        if self.binary_mode:
            img_frame = binarize_frame(img_frame, binary_thresholds=self.binary_thresholds)
        self.imshow_obj = ax.imshow(np.flipud(img_frame), cmap='gray')
        # ax.imshow(img)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        # rects = ax.bar(range(10), 20*np.random.rand(10))
        left = self.x_bottom_left
        bottom = self.y_bottom_left
        width = self.size_rect[0]
        height = self.size_rect[1]
        self.rect = plt.Rectangle((left, bottom), width, height, fill=False, color="red", lw=1)
        # rect.set_transform(ax.transAxes)
        # rect.set_clip_on(False)
        ax.add_patch(self.rect)
        drs = []
        dr = DraggableRectangle(self.rect)
        dr.connect()
        drs.append(dr)

        plt.show()
        return [int(value) for value in dr.rect.xy]

    def print_keyboard_shortcuts(self):
        print("GUI keyboard shortcuts")
        print(f"r -> display a random frame")
        print(f"e -> Last frame displayed")
        print(f"m -> Display the frame just before in time")
        print(f"p -> Display the frame just after in time")
        print(f"b -> In/out binary mode")
        print(f"c -> Change binary thresholds")
        print(f"left -> Reduce width of the rectangle")
        print(f"right -> Increase width of the rectangle")
        print(f"down -> Reduce height of the rectangle")
        print(f"up -> Increase height of the rectangle")
        print(" ")

    def press(self, event):
        # print('press', event.key)
        sys.stdout.flush()
        if event.key == 'r':
            self.last_frames.append(self.current_frame)
            self.display_new_frame(self.pick_random_frame())
        elif event.key == 'p':
            # like plus one
            self.last_frames.append(self.current_frame)
            self.display_new_frame(self.current_frame+1)
        elif event.key == 'm':
            # like minus one
            self.last_frames.append(self.current_frame)
            self.display_new_frame(self.current_frame-1)
        elif event.key == 'e':
            # last img displayed
            if len(self.last_frames) > 0:
                last_frame = self.last_frames[-1]
                self.last_frames = self.last_frames[:-1]
                self.display_new_frame(last_frame)
        elif event.key == 'b':
            self.binary_mode = not self.binary_mode
            self.display_new_frame(self.current_frame)
        elif event.key == "c":
            self.change_binary_thresholds()
        elif event.key == 'left':
            self.change_width_rect(pixels_diff=self.step_rect_dim * -1)
        elif event.key == 'right':
            self.change_width_rect(pixels_diff=self.step_rect_dim)
        elif event.key == 'up':
            self.change_height_rect(pixels_diff=self.step_rect_dim)
        elif event.key == 'down':
            self.change_height_rect(pixels_diff=self.step_rect_dim * -1)

    def change_binary_thresholds(self):
        print(" ")
        print(f"Actual binary thresholds: {self.binary_thresholds}")
        input_tresholds = input("Binary thresholds:")
        split_values = input_tresholds.split(",")
        new_thresholds = []
        for split_value in split_values:
            try:
                thr = int(split_value)
                if thr < 1 or thr > 254:
                    print(f"{thr} out of bounds 1-254")
                    return
                if (len(new_thresholds) > 0) and (thr <= new_thresholds[-1]):
                    print(f"{split_values}: should be in ascending order")
                    return
                new_thresholds.append(thr)
            except ValueError:
                print(f"{split_value} is not an integer")
                return

        if len(new_thresholds) == 0 or len(new_thresholds) > 2:
            print(f"{new_thresholds} len should be 1 or 2")
            return

        self.binary_thresholds = new_thresholds
        print(f"New binary thresholds: {self.binary_thresholds}")
        if self.binary_mode:
            # we update the image
            self.display_new_frame(frame = self.current_frame)

    def change_width_rect(self, pixels_diff):
        width_rect = self.rect.get_width()
        self.size_rect[0] = max(20, self.size_rect[0] + pixels_diff)
        self.rect.set_width(w=max(20, width_rect + pixels_diff))
        self.fig.canvas.draw()

    def change_height_rect(self, pixels_diff):
        height_rect = self.rect.get_height()
        self.size_rect[1] = max(20, self.size_rect[1] + pixels_diff)
        self.rect.set_height(h=max(20, height_rect + pixels_diff))
        self.fig.canvas.draw()

    def display_new_frame(self, frame):
        self.current_frame = frame
        img_frame = self.video_reader.get_frame(frame)
        if self.binary_mode:
            img_frame = binarize_frame(img_frame,
                                       binary_thresholds=self.binary_thresholds)
        self.imshow_obj.set_data(np.flipud(img_frame))
        self.fig.canvas.draw()

    def pick_random_frame(self):
        return randrange(self.n_frames)

    def get_size_rect(self):
        return self.size_rect

    def get_binary_thresholds(self):
        return self.binary_thresholds
#
# def plot_img_with_rect(img, title="", size_rect=(1600, 1000)):
#     """
#     Build a plot with a image a rectangle of size_rect that can be moved over the image.
#     It returns the bottom_left coin coordinates
#     Args:
#         img:
#         title:
#         size_rect:
#
#     Returns:
#
#     """
#     fig = plt.figure()
#     plt.title(title)
#     ax = fig.add_subplot(111)
#     ax.set_ylim(img.shape[0])
#     ax.set_xlim(img.shape[1])
#     # ax.imshow(np.fliplr(img), origin='lower')
#     ax.imshow(np.flipud(img))
#     # ax.imshow(img)
#     plt.gca().invert_xaxis()
#     plt.gca().invert_yaxis()
#     # rects = ax.bar(range(10), 20*np.random.rand(10))
#     left = 10
#     bottom = 10
#     width = size_rect[0]
#     height = size_rect[1]
#     rect = plt.Rectangle((left, bottom), width, height, fill=False, color="red", lw=1)
#     # rect.set_transform(ax.transAxes)
#     # rect.set_clip_on(False)
#     ax.add_patch(rect)
#     drs = []
#     dr = DraggableRectangle(rect)
#     dr.connect()
#     drs.append(dr)
#
#     plt.show()
#     return [int(value) for value in dr.rect.xy]

def norm01(data):
    min_value = np.min(data)
    max_value = np.max(data)

    difference = max_value - min_value

    data -= min_value

    if difference > 0:
        data = data / difference

    return data

class VideoReaderWrapper:
    """
        An abstract class that should be inherited in order to create a specific video format wrapper.
        A class can be created using either different packages or aim at specific format.

    """

    def __init__(self):
        self._length = None
        self._width = None
        self._height = None
        self._fps = None

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def fps(self):
        return self._fps

    @abstractmethod
    def close_reader(self):
        pass

class OpenCvVideoReader(VideoReaderWrapper):
    """
    Use OpenCv to read video
    """

    def __init__(self, video_file_name):
        VideoReaderWrapper.__init__(self)

        self.video_file_name = video_file_name
        self.basename_video_file_name = os.path.basename(video_file_name)

        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        self.video_capture = VideoCapture(self.video_file_name)

        if self.video_capture.isOpened() == False:
            raise Exception(f"Error opening video file {self.video_file_name}")

        # length in frames
        self._length = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # in pixels
        self._width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # frame per seconds
        self._fps = self.video_capture.get(cv2.CAP_PROP_FPS)

        print(f"OpenCvVideoReader init for {self.basename_video_file_name}: "
              f"self.width {self.width}, self.height {self.height}, n frames {self._length}")

    def get_frame(self, frame_index):
        if (frame_index >= self._length) or (frame_index < 0):
            return None

        # The first argument of cap.set(), number 2 defines that parameter for setting the frame selection.
        # Number 2 defines flag CAP_PROP_POS_FRAMES which is
        # a 0-based index of the frame to be decoded/captured next.
        # The second argument defines the frame number in range 0.0-1.0
        # frame_no = frame_index / self._length
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read the next frame from the video. If you set frame 749 above then the code will return the last frame.
        # 'res' is boolean result of operation, one may use it to check if frame was successfully read.
        res, frame = self.video_capture.read()

        if res:
            return frame
        else:
            return None

    def close_reader(self):
        # When everything done, release the capture
        self.video_capture.release()
        cv2.destroyAllWindows()


def find_nearest(array, value, is_sorted=True):
    """
    Return the index of the nearest content in array of value.
    from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    return -1 or len(array) if the value is out of range for sorted array
    Args:
        array:
        value:
        is_sorted:

    Returns:

    """
    if len(array) == 0:
        return -1

    if is_sorted:
        if value < array[0]:
            return -1
        elif value > array[-1]:
            return len(array)
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
            return idx - 1
        else:
            return idx
    else:
        array = np.asarray(array)
        idx = (np.abs(array - value)).idxmin()
        return idx


def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax


def get_intervals_names(nwb_file):
    """
    Return a list representing the intervals contains in this data
    Returns:

    """
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()

    if nwb_data.intervals is None:
        io.close()
        return []

    intervals = []
    for name_interval in nwb_data.intervals.keys():
        intervals.append(name_interval)

    io.close()
    return intervals


def get_interval_times(nwb_file, interval_name):
    """
    Return an interval times (start and stop in seconds) as a numpy array of 2*n_times.
    Args:
        interval_name: Name of the interval to retrieve

    Returns: None if the interval doesn't exists or a 2d array

    """
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()

    if interval_name not in nwb_data.intervals:
        io.close()
        return None

    df = nwb_data.intervals[interval_name].to_dataframe()

    # TODO: See to make it more modulable in case someone will use another name
    if ("start_time" not in df) or \
            ("stop_time" not in df):
        return None

    # time series
    start_time_ts = df["start_time"]
    stop_time_ts = df["stop_time"]

    # it shouldn't be the case
    if len(start_time_ts) != len(stop_time_ts):
        print(f"len(start_time_ts) {len(start_time_ts)} != {len(stop_time_ts)} len(stop_time_ts)")
        return None

    data = np.zeros((2, len(start_time_ts)))
    data[0] = np.array(start_time_ts)
    data[1] = np.array(stop_time_ts)

    io.close()

    return data

def get_ci_movie_time_stamps(nwb_file):
    """
    return a np.array with the timestamps of each frame of the CI movie
    return None if non available
    Returns:

    """
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()

    if "ci_frames" not in nwb_data.acquisition:
        return None
    ci_frames_timestamps = np.array(nwb_data.acquisition["ci_frames"].timestamps)

    io.close()

    return ci_frames_timestamps

def get_behaviors_movie_time_stamps(nwb_file, cam_id):
    """

    Args:
        nwb_file:
        cam_id: '22983298' or '23109588'

    Returns:

    """
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()

    for name, acquisition_data in nwb_data.acquisition.items():
        if name.startswith(f"cam_{cam_id}"):
            return np.array(acquisition_data.timestamps)

    io.close()

    return None

def find_files_with_ext(dir_to_explore, extension):
    subfiles = []
    for (dirpath, dirnames, filenames) in os.walk(dir_to_explore):
        subfiles = [os.path.join(dirpath, filename) for filename in filenames]
        break
    return [f for f in subfiles if f.endswith(extension)]


def circle_points(resolution, center, radius):
    """
    Generate points which define a circle on an image.Centre refers to the centre of the circle
    """
    # from: https://towardsdatascience.com/image-segmentation-using-pythons-scikit-image-module-533a61ecc980
    radians = np.linspace(0, 2 * np.pi, resolution)
    c = center[1] + radius * np.cos(radians)  # polar co-ordinates
    r = center[0] + radius * np.sin(radians)

    return np.array([c, r]).T

def randomw_walker(img_test):
    # TOO SLOW
    print(f"img_test.shape {img_test.shape}")
    # into grey_scale
    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)

    # img_test = img_test.astype(np.uint8)
    img_test = ((img_test / img_test.max(axis=0).max(axis=0)) * 255).astype(np.uint8)

    # plt.imshow(img_test, 'gray')
    # plt.show()
    # return
    # The range of the binary image spans over (-1, 1).
    # We choose the hottest and the coldest pixels as markers.
    markers = np.zeros(img_test.shape, dtype=np.uint)
    # markers[data < -0.95] = 1
    # markers[data > 0.95] = 2

    # markers[data < -0.95] = 1
    # markers[500, 650] = 2

    # image_labels = np.zeros(image_gray.shape, dtype=np.uint8)

    # Exclude last point because a closed path should not have duplicate points
    points = circle_points(200, [500, 620], 300)[:-1]
    # points = circle_points(200, [80, 250], 80)[:-1]
    indices = draw.circle_perimeter(500, 620, 35)  # from here
    markers[indices] = 1
    markers[points[:, 1].astype(np.int), points[:, 0].astype(np.int)] = 2

    # Run random walker algorithm
    # labels = random_walker(img_test, markers, beta=10, mode='bf')
    image_segmented = random_walker(img_test, markers)

    fig, ax = image_show(img_test)
    ax.imshow(image_segmented == 1, alpha=0.3)
    plt.show()

    # Plot results
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
    #                                     sharex=True, sharey=True)
    # ax1.imshow(img_test, cmap='gray')
    # ax1.axis('off')
    # ax1.set_title('Noisy data')
    # ax2.imshow(markers, cmap='magma')
    # ax2.axis('off')
    # ax2.set_title('Markers')
    # ax3.imshow(labels, cmap='gray')
    # ax3.axis('off')
    # ax3.set_title('Segmentation')
    #
    # fig.tight_layout()
    # plt.show()
    return


def binarize_frame(img_frame, binary_thresholds):
    """

    Args:
        img_frame:
        binary_thresholds: list of 1 to 2 int, between 0 and 255, in ascending order, representing the different
            level of gray

    Returns:

    """
    # into grey_scale
    img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)

    # plt.imshow(img_frame, 'gray')
    # plt.show()

    # img_frame = img_frame.astype(np.uint8)
    img_frame = ((img_frame / img_frame.max(axis=0).max(axis=0)) * 255).astype(np.uint8)

    # bluring the image a bit
    img_frame = cv2.medianBlur(img_frame, 5)

    # Applying Histogram Equalization
    # img_frame = cv2.equalizeHist(img_frame)

    if len(binary_thresholds) == 1:
        # used to be 75
        ret, th1 = cv2.threshold(img_frame, binary_thresholds[0], 255, cv2.THRESH_BINARY)
    else:
        if len(binary_thresholds) != 2:
            raise Exception("Only 1 or 2 binary_thresholds are supported yet")
        # used to be [65, 150]
        th1 = np.zeros_like(img_frame)
        th1[img_frame < binary_thresholds[0]] = 0
        th1[np.logical_and(img_frame >= binary_thresholds[0], img_frame < binary_thresholds[1])] = 127
        th1[img_frame >= binary_thresholds[1]] = 255
    # print(f"np.unique(th1) {np.unique(th1)}")
    # Otsu's thresholding
    # ret2, th1 = cv2.threshold(img_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    th1 = np.array(th1).astype(float)
    # th3 = cv2.adaptiveThreshold(img_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                             cv2.THRESH_BINARY, 11, 2)

    """
                th2 = cv2.adaptiveThreshold(img_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
                th3 = cv2.adaptiveThreshold(img_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

                titles = ['Original Image', 'Global Thresholding (v = 127)',
                          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
                images = [img_frame, th1, th2, th3]

                for i in range(4):
                    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
                    plt.title(titles[i])
                    plt.xticks([]), plt.yticks([])
                plt.show()
    """

    return th1

def ploting_power_spectrum(filtered_signal, raw_signal, title):
    sampling_rate = 20
    fourier_transform = np.fft.rfft(filtered_signal)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, sampling_rate / 2, len(power_spectrum))

    fourier_transform = np.fft.rfft(raw_signal)
    abs_fourier_transform = np.abs(fourier_transform)
    original_power_spectrum = np.square(abs_fourier_transform)
    original_frequency = np.linspace(0, sampling_rate / 2, len(original_power_spectrum))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
    ax1.plot(filtered_signal, c="blue", zorder=2)
    ax1.plot(raw_signal, c="red", zorder=1)
    ax2.plot(frequency, power_spectrum, c="blue", zorder=2)
    ax2.plot(original_frequency, original_power_spectrum, c="red", zorder=1)
    plt.title(title)
    plt.show()

def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = signal.butter(order, [low, high], btype='bandstop')
    # y = signal.lfilter(i, u, data)
    # TODO: see if there is also a shift here
    # to correct shift (https://stackoverflow.com/questions/45098384/butterworth-filter-x-shift)
    y = signal.filtfilt(i, u, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    # produce a shift in the signal
    # y = signal.lfilter(b, a, data)
    # to correct shift (https://stackoverflow.com/questions/45098384/butterworth-filter-x-shift)
    y = signal.filtfilt(b, a, data)
    return y

def main():
    root_path = "/media/julien/Not_today/hne_not_today/"
    # root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    data_path = os.path.join(root_path, "data/tada_data/for_training/")

    results_path = os.path.join(root_path, "data/tada_data/results_tada/")
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    #
    results_path = os.path.join(results_path, time_str)
    os.mkdir(results_path)

    data_id = "p5_191127_191202_191202_a001"

    data_path = os.path.join(data_path, data_id)

    npz_data_path = os.path.join(data_path, "behaviors_timestamps")

    nwb_file = find_files_with_ext(data_path, "nwb")[0]

    npz_file = find_files_with_ext(npz_data_path, "npz")[0]

    npz_base_name = os.path.basename(npz_file)
    new_npz_base_name = npz_base_name[:-4] + "_in_frames" + ".npz"
    new_npz_file = os.path.join(data_path, new_npz_base_name)

    avi_files = find_files_with_ext(data_path, "avi")

    for avi_file in avi_files:
        if "22983298" in avi_file:
            right_movie_file = avi_file
        else:
            left_movie_file = avi_file

    # 672
    right_behavior_time_stamps = get_behaviors_movie_time_stamps(nwb_file, cam_id="22983298")

    left_behavior_time_stamps = get_behaviors_movie_time_stamps(nwb_file, cam_id="23109588")

    ci_movie_time_stamps = get_ci_movie_time_stamps(nwb_file)

    intervals_names = get_intervals_names(nwb_file)
    if "ci_recording_on_pause" not in intervals_names:
        raise Exception("ci_recording_on_pause epochs not in the NWB")

    pause_epochs = get_interval_times(nwb_file, "ci_recording_on_pause")
    print(f"pause_epochs {pause_epochs.shape}")

    n_frames_dict = dict()
    movie_dict = dict()
    behavior_time_stamps_dict = dict()

    right_movie = OpenCvVideoReader(right_movie_file)
    n_frames_right = right_movie.length

    left_movie = OpenCvVideoReader(left_movie_file)
    n_frames_left = left_movie.length

    # for looping over both sides
    n_frames_dict["right"] = n_frames_right
    movie_dict["right"] = right_movie
    behavior_time_stamps_dict["right"] = right_behavior_time_stamps

    n_frames_dict["left"] = n_frames_left
    movie_dict["left"] = left_movie
    behavior_time_stamps_dict["left"] = left_behavior_time_stamps

    if len(left_behavior_time_stamps) != n_frames_left:
        print(f"len(left_behavior_time_stamps) {len(left_behavior_time_stamps)} vs {n_frames_left} n_frames")

    if len(right_behavior_time_stamps) != n_frames_right:
        print(f"len(right_behavior_time_stamps) {len(right_behavior_time_stamps)} vs {n_frames_right} n_frames")

    first_ci_frame_dict = dict()
    last_ci_frame_dict = dict()
    # indicate when the laser if off (epochs in frames), 2d array (2, n_epochs): first_frame, last_frame (included)
    pause_epochs_dict = dict()
    pause_epochs_dict["right"] = np.zeros((2, pause_epochs.shape[1]+2), dtype="int64")
    pause_epochs_dict["left"] = np.zeros((2, pause_epochs.shape[1]+2), dtype="int64")

    # finding which frame from the behavior matches the CI activation (for z-score)
    first_ci_frame_right = int(find_nearest(right_behavior_time_stamps, ci_movie_time_stamps[0]))
    first_ci_frame_dict["right"] = first_ci_frame_right
    last_ci_frame_right = int(find_nearest(right_behavior_time_stamps, ci_movie_time_stamps[-1]))
    last_ci_frame_dict["right"] = last_ci_frame_right

    pause_epochs_dict["right"][0, 0] = 0
    pause_epochs_dict["right"][1, 0] = first_ci_frame_right - 1
    for i in range(pause_epochs.shape[1]):
        pause_epochs_dict["right"][0, i+1] = int(find_nearest(right_behavior_time_stamps, pause_epochs[0, i]))
        pause_epochs_dict["right"][1, i+1] = int(find_nearest(right_behavior_time_stamps, pause_epochs[1, i]))
    pause_epochs_dict["right"][0, pause_epochs.shape[1]] = last_ci_frame_right
    pause_epochs_dict["right"][1, 0] = len(right_behavior_time_stamps) - 1

    # print(f"first_ci_frame_right {first_ci_frame_right}")
    first_ci_frame_left = int(find_nearest(left_behavior_time_stamps, ci_movie_time_stamps[0]))
    first_ci_frame_dict["left"] = first_ci_frame_left
    last_ci_frame_left = int(find_nearest(left_behavior_time_stamps, ci_movie_time_stamps[-1]))
    last_ci_frame_dict["left"] = last_ci_frame_left

    pause_epochs_dict["left"][0, 0] = 0
    pause_epochs_dict["left"][1, 0] = first_ci_frame_left - 1
    for i in range(pause_epochs.shape[1]):
        pause_epochs_dict["left"][0, i + 1] = int(find_nearest(left_behavior_time_stamps, pause_epochs[0, i]))
        pause_epochs_dict["left"][1, i + 1] = int(find_nearest(left_behavior_time_stamps, pause_epochs[1, i]))
    pause_epochs_dict["left"][0, pause_epochs.shape[1]] = last_ci_frame_left
    pause_epochs_dict["left"][1, 0] = len(left_behavior_time_stamps) - 1

    # if left_behavior_time_stamps[0] > right_behavior_time_stamps[0]:
    #     first_time_stamp = left_behavior_time_stamps[0]
    # else:
    #     first_time_stamp = right_behavior_time_stamps[0]
    #
    # if left_behavior_time_stamps[-1] < right_behavior_time_stamps[-1]:
    #     last_time_stamp = left_behavior_time_stamps[-1]
    # else:
    #     last_time_stamp = right_behavior_time_stamps[-1]
    #
    #     # now we cut time_stamps and frames, so both side are synchronized
    #     # first left side
    # if left_behavior_time_stamps[0] != first_time_stamp:
    #     first_frame_left = int(find_nearest(left_behavior_time_stamps, first_time_stamp))
    #     first_frame_right = 0
    # else:
    #     first_frame_right = int(find_nearest(right_behavior_time_stamps, first_time_stamp))
    #     first_frame_left = 0
    #
    # if left_behavior_time_stamps[-1] != last_time_stamp:
    #     last_frame_left = int(find_nearest(left_behavior_time_stamps, last_time_stamp))
    #     last_frame_left = min(last_frame_left, n_frames_left - 1)
    #     last_frame_right = n_frames_right - 1
    # else:
    #     last_frame_left = n_frames_left - 1
    #     last_frame_right = int(find_nearest(right_behavior_time_stamps, last_time_stamp))
    #     last_frame_right = min(last_frame_right, n_frames_right - 1)
    #
    #     # list containing as many time_stamps as the movie after removing frames before and after first_frame and last_frame
    # new_time_stamps = left_behavior_time_stamps[first_frame_left:last_frame_left + 1]
    # new_time_stamps_right = right_behavior_time_stamps[first_frame_right:last_frame_right + 1]
    # print(f"First left timestamps {first_frame_left}")
    # print(f"Last left timestamps {last_frame_left}")
    # print(f"First right timestamps {first_frame_right}")
    # print(f"Last right timestamps {last_frame_right}")
    # print(f"len(new_time_stamps) {len(new_time_stamps)}, len(new_time_stamps_right) {len(new_time_stamps_right)}")
    #
    #"left"
    # width, height

    bodyparts = ["hindlimb", "forelimb",  "tail"]
    # bodyparts = ["forelimb"]
    size_rect_legs = [350, 250]
    size_rect_tail = [400, 250]
    default_binary_thresholds = [65, 150]
    # kzy is a string f"{side}_{bodypart}"
    bodyparts_config_yaml_file = os.path.join(data_path, "gui_config", f'{data_id}_bodyparts_config_mvt_gui.yaml')
    if os.path.isfile(bodyparts_config_yaml_file):
        print(f"Loading gui config {data_id}_bodyparts_config_mvt_gui.yaml")
        with open(bodyparts_config_yaml_file, 'r') as stream:
            bodyparts_config = yaml.load(stream, Loader=yaml.FullLoader)
    else:
        bodyparts_config = dict()

    actual_binary_thresholds = default_binary_thresholds
    for bodypart in bodyparts:
        for side in ["right", "left"]:
            # to make a copy and avoid problem when dumping config in the yaml
            actual_binary_thresholds = list(actual_binary_thresholds)
            img_frame = movie_dict[side].get_frame(0)

            if f"{side}_{bodypart}" in bodyparts_config:
                x_bottom_left, y_bottom_left, size_rect, \
                actual_binary_thresholds = bodyparts_config[f"{side}_{bodypart}"]
                y_bottom_left = img_frame.shape[0] - y_bottom_left
            else:
                if bodypart == "tail":
                    size_rect = size_rect_tail
                else:
                    size_rect = size_rect_legs
                x_bottom_left, y_bottom_left = (50, 10)
            image_area_selection = ImageAreaSelection(video_reader=movie_dict[side],
                                                      initial_frame=0, n_frames=n_frames_dict[side],
                                                      size_rect=size_rect, title=f"{side} {bodypart}",
                                                      x_bottom_left=x_bottom_left,
                                                      y_bottom_left=y_bottom_left,
                                                      binary_thresholds=actual_binary_thresholds)
            image_area_selection.print_keyboard_shortcuts()
            x_bottom_left, y_bottom_left = image_area_selection.select_area()
            # x_bottom_left, y_bottom_left = plot_img_with_rect(img_frame, title="", size_rect=size_rect)
            y_bottom_left = img_frame.shape[0] - y_bottom_left
            size_rect = image_area_selection.get_size_rect()
            # print(f"size_rect {size_rect}")
            binary_thresholds = image_area_selection.get_binary_thresholds()
            bodyparts_config[f"{side}_{bodypart}"] = [x_bottom_left, y_bottom_left, size_rect, binary_thresholds]
            actual_binary_thresholds = binary_thresholds

    for side in ["right", "left"]:
        # ----------------------------------------------------
        # ----------------------------------------------------
        # ----------------------------------------------------
        frames = np.arange(1000)

        # ----------------------------------------------------
        # ----------------------------------------------------
        # ----------------------------------------------------
        # frames = np.arange(n_frames_dict[side])
        for bodypart in bodyparts:
            # diff of pixels between 2 consecutive binary frames
            mvt = np.zeros(n_frames_dict[side], dtype="float")
            last_frame = None
            bodypart_config = bodyparts_config[f"{side}_{bodypart}"]
            print(f"bodypart_config {side}_{bodypart} : {bodypart_config}")
            x_bottom_left, y_bottom_left, size_rect, binary_thresholds = bodypart_config

            for frame in frames:
                if frame % 1000 == 0:
                    print(f"Processing frame {frame}")
                img_frame = movie_dict[side].get_frame(frame)
                # plt.imshow(img_frame, 'gray')
                # plt.show()
                img_frame = img_frame[y_bottom_left - size_rect[1]:y_bottom_left,
                            x_bottom_left:x_bottom_left + size_rect[0]]
                # start_time = time.time()
                img_frame = binarize_frame(img_frame, binary_thresholds)
                # stop_time = time.time()
                # print(f"Time to binarize one frame: "
                #       f"{np.round(stop_time - start_time, 5)} s")
                # plt.imshow(img_frame, 'gray')
                # plt.show()

                # diff between two frames, to build the movement 1d array
                # TODO: put to zero if transition with laser on/off
                if last_frame is not None:
                    mvt[frame] = np.sum(np.abs(np.subtract(last_frame, img_frame)))
                    # print(f"{np.sum(np.abs(np.subtract(last_frame, img_frame)))} / {np.sum(last_frame == img_frame)}")
                    # mvt[frame] = np.sum(last_frame == img_frame)
                last_frame = img_frame

            # mvt = np.sum(diff_movie, axis=1)
            # mvt = norm01(mvt)

            # normalization of the signal using z-score
            #last_ci_frame_dict
            mvt[0] = np.mean(mvt[1:first_ci_frame_dict[side]])
            mvt[:first_ci_frame_dict[side]] = stats.zscore(mvt[:first_ci_frame_dict[side]])
            mvt[first_ci_frame_dict[side]:frames[-1]+1] = stats.zscore(mvt[first_ci_frame_dict[side]:frames[-1]+1])

            original_mvt = np.copy(mvt)
            use_low_band_pass_filter = True
            if use_low_band_pass_filter:
                # remove frequency higher than 2 Hz
                mvt = butter_lowpass_filter(data=mvt, cutoff=2, fs=20, order=10)
            else:
                # bandstop filter
                # issue with bandstop, session without laser have a ten-fold lower amplitude
                mvt = butter_bandstop_filter(data=mvt, lowcut=2.8, highcut=3.5, fs=20, order=6)

            # second normalization after filtering
            mvt[:first_ci_frame_dict[side]] = stats.zscore(mvt[:first_ci_frame_dict[side]])
            mvt[first_ci_frame_dict[side]:frames[-1]+1] = stats.zscore(mvt[first_ci_frame_dict[side]:frames[-1]+1])

            # mvt = butter_bandstop_filter(data=mvt, lowcut=5.3, highcut=5.5, fs=20, order=6)
            # mvt = butter_bandstop_filter(data=mvt, lowcut=8, highcut=8.6, fs=20, order=6)
            # noise at 3.05hz
            ploting_power_spectrum(filtered_signal=mvt, raw_signal=original_mvt, title=f"{side}_{bodypart}")

            arg_dict = {f"{side}_{bodypart}": mvt,
                        "timestamps": behavior_time_stamps_dict[side]}
            np.savez(os.path.join(results_path, f"{data_id}_{side}_{bodypart}_mvt.npz"), **arg_dict)

            # to plot movement
            # plt.plot(mvt)
            # plt.title(f"{side}_{bodypart}")
            # plt.show()

    with open(os.path.join(results_path, f'{data_id}_bodyparts_config_mvt_gui.yaml'), 'w') as outfile:
        yaml.dump(bodyparts_config, outfile, default_flow_style=False)

if __name__ == "__main__":
    main()
