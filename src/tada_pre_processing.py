"""
This idea is to prepare the data so they can be processed by deeptada
It takes a directory that should contain the 2 behavioral movies and the nwb corresponding.
It will create a .yaml file that will allows to know which frames to keep so both movie frames are synchronized and
the coordinates of the cropping (using a mini-GUI).
"""

from pynwb import NWBHDF5IO

import numpy as np
# import time

from abc import ABC, abstractmethod
import os

from cv2 import VideoCapture
import cv2

import yaml
import math

import numpy as np
import matplotlib.pyplot as plt

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

def plot_img_with_rect(img, title="", size_rect=(1600, 1000)):
    """
    Build a plot with a image a rectangle of size_rect that can be moved over the image.
    It returns the bottom_left coin coordinates
    Args:
        img:
        title:
        size_rect:

    Returns:

    """
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(111)
    ax.set_ylim(img.shape[0])
    ax.set_xlim(img.shape[1])
    # ax.imshow(np.fliplr(img), origin='lower')
    ax.imshow(np.flipud(img))
    # ax.imshow(img)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    # rects = ax.bar(range(10), 20*np.random.rand(10))
    left = 10
    bottom = 10
    width = size_rect[0]
    height = size_rect[1]
    rect = plt.Rectangle((left, bottom), width, height, fill=False, color="red", lw=1)
    # rect.set_transform(ax.transAxes)
    # rect.set_clip_on(False)
    ax.add_patch(rect)
    drs = []
    dr = DraggableRectangle(rect)
    dr.connect()
    drs.append(dr)

    plt.show()
    return [int(value) for value in dr.rect.xy]

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

    return None


def find_files_with_ext(dir_to_explore, extension):
    subfiles = []
    for (dirpath, dirnames, filenames) in os.walk(dir_to_explore):
        subfiles = [os.path.join(dirpath, filename) for filename in filenames]
        break
    return [f for f in subfiles if f.endswith(extension)]


def convert_npz_cicada_in_frames(npz_file, time_stamps, new_npz_file):
    """
    Convert epochs in npz_file in frames, according to new_time_stamps array.
    Epochs that are starting or finishing after the time limits in new_time_stamps will be removed
    The new version is registered in new_npz_file
    Args:
        npz_file:
        time_stamps:
        new_npz_file:

    Returns:

    """
    npz_content = np.load(npz_file)

    new_content_dict = dict()

    for tag_name, epochs in npz_content.items():
        print(f"- {tag_name}")
        tmp_epochs = []
        for epoch_index in range(epochs.shape[1]):
            # first we make sure we don't go over the limits
            if epochs[0, epoch_index] < time_stamps[0]:
                continue
            if epochs[1, epoch_index] > time_stamps[-1]:
                continue
            first_frame = find_nearest(time_stamps, epochs[0, epoch_index])
            last_frame = find_nearest(time_stamps, epochs[1, epoch_index])
            tmp_epochs.append((first_frame, last_frame))
        new_content_dict[tag_name] = np.zeros((2, len(tmp_epochs)), dtype="int64")
        for epoch_index, epoch in enumerate(tmp_epochs):
            new_content_dict[tag_name][0, epoch_index] = epoch[0]
            new_content_dict[tag_name][1, epoch_index] = epoch[1]

    np.savez(new_npz_file, **new_content_dict)

def main():

    root_path = "/media/julien/Not_today/hne_not_today/"
    # root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    data_path = os.path.join(root_path, "data/tada_data/for_training/")

    data_id = "p5_191205_191210_0_191210_a001"

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

    right_movie = OpenCvVideoReader(right_movie_file)
    n_frames_right = right_movie.length

    left_movie = OpenCvVideoReader(left_movie_file)
    n_frames_left = left_movie.length

    right_behavior_time_stamps = get_behaviors_movie_time_stamps(nwb_file, cam_id="22983298")

    left_behavior_time_stamps = get_behaviors_movie_time_stamps(nwb_file, cam_id="23109588")

    if len(left_behavior_time_stamps) != n_frames_left:
        print(f"len(left_behavior_time_stamps) {len(left_behavior_time_stamps)} vs {n_frames_left} n_frames")

    if len(right_behavior_time_stamps) != n_frames_right:
        print(f"len(right_behavior_time_stamps) {len(right_behavior_time_stamps)} vs {n_frames_right} n_frames")

    if left_behavior_time_stamps[0] > right_behavior_time_stamps[0]:
        first_time_stamp = left_behavior_time_stamps[0]
    else:
        first_time_stamp = right_behavior_time_stamps[0]

    if left_behavior_time_stamps[-1] < right_behavior_time_stamps[-1]:
        last_time_stamp = left_behavior_time_stamps[-1]
    else:
        last_time_stamp = right_behavior_time_stamps[-1]

    # TODO: change the frame index according to movie number of frames

    # now we cut time_stamps and frames, so both side are synchronized
    # first left side
    if left_behavior_time_stamps[0] != first_time_stamp:
        first_frame_left = int(find_nearest(left_behavior_time_stamps, first_time_stamp))
        first_frame_right = 0
    else:
        first_frame_right = int(find_nearest(right_behavior_time_stamps, first_time_stamp))
        first_frame_left = 0

    if left_behavior_time_stamps[-1] != last_time_stamp:
        last_frame_left = int(find_nearest(left_behavior_time_stamps, last_time_stamp))
        last_frame_left = min(last_frame_left, n_frames_left-1)
        last_frame_right = n_frames_right-1
    else:
        last_frame_left = n_frames_left-1
        last_frame_right = int(find_nearest(right_behavior_time_stamps, last_time_stamp))
        last_frame_right = min(last_frame_right, n_frames_right-1)

    # list containing as many time_stamps as the movie after removing frames before and after first_frame and last_frame
    new_time_stamps = left_behavior_time_stamps[first_frame_left:last_frame_left+1]

    size_rect = (1600, 1000)
    x_left, y_left = plot_img_with_rect(left_movie.get_frame(1000), title="left_movie", size_rect=size_rect)
    print(f"x_left {x_left}, y_left {y_left}")
    x_right, y_right = plot_img_with_rect(right_movie.get_frame(1000), title="right_movie", size_rect=size_rect)
    print(f"x_right {x_right}, y_right {y_right}")

    new_n_frames_left = last_frame_left - first_frame_left + 1
    print(f"N frames in left: {new_n_frames_left}")
    new_n_frames_right = last_frame_right - first_frame_right + 1
    print(f"N frames in right: {new_n_frames_right}")

    yaml_data = dict()
    # the yaml file has an entry for each movie with 7 values so far, frames to keep, bottom left corner coordinates
    # for cropping, the size of the rect used to chose the cropping and the input_order
    # input_order allows to know in which order the inputs should be given to the model so that it is always the same
    yaml_data[os.path.basename(left_movie_file)] = [first_frame_left, last_frame_left, x_left, y_left,
                                                    size_rect[0], size_rect[1], 0]
    yaml_data[os.path.basename(right_movie_file)] = [first_frame_right, last_frame_right, x_right, y_right,
                                                    size_rect[0], size_rect[1], 1]

    with open(os.path.join(data_path, f'{data_id}_config.yaml'), 'w') as outfile:
        yaml.dump(yaml_data, outfile, default_flow_style=False)

    convert_npz_cicada_in_frames(npz_file, new_time_stamps, new_npz_file)

    # TODO: make config yaml file to indicate which behaviors tag to keep for training

if __name__ == "__main__":
    main()
