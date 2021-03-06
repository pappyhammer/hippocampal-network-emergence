import numpy as np
import os
from PIL import Image
import tifffile
from abc import ABC, abstractmethod
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import time
from ScanImageTiffReader import ScanImageTiffReader
import PIL
from PIL import ImageSequence
# matplotlib.use('TkAgg')
import matplotlib
# useful on mac to create movie from fig
matplotlib.use('agg')
import matplotlib.pyplot as plt

from matplotlib import patches

def loading_tiff_movie(tiff_file_name):
    # loading the movie
    try:
        start_time = time.time()
        tiff_movie = ScanImageTiffReader(tiff_file_name).data()
        stop_time = time.time()
        print(f"tiff_movie.dtype.name {tiff_movie.dtype.name}")
        raise Exception("titi")
        print(f"Time for loading movie {value['tiff_file']} with scan_image_tiff: "
              f"{np.round(stop_time - start_time, 3)} s")
        return tiff_movie
    except:
        start_time = time.time()
        im = PIL.Image.open(tiff_file_name)
        n_frames = len(list(ImageSequence.Iterator(im)))
        dim_x, dim_y = np.array(im).shape
        print(f"n_frames {n_frames}, dim_x {dim_x}, dim_y {dim_y}")
        tiff_movie = np.zeros((n_frames, dim_x, dim_y), dtype="uint16")
        for frame, page in enumerate(ImageSequence.Iterator(im)):
            tiff_movie[frame] = np.array(page)
        stop_time = time.time()
        print(f"Time for loading movie: "
              f"{np.round(stop_time - start_time, 3)} s")
        print(f"PIL tiff_movie.dtype.name {tiff_movie.dtype.name}")
        return tiff_movie

def normalize_array_0_255(img_array):
    minv = np.amin(img_array)
    # minv = 0
    maxv = np.amax(img_array)
    if maxv - minv == 0:
        print(f"maxv - min == 0")
        img_array = img_array.astype(np.uint8)
    else:
        img_array = (255 * (img_array - minv) / (maxv - minv)).astype(np.uint8)
    return img_array


def normalize_array_16_bits(img_array):
    minv = np.amin(img_array)
    # minv = 0
    maxv = np.amax(img_array)
    if maxv - minv == 0:
        img_array = img_array.astype(np.uint16)
    else:
        img_array = (((2**16)- 1) * (img_array - minv) / (maxv - minv)).astype(np.uint16)
    return img_array


def make_video(images, outvid=None, fps=5, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.

    @param      outvid      output video file_name
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        # if not os.path.exists(image):
        #     raise FileNotFoundError(image)
        # img = imread(image)
        img = image
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    # print(f"buf.shape {buf.shape}")
    # raise Exception("toto")
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())  # "RGBA"


# instances of this class will be use to fill the boxes in HNEAnimation
class AnimationBox(ABC):
    def __init__(self, description):
        self.width = None
        self.height = None
        self.boundaries = None
        # key: left, right, top, bottom
        self.neighbors = dict()
        self.description = description

    def add_neighbor(self, direction, box):
        if direction not in ["bottom", "top", "left", "right"]:
            print(f"unknown direction {direction}")
        self.neighbors[direction] = box

    def set_boundaries(self, top_left_coord):
        """

        :param top_left_coord: tuple(x, y)
        :return:
        """
        if self.boundaries is not None:
            return
        # list of 4 tuples, in order top left, top right, bottom left, bottom right
        self.boundaries = []
        self.boundaries.append(top_left_coord)
        top_right_coord = (top_left_coord[0]+self.width-1, top_left_coord[1])
        self.boundaries.append(top_right_coord)
        bottom_left_coord = (top_left_coord[0], top_left_coord[1] + self.height-1)
        self.boundaries.append(bottom_left_coord)
        bottom_right_coord = (bottom_left_coord[0]+self.width-1, bottom_left_coord[1])
        self.boundaries.append(bottom_right_coord)

        if "bottom" in self.neighbors:
            self.neighbors["bottom"].set_boundaries(top_left_coord=(bottom_left_coord[0], bottom_left_coord[1]+1))
        if "right" in self.neighbors:
            self.neighbors["right"].set_boundaries(top_left_coord=(top_right_coord[0] + 1, top_right_coord[1]))

    @abstractmethod
    def get_frame_img(self, frame):
        pass

    @abstractmethod
    def set_frames_range(self, frames):
        # define a range, that is serve for exemple to normalize over a range of frames
        pass


class EmptyBox(AnimationBox):
    def __init__(self, width, height):
        super().__init__(description="empty_box")
        self.width = width
        self.height = height

    def get_frame_img(self, frame):
        return np.zeros((self.width, self.height))

    def set_frames_range(self, frames):
        pass


class RawMovieBox(AnimationBox):
    def __init__(self, tiff_file_name, zoom_factor=1, cells_groups_to_color=None, colors_for_cells_groups=None,
                 coord_obj=None, cells_groups_alpha=None, raster_dur=None):
        super().__init__(description="raw movie")
        self.tiff_movie = loading_tiff_movie(tiff_file_name)
        self.zoom_factor=zoom_factor
        self.dpi = 100*self.zoom_factor
        self.width = self.tiff_movie.shape[2]*self.zoom_factor
        self.height = self.tiff_movie.shape[1]*self.zoom_factor
        self.n_frames = self.tiff_movie.shape[0]
        self.cells_groups_to_color = cells_groups_to_color
        self.colors_for_cells_groups = colors_for_cells_groups
        self.cells_groups_alpha = cells_groups_alpha
        self.coord_obj = coord_obj
        self.raster_dur = raster_dur

        self.do_color_cells = (self.cells_groups_to_color is not None) and \
                              (self.colors_for_cells_groups is not None) and (self.coord_obj is not None) \
                              and (self.raster_dur is not None)

    # def get_frame_img(self, frame):
    #     return self.tiff_movie[frame]

    def set_frames_range(self, frames):
        pass

    def get_frame_img(self, frame):
        # return self.tiff_movie[frame]
        fig_size_width = 10
        background_color = "black"
        fig_size_height = fig_size_width * (self.height / self.width)
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                gridspec_kw={'height_ratios': [1], 'width_ratios': [1]},
                                # figsize=plt.figaspect(self.tiff_movie[frame]))
                                # figsize=(self.width*dpi, self.height*dpi), dpi=dpi)
                                figsize=(fig_size_width, fig_size_height), dpi=self.dpi)
        # fig, ax = plt.subplots(figsize=plt.figaspect(a))
        # fig.subplots_adjust(0, 0, 1, 1)
        fig.patch.set_facecolor(background_color)
        ax1.imshow(self.tiff_movie[frame],
                                     cmap=plt.get_cmap('gray'))
        if self.do_color_cells:
            cells_alpha = 0.4
            line_width = 0
            edge_color = None
            z_order_cells = 10
            for group_index, cell_group in enumerate(self.cells_groups_to_color):
                for cell in cell_group:
                    if self.raster_dur[cell, frame] == 0:
                        continue
                    xy = self.coord_obj.coord[cell].transpose()
                    # if with_edge:
                    #     line_width = edge_line_width
                    #     if cells_groups_edge_colors is None:
                    #         edge_color = default_edge_color
                    #     else:
                    #         edge_color = cells_groups_edge_colors[group_index]
                    # else:
                    #     edge_color = cells_groups_colors[group_index]
                    #     line_width = 0
                    # allow to set alpha of the edge to 1
                    face_color = list(self.colors_for_cells_groups[group_index])
                    # changing alpha
                    if self.cells_groups_alpha is not None:
                        face_color[3] = self.cells_groups_alpha[group_index]
                    else:
                        face_color[3] = cells_alpha
                    face_color = tuple(face_color)
                    cell_contour = patches.Polygon(xy=xy,
                                                   fill=True, linewidth=0,
                                                   facecolor=face_color,
                                                   edgecolor=edge_color,
                                                   zorder=z_order_cells)  # lw=2
                    ax1.add_patch(cell_contour)

        # ax1.legend()
        axes_to_clean = [ax1]
        # plt.setp(ax1.spines.values(), color="black")
        for ax in axes_to_clean:
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # ax.spines['left'].set_color('black')
            ax.spines['left'].set_visible(False)
            ax.margins(0)
        fig.tight_layout()
        # plt.show()

        im = fig2img(fig)
        plt.close()
        # im = im.convert('L')
        # im = im.convert('RGB')
        # print(f"coord[0].shape {coord[0].shape}")

        # im.thumbnail((self.width, self.height), Image.ANTIALIAS)
        im = im.resize((self.width, self.height), Image.ANTIALIAS)
        # im.show()
        # raise Exception("trial")
        # im_array = np.asarray(im)
        return im

class FrameCountBox(AnimationBox):
    def __init__(self):
        super().__init__(description="frame count")

    def get_frame_img(self, frame):
        return None


class PlotBox(AnimationBox):
    def __init__(self, width, height, values_array, color_past_and_present="red", color_future="white",
                 show_v_line_at_active_frame=False, n_frames_to_display=100):
        """

        :param width:
        :param height:
        :param raster:
        :param show_sum_spikes_as_percentage:
        :param n_frames_to_display: number of frames to display, should an even number
        """
        super().__init__(description="activity sum")
        self.width = width
        self.height = height
        self.n_frames_to_display = n_frames_to_display
        self.color_past_and_present = color_past_and_present
        self.color_future = color_future
        self.show_v_line_at_active_frame = show_v_line_at_active_frame

        self.values_array = values_array
        self.n_times = len(self.values_array)
        self.min_value = np.min(self.values_array)
        self.max_value = np.max(self.values_array)
        # print(f"bounds : {(self.min_value, self.max_value)}")

    def set_frames_range(self, frames):
        self.min_value = np.min(self.values_array[frames])
        self.max_value = np.max(self.values_array[frames])

    def get_frame_img(self, frame):
        fig_size_width = 12
        background_color = "black"
        fig_size_height = fig_size_width * (self.height / self.width)
        fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex='col',
                                gridspec_kw={'height_ratios': [1], 'width_ratios': [1]},
                                figsize=(fig_size_width, fig_size_height))
        fig.patch.set_facecolor(background_color)
        ax1.set_facecolor(background_color)
        if (frame - (self.n_frames_to_display // 2)) >= 0:
            x_value = np.arange(0, (self.n_frames_to_display // 2)+1)
            sum_spikes = self.values_array[frame - (self.n_frames_to_display // 2):frame + 1]
        else:
            n_frames = np.abs((frame - (self.n_frames_to_display // 2)))
            x_value = np.arange(n_frames, (self.n_frames_to_display // 2)+1)
            sum_spikes = self.values_array[0:frame + 1]
        # print(f"frame {frame}")
        # print(f"x_value {x_value}")
        # print(f"sum_spikes {sum_spikes}")
        ax1.fill_between(x_value, 0, sum_spikes, facecolor=self.color_past_and_present)

        if frame < self.n_times - 1:
            if (frame + (self.n_frames_to_display // 2)) < self.n_times:
                x_value = np.arange((self.n_frames_to_display // 2), self.n_frames_to_display)
                sum_spikes = self.values_array[frame: frame + (self.n_frames_to_display // 2)]
            else:
                n_frames = self.n_times - frame
                x_value = np.arange((self.n_frames_to_display // 2), (self.n_frames_to_display // 2) + n_frames)
                sum_spikes = self.values_array[frame:]
            # print(f"----")
            # print(f"frame {frame}")
            # print(f"x_value {x_value} {len(x_value)}")
            # print(f"sum_spikes {sum_spikes} {len(sum_spikes)}")
            ax1.fill_between(x_value, 0, sum_spikes, facecolor=self.color_future)

        ax1.set_ylim(self.min_value, self.max_value)
        ax1.set_xlim(0, self.n_frames_to_display-1)

        # ax1.legend()
        axes_to_clean = [ax1]
        plt.setp(ax1.spines.values(), color="black")
        for ax in axes_to_clean:
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # ax.spines['left'].set_color('black')
            ax.spines['left'].set_visible(False)
            ax.margins(0)
        fig.tight_layout()
        # plt.show()

        im = fig2img(fig)
        plt.close()
        # im = im.convert('L')
        # im = im.convert('RGB')
        # print(f"coord[0].shape {coord[0].shape}")

        # im.thumbnail((self.width, self.height), Image.ANTIALIAS)
        im = im.resize((self.width, self.height), Image.ANTIALIAS)
        # im.show()
        # raise Exception("trial")
        # im_array = np.asarray(im)
        return im

class HNEAnimation:
    def __init__(self, n_frames, n_rows=1, n_cols=1):
        if n_rows < 1:
            raise Exception("n_rows must be at least 1")
        if n_cols < 1:
            raise Exception("n_cols must be at least 1")
        self.n_frames = n_frames
        self.n_rows = n_rows
        self.n_cols = n_cols
        # key is a tuple representing a box as it line and col
        self.boxes = dict()
        for row in np.arange(n_rows):
            for col in np.arange(n_cols):
                self.boxes[(row, col)] = None
        # width and height in pixels
        self.width = None
        self.height = None
        # dict with key being a tuple representing a box, and value is a tuple of 4 tuple of 2 int representing the
        # boundaries of the box
        self.boundaries = None

    def add_box(self, box, row, col):
        """

        :param box:
        :param row:
        :param col:
        :return:
        """
        if (row < 0) or (row >= self.n_rows):
            print(f"invalid row")
            return
        if(col < 0) or (col >= self.n_cols):
            print(f"invalid col")
            return

        self.boxes[(row, col)] = box

    def set_size(self):
        """
        Initialize the size of the HNEAnimation from the size of all the boxes that it contains
        It will fill non occupy space by empty_boxes
        It will set the boundaries for each box
        :return:
        """
        self.width = 0
        self.height = 0
        width_by_col = np.zeros(self.n_cols, dtype="uint16")
        height_by_row = np.zeros(self.n_rows, dtype="uint16")
        # determining the dimensions
        for row in np.arange(self.n_rows):
            for col in np.arange(self.n_cols):
                box = self.boxes[(row, col)]
                if box is not None:
                    if (width_by_col[col] != 0) and (box.width != width_by_col[col]):
                        print(f"box {box.description} at position {(row, col)} has a width that doesn't match"
                              f" the width of other boxes")
                    width_by_col[col] = box.width
                    if (height_by_row[row] != 0) and (box.height != height_by_row[row]):
                        print(f"box {box.description} at position {(row, col)} has a width that doesn't match"
                              f" the width of other boxes")
                    height_by_row[row] = box.height
        print(f"width_by_col {width_by_col}")
        print(f"height_by_row {height_by_row}")
        self.width = np.sum(width_by_col)
        self.height = np.sum(height_by_row)

        # then filling the empty space by empty boxes
        for row in np.arange(self.n_rows):
            for col in np.arange(self.n_cols):
                box = self.boxes[(row, col)]
                if box is None:
                    self.boxes[(row, col)] = EmptyBox(width=width_by_col[col], height=height_by_row[row])

        # then connecting the boxes
        for row in np.arange(self.n_rows):
            for col in np.arange(self.n_cols):
                box = self.boxes[(row, col)]
                if row < (self.n_rows - 1):
                    box.add_neighbor(direction="bottom", box=self.boxes[(row + 1, col)])
                if row > 0:
                    box.add_neighbor(direction="top", box=self.boxes[(row - 1, col)])
                if col < (self.n_cols - 1):
                    box.add_neighbor(direction="right", box=self.boxes[(row, col + 1)])
                if col > 0:
                    box.add_neighbor(direction="left", box=self.boxes[(row, col - 1)])

        # finally setting boundaries of the top left box, the other will be recursively updated
        self.boxes[(0, 0)].set_boundaries(top_left_coord=(0, 0))

    def get_frame_img(self, frame):
        if (self.height is None) or (self.height is None):
            self.set_size()
        img = np.zeros((self.height, self.width, 4))
        # print(f"img.shape {img.shape}")
        for row in np.arange(self.n_rows):
            for col in np.arange(self.n_cols):
                box = self.boxes[(row, col)]
                if box is not None:
                    img_box = box.get_frame_img(frame=frame)
                    img_box = np.asarray(img_box)
                    if len(img_box.shape) < 3:
                        # print(f"img_box.shape {img_box.shape}")
                        # stacked_img = np.stack((img_box,) * 4, axis=-1)
                        # img_box = stacked_img
                        img_box = Image.fromarray(obj=img_box)
                        img_box = img_box.convert("RGBA", dither=Image.FLOYDSTEINBERG, colors=2**16)
                        print(f"post PIL: img_box.dtype.name {np.asarray(img_box, dtype='uint16').dtype.name}")
                        # print(f"post PIL: img_box.shape {np.asarray(img_box).shape}")
                    x_top_left = box.boundaries[0][0]
                    y_top_left = box.boundaries[0][1]
                    # print(f"x_top_left {x_top_left}, y_top_left {y_top_left}")
                    # print(f"dim {(self.width, self.height)}")
                    #
                    # print(f"dim box {(box.width, box.height)}")
                    img[y_top_left:y_top_left+box.height, x_top_left:x_top_left+box.width] = img_box
        return img

    def produce_animation(self, path_results, file_name, save_formats, frames_to_display=None):
        # TODO: produce in the console a progression barline
        if self.width is None:
            self.set_size()
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        output_filenames = dict()
        right_format_found = False
        for format in save_formats:
            if format.lower() in ["avi", "tif", "tiff"]:
                if format.lower() == "avi":
                    output_filenames["avi"] = os.path.join(path_results, f"{file_name}.avi")
                    right_format_found = True
                else:
                    output_filenames["tiff"] = os.path.join(path_results, f"{file_name}.tiff")
                    right_format_found = True
        if not right_format_found:
            print(f"no correct format detected")
            return

        if frames_to_display is None:
            frames_to_display = np.arange(self.n_frames)
        else:
            if np.max(frames_to_display) >= self.n_frames:
                print(f"Max frame index is {self.n_frames - 1}")
                return
            if np.min(frames_to_display) < 0:
                print(f"min frame index is 0")
                return

        for row in np.arange(self.n_rows):
            for col in np.arange(self.n_cols):
                self.boxes[(row, col)].set_frames_range(frames_to_display)

        if 'avi' in output_filenames:
            fourcc = VideoWriter_fourcc(*"XVID")
        else:
            fourcc = None
        vid_avi = None
        size_avi = None
        fps_avi = 10
        is_color = True

        if 'tiff' in output_filenames:
            with tifffile.TiffWriter(output_filenames['tiff']) as tiff:
                for frame in frames_to_display:
                    img = self.get_frame_img(frame=frame)
                    # print(f"img.shape {img.shape}")
                    # normalized_img = normalize_array_16_bits(img)
                    normalized_img = img  # *256
                    # normalized_img = normalized_img.astype(dtype="uint8")
                    # print(f"normalized_img.shape {normalized_img.shape}")
                    # normalized_img = np.uint8(255 * img)
                    normalized_img = normalize_array_0_255(img)
                    tiff.save(normalized_img, compress=6) #, compress=6, photometric='RGB')
                    if 'avi' in output_filenames:
                        if vid_avi is None:
                            if size_avi is None:
                                size_avi = img.shape[1], img.shape[0]
                            vid_avi = VideoWriter(output_filenames['avi'], fourcc, float(fps_avi), size_avi, is_color)
                        if size_avi[0] != img.shape[1] and size_avi[1] != img.shape[0]:
                            img = resize(img, size_avi)
                        # normalized_img = np.uint8(255 * normalized_img)
                        vid_avi.write(normalized_img)
        elif 'avi' in output_filenames:
            for frame in frames_to_display:
                img = self.get_frame_img(frame=frame)
                if 'avi' in output_filenames:
                    if vid_avi is None:
                        if size_avi is None:
                            size_avi = img.shape[1], img.shape[0]
                        vid_avi = VideoWriter(output_filenames['avi'], fourcc, float(fps_avi), size_avi, is_color)
                    if size_avi[0] != img.shape[1] and size_avi[1] != img.shape[0]:
                        img = resize(img, size_avi)
                        vid_avi.write(img)
        if 'avi' in output_filenames:
            vid_avi.release()

