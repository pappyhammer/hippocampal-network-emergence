from shapely.geometry import MultiPoint, LineString
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from matplotlib import patches
from shapely import geometry
import PIL
from PIL import ImageDraw
import math
import scipy.stats as stats
import time
from ScanImageTiffReader import ScanImageTiffReader
import scipy.signal as signal
try:
    import Image
except ImportError:
    from PIL import Image
from PIL import ImageSequence
import os

from pattern_discovery.tools.signal import smooth_convolve

class CoordClass:
    def __init__(self, coord, nb_lines=None, nb_col=None, from_suite_2p=False):
        # contour coord
        self.coord = coord
        if nb_lines is None:
            self.nb_lines = 200
        else:
            self.nb_lines = nb_lines
        if nb_col is None:
            self.nb_col = 200
        else:
            self.nb_col = nb_col
        self.n_cells = len(self.coord)
        # dict of tuples, key is the cell #, cell center coord x and y (x and y are inverted for imgshow)
        self.center_coord = dict()
        self.img_filled = None
        self.img_contours = None
        # used in case some cells would be removed, to we can update the centers accordingly
        self.neurons_removed = []
        # compute_center_coord() will be called when a mouseSession will be created
        # self.compute_center_coord()
        self.cells_groups = None
        self.cells_groups_colors = None
        # shapely polygons
        self.cells_polygon = dict()
        # first key is an int representing the number of the cell, and value is a list of cells it interesects
        self.intersect_cells = dict()

        for cell, c in enumerate(self.coord):

            if not from_suite_2p:
                # it is necessary to remove one, as data comes from matlab, starting from 1 and not 0
                c = c - 1
            c = c.astype(int)
            self.coord[cell] = c

            if c.shape[0] == 0:
                print(f'Error: {cell} c.shape {c.shape}')
                continue

            c_filtered = c.astype(int)
            bw = np.zeros((self.nb_col, self.nb_lines), dtype="int8")
            # morphology.binary_fill_holes(input
            bw[c_filtered[0, :], c_filtered[1, :]] = 1

            n_coord = len(c_filtered[0, :])
            coord_list_tuple = []
            for n in np.arange(n_coord):
                coord_list_tuple.append((c_filtered[0, n], c_filtered[1, n]))

            # buffer(0) or convex_hull could be used if the coord are a list of points not
            # in the right order. However buffer(0) return a MultiPolygon with no coord available.
            if len(coord_list_tuple) < 3:
                list_points = []
                for coord in coord_list_tuple:
                    list_points.append(geometry.Point(coord))
                self.cells_polygon[cell] = geometry.LineString(list_points)
                # print(f"len(coord_list_tuple) {len(coord_list_tuple)}")
            else:
                self.cells_polygon[cell] = geometry.Polygon(coord_list_tuple)  # .convex_hull # buffer(0)
            # self.coord[cell] = np.array(self.cells_polygon[cell].exterior.coords).transpose()

            c_x, c_y = ndimage.center_of_mass(bw)
            self.center_coord[cell] = (c_x, c_y)

            # if (cell == 0) or (cell == 159):
            #     print(f"cell {cell} fig")
            #     fig, ax = plt.subplots(nrows=1, ncols=1,
            #                            gridspec_kw={'height_ratios': [1]},
            #                            figsize=(5, 5))
            #     ax.imshow(bw)
            #     plt.show()
            #     plt.close()

        for cell_1 in np.arange(self.n_cells-1):
            if cell_1 not in self.intersect_cells:
                if cell_1 not in self.cells_polygon:
                    continue
                self.intersect_cells[cell_1] = set()
            for cell_2 in np.arange(cell_1+1, self.n_cells):
                if cell_2 not in self.cells_polygon:
                    continue
                if cell_2 not in self.intersect_cells:
                    self.intersect_cells[cell_2] = set()
                poly_1 = self.cells_polygon[cell_1]
                poly_2 = self.cells_polygon[cell_2]
                # if it intersects and not only touches if adding and (not poly_1.touches(poly_2))
                # try:
                if poly_1.intersects(poly_2):
                    self.intersect_cells[cell_2].add(cell_1)
                    self.intersect_cells[cell_1].add(cell_2)
                # except shapely.errors.TopologicalError:
                #     print(f"cell_1 {cell_1}, cell_2 {cell_2}")
                #     print(f"cell_1 {poly_1.is_valid}, cell_2 {poly_2.is_valid}")
                #     poly_1 = poly_1.buffer(0)
                #     poly_2 = poly_2.buffer(0)
                #     print(f"cell_1 {poly_1.is_valid}, cell_2 {poly_2.is_valid}")
                #     raise Exception("shapely.errors.TopologicalError")

        # print(f"n_cells {self.n_cells}")
        # n_intersecting_cells = 0
        # n_dict = dict()
        # for cell in np.arange(self.n_cells):
        #     n = len(self.intersect_cells[cell])
        #     if n > 0:
        #         n_intersecting_cells += 1
        #         n_dict[n] = n_dict.get(n, 0) + 1
        # print(f"n_intersecting_cells {n_intersecting_cells}")
        # print(f"n_dict {n_dict}")

    def get_cell_mask(self, cell, dimensions):
        """

        :param cell:
        :param dimensions: height x width
        :return:
        """
        poly_gon = self.cells_polygon[cell]
        img = Image.new('1', (dimensions[1], dimensions[0]), 0)
        if isinstance(poly_gon, geometry.LineString):
            ImageDraw.Draw(img).polygon(list(poly_gon.coords), outline=1,
                                        fill=1)
        else:
            ImageDraw.Draw(img).polygon(list(poly_gon.exterior.coords), outline=1,
                                    fill=1)
        return np.array(img)

    def match_cells_indices(self, coord_obj, param, plot_title_opt=""):
        """

        :param coord_obj: another instanc of coord_obj
        :return: a 1d array, each index corresponds to the index of a cell of coord_obj, and map it to an index to self
        or -1 if no cell match
        """
        mapping_array = np.zeros(len(coord_obj.coord), dtype='int16')
        for cell in np.arange(len(coord_obj.coord)):
            c_x, c_y = coord_obj.center_coord[cell]
            distances = np.zeros(len(self.coord))
            for self_cell in np.arange(len(self.coord)):
                self_c_x, self_c_y = self.center_coord[self_cell]
                # then we calculte the cartesian distance to all other cells
                distances[self_cell] = math.sqrt((self_c_x - c_x) ** 2 + (self_c_y - c_y) ** 2)
            if np.min(distances) <= 2:
                mapping_array[cell] = np.argmin(distances)
            else:
                mapping_array[cell] = -1
        plot_result = True
        if plot_result:
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   gridspec_kw={'height_ratios': [1]},
                                   figsize=(20, 20))

            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")

            # dark blue
            other_twin_color = list((0.003, 0.313, 0.678, 1.0))
            n_twins = 0
            # red
            other_orphan_color = list((1, 0, 0, 1.0))
            n_other_orphans = 0
            # light blue
            self_twin_color = list((0.560, 0.764, 1, 1.0))
            # green
            self_orphan_color = list((0.278, 1, 0.101, 1.0))
            n_self_orphans = 0
            # blue = "cornflowerblue"
            # cmap.set_over('red')
            with_edge = True
            edge_line_width = 1
            z_order_cells = 12
            for cell in np.arange(len(coord_obj.coord)):
                xy = coord_obj.coord[cell].transpose()
                if with_edge:
                    line_width = edge_line_width
                    edge_color = "white"
                else:
                    edge_color = "white"
                    line_width = 0
                # allow to set alpha of the edge to 1
                if mapping_array[cell] >= 0:
                    # dark blue
                    face_color = other_twin_color
                    n_twins += 1
                else:
                    # red
                    face_color = other_orphan_color
                    n_other_orphans += 1
                face_color[3] = 0.8
                face_color = tuple(face_color)
                cell_contour = patches.Polygon(xy=xy,
                                                    fill=True, linewidth=line_width,
                                                    facecolor=face_color,
                                                    edgecolor=edge_color,
                                                    zorder=z_order_cells)  # lw=2
                ax.add_patch(cell_contour)

            for cell in np.arange(len(self.coord)):
                xy = self.coord[cell].transpose()
                if with_edge:
                    line_width = edge_line_width
                    edge_color = "white"
                else:
                    edge_color = "white"
                    line_width = 0
                # allow to set alpha of the edge to 1
                if cell in mapping_array:
                    # light blue
                    face_color = self_twin_color
                else:
                    # green
                    face_color = self_orphan_color
                    n_self_orphans += 1
                face_color[3] = 0.8
                face_color = tuple(face_color)
                cell_contour = patches.Polygon(xy=xy,
                                                    fill=True, linewidth=line_width,
                                                    facecolor=face_color,
                                                    edgecolor=edge_color,
                                                    zorder=z_order_cells)  # lw=2
                ax.add_patch(cell_contour)
            fontsize = 12
            plt.text(x=190, y=180,
                     s=f"{n_twins}", color=self_twin_color, zorder=22,
                     ha='center', va="center", fontsize=fontsize, fontweight='bold')
            plt.text(x=190, y=185,
                     s=f"{n_self_orphans}", color=self_orphan_color, zorder=22,
                     ha='center', va="center", fontsize=fontsize, fontweight='bold')
            plt.text(x=190, y=190,
                     s=f"{n_twins}", color=other_twin_color, zorder=22,
                     ha='center', va="center", fontsize=fontsize, fontweight='bold')
            plt.text(x=190, y=195,
                     s=f"{n_other_orphans}", color=other_orphan_color, zorder=22,
                     ha='center', va="center", fontsize=fontsize, fontweight='bold')
            ax.set_ylim(0, self.nb_lines)
            ax.set_xlim(0, self.nb_col)
            ylim = ax.get_ylim()
            # invert Y
            ax.set_ylim(ylim[::-1])
            plt.setp(ax.spines.values(), color="black")
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            save_format = "png"
            fig.savefig(f'{param.path_results}/cells_map_{plot_title_opt}'
                        f'_{param.time_str}.{save_format}',
                        format=f"{save_format}",
                            facecolor=fig.get_facecolor())
            # plt.show()
            plt.close()
        return mapping_array

    def plot_cells_map(self, param, data_id, title_option="", connections_dict=None,
                       background_color=(0, 0, 0, 1), default_cells_color=(1, 1, 1, 1.0),
                       default_edge_color="white",
                       dont_fill_cells_not_in_groups=False,
                       link_connect_color="white", link_line_width=1,
                       cell_numbers_color="dimgray", show_polygons=False,
                       cells_to_link=None, edge_line_width=2, cells_alpha=1,
                       fill_polygons=True, cells_groups=None, cells_groups_colors=None,
                       cells_groups_alpha=None,
                       cells_to_hide=None,
                       cells_groups_edge_colors=None, with_edge=False,
                       with_cell_numbers=False, save_formats="png",
                       save_plot=True, return_fig=False):
        """

        :param connections_dict: key is an int representing a cell number, and value is a dict representing the cells it
        connects to. The key is a cell is connected too, and the value represent the strength of the connection (like how
        many
        times it connects to it)
        :param plot_option: if 0: plot n_out and n_int, if 1 only n_out, if 2 only n_in, if 3: only n_out with dotted to
        show the commun n_in and n_out, if 4: only n_in with dotted to show the commun n_in and n_out,
        :return:
        """

        cells_center = self.center_coord
        n_cells = len(self.coord)
        if cells_to_hide is None:
            cells_to_hide = []

        cells_in_groups = []
        if cells_groups is not None:
            for group_id, cells_group in enumerate(cells_groups):
                cells_in_groups.extend(cells_group)
        cells_in_groups = np.array(cells_in_groups)
        cells_not_in_groups = np.setdiff1d(np.arange(n_cells), cells_in_groups)

        fig, ax = plt.subplots(nrows=1, ncols=1,
                               gridspec_kw={'height_ratios': [1]},
                               figsize=(20, 20))
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)

        # blue = "cornflowerblue"
        # cmap.set_over('red')
        z_order_cells = 12
        for group_index, cell_group in enumerate(cells_groups):
            for cell in cell_group:
                if cell in cells_to_hide:
                    continue

                xy = self.coord[cell].transpose()
                if with_edge:
                    line_width = edge_line_width
                    if cells_groups_edge_colors is None:
                        edge_color = default_edge_color
                    else:
                        edge_color = cells_groups_edge_colors[group_index]
                else:
                    edge_color = cells_groups_colors[group_index]
                    line_width = 0
                # allow to set alpha of the edge to 1
                face_color = list(cells_groups_colors[group_index])
                # changing alpha
                if cells_groups_alpha is not None:
                    face_color[3] = cells_groups_alpha[group_index]
                else:
                    face_color[3] = cells_alpha
                face_color = tuple(face_color)
                self.cell_contour = patches.Polygon(xy=xy,
                                                    fill=True, linewidth=line_width,
                                                    facecolor=face_color,
                                                    edgecolor=edge_color,
                                                    zorder=z_order_cells) # lw=2
                ax.add_patch(self.cell_contour)
                if with_cell_numbers:
                    self.plot_text_cell(cell=cell, cell_numbers_color=cell_numbers_color)

        for cell in cells_not_in_groups:
            if cell in cells_to_hide:
                continue
            xy = self.coord[cell].transpose()
            # face_color = default_cells_color
            # if dont_fill_cells_not_in_groups:
            #     face_color = None
            self.cell_contour = patches.Polygon(xy=xy,
                                                fill=not dont_fill_cells_not_in_groups,
                                                linewidth=0, facecolor=default_cells_color,
                                                edgecolor=default_cells_color,
                                                zorder=z_order_cells, lw=2)
            ax.add_patch(self.cell_contour)

            if with_cell_numbers:
                self.plot_text_cell(cell=cell, cell_numbers_color=cell_numbers_color)

        ax.set_ylim(0, self.nb_lines)
        ax.set_xlim(0, self.nb_col)
        ylim = ax.get_ylim()
        # invert Y
        ax.set_ylim(ylim[::-1])

        if (connections_dict is not None) :
            zorder_lines = 15
            for neuron in connections_dict.keys():
                # plot a line to all out of the neuron
                for connected_neuron, nb_connexion in connections_dict[neuron].items():
                    line_width = link_line_width + np.log(nb_connexion)

                    c_x = cells_center[neuron][0]
                    c_y = cells_center[neuron][1]
                    c_x_c = cells_center[connected_neuron][0]
                    c_y_c = cells_center[connected_neuron][1]

                    line = plt.plot((c_x, c_x_c), (c_y, c_y_c), linewidth=line_width, c=link_connect_color,
                                    zorder=zorder_lines)[0]

        if (self.cells_groups is not None) and show_polygons:
            for group_id, cells in enumerate(self.cells_groups):
                points = np.zeros((2, len(cells)))
                for cell_id, cell in enumerate(cells):
                    c_x, c_y = cells_center[cell]
                    points[0, cell_id] = c_x
                    points[1, cell_id] = c_y
                # finding the convex_hull for each group
                xy = convex_hull(points=points)
                # xy = xy.transpose()
                # print(f"xy {xy}")
                # xy is a numpy array with as many line as polygon point
                # and 2 columns: x and y coord of each point
                face_color = list(self.cells_groups_colors[group_id])
                # changing alpha
                face_color[3] = 0.3
                face_color = tuple(face_color)
                # edge alpha will be 1
                poly_gon = patches.Polygon(xy=xy,
                                           fill=fill_polygons, linewidth=0, facecolor=face_color,
                                           edgecolor=self.cells_groups_colors[group_id],
                                           zorder=15, lw=3)
                ax.add_patch(poly_gon)

        # plt.title(f"Cells map {data_id} {title_option}")

        # ax.set_frame_on(False)

        # invert Y
        ax.set_ylim(ylim[::-1])
        plt.setp(ax.spines.values(), color=background_color)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        # ax.xaxis.set_ticks_position('none')
        # ax.yaxis.set_ticks_position('none')
        #  :param plot_option: if 0: plot n_out and n_int, if 1 only n_out, if 2 only n_in, if 3: only n_out with dotted to
        # show the commun n_in and n_out, if 4: only n_in with dotted to show the commun n_in and n_out,
        if save_plot:
            if isinstance(save_formats, str):
                save_formats = [save_formats]
            for save_format in save_formats:
                fig.savefig(f'{param.path_results}/{data_id}_cell_maps_{title_option}'
                            f'_{param.time_str}.{save_format}',
                            format=f"{save_format}",
                            facecolor=fig.get_facecolor())
        if return_fig:
            return fig
        else:
            plt.close()

    def plot_text_cell(self, cell, cell_numbers_color, ax_to_use=None):
        fontsize = 6
        if cell >= 100:
            fontsize = 4
        elif cell >= 10:
            fontsize = 5

        c_x_c = self.center_coord[cell][0]
        c_y_c = self.center_coord[cell][1]
        if ax_to_use is None:
            plt.text(x=c_x_c, y=c_y_c,
                     s=f"{cell}", color=cell_numbers_color, zorder=22,
                     ha='center', va="center", fontsize=fontsize + 2, fontweight='bold')
        else:
            ax_to_use.text(x=c_x_c, y=c_y_c,
                     s=f"{cell}", color=cell_numbers_color, zorder=22,
                     ha='center', va="center", fontsize=fontsize + 2, fontweight='bold')

    def get_cell_new_coord_in_source(self, cell, minx, miny):
        coord = self.coord[cell]
        # coord = coord - 1
        coord = coord.astype(int)
        n_coord = len(coord[0, :])
        xy = np.zeros((n_coord, 2))
        for n in np.arange(n_coord):
            # shifting the coordinates in the square size_square+1
            xy[n, 0] = coord[0, n] - minx
            xy[n, 1] = coord[1, n] - miny
        return xy

    def scale_polygon_to_source(self, poly_gon, minx, miny):
        coords = list(poly_gon.exterior.coords)
        scaled_coords = []
        for coord in coords:
            scaled_coords.append((coord[0] - minx, coord[1] - miny))
        # print(f"scaled_coords {scaled_coords}")
        return geometry.Polygon(scaled_coords)


    def get_source_profile(self, cell, tiff_movie, traces, peak_nums, spike_nums,
                           pixels_around=0, bounds=None, buffer=None, with_full_frame=False):
        """
        Return the source profile of a cell
        :param cell:
        :param pixels_around:
        :param bounds: how much padding around the cell pretty much, coordinate of the frame covering the source profile
        4 int list
        :param buffer:
        :param with_full_frame:  Average the full frame
        :return:
        """
        # print("get_source_profile")
        len_frame_x = tiff_movie[0].shape[1]
        len_frame_y = tiff_movie[0].shape[0]

        # determining the size of the square surrounding the cell
        poly_gon = self.cells_polygon[cell]
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
        peaks = np.where(peak_nums[cell, :] > 0)[0]
        threshold = np.percentile(traces[cell, peaks], 95)
        selected_peaks = peaks[np.where(traces[cell, peaks] > threshold)[0]]
        # max 10 peaks, min 5 peaks
        if len(selected_peaks) > 10:
            p = 10 / len(peaks)
            threshold = np.percentile(traces[cell, peaks], (1 - p) * 100)
            selected_peaks = peaks[np.where(traces[cell, peaks] > threshold)[0]]
        elif (len(selected_peaks) < 5) and (len(peaks) > 5):
            p = 5 / len(peaks)
            threshold = np.percentile(traces[cell, peaks], (1 - p) * 100)
            selected_peaks = peaks[np.where(traces[cell, peaks] > threshold)[0]]

        # print(f"threshold {threshold}")
        # print(f"n peaks: {len(selected_peaks)}")

        onsets_frames = np.where(spike_nums[cell, :] > 0)[0]
        pos_traces = np.copy(traces)
        pos_traces += abs(np.min(traces))
        for peak in selected_peaks:
            tmp_source_profile = np.zeros((len_y, len_x))
            onsets_before_peak = np.where(onsets_frames <= peak)[0]
            if len(onsets_before_peak) == 0:
                # shouldn't arrive
                continue
            onset = onsets_frames[onsets_before_peak[-1]]
            # print(f"onset {onset}, peak {peak}")
            frames_tiff = tiff_movie[onset:peak + 1]
            for frame_index, frame_tiff in enumerate(frames_tiff):
                tmp_source_profile += (frame_tiff[miny:maxy + 1, minx:maxx + 1] * pos_traces[cell, onset + frame_index])
            # averaging
            tmp_source_profile = tmp_source_profile / (np.sum(pos_traces[cell, onset:peak + 1]))
            source_profile += tmp_source_profile
        if len(selected_peaks) > 0:
            source_profile = source_profile / len(selected_peaks)

        return source_profile, minx, miny, mask

    def get_transient_profile(self, cell, transient, tiff_movie, traces,
                              pixels_around=0, bounds=None):
        len_frame_x = tiff_movie[0].shape[1]
        len_frame_y = tiff_movie[0].shape[0]

        # determining the size of the square surrounding the cell
        if bounds is None:
            poly_gon = self.cells_polygon[cell]
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
        frames_tiff = tiff_movie[transient[0]:transient[-1] + 1]
        # print(f"transient[0] {transient[0]}, transient[1] {transient[1]}")
        # now we do the weighted average
        raw_traces = np.copy(traces)
        # so the lowest value is zero
        raw_traces += abs(np.min(raw_traces))

        for frame_index, frame_tiff in enumerate(frames_tiff):
            transient_profile += (
                    frame_tiff[miny:maxy + 1, minx:maxx + 1] * raw_traces[cell, transient[0] + frame_index])
        # averaging
        transient_profile = transient_profile / (np.sum(raw_traces[cell, transient[0]:transient[-1] + 1]))

        return transient_profile, minx, miny

    def corr_between_source_and_transient(self, cell, transient, source_profile_dict, tiff_movie, traces,
                                          source_profile_corr_dict=None,
                                          pixels_around=1):
        """
        Measure the correlation (pearson) between a source and transient profile for a giveb cell
        :param cell:
        :param transient:
        :param source_profile_dict should contains cell as key, and results of get_source_profile avec values
        :param pixels_around:
        :param source_profile_corr_dict: if not None, used to save the correlation of the source profile, f
        for memory and computing proficiency
        :return:
        """
        # print('corr_between_source_and_transient')
        poly_gon = self.cells_polygon[cell]

        # Correlation test
        bounds_corr = np.array(list(poly_gon.bounds)).astype(int)
        # looking if this source has been computed before for correlation
        if (source_profile_corr_dict is not None) and (cell in source_profile_corr_dict):
            source_profile_corr, mask_source_profile = source_profile_corr_dict[cell]
        else:
            source_profile_corr, minx_corr, \
                miny_corr, mask_source_profile, xy_source = source_profile_dict[cell]
            # normalizing
            source_profile_corr = source_profile_corr - np.mean(source_profile_corr)
            # we want the mask to be at ones over the cell
            mask_source_profile = (1 - mask_source_profile).astype(bool)
            if source_profile_corr_dict is not None:
                source_profile_corr_dict[cell] = (source_profile_corr, mask_source_profile)

        transient_profile_corr, minx_corr, miny_corr = self.get_transient_profile(cell=cell,
                                                                                  transient=transient,
                                                                                  tiff_movie=tiff_movie, traces=traces,
                                                                                  pixels_around=pixels_around,
                                                                                  bounds=bounds_corr)
        transient_profile_corr = transient_profile_corr - np.mean(transient_profile_corr)

        pearson_corr, pearson_p_value = stats.pearsonr(source_profile_corr[mask_source_profile],
                                                       transient_profile_corr[mask_source_profile])

        return pearson_corr

def _angle_to_point(point, centre):
    '''calculate angle in 2-D between points and x axis'''
    delta = point - centre
    res = np.arctan(delta[1] / delta[0])
    if delta[0] < 0:
        res += np.pi
    return res


def area_of_triangle(p1, p2, p3):
    '''calculate area of any triangle given co-ordinates of the corners'''
    return np.linalg.norm(np.cross((p2 - p1), (p3 - p1))) / 2.


def convex_hull(points, smidgen=0.0075):
    '''
    from: https://stackoverflow.com/questions/17553035/draw-a-smooth-polygon-around-data-points-in-a-scatter-plot-in-matplotlib
    Calculate subset of points that make a convex hull around points
    Recursively eliminates points that lie inside two neighbouring points until only convex hull is remaining.

    :Parameters:
    points : ndarray (2 x m)
    array of points for which to find hull
    use pylab to show progress?
    smidgen : float
    offset for graphic number labels - useful values depend on your data range

    :Returns:
    hull_points : ndarray (2 x n)
    convex hull surrounding points
    '''

    n_pts = points.shape[1]
    # assert(n_pts > 5)
    centre = points.mean(1)

    angles = np.apply_along_axis(_angle_to_point, 0, points, centre)
    pts_ord = points[:, angles.argsort()]

    pts = [x[0] for x in zip(pts_ord.transpose())]
    prev_pts = len(pts) + 1
    k = 0
    while prev_pts > n_pts:
        prev_pts = n_pts
        n_pts = len(pts)
        i = -2
        while i < (n_pts - 2):
            Aij = area_of_triangle(centre, pts[i], pts[(i + 1) % n_pts])
            Ajk = area_of_triangle(centre, pts[(i + 1) % n_pts], pts[(i + 2) % n_pts])
            Aik = area_of_triangle(centre, pts[i], pts[(i + 2) % n_pts])
            if Aij + Ajk < Aik:
                del pts[i + 1]
            i += 1
            n_pts = len(pts)
        k += 1
    return np.asarray(pts)


def get_coord_obj(data_path, movie_len_x=None, movie_len_y=None):
    if not os.path.isdir(data_path):
        print(f"Suite 2p data could not be loaded for , "
              f"the directory {data_path}: doesn't exist")
        return


    suite2p_data = dict()
    if os.path.isfile(os.path.join(data_path, 'F.npy')):
        f = np.load(os.path.join(data_path, 'F.npy'), allow_pickle=True)
        suite2p_data["F"] = f
    if os.path.isfile(os.path.join(data_path, 'Fneu.npy')):
        f_neu = np.load(os.path.join(data_path, 'Fneu.npy'), allow_pickle=True)
        suite2p_data["Fneu"] = f_neu
    if os.path.isfile(os.path.join(data_path, 'spks.npy')):
        spks = np.load(os.path.join(data_path, 'spks.npy'), allow_pickle=True)
        suite2p_data["spks"] = spks

    stat = np.load(os.path.join(data_path, 'stat.npy'), allow_pickle=True)
    suite2p_data["stat"] = stat

    if os.path.isfile(os.path.join(data_path, 'ops.npy')):
        ops = np.load(os.path.join(data_path, 'ops.npy'), allow_pickle=True)
        ops = ops.item()
        suite2p_data["ops"] = ops

    is_cell = np.load(os.path.join(data_path, 'iscell.npy'), allow_pickle=True)
    suite2p_data["is_cell"] = is_cell

    coord = []
    for cell in np.arange(len(stat)):
        if is_cell[cell][0] == 0:
            continue
        list_points_coord = [(x, y) for x, y in zip(stat[cell]["xpix"], stat[cell]["ypix"])]
        convex_hull = MultiPoint(list_points_coord).convex_hull
        if isinstance(convex_hull, LineString):
            coord_shapely = MultiPoint(list_points_coord).convex_hull.coords
        else:
            coord_shapely = MultiPoint(list_points_coord).convex_hull.exterior.coords
        coord.append(np.array(coord_shapely).transpose())
    suite2p_data["coord"] = coord


    coord = coord
    coord_obj = CoordClass(coord=coord, nb_col=movie_len_x,
                           nb_lines=movie_len_y, from_suite_2p=True)

    print(f"suite2p data loaded")
    return coord_obj

def build_raw_traces_from_movie(tiff_movie, coord_obj):
    print(f"building_raw_traces_from_movie")
    raw_traces = np.zeros((coord_obj.n_cells, tiff_movie.shape[0]))
    for cell in np.arange(coord_obj.n_cells):
        mask = coord_obj.get_cell_mask(cell=cell,
                                      dimensions=(tiff_movie.shape[1], tiff_movie.shape[2]))
        raw_traces[cell, :] = np.mean(tiff_movie[:, mask], axis=1)
    return raw_traces


def load_tiff_movie(movie_file_name):
    if movie_file_name is None:
        raise Exception("movie_file_name is None")

    print(f"Loading movie {movie_file_name}")
    # using_scan_image_tiff = False
    # if using_scan_image_tiff:
    try:
        start_time = time.time()
        tiff_movie = ScanImageTiffReader(movie_file_name).data()
        stop_time = time.time()
        print(f"Time for loading movie with scan_image_tiff: "
              f"{np.round(stop_time - start_time, 3)} s")
    except Exception as e:
        start_time = time.time()
        im = Image.open(movie_file_name)
        n_frames = len(list(ImageSequence.Iterator(im)))
        dim_x, dim_y = np.array(im).shape
        print(f"n_frames {n_frames}, dim_x {dim_x}, dim_y {dim_y}")
        tiff_movie = np.zeros((n_frames, dim_x, dim_y), dtype="uint16")
        for frame, page in enumerate(ImageSequence.Iterator(im)):
            tiff_movie[frame] = np.array(page)
        stop_time = time.time()
        print(f"Time for loading movie: "
              f"{np.round(stop_time - start_time, 3)} s")

        return tiff_movie

def do_traces_smoothing(traces):
    smooth_traces = np.copy(traces)
    # smoothing the trace
    windows = ['hanning', 'hamming', 'bartlett', 'blackman']
    i_w = 1
    window_length = 7  # 11
    for i in np.arange(traces.shape[0]):
        smooth_signal = smooth_convolve(x=smooth_traces[i], window_len=window_length,
                                        window=windows[i_w])
        beg = (window_length - 1) // 2
        smooth_traces[i] = smooth_signal[beg:-beg]
    return smooth_traces

def detect_onsets_and_peaks(traces):
    # automatic detection
    n_cells = traces.shape[0]
    n_times = traces.shape[1]
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
    return spike_nums, peak_nums

def plot_cell_trace(traces, cell):
    fig = plt.figure()
    plt.plot(traces[cell,:])
    plt.show()

def main():
    root_path = "/home/julien/these_inmed/hne_project/"

    tiff_movie = load_tiff_movie(movie_file_name=os.path.join(root_path,
                                              "data/p12/p12_17_11_10_a000/p12_17_11_10_a000.tif"))
    movie_len_x = tiff_movie.shape[1]
    movie_len_y = tiff_movie.shape[2]
    suite_2p_path = os.path.join(root_path, "data/p12/p12_17_11_10_a000/suite2p")

    coord_obj = get_coord_obj(suite_2p_path,
                              movie_len_x=movie_len_y,
                              movie_len_y=movie_len_x)

    raw_traces = build_raw_traces_from_movie(tiff_movie, coord_obj)
    plot_cell_trace(raw_traces, 2)
    smooth_traces = do_traces_smoothing(raw_traces)
    plot_cell_trace(smooth_traces, 2)
    # onsets, peaks = detect_onsets_and_peaks(smooth_traces)


main()