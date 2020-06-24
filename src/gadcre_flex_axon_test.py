import PIL
from ScanImageTiffReader import ScanImageTiffReader
from PIL import ImageSequence
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

from skimage.morphology import skeletonize, medial_axis, remove_small_objects
from skimage.measure import label
from skimage.color import label2rgb
from tifffile import imsave
import hdbscan
import math
import scipy.stats
import networkx as nx
from cicada.utils.display.cells_map_utils import CellsCoord

def plot_hist_distribution(distribution_data,
                           filename=None,
                           values_to_scatter=None,
                           n_bins=None,
                           use_log=False,
                           x_range=None,
                           labels=None,
                           scatter_shapes=None,
                           colors=None,
                           tight_x_range=False,
                           twice_more_bins=False,
                           scale_them_all=False,
                           background_color="black",
                           hist_facecolor="white",
                           hist_edgeccolor="white",
                           axis_labels_color="white",
                           axis_color="white",
                           axis_label_font_size=20,
                           ticks_labels_color="white",
                           ticks_label_size=14,
                           xlabel=None,
                           ylabel=None,
                           fontweight=None,
                           fontfamily=None,
                           size_fig=None,
                           dpi=100,
                           path_results=None,
                           save_formats="pdf",
                           ax_to_use=None,
                           color_to_use=None, legend_str=None,
                           density=False,
                           save_figure=False,
                           with_timestamp_in_file_name=True,
                           max_value=None):
    """
    Plot a distribution in the form of an histogram, with option for adding some scatter values
    :param distribution_data:
    :param description:
    :param param:
    :param values_to_scatter:
    :param labels:
    :param scatter_shapes:
    :param colors:
    :param tight_x_range:
    :param twice_more_bins:
    :param xlabel:
    :param ylabel:
    :param save_formats:
    :return:
    """
    distribution = np.array(distribution_data)

    if x_range is not None:
        min_range = x_range[0]
        max_range = x_range[1]
    elif tight_x_range:
        max_range = np.max(distribution)
        min_range = np.min(distribution)
    else:
        max_range = 100
        min_range = 0
    weights = (np.ones_like(distribution) / (len(distribution))) * 100
    # weights=None

    if ax_to_use is None:
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                gridspec_kw={'height_ratios': [1]},
                                figsize=size_fig, dpi=dpi)
        ax1.set_facecolor(background_color)
        fig.patch.set_facecolor(background_color)
    else:
        ax1 = ax_to_use
    if n_bins is not None:
        bins = n_bins
    else:
        bins = int(np.sqrt(len(distribution)))
        if twice_more_bins:
            bins *= 2

    hist_color = hist_facecolor
    if bins > 100:
        edge_color = hist_color
    else:
        edge_color = hist_edgeccolor
    ax1.spines['bottom'].set_color(axis_color)
    ax1.spines['left'].set_color(axis_color)

    hist_plt, edges_plt, patches_plt = ax1.hist(distribution, bins=bins, range=(min_range, max_range),
                                                facecolor=hist_color, log=use_log,
                                                edgecolor=edge_color, label=legend_str,
                                                weights=weights, density=density)
    if values_to_scatter is not None:
        scatter_bins = np.ones(len(values_to_scatter), dtype="int16")
        scatter_bins *= -1

        for i, edge in enumerate(edges_plt):
            # print(f"i {i}, edge {edge}")
            if i >= len(hist_plt):
                # means that scatter left are on the edge of the last bin
                scatter_bins[scatter_bins == -1] = i - 1
                break

            if len(values_to_scatter[values_to_scatter <= edge]) > 0:
                if (i + 1) < len(edges_plt):
                    bool_list = values_to_scatter < edge  # edges_plt[i + 1]
                    for i_bool, bool_value in enumerate(bool_list):
                        if bool_value:
                            if scatter_bins[i_bool] == -1:
                                new_i = max(0, i - 1)
                                scatter_bins[i_bool] = new_i
                else:
                    bool_list = values_to_scatter < edge
                    for i_bool, bool_value in enumerate(bool_list):
                        if bool_value:
                            if scatter_bins[i_bool] == -1:
                                scatter_bins[i_bool] = i

        decay = np.linspace(1.1, 1.15, len(values_to_scatter))
        for i, value_to_scatter in enumerate(values_to_scatter):
            if i < len(labels):
                ax1.scatter(x=value_to_scatter, y=hist_plt[scatter_bins[i]] * decay[i], marker=scatter_shapes[i],
                            color=colors[i], s=60, zorder=20, label=labels[i])
            else:
                ax1.scatter(x=value_to_scatter, y=hist_plt[scatter_bins[i]] * decay[i], marker=scatter_shapes[i],
                            color=colors[i], s=60, zorder=20)
    ax1.legend()

    if tight_x_range:
        ax1.set_xlim(min_range, max_range)
    else:
        ax1.set_xlim(0, 100)
        xticks = np.arange(0, 110, 10)

        ax1.set_xticks(xticks)
        # sce clusters labels
        ax1.set_xticklabels(xticks)
    ax1.yaxis.set_tick_params(labelsize=ticks_label_size)
    ax1.xaxis.set_tick_params(labelsize=ticks_label_size)
    ax1.tick_params(axis='y', colors=axis_labels_color)
    ax1.tick_params(axis='x', colors=axis_labels_color)
    # TO remove the ticks but not the labels
    # ax1.xaxis.set_ticks_position('none')

    if ylabel is None:
        ax1.set_ylabel("Distribution (%)", fontsize=axis_label_font_size, labelpad=20, fontweight=fontweight,
                       fontfamily=fontfamily)
    else:
        ax1.set_ylabel(ylabel, fontsize=axis_label_font_size, labelpad=20, fontweight=fontweight, fontfamily=fontfamily)
    ax1.set_xlabel(xlabel, fontsize=axis_label_font_size, labelpad=20, fontweight=fontweight, fontfamily=fontfamily)

    ax1.xaxis.label.set_color(axis_labels_color)
    ax1.yaxis.label.set_color(axis_labels_color)

    if ax_to_use is None:
        # padding between ticks label and  label axis
        # ax1.tick_params(axis='both', which='major', pad=15)
        fig.tight_layout()
        if save_figure and (path_results is not None):
            # transforming a string in a list
            if isinstance(save_formats, str):
                save_formats = [save_formats]
            time_str = ""
            if with_timestamp_in_file_name:
                time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
            for save_format in save_formats:
                if not with_timestamp_in_file_name:
                    fig.savefig(os.path.join(f'{path_results}', f'{filename}.{save_format}'),
                                format=f"{save_format}",
                                facecolor=fig.get_facecolor())
                else:
                    fig.savefig(os.path.join(f'{path_results}', f'{filename}{time_str}.{save_format}'),
                                format=f"{save_format}",
                                facecolor=fig.get_facecolor())
        plt.close()

def load_tiff_movie(tiff_file_name):
    """
    Load a tiff movie from tiff file name.
    Args:
        tiff_file_name:

    Returns: a 3d array: n_frames * width_FOV * height_FOV

    """
    try:
        # start_time = time.time()
        tiff_movie = ScanImageTiffReader(tiff_file_name).data()
        # stop_time = time.time()
        # print(f"Time for loading movie with ScanImageTiffReader: "
        #       f"{np.round(stop_time - start_time, 3)} s")
    except Exception as e:
        im = PIL.Image.open(tiff_file_name)
        n_frames = len(list(ImageSequence.Iterator(im)))
        dim_y, dim_x = np.array(im).shape
        tiff_movie = np.zeros((n_frames, dim_y, dim_x), dtype="uint16")
        for frame, page in enumerate(ImageSequence.Iterator(im)):
            tiff_movie[frame] = np.array(page)
    return tiff_movie

def skeleton(im, results_path):
    """Return the skeleton of an image
        Parameters
        ----------
        im : np.array
            An image
        Returns
        -------
        im_skel_clean : np.array
            An image of the skeletonized sequence
        """

    # Applying Histogram Equalization
    im_eq = cv2.equalizeHist(im)
    #cv2.imshow("im_eq", im_eq)
    cv2.imwrite(os.path.join(results_path, "im_equalized.png"), im_eq)


    # Denoising
    im_denoised = cv2.fastNlMeansDenoising(im_eq)
    #cv2.imshow("im_den", im_denoised)
    cv2.imwrite(os.path.join(results_path, "im_denoised.png"), im_denoised)

    # Gaussian Local Thresholding
    im_thresh = cv2.adaptiveThreshold(im_denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 11, 0)
    #cv2.imshow("gauss_thresh", th2)
    cv2.imwrite(os.path.join(results_path, "im_thresh.png"), im_thresh)


    # Kernel for Morphological operations
    kernel = np.ones((2, 2), np.uint8)


    # Closing
    im_close = cv2.morphologyEx(im_thresh, cv2.MORPH_CLOSE, kernel)
    #im_close = im_thresh #(in case the closing denatures the image)
    #cv2.imshow("im_close", im_close)
    cv2.imwrite(os.path.join(results_path, "im_close.png"), im_close)


    # Skeletonization
    im_close = (im_close/255).astype(np.uint8) # skeletonization requires a binary image
    im_skel_lee = skeletonize(im_close, method='lee')
    im_skel_lee = im_skel_lee.astype(np.uint8) # skeletonization outputs a boolean matrix
    #cv2.imshow("im_skel_lee", 255 * im_skel_lee) # homothety for display
    cv2.imwrite(os.path.join(results_path, "im_skel_lee.png"), 255 * im_skel_lee)


    # Skeleton cleaning
    im_skel_clean = remove_small_objects(im_skel_lee.astype(bool), 10, connectivity=2)
    #cv2.imshow('im_skel_clean', im_skel_clean.astype(np.uint8)*255)
    cv2.imwrite(os.path.join(results_path, "im_skel_clean.png"), im_skel_clean.astype(np.uint8) * 255)


    # Label image regions
    im_label = label(im_skel_clean)
    im_label_overlay = label2rgb(im_label, image=im, alpha=0.5, image_alpha=1, bg_label=0)
    #cv2.imshow('im_label', im_label_overlay)
    cv2.imwrite(os.path.join(results_path, "im_label.png"), im_label_overlay*255)

    # Closing windows
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return im_skel_clean, im_skel_lee.astype(bool)

def build_skeleton_connectivty_graph(x_pos, y_pos, tif_seq,
                                     with_subgraph_connectivity=False, r_threshold=0.4, results_path=None):
    # for each pair of pixels, if they are close, we compute the pearson corr, if the pearson is high,
    # we build a connexion
    # n_pixels in skeleton
    n_pixels = len(x_pos)
    # averaging everything 5 frames
    tif_seq = np.mean(tif_seq.reshape(-1, 5, tif_seq.shape[1], tif_seq.shape[2]), axis=1)
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(n_pixels))

    all_rs = []

    for first_pixel in range(n_pixels-1):
        if first_pixel % 1000 == 0:
            print(f"first_pixel: {first_pixel}")
        for second_pixel in np.arange(first_pixel+1, n_pixels):
            dist = math.sqrt((x_pos[first_pixel] - x_pos[second_pixel]) ** 2 +
                             (y_pos[first_pixel] - y_pos[second_pixel]) ** 2)
            if dist > 3:
                continue

            r, p_value = scipy.stats.pearsonr(tif_seq[:, x_pos[first_pixel], y_pos[first_pixel]],
                                                tif_seq[:, x_pos[second_pixel], y_pos[second_pixel]])
            all_rs.append(r)
            if r >= r_threshold:
                if with_subgraph_connectivity:
                    desc_first = nx.descendants(graph, first_pixel)
                    desc_second = nx.descendants(graph, second_pixel)
                    list_of_nodes = []
                    node_to_compare = None
                    if len(desc_first) == 0 and len(desc_second) > 0:
                        list_of_nodes = desc_second
                        node_to_compare = first_pixel
                    elif len(desc_second) == 0 and len(desc_first) > 0:
                        list_of_nodes = desc_first
                        node_to_compare = second_pixel
                    if node_to_compare is not None:
                        # checking if the pixel is correlated with all other in the subgraph
                        pass_all_corr = True
                        for node in list_of_nodes:
                            r, p_value = scipy.stats.pearsonr(tif_seq[:, x_pos[node], y_pos[node]],
                                                              tif_seq[:, x_pos[node_to_compare], y_pos[node_to_compare]])
                            if r < r_threshold:
                                pass_all_corr = False
                                break
                        if pass_all_corr:
                            graph.add_edge(first_pixel, second_pixel)
                    else:
                        graph.add_edge(first_pixel, second_pixel)
                else:
                    graph.add_edge(first_pixel, second_pixel)

    if results_path is not None:
        plot_hist_distribution(distribution_data=all_rs, save_figure=True, tight_x_range=True,
                               filename="pearson_r_distribution", path_results=results_path)

    return graph

def pairwise_distances(x_pos, y_pos, tif_seq):
    # for each pair of pixels, if they are close, we compute the pearson corr
    n_pixels = len(x_pos)

    corr_matrix = np.zeros((n_pixels, n_pixels))
    np.fill_diagonal(corr_matrix, 1)

    # TODO: average every 5 frames

    for first_pixel in range(n_pixels-1):
        if first_pixel % 1000 == 0:
            print(f"first_pixel: {first_pixel}")
        for second_pixel in np.arange(first_pixel+1, n_pixels):
            dist = math.sqrt((x_pos[first_pixel] - x_pos[second_pixel]) ** 2 +
                             (y_pos[first_pixel] - y_pos[second_pixel]) ** 2)
            if dist > 2:
                corr_matrix[first_pixel, second_pixel] = np.inf
                continue
            r, p_value = scipy.stats.pearsonr(tif_seq[:, x_pos[first_pixel], y_pos[first_pixel]],
                                              tif_seq[:, x_pos[second_pixel], y_pos[second_pixel]])

            corr_matrix[first_pixel, second_pixel] = r

    return corr_matrix
    # binary_data = np.zeros((len(x_pos), n_frames), dtype="float")
    # for index in range(len(x_pos)):
    #     binary_data[index] = movie[:, x_pos[index], y_pos[index]]


if __name__ == '__main__':
    # root_path = "/Users/pappyhammer/Documents/academique/these_inmed/tbi_microglia_github/"
    root_path = "/media/julien/Not_today/hne_not_today/"
    data_path = os.path.join(root_path, "data/")

    results_path = os.path.join(root_path, "results_hne")
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    #
    results_path = os.path.join(results_path, time_str)
    os.mkdir(results_path)


    try_axon_flex = False

    if try_axon_flex:

        tiff_file_name = "/media/julien/Not_today/hne_not_today/data/axon_flex/p12_19_02_08_a003_part1.tif"
        movie = load_tiff_movie(tiff_file_name=tiff_file_name)
        print(movie.shape)

        mean_img = np.mean(movie, axis=0)

        percentile_threshold = 70
        threshold_value = np.percentile(mean_img, percentile_threshold)

        binary_frame = binarized_frame(movie_frame=mean_img, filled_value=1, threshold_value=None,
                                       percentile_threshold=percentile_threshold, with_uint=False,
                                       with_4_blocks=True)
        test_display = True

        if test_display:
            print(f"threshold_value {threshold_value}")
            plt.imshow(binary_frame, cmap=cm.Greys)
            plt.show()

    else:
        # ring analysis
        session_id = "190127_190207_190208_a003"
        tiff_file_name = f"/media/julien/Not_today/hne_not_today/data/tom_data/{session_id}_MotCorr.tif"
        seq_tif = load_tiff_movie(tiff_file_name=tiff_file_name)

        print(f"seq_tif.shape {seq_tif.shape}")
        n_frames = seq_tif.shape[0]
        seq_tif = seq_tif[:, :150, :150]
        im_mean = np.mean(seq_tif, axis=0).astype(np.uint8)

        cv2.imwrite(os.path.join(results_path, "im_mean.png"), im_mean)

        # Extracting the skeleton image
        im_skel, im_skel_lee = skeleton(im_mean, results_path=results_path)
        print(im_skel.shape)
        # plt.imshow(im_skel, cmap=cm.Greys)
        # plt.show()
        print(f"np.sum(im_skel_lee) {np.sum(im_skel_lee)}")

        where_pos = np.where(im_skel_lee)
        x_pos = where_pos[0]
        y_pos = where_pos[1]

        try_dbscan_option = False

        if try_dbscan_option:
            distance_matrix = pairwise_distances(x_pos=x_pos, y_pos=y_pos, tif_seq=seq_tif)
            clusterer = hdbscan.HDBSCAN(metric='precomputed')
            clusterer.fit(distance_matrix)
            print(f"{np.unique(clusterer.labels_)}")
            # writing an image for each labels

            for label in np.unique(clusterer.labels_):
                tmp_img = np.zeros_like(im_skel_lee)
                label_indices = np.where(clusterer.labels_ == label)[0]
                for label_index in label_indices:
                    tmp_img[x_pos[label_index], y_pos[label_index]] = 1
                cv2.imwrite(os.path.join(results_path, f"im_skel_label_{label}.png"), tmp_img.astype(np.uint8) * 255)
        else:
            # building graph based on pearson correlation
            # each pixel of the skeleton is a node
            # an edge is put between two nodes if their correlation passed the threshold r_threshold
            # if the pixels is already connected to other, then there is an option so to add an edge
            # the new node should be also correlated to all those in the subgraph
            graph = build_skeleton_connectivty_graph(x_pos=x_pos, y_pos=y_pos, tif_seq=seq_tif,
                                                     with_subgraph_connectivity=False, r_threshold=0.6,
                                                     results_path=results_path)
            sub_graphs = list(nx.connected_component_subgraphs(graph))
            print(f"{len(sub_graphs)} sub graphs found")
            print(" ")
            n_final_sub_graphs = 0
            n_one_node_sg = 0
            # min number of pixels that should be connected
            min_sub_graph_size = 10
            n_nodes_in_subgraphs = []
            pixel_masks = []
            components_results_path = os.path.join(results_path, "components")
            os.mkdir(components_results_path)
            for sub_graph_index, sub_graph in enumerate(sub_graphs):
                if sub_graph.number_of_nodes() < min_sub_graph_size:
                    n_one_node_sg += 1
                    continue
                n_final_sub_graphs += 1
                # print(f"sub_graph {sub_graph_index}")
                # print(f"{sub_graph.number_of_nodes()} nodes")
                # print(f"nodes {sub_graph.nodes(data=False)}")
                n_nodes_in_subgraphs.append(sub_graph.number_of_nodes())
                tmp_img = np.zeros_like(im_skel_lee)
                sub_pixel_masks = []
                for node in sub_graph.nodes(data=False):
                    tmp_img[x_pos[node], y_pos[node]] = 1
                    sub_pixel_masks.append((x_pos[node], y_pos[node]))
                pixel_masks.append(sub_pixel_masks)
                cv2.imwrite(os.path.join(components_results_path, f"im_skel_sg_{sub_graph_index}.png"),
                            tmp_img.astype(np.uint8) * 255)

            coord_obj = CellsCoord(pixel_masks=pixel_masks)
            coord_obj.plot_cells_map(path_results=results_path, use_welsh_powell_coloring=True, data_id="ring")
            coord_obj.save_coords(file_name=os.path.join(results_path, f"{session_id}_rings_coord"))
            plot_hist_distribution(distribution_data=n_nodes_in_subgraphs, save_figure=True, tight_x_range=True,
                                   twice_more_bins=True,
                                   filename="n_nodes_in_sg_distribution", path_results=results_path)
            print(f"{n_final_sub_graphs} n_final_sub_graphs")
            print(" ")

        # Projecting the activity on the skeleton
        seq_skel = np.zeros_like(seq_tif)
        for i in range(len(seq_skel)):
            seq_skel[i] = (im_skel != 0) * seq_tif[i]
        imsave(os.path.join(results_path, 'skel_seq.tif'), 100 * seq_skel)  # Rk : *100 for the vizualization

        """
        percentile_threshold = 60
        threshold_value = np.percentile(mean_img, percentile_threshold)

        binary_frame = binarized_frame(movie_frame=mean_img, filled_value=1, threshold_value=None,
                                       percentile_threshold=percentile_threshold, with_uint=False,
                                       with_4_blocks=True)

        where_pos = np.where(binary_frame)
        x_pos = where_pos[0]
        y_pos = where_pos[1]
        binary_data = np.zeros((len(x_pos), n_frames), dtype="float")
        for index in range(len(x_pos)):
            binary_data[index] = movie[:, x_pos[index], y_pos[index]]

        ica = FastICA(n_components=10)
        S_ = ica.fit_transform(binary_data)  # Reconstruct signals
        A_ = ica.mixing_  # Get estimated mixing matrix
        print(f"S_.shape {S_.shape}, A_.shape {A_.shape}")
        print(f"S_ {S_[:3, :]}")
        test_display = False

        if test_display:
            print(f"threshold_value {threshold_value}")
            plt.imshow(binary_frame, cmap=cm.Greys)
            plt.show()
        
        """
# import tifffile
# from bisect import bisect_right
# from sklearn.decomposition import FastICA, PCA

def binarized_frame(movie_frame, filled_value=1, percentile_threshold=90,
                    threshold_value=None, with_uint=True, with_4_blocks=False):
    """
    Take a 2d-array and return a binarized version, thresholding using a percentile value.
    It could be filled with 1 or another value
    Args:
        movie_frame:
        filled_value:
        percentile_threshold:
        with_4_blocks: make 4 square of the images

    Returns:

    """
    img = np.copy(movie_frame)
    if threshold_value is None:
        threshold = np.percentile(img, percentile_threshold)
    else:
        threshold = threshold_value

    img[img < threshold] = 0
    img[img >= threshold] = filled_value

    if with_uint:
        img = img.astype("uint8")
    else:
        img = img.astype("int8")
    return img
