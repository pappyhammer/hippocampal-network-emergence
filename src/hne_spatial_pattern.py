import numpy as np
import networkx as nx
import os
from fa2 import ForceAtlas2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pattern_discovery.display.misc import plot_hist_distribution
import matplotlib.cm as cm
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D


def build_centroid_vector(activity_grid):
    """

    :param activity_grid: a N*M*T array, T being the number of timesteps, N & M represent the number of line and columns
    in the grid
    :return: a vector of size T
    """
    n_times = activity_grid.shape[2]
    activity_grid = np.reshape(activity_grid, (activity_grid.shape[0] * activity_grid.shape[1], n_times))
    centroid_vector = np.ones(n_times, dtype="int16")
    centroid_vector = centroid_vector * -1
    for t in np.arange(n_times):
        if np.max(activity_grid[:, t]) > 0:
            centroid_vector[t] = np.argmax(activity_grid[:, t])
    return centroid_vector


def build_graph_from_vector(centroid_vector):
    """

    :param centroid_vector: vector of T size (T being the number of timesteps), each value of the vector
    represent a node in the graph. Each adjacent node in the vector are connected.
    :return: Return a weighted directed graph.
    """
    graph = nx.DiGraph()
    nodes = np.unique(centroid_vector)
    nodes = np.setdiff1d(centroid_vector, np.array([-1]))
    graph.add_nodes_from(nodes)
    # now we build the edges, first using a dict
    # keys are tuple of nodes, value is the weight
    edges_dict = dict()
    for index, node in enumerate(centroid_vector[:-1]):
        if (node == -1) or (centroid_vector[index + 1] == -1):
            continue
        if node == centroid_vector[index + 1]:
            continue
        node_tuple = (node, centroid_vector[index + 1])
        edges_dict[node_tuple] = edges_dict.get(node_tuple, 0) + 1

    # TODO: see to normalize edges weights

    edges_list = []
    for node_tuple, weight in edges_dict.items():
        edges_list.append((node_tuple[0], node_tuple[1], weight))
    graph.add_weighted_edges_from(edges_list)

    return graph


def plot_graph_weights_distribution(graph, description, path_results, xlabel, save_formats):
    """

    :param graph:
    :param description:
    :param path_results:
    :param xlabel:
    :param save_formats:
    :return:
    """
    distribution = []
    for edge_tuple in graph.edges:
        distribution.append(graph[edge_tuple[0]][edge_tuple[1]]['weight'])
    plot_hist_distribution(distribution_data=distribution,
                           description=description,
                           path_results=path_results,
                           tight_x_range=True, twice_more_bins=True, xlabel=xlabel,
                           save_formats=save_formats)


def plot_graph_n_edges_by_node_distribution(graph, description, path_results, xlabel, save_formats):
    """

    :param graph:
    :param description:
    :param path_results:
    :param xlabel:
    :param save_formats:
    :return:
    """
    distribution = []
    for node in list(graph.nodes):
        distribution.append(len(graph[node]))

    plot_hist_distribution(distribution_data=distribution,
                           description=description,
                           path_results=path_results,
                           tight_x_range=True, twice_more_bins=True, xlabel=xlabel,
                           save_formats=save_formats)


def spatial_pattern_detector(ci_movie, coord_obj, raster_dur, subject_description, path_results, grid_size):
    """
    Aimed at detecting spatial wave
    :param ci_movie:
    :param coord_obj:
    :param raster_dur:
    :param subject_description:
    :param path_results
    :return:
    """

    print(f"spatial_pattern_detector grid_size {grid_size}")

    min_size_movie = min(ci_movie.shape[1], ci_movie.shape[2])
    n_frames = ci_movie.shape[0]
    n_cells = coord_obj.n_cells
    n_lines = min_size_movie // grid_size
    print(f"### N grids {n_lines * n_lines}")
    # signal fluorescence using the grid as ROIs
    # TODO: see to add only ROIs of the cell in the grid
    grid_traces = np.zeros((n_lines, n_lines, n_frames))
    # number of cell active in the grid for any given frames
    grid_cells = np.zeros((n_lines, n_lines, n_frames), dtype="int16")

    # traces = np.zeros((n_frames, n_lines*n_lines))
    for x_box in np.arange(n_lines):
        for y_box in np.arange(n_lines):
            mask = np.zeros((ci_movie.shape[1], ci_movie.shape[2]), dtype="bool")
            mask[y_box * grid_size:(y_box + 1) * grid_size, x_box * grid_size:(x_box + 1) * grid_size] = True
            grid_traces[y_box, x_box, :] = np.nanmean(ci_movie[:, mask], axis=1)

    # TODO: See to normalize the traces grid

    # filling grid_cells
    # if the centroid of the cell is in the grid, we add it
    for cell in np.arange(n_cells):
        centroid = coord_obj.center_coord[cell]
        grid_x = int(centroid[0] // grid_size)
        grid_y = int(centroid[1] // grid_size)
        if (grid_x >= n_lines) or (grid_y >= n_lines):
            continue
        # print(f"grid_x {grid_x}, grid_y {grid_y}")
        # print(f"len(grid_cells[grid_x, grid_y, :]) {len(grid_cells[grid_x, grid_y, :])}, "
        #       f"len(raster_dur[cell]) {raster_dur[cell]}")
        grid_cells[grid_x, grid_y, :] = grid_cells[grid_x, grid_y, :] + raster_dur[cell]

    # now we want to use a sliding window to correlate wave between them
    # also we choose a binning for activity
    cells_vector = build_centroid_vector(grid_cells)
    cells_graph = build_graph_from_vector(cells_vector)

    traces_vector = build_centroid_vector(grid_traces)
    traces_graph = build_graph_from_vector(traces_vector)

    with_3d_plot = False
    if with_3d_plot:
        for index_frame in np.arange(0, n_frames, 2500):
            frames = np.arange(index_frame, index_frame+2500)
            plot_3d_plot(centroid_vector=cells_vector, path_results=path_results,
                         frames=frames,
                         description=f"grid_{grid_size}_graph_cells_{subject_description}_3d_plot_frame_{index_frame}",
                         n_grid_by_line=n_lines)

            plot_3d_plot(centroid_vector=traces_vector, path_results=path_results,
                         frames=frames,
                         description=f"grid_{grid_size}_graph_fluorescence_{subject_description}_3d_"
                         f"plot_frame_{index_frame}",
                         n_grid_by_line=n_lines)

    # graph analyses
    analyses_on_graph(graph=traces_graph, graph_keyword="fluorescence", path_results=path_results,
                      subject_description=subject_description, grid_size=grid_size, n_lines=n_lines,
                      coord_obj=coord_obj)
    analyses_on_graph(graph=cells_graph, graph_keyword="cells", path_results=path_results,
                      subject_description=subject_description, grid_size=grid_size, n_lines=n_lines,
                      coord_obj=coord_obj)


def analyses_on_graph(graph, graph_keyword, path_results, subject_description, grid_size, n_lines, coord_obj):
    """
    Analyses to do on a graph
    :param graph:
    :param graph_keyword: string representing the graph
    :return:
    """
    print(f"### N nodes {graph_keyword}: {len(list(graph.nodes))}")

    plot_graph_weights_distribution(graph=graph,
                                    description=f"grid_{grid_size}_graph_{graph_keyword}_{subject_description}_"
                                    f"distribution_weights",
                                    path_results=path_results,
                                    xlabel="weights", save_formats="pdf")

    plot_graph_n_edges_by_node_distribution(graph=graph,
                                            description=f"grid_{grid_size}_graph_{graph_keyword}_{subject_description}_"
                                            f"distribution_edges_by_node",
                                            path_results=path_results,
                                            xlabel="N edges by node", save_formats="pdf")

    # to open with Cytoscape
    nx.write_graphml(graph, os.path.join(path_results,
                                         f"grid_{grid_size}_graph_{graph_keyword}_{subject_description}.graphml"))

    nx.write_gexf(graph, os.path.join(path_results,
                                      f"grid_{grid_size}_graph_{graph_keyword}_{subject_description}.gexf"))

    plot_graph(graph=graph, n_grid_by_line=int(n_lines), grid_size=grid_size, coord_obj=coord_obj,
               path_results=path_results,
               file_name=f"grid_{grid_size}_graph_{graph_keyword}_{subject_description}")


def plot_3d_plot(centroid_vector, path_results, description, n_grid_by_line, frames, save_formats="png"):
    """

    :param centroid_vector: vector of T size (T being the number of timesteps), each value of the vector
    represent a node in the graph. Each adjacent node in the vector are connected.
    :return: Return a weighted directed graph.
    """
    # from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = Axes3D(fig)

    # now we build the edges, first using a dict
    # keys are tuple of nodes, value is the weight
    edges_dict = dict()
    xs, ys, zs = [], [], []
    for index, node in enumerate(centroid_vector[frames[:-1]]):
        index = frames[index]
        if (node == -1) or (centroid_vector[index + 1] == -1) or (node == centroid_vector[index + 1]):
            if len(xs) > 1:
                ax.plot(xs=xs, ys=ys, zs=zs)
            else:
                xs, ys, zs = [], [], []
                continue
        # if node == centroid_vector[index + 1]:
        #     continue
        line = int(node // n_grid_by_line)
        col = node % n_grid_by_line
        zs.append(index)
        xs.append(line)
        ys.append(col)
    if len(xs) > 1:
        ax.plot(xs=xs, ys=ys, zs=zs)

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{path_results}/{description}.{save_format}', format=f"{save_format}")

    plt.close()


def plot_graph(graph, path_results, n_grid_by_line, grid_size, coord_obj, file_name="", iterations=2000, save_raster=True,
               color=None,
               with_labels=True, title=None, ax_to_use=None,
               save_formats="pdf", show_plot=False):
    types_of_draw = ["normal", "circular", "fa2", "grid"]
    types_of_draw = ["grid"]

    if "fa2" in types_of_draw:
        forceatlas2 = ForceAtlas2(
            # Behavior alternatives
            outboundAttractionDistribution=False,  # Dissuade hubs
            linLogMode=False,  # NOT IMPLEMENTED
            adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
            edgeWeightInfluence=1.0,

            # Performance
            jitterTolerance=1.0,  # Tolerance
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            multiThreaded=False,  # NOT IMPLEMENTED

            # Tuning
            scalingRatio=3.0,
            strongGravityMode=False,
            gravity=1.0,

            # Log
            verbose=False)
        positions_fa2 = forceatlas2.forceatlas2_networkx_layout(graph, pos=None, iterations=iterations)
        # print(f"positions_fa2 {positions_fa2}")
    if "grid" in types_of_draw:
        positions_grid = dict()
        half_fov_size = int((n_grid_by_line * grid_size) // 2)
        # dividing the space so none of the grid centroid share the same x, y coordinates
        pos_in_the_grid = np.linspace(grid_size / n_grid_by_line, grid_size - (grid_size / n_grid_by_line),
                                      n_grid_by_line)
        np.random.shuffle(pos_in_the_grid)
        # print(f"pos_in_the_grid {pos_in_the_grid}")
        for node in list(graph.nodes):
            # first we calculate in which line and column is the node (grid)
            line = int(node // n_grid_by_line)
            col = node % n_grid_by_line
            coordinates = ((col * grid_size) + pos_in_the_grid[line] - half_fov_size,
                           (line * grid_size) + pos_in_the_grid[col] - half_fov_size)
            positions_grid[node] = coordinates

    for type_of_draw in types_of_draw:
        if ax_to_use is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            fig.tight_layout()
            ax.set_facecolor("white")
            ax.axis('off')
        else:
            ax = ax_to_use
        if color is None:
            color = "cornflowerblue"

        grid_color = "black"
        font_color = "black"

        edges = graph.edges()
        # colors = [G[u][v]['color'] for u, v in edges]
        weights = [graph[u][v]['weight'] for u, v in edges]
        # first we normalize the weights so the max value is not to big
        max_weight_to_set = 5
        weights = np.array(weights)
        min_value = np.min(weights)
        max_value = np.max(weights)

        difference = max_value - min_value

        weights -= min_value

        if difference > 0:
            weights = weights / difference
        weights_01 = np.copy(weights)
        weights = weights * max_weight_to_set
        # minimum is set to one
        weights = weights + 0.2

        use_color_gradient = True
        if use_color_gradient:
            color_map = cm.Reds
            edge_color = []
            sorted_weights = np.argsort(weights)
            # print(f"weights[sorted_weights] {weights[sorted_weights]}")
            # print(f"np.max(sorted_weights) {np.max(sorted_weights)}, len(weights) {len(weights)}")
            n_unique_weights = len(np.unique(weights))
            for index_weight in np.arange(len(weights)):
                c = color_map(sorted_weights[index_weight] / (len(weights)-1))
                edge_color.append(c)
        else:
            edge_color = "red"
        font_size = 3

        if type_of_draw == "fa2":
            nx.draw_networkx(graph, pos=positions_fa2, node_size=20, edge_color=edge_color,
                             node_color=color, arrowsize=4, width=weights,
                             with_labels=with_labels, arrows=True,
                             font_size=font_size, font_color=font_color,
                             ax=ax)
        elif type_of_draw == "circular":
            nx.draw_circular(graph, node_size=20, edge_color=edge_color,
                             node_color=color, arrowsize=4, width=weights,
                             with_labels=with_labels, arrows=True,
                             font_size=font_size, font_color=font_color,
                             ax=ax)
        elif type_of_draw == "normal":
            nx.draw(graph, node_size=20, edge_color=edge_color,
                    node_color=color, arrowsize=4, width=weights,
                    with_labels=with_labels, arrows=True,
                    font_size=font_size, font_color=font_color,
                    ax=ax)
        elif type_of_draw == "grid":
            nx.draw_networkx(graph, pos=positions_grid, node_size=20, edge_color=weights_01,
                             edge_cmap=cm.Reds, edge_vmin=0, edge_vmax=np.max(weights_01),
                             node_color=color, arrowsize=4, width=weights,
                             with_labels=with_labels, arrows=True,
                             font_size=font_size, font_color=font_color,
                             ax=ax)
            # plotting the grid
            lw_grid = 0.2
            for line in np.arange(n_grid_by_line+1):
                ax.plot([(line * grid_size) - half_fov_size, (line * grid_size) - half_fov_size],
                        [-half_fov_size, half_fov_size], color=grid_color, lw=lw_grid, zorder=1)
                ax.plot([-half_fov_size, half_fov_size],
                        [(line * grid_size) - half_fov_size, (line * grid_size) - half_fov_size],
                        color=grid_color, lw=lw_grid, zorder=1)
            for cell in np.arange(coord_obj.n_cells):
                xy = coord_obj.coord[cell].transpose()
                # print(f"xy.shape {xy.shape}")
                # print(f"xy {xy}")
                xy = xy - half_fov_size
                xy_copy = np.copy(xy)
                xy[:, 0] = xy_copy[:, 1]
                xy[:, 1] = xy_copy[:, 0]
                cell_contour = patches.Polygon(xy=xy,
                                               fill=False, facecolor="black",
                                               edgecolor="black",
                                               zorder=1, lw=1)
                ax.add_patch(cell_contour)
            # coord_obj.plot_cells_map(ax_to_use=ax,
            #                          data_id="", show_polygons=False,
            #                          fill_polygons=False, connections_dict=None,
            #                             param=None,
            #                          # img_on_background=avg_cell_map_img,
            #                          default_cells_color=(0, 0, 0, 1.0),
            #                          default_edge_color="black",
            #                          with_edge=True,
            #                          dont_fill_cells_not_in_groups=False,
            #                          with_cell_numbers=False, save_formats=["png", "pdf"],
            #                          save_plot=False, return_fig=False)

        # nx.draw_networkx(graph, node_size=10, edge_color="white",
        #                  node_color="cornflowerblue",
        #                  with_labels=with_labels, arrows=True,
        #                  ax=ax)
        if ax_to_use is not None:
            legend_elements = []
            legend_elements.append(Patch(facecolor=color,
                                         edgecolor=edge_color, label=f'{title}'))
            ax.legend(handles=legend_elements)

        if (title is not None) and (ax_to_use is None):
            plt.title(title)

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if ax_to_use is None:
            if show_plot:
                plt.show()

            if save_raster:
                # transforming a string in a list
                if isinstance(save_formats, str):
                    save_formats = [save_formats]
                for save_format in save_formats:
                    fig.savefig(f'{path_results}/{file_name}_{type_of_draw}.{save_format}', format=f"{save_format}")

            plt.close()
