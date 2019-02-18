import numpy as np
from datetime import datetime
from mouse_session_loader import load_mouse_sessions
import pattern_discovery.tools.param as p_disc_tools_param
import scipy.io as sio
import os
import hdf5storage
from pattern_discovery.display.cells_map_module import CoordClass
from mouse_session import MouseSession


class DataForMs(p_disc_tools_param.Parameters):
    def __init__(self, path_data, path_results, time_str=None):
        if time_str is None:
            self.time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        else:
            self.time_str = time_str
        super().__init__(path_results=path_results, time_str=self.time_str, bin_size=1)
        self.path_data = path_data
        self.cell_assemblies_data_path = None
        self.best_order_data_path = None


def produce_cell_coord_from_cnn_validated_cells(param):
    path_cnn_classifier = "cell_classifier_results_txt/"

    ms_to_use = ["p7_171012_a000_ms", "p8_18_10_24_a005_ms", "p9_18_09_27_a003_ms", "p11_17_11_24_a000_ms",
                 "p12_171110_a000_ms", "p13_18_10_29_a001_ms"]

    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_to_use,
                                            param=param,
                                            load_traces=False, load_abf=False,
                                            for_transient_classifier=True)
    coords_to_keep = []

    for ms in ms_str_to_ms_dict.values():
        path_data = param.path_data

        cnn_file_name = None
        # finding the cnn_file coresponding to the ms
        for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(path_data, path_cnn_classifier)):
            for file_name in local_filenames:
                if file_name.endswith(".txt"):
                    if ms.description.lower() in file_name.lower():
                        cnn_file_name = file_name
                        break
            # looking only in the top directory
            break

        if cnn_file_name is None:
            print(f"{ms.description} no cnn file_name")
            continue

        cell_cnn_predictions = []
        with open(os.path.join(path_data, path_cnn_classifier, cnn_file_name), "r", encoding='UTF-8') as file:
            for nb_line, line in enumerate(file):
                line_list = line.split()
                cells_list = [float(i) for i in line_list]
                cell_cnn_predictions.extend(cells_list)
        cell_cnn_predictions = np.array(cell_cnn_predictions)
        cells_predicted_as_true = np.where(cell_cnn_predictions >= 0.5)[0]

        print(f"ms.coord_obj.coord[0].shape {ms.coord_obj.coord[0].shape}")
        print(f"{ms.description}: n_cells: {ms.coord_obj.n_cells}")
        print(f"{ms.description}: cells_predicted_as_true: {len(cells_predicted_as_true)}")

        for cell in np.arange(ms.coord_obj.n_cells):
            if cell in cells_predicted_as_true:
                coords_to_keep.append(ms.coord_obj.coord[cell])

    print(f"len(coords_to_keep): {len(coords_to_keep)}")
    coords_matlab_style = np.empty((len(coords_to_keep),), dtype=np.object)
    for i in range(len(coords_to_keep)):
        coords_matlab_style[i] = coords_to_keep[i]
    sio.savemat(os.path.join(param.path_results, "test_coords_cnn.mat"), {"coord_python": coords_matlab_style})

def main():
    """
    Objective is to produce fake movies of let's say 1000 frames with like 50 cells with targeted cells that would have
    between 1 and 3 overlaping cells.
    The function should produce a tiff movie and a file .mat with a coord variable containing the coords of the cells
    :return:
    """
    root_path = None
    with open("param_hne.txt", "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split('=')
            root_path = line_list[1]
    if root_path is None:
        raise Exception("Root path is None")
    path_data = root_path + "data/"
    path_results = root_path + "results_hne/"
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    path_results = path_results + "/" + time_str
    if not os.path.isdir(path_results):
        os.mkdir(path_results)

    param = DataForMs(path_data=path_data, path_results=path_results, time_str=time_str)

    data = hdf5storage.loadmat(os.path.join(path_data, "artificial_movie_generator", "test_coords_cnn.mat"))
    coords = data["coord_python"][0]
    # coord_obj = CoordClass(coord=coords, nb_col=200,
    #                             nb_lines=200)

    # ms_fusion = MouseSession(age=10, session_id="fusion", nb_ms_by_frame=100, param=param)
    # ms_fusion.coord_obj = coord_obj
    # ms_fusion.plot_all_cells_on_map()

    print(f"coord[0].shape {coord[0].shape}")

    produce_cell_coords = False

    if produce_cell_coords:
        produce_cell_coord_from_cnn_validated_cells(param)

    coords_left, map_coords, cells_with_overlap = \
        generate_artificial_map(coords_to_use=coords, dimensions=(200, 200), n_cells=50)

main()