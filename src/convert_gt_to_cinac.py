import os
from mouse_session_loader import load_mouse_sessions
from datetime import datetime
import pattern_discovery.tools.param as p_disc_tools_param
from deepcinac.utils.cinac_file_utils import CinacFileWriter
from deepcinac.utils.utils import get_source_profile_param, scale_polygon_to_source
import numpy as np
from shapely import geometry


# ## aim at converting the first version of ground truth to the new format, .cinac


class DataForMs(p_disc_tools_param.Parameters):
    def __init__(self, path_data, result_path, time_str=None):
        if time_str is None:
            self.time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        else:
            self.time_str = time_str
        super().__init__(path_results=result_path, time_str=self.time_str, bin_size=1)
        self.path_data = path_data
        self.cell_assemblies_data_path = None
        self.best_order_data_path = None


def create_cinac_file(ms, session_dict, param, bonus_str):
    file_name = os.path.join(param.path_results, f"gt_{session_dict['id']}_{bonus_str}.cinac")

    cinac_writer = CinacFileWriter(file_name=file_name)

    # not putting the name of the original movie in the file
    invalid_cells = np.zeros(ms.coord_obj.n_cells, dtype="bool")
    if (ms.cells_to_remove is not None) and (len(ms.cells_to_remove) > 0):
        print(f"create_cinac_file ms.cells_to_remove {ms.cells_to_remove}")
        for cell in np.arange(ms.coord_obj.n_cells):
            if cell in ms.cells_to_remove:
                invalid_cells[cell] = True

    # list of tuple of 3 ints
    segments_to_add = []
    raster_dur = ms.spike_struct.spike_nums_dur
    n_frames = raster_dur.shape[1]
    ms.normalize_traces()

    cinac_writer.create_full_data_group(save_ci_movie_info=False, save_only_movie_ref=True,
                                        n_cells=len(ms.coord_obj.coord), n_frames=n_frames,
                                        cells_contour=ms.coord_obj.coord, invalid_cells=invalid_cells)



    if (n_frames == 12500) or (n_frames == 10000):

        if ms.doubtful_frames_nums is None:
            # then we create it
            ms.doubtful_frames_nums = np.zeros(raster_dur.shape, dtype="int8")
        # we put the first doubt_window and last doubt_window frames in doubt
        doubt_window = 10
        ms.doubtful_frames_nums[:, :doubt_window] = 1
        ms.doubtful_frames_nums[:, -doubt_window:] = 1
        concat_indices = [2500, 5000, 7500, 10000]
        if n_frames == 10000:
            concat_indices = [2500, 5000, 7500]
        for concat_index in concat_indices:
            ms.doubtful_frames_nums[:, concat_index - doubt_window:concat_index] = 1
            ms.doubtful_frames_nums[:, concat_index:concat_index + doubt_window] = 1

    # then first we add the cell that were fully ground truth
    if "gt_cells" in session_dict:
        cells = session_dict["gt_cells"]
        for cell in cells:
            segments_to_add.append((cell, 0, n_frames-1))

    # then the segments
    if "segments_folder" in session_dict:
        file_names_to_load = []
        dir_of_files = []
        dir_to_load = session_dict["segments_folder"]
        path_data = param.path_data
        path_data = os.path.join(path_data, session_dict["path"])
        for directory in dir_to_load:
            for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(path_data, directory)):
                for file_name in local_filenames:
                    if file_name.endswith(".npy"):
                        file_names_to_load.append(file_name)
                        dir_of_files.append(os.path.join(path_data, directory))
                break
        print(f"n file_names_to_load in segments_folder == {len(file_names_to_load)}")
        for file_index, file_name in enumerate(file_names_to_load):
            underscores_pos = [pos for pos, char in enumerate(file_name) if char == "_"]
            if len(underscores_pos) < 4:
                continue
            # the last 4 indicates how to get cell number and frames
            # middle_frame = int(file_name[underscores_pos[-1] + 1:-4])
            last_frame = int(file_name[underscores_pos[-2] + 1:underscores_pos[-1]]) - 1
            first_frame = int(file_name[underscores_pos[-3] + 1:underscores_pos[-2]])
            cell = int(file_name[underscores_pos[-4] + 1:underscores_pos[-3]])
            # ms_str = file_name[:underscores_pos[-4]].lower()
            # if ms_str == "p7_17_10_12_a000":
            #     ms_str = "p7_171012_a000"
            # ms_str += "_ms"

            segments_to_add.append((cell, first_frame, last_frame))

            segment_raster_dur = np.load(os.path.join(dir_of_files[file_index], file_name))

            raster_dur[cell, first_frame:last_frame+1] = segment_raster_dur

    segment_window_in_pixels = 25

    for segment in segments_to_add:
        cell = segment[0]
        first_frame = segment[1]
        last_frame = segment[2]
        raster_dur_to_save = raster_dur[cell, first_frame:last_frame+1]

        ms.load_tiff_movie_in_memory()
        ms.normalize_movie()
        # now getting the movie patch surroung the cell so we save in the file only the pixels
        # surrounding the cell for the given frames
        mask_source_profiles, coords = get_source_profile_param(cell=cell,
                                                                movie_dimensions=ms.tiff_movie_normalized.shape[1:],
                                                                coord_obj=ms.coord_obj,
                                                                pixels_around=0,
                                                                buffer=1,
                                                                max_width=segment_window_in_pixels,
                                                                max_height=segment_window_in_pixels,
                                                                with_all_masks=True)
        # we save the normalized movie version, as normalizing movie won't be possible without the full movie
        # later on
        frames = np.arange(first_frame, last_frame + 1)
        minx, maxx, miny, maxy = coords
        # frames that contains all the pixels surrounding our cell and the overlapping one
        # with a max size of self.segment_window_in_pixels
        # Important to use the normalized movie here
        source_profile_frames = ms.tiff_movie_normalized[frames, miny:maxy + 1, minx:maxx + 1]

        # then we fit it the frame use by the network, padding the surrounding by zero if necessary
        profile_fit = np.zeros((len(frames), segment_window_in_pixels, segment_window_in_pixels))
        # we center the source profile
        y_coord = (profile_fit.shape[1] - source_profile_frames.shape[1]) // 2
        x_coord = (profile_fit.shape[2] - source_profile_frames.shape[2]) // 2
        profile_fit[:, y_coord:source_profile_frames.shape[1] + y_coord,
        x_coord:source_profile_frames.shape[2] + x_coord] = \
            source_profile_frames

        # changing the corner coordinates, used to scale the scale coordinates of cells contour
        minx = minx - x_coord
        miny = miny - y_coord

        # now we want to compute the new cells coordinates in this window and see if some of the overlapping
        # cells are invalid
        overlapping_cells = ms.coord_obj.intersect_cells[cell]
        coords_to_register = []
        cells_to_register = [cell]
        cells_to_register.extend(overlapping_cells)
        invalid_cells = np.zeros(len(cells_to_register), dtype="bool")
        if (ms.cells_to_remove is not None) and (len(ms.cells_to_remove) > 0):
            for cell_to_register_index, cell_to_register in enumerate(cells_to_register):
                if cell_to_register in ms.cells_to_remove:
                    print(f"For segment {segment}, cell_to_register {cell_to_register} "
                          f"cell_to_register_index {cell_to_register_index} invalid ")
                    invalid_cells[cell_to_register_index] = True
        for cell_to_register_index, cell_to_register in enumerate(cells_to_register):
            polygon = ms.coord_obj.cells_polygon[cell_to_register]
            scaled_polygon = scale_polygon_to_source(polygon=polygon, minx=minx, miny=miny)
            if isinstance(scaled_polygon, geometry.LineString):
                coord_shapely = list(scaled_polygon.coords)
            else:
                coord_shapely = list(scaled_polygon.exterior.coords)
            # changing the format of coordinates so it matches the usual one
            coords_to_register.append(np.array(coord_shapely).transpose())

        # registering invalid cells among the overlaping cells
        # the cell registered can't be invalid
        invalid_cells[0] = 0

        doubtful_frames = ms.doubtful_frames_nums[cell, first_frame:last_frame + 1]
        group_name = cinac_writer.add_segment_group(cell=cell, first_frame=first_frame,
                                                    last_frame=last_frame, raster_dur=raster_dur_to_save,
                                                    doubtful_frames=doubtful_frames, ci_movie=profile_fit,
                                                    pixels_around=0,
                                                    buffer=1,
                                                    smooth_traces=ms.z_score_smooth_traces[cell, first_frame:last_frame + 1],
                                                    raw_traces=ms.z_score_raw_traces[cell, first_frame:last_frame + 1],
                                                    cells_contour=coords_to_register,
                                                    invalid_cells=invalid_cells)
    cinac_writer.close_file()

def main_convert_gt_to_cinac():
    # import pyqtgraph.examples
    # pyqtgraph.examples.run()
    # return
    # root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    root_path = '/media/julien/Not_today/hne_not_today/'
    path_data = os.path.join(root_path, "data/")
    result_path = os.path.join(root_path, "results_hne/")
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    result_path = result_path + "/" + time_str
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    param = DataForMs(path_data=path_data, result_path=result_path, time_str=time_str)

    # first key will be the ms id
    data_dict = dict()

    data_dict["p5_19_03_25_a001_ms"] = dict()
    data_dict["p5_19_03_25_a001_ms"]["id"] = "p5_19_03_25_a001"
    data_dict["p5_19_03_25_a001_ms"]["path"] = "p5/p5_19_03_25_a001"
    data_dict["p5_19_03_25_a001_ms"]["segments_folder"] = ["transients_to_add_for_rnn"]
    # data_dict["p5_19_03_25_a001_ms"]["segmentation_tool"] = "suite2p"

    data_dict["p7_171012_a000_ms"] = dict()
    data_dict["p7_171012_a000_ms"]["id"] = "p7_17_10_12_a000"
    data_dict["p7_171012_a000_ms"]["path"] = "p7/p7_17_10_12_a000"
    # data_dict["p7_171012_a000_ms"]["gt_cells"] = [3, 8, 11, 12, 14, 17, 18, 24] # for training
    data_dict["p7_171012_a000_ms"]["gt_cells"] = [2, 25] # for benchmarks
    # data_dict["p7_171012_a000_ms"]["gt_cells"] = [2, 3, 8, 11, 12, 14, 17, 18, 24, 25] # all
    # data_dict["p7_171012_a000_ms"]["segments_folder"] = ["transients_to_add_for_rnn"] # TODO: comment for benchmark
    # data_dict["p7_171012_a000_ms"]["segmentation_tool"] = "caiman"

    data_dict["p8_18_10_24_a005_ms"] = dict()
    data_dict["p8_18_10_24_a005_ms"]["id"] = "p8_18_10_24_a005"
    data_dict["p8_18_10_24_a005_ms"]["path"] = "p8/p8_18_10_24_a005"
    # data_dict["p8_18_10_24_a005_ms"]["gt_file"] = "p8_18_10_24_a005_fusion_validation.mat"
    data_dict["p8_18_10_24_a005_ms"]["gt_cells"] = [0, 1, 9, 10, 13, 15, 28, 41, 42, 110, 207, 321] # all with segments
    # data_dict["p8_18_10_24_a005_ms"]["segments_folder"] = ["transients_to_add_for_rnn"] # TODO: comment for benchmarks
    # data_dict["p8_18_10_24_a005_ms"]["segmentation_tool"] = "caiman"

    data_dict["p8_18_10_24_a006_ms"] = dict()
    data_dict["p8_18_10_24_a006_ms"]["id"] = "p8_18_10_24_a006"
    data_dict["p8_18_10_24_a006_ms"]["path"] = "p8/p8_18_10_24_a006"
    # data_dict["p8_18_10_24_a006_ms"]["gt_cells"] = [0, 1, 6, 7, 9, 10, 11, 18, 24]  # for training
    data_dict["p8_18_10_24_a006_ms"]["gt_cells"] = [28, 32, 33] # for benchmarks # TODO: use RD GT
    # data_dict["p8_18_10_24_a006_ms"]["segments_folder"] = ["transients_to_add_for_rnn"] # TODO: comment for benchmarks
    # data_dict["p8_18_10_24_a006_ms"]["segmentation_tool"] = "caiman"

    data_dict["p11_17_11_24_a000_ms"] = dict()
    data_dict["p11_17_11_24_a000_ms"]["id"] = "p11_17_11_24_a000"
    data_dict["p11_17_11_24_a000_ms"]["path"] = "p11/p11_17_11_24_a000"
    # data_dict["p11_17_11_24_a000_ms"]["gt_cells"] = [3, 17, 22, 24, 25, 29, 30, 33, 45] # all
    # data_dict["p11_17_11_24_a000_ms"]["gt_cells"] = [17, 22, 24, 25, 29, 30, 33] # for training
    data_dict["p11_17_11_24_a000_ms"]["gt_cells"] = [3, 45] # for benchmarks
    # data_dict["p11_17_11_24_a000_ms"]["segments_folder"] = ["transients_to_add_for_rnn"] # TODO: comment for benchmarks
    # data_dict["p11_17_11_24_a000_ms"]["segmentation_tool"] = "caiman"

    data_dict["p12_171110_a000_ms"] = dict()
    data_dict["p12_171110_a000_ms"]["id"] = "p12_17_11_10_a000"
    data_dict["p12_171110_a000_ms"]["path"] = "p12/p12_17_11_10_a000"
    # data_dict["p12_171110_a000_ms"]["gt_cells"] = [0, 3, 6, 7, 12, 14, 15, 19] # for training
    data_dict["p12_171110_a000_ms"]["gt_cells"] = [9, 10] # for benchmarks
    # data_dict["p12_171110_a000_ms"]["gt_cells"] = [0, 3, 6, 7, 9, 10, 12, 14, 15, 19] # all
    # data_dict["p12_171110_a000_ms"]["segmentation_tool"] = "caiman"

    data_dict["p13_18_10_29_a001_ms"] = dict()
    data_dict["p13_18_10_29_a001_ms"]["id"] = "p13_18_10_29_a001"
    data_dict["p13_18_10_29_a001_ms"]["path"] = "p13/p13_18_10_29_a001"
    # data_dict["p13_18_10_29_a001_ms"]["gt_cells"] = [0, 2, 5, 12, 13, 31, 42, 44, 48, 51]  # for training
    # data_dict["p13_18_10_29_a001_ms"]["gt_cells"] = [0, 2, 5, 12, 13, 31, 42, 44, 48, 51] # all
    data_dict["p13_18_10_29_a001_ms"]["gt_cells"] = [77, 117] # TODO for benchmarks, but load RD GT
    # data_dict["p13_18_10_29_a001_ms"]["segments_folder"] = ["transients_to_add_for_rnn"] # TODO: comment for benchmarks
    # data_dict["p13_18_10_29_a001_ms"]["segmentation_tool"] = "caiman"

    data_dict["artificial_ms_1"] = dict()
    data_dict["artificial_ms_1"]["id"] = "artificial_ms_1"
    data_dict["artificial_ms_1"]["path"] = "artificial_movies/1"
    data_dict["artificial_ms_1"]["gt_cells"] = [0, 11, 22, 31, 38, 43, 56, 64, 70, 79, 86, 96,
                                                110, 118, 131, 136]  # for training

    data_dict["artificial_ms_2"] = dict()
    data_dict["artificial_ms_2"]["id"] = "artificial_ms_2"
    data_dict["artificial_ms_2"]["path"] = "artificial_movies/2"
    data_dict["artificial_ms_2"]["gt_cells"] = [0, 9, 18, 26, 34, 41, 46, 56, 62, 77, 88, 101,
                                                116, 127, 140, 150]  # for training

    data_dict["p10_19_02_21_a005_ms"] = dict()
    data_dict["p10_19_02_21_a005_ms"]["id"] = "p10_19_02_21_a005"
    data_dict["p10_19_02_21_a005_ms"]["path"] = "p10/p10_19_02_21_a005"
    data_dict["p10_19_02_21_a005_ms"]["segments_folder"] = ["transients_to_add_for_rnn"]
    # data_dict["p10_19_02_21_a005_ms"]["segmentation_tool"] = "suite2p"

    data_dict["p7_19_03_05_a000_ms"] = dict()
    data_dict["p7_19_03_05_a000_ms"]["id"] = "p7_19_03_05_a000"
    data_dict["p7_19_03_05_a000_ms"]["path"] = "p7/p7_19_03_05_a000"
    data_dict["p7_19_03_05_a000_ms"]["segments_folder"] = ["transients_to_add_for_rnn"]
    # data_dict["p7_19_03_05_a000_ms"]["segmentation_tool"] = "suite2p"

    data_dict["p7_19_03_27_a000_ms"] = dict()
    data_dict["p7_19_03_27_a000_ms"]["id"] = "p7_19_03_27_a000"
    data_dict["p7_19_03_27_a000_ms"]["path"] = "p7/p7_19_03_27_a000"
    data_dict["p7_19_03_27_a000_ms"]["segments_folder"] = ["transients_to_add_for_rnn"]
    # data_dict["p7_19_03_27_a000_ms"]["segmentation_tool"] = "suite2p"

    data_dict["p16_18_11_01_a002_ms"] = dict()
    data_dict["p16_18_11_01_a002_ms"]["id"] = "p16_18_11_01_a002"
    data_dict["p16_18_11_01_a002_ms"]["path"] = "p16/p16_18_11_01_a002"
    data_dict["p16_18_11_01_a002_ms"]["segments_folder"] = ["transients_to_add_for_rnn"]
    # data_dict["p16_18_11_01_a002_ms"]["segmentation_tool"] = "suite2p"

    data_dict["p9_19_03_14_a001_ms"] = dict()
    data_dict["p9_19_03_14_a001_ms"]["id"] = "p9_19_03_14_a001"
    data_dict["p9_19_03_14_a001_ms"]["path"] = "p9/p9_19_03_14_a001"
    data_dict["p9_19_03_14_a001_ms"]["segments_folder"] = ["transients_to_add_for_rnn"]
    # data_dict["p9_19_03_14_a001_ms"]["segmentation_tool"] = "suite2p"

    data_dict["p5_19_09_02_a000_ms"] = dict()
    data_dict["p5_19_09_02_a000_ms"]["id"] = "p5_19_09_02_a000"
    data_dict["p5_19_09_02_a000_ms"]["path"] = "p5/p5_19_09_02_a000"
    data_dict["p5_19_09_02_a000_ms"]["segments_folder"] = ["transients_to_add_for_rnn"]
    # data_dict["p5_19_09_02_a000_ms"]["segmentation_tool"] = "suite2p"

    data_dict["p10_19_03_08_a001_ms"] = dict()
    data_dict["p10_19_03_08_a001_ms"]["id"] = "p10_19_03_08_a001"
    data_dict["p10_19_03_08_a001_ms"]["path"] = "p10/p10_19_03_08_a001"
    data_dict["p10_19_03_08_a001_ms"]["segments_folder"] = ["transients_to_add_for_rnn"]
    # data_dict["p10_19_03_08_a001_ms"]["segmentation_tool"] = "suite2p"

    data_dict["p12_17_11_10_a002_ms"] = dict()
    data_dict["p12_17_11_10_a002_ms"]["id"] = "p12_17_11_10_a002"
    data_dict["p12_17_11_10_a002_ms"]["path"] = "p12/p12_17_11_10_a002"
    data_dict["p12_17_11_10_a002_ms"]["segments_folder"] = ["transients_to_add_for_rnn"]
    # data_dict["p12_17_11_10_a002_ms"]["segmentation_tool"] = "suite2p"

    # ms_to_use = ["p16_18_11_01_a002_ms", "p9_19_03_14_a001_ms", "p5_19_09_02_a000_ms", "p10_19_03_08_a001_ms", "p12_17_11_10_a002_ms"]
    # # for training
    # bonus_str = "for_training"
    # ms_to_use = list(data_dict.keys())
    bonus_str = "for_benchmarks"
    ms_to_use = ["p7_171012_a000_ms", "p8_18_10_24_a006_ms", "p11_17_11_24_a000_ms", "p12_171110_a000_ms",
                 "p13_18_10_29_a001_ms", "p8_18_10_24_a005_ms"]
    # ms_to_use = ["p8_18_10_24_a005_ms"]

    # we need to change in mouse_session_loader the type of segmentation to use

    # TODO: deactivate clean_data_using_cells_to_remove() for conversion
    #  otherwise the cells to removed will have been removed already and the cells_contour
    #  saved won't be right
    # TODO: otherwise, removing the cells, saving the new coords, but using
    #  ms.get_new_cell_indices_if_cells_removed() to register the right cells
    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_to_use,
                                            param=param,
                                            load_traces=True, load_abf=False,
                                            for_transient_classifier=True)

    # for each given mouse session, we create a unique .cinac file

    for ms_id in ms_to_use:
        create_cinac_file(ms=ms_str_to_ms_dict[ms_id], session_dict=data_dict[ms_id], param=param,
                          bonus_str=bonus_str)

if __name__ == "__main__":
    main_convert_gt_to_cinac()