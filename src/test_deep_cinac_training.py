import os
from deepcinac.cinac_model import *
from deepcinac.cinac_predictor import *
import hdf5storage


def build_spike_nums_dur(spike_nums, peak_nums):
    n_cells = len(spike_nums)
    n_frames = spike_nums.shape[1]
    spike_nums_dur = np.zeros((n_cells, n_frames), dtype="int8")
    for cell in np.arange(n_cells):
        peaks_index = np.where(peak_nums[cell, :])[0]
        onsets_index = np.where(spike_nums[cell, :])[0]

        for onset_index in onsets_index:
            peaks_after = np.where(peaks_index > onset_index)[0]
            if len(peaks_after) == 0:
                continue
            peaks_after = peaks_index[peaks_after]
            peak_after = peaks_after[0]
            spike_nums_dur[cell, onset_index:peak_after + 1] = 1
    return spike_nums_dur

if __name__ == '__main__':
    # root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    root_path = "/media/julien/Not_today/hne_not_today/"

    data_path = os.path.join(root_path, "data/")
    results_path = os.path.join(root_path, "results_hne")
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    results_path = os.path.join(results_path, time_str)
    os.mkdir(results_path)

    """
    To start training from a full saved model from another training, it is necessary to:
    - During the previous training, to put as argument: save_only_the_weitghs=False
    - then specify the .h5 containing the model using partly_trained_model
    - finally setting the learning rate so it is the same as the last epoch trained, using learning_rate_start
    """

    partly_trained_model = "/media/julien/Not_today/hne_not_today/data/test_cinac_gui/transient_classifier_full_model_02-0.9779.h5"
    cinac_model = CinacModel(results_path=results_path, n_epochs=2, verbose=1,
                               partly_trained_model=partly_trained_model, save_only_the_weitghs=False,
                               learning_rate_start=0.001)

    artificial_movie_1 = os.path.join(data_path, "test_cinac_gui/artificial_movie_1_cell_0.cinac")

    cinac_model.add_input_data(cinac_file_name=artificial_movie_1)
    cinac_model.prepare_model()
    cinac_model.fit()


def just_for_test_to_forget():
    #  ---- ADDING DATA TO MODEL ------------
    artificial_movies_id = ["artificial_ms_1", "artificial_ms_2"]
    art_movie_file_names = {"artificial_ms_1": os.path.join(data_path, "artificial_movies", "1", "p1_artificial_1.tiff"),
                            "artificial_ms_2": os.path.join(data_path, "artificial_movies", "2_suite2p_contours",
                                                            "p2_artificial_2.tiff")}
    art_movie_cell_contours = {"artificial_ms_1": os.path.join(data_path, "artificial_movies", "1", "map_coords.mat"),
                            "artificial_ms_2": os.path.join(data_path, "artificial_movies", "2_suite2p_contours",
                                                            "map_coords.mat")}
    art_movie_raster = {"artificial_ms_1": os.path.join(data_path, "artificial_movies", "1", "gui_data.mat"),
                            "artificial_ms_2": os.path.join(data_path, "artificial_movies", "2_suite2p_contours",
                                                            "gui_data.mat")}
    cell_to_load_by_am = {"artificial_ms_1":
                              np.array([0, 11, 31, 38, 43, 56, 64, 70, 79, 96, 110, 118, 131, 136]),
                          # keeping cell 22, 86 for test
                          "artificial_ms_2":
                              np.array([0, 18, 26, 34, 41, 46, 56, 62, 88, 101, 116, 127, 140, 150])}
                            # keeping cell 9, 77 for test
    # cinac_files_to_use = []
    # now adding data

    for artificial_movie_id in artificial_movies_id:
        # ############
        # Creating an instance of CinacRecording
        # this class will be use to link the calcium imaging movie and the ROIs.
        # ############

        cinac_recording = CinacRecording(identifier=artificial_movie_id)
        # TODO: to test
        # cinac_movie = CinacSplitedTiffMovie(identifier=artificial_movie, tiffs_dirname="")

        # Creating and adding to cinac_recording the calcium imaging movie data
        cinac_movie = CinacTiffMovie(tiff_file_name=art_movie_file_names[artificial_movie_id])
        cinac_recording.set_movie(cinac_movie)

        # -----------------------------
        # options 1: suite2p data
        # -----------------------------

        """
        Segmenting your data will produce npy files used to build the ROIs
        the file iscell.npy indicated which roi represents a real cell, only those will be used.
        stat.npy will conain the ROIs coordinates. 
        """

        # is_cell_suite2p_file_name = os.path.join(data_path, "suite2p", "demo_deepcinac_iscell_1.npy")
        # stat_suite2p_file_name = os.path.join(data_path, "suite2p", "demo_deepcinac_stat_1.npy")
        # cinac_recording.set_rois_from_suite_2p(is_cell_file_name=is_cell_suite2p_file_name,
        #                                        stat_file_name=stat_suite2p_file_name)

        # ------------------------------------------------------
        # options 2: contours coordinate (such as CaImAn, Fiji)
        # ------------------------------------------------------

        """
        Args:
            coord: numpy array of 2d, first dimension of length 2 (x and y) and 2nd dimension 
                   of length the number of
                   cells. Could also be a list of lists or tuples of 2 integers
            from_matlab: Indicate if the data has been computed by matlab, 
                         then 1 will be removed to the coordinates so that it starts at zero.
            """
        data = hdf5storage.loadmat(art_movie_cell_contours[artificial_movie_id])
        coord = data["coord_python"][0]
        # variables_mapping = {"coord": "coord_python"}
        # old artifical movie where having matlab indexing
        cinac_recording.set_rois_2d_array(coord=coord, from_matlab=True)

        # variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
        #                      "peak_nums": "LocPeakMatrix_Python"}
        # artificial_ms.load_data_from_file(file_name_to_load=
        #                                   "artificial_movies/2_suite2p_contours/gui_data.mat",
        #                                   variables_mapping=variables_mapping,
        #                                   from_gui=True)
        #
        # artificial_ms.build_spike_nums_dur()
        data = hdf5storage.loadmat(art_movie_raster[artificial_movie_id])
        spike_nums = data["Bin100ms_spikedigital_Python"].astype(int)
        peak_nums = data["LocPeakMatrix_Python"].astype(int)
        raster_dur = build_spike_nums_dur(spike_nums, peak_nums)

        for cell in cell_to_load_by_am[artificial_movie_id]:
            cinac_model.add_input_data(cinac_recording=cinac_recording, cell=cell,
                                       ground_truth=raster_dur[cell, :], doubtful_frames=None, cells_to_remove=None)

    # TODO: For non artificial movies, add DOUBT at concatenation and beg and end of movie
    # cinac_model.add_input_data(cinac_recording=cinac_recording, cell, frames_to_add,
    #                            ground_truth, doubtful_frames, cells_to_remove)

