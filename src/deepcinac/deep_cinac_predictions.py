from deepcinac.cinac_predictor import *
from deepcinac.cinac_structures import *
from deepcinac.cinac_benchmarks import benchmark_neuronal_activity_inferences
from deepcinac.utils.cinac_file_utils import create_tiffs_from_movie
import tensorflow as tf
import numpy as np
import hdf5storage
import os

"""
We're going to guide you on how using DeepCINAC to infer the neuronal activity from your calcium imaging data.

Here is a link to our gitlab page for more information about our package: https://gitlab.com/cossartlab/deepcinac

So far, to run this code, you will need some calcium imaging data to work on (in tiff format) and 
some segmentation data (ROIs) indicating the contours or pixels 
that compose your cells (compatible format are Caiman, suite2p, Fiji or NWB outputs).

The filenames of the data in the code corresponds to the demo data available on our gitlab repository, 
using suite2p ROIs data: https://gitlab.com/cossartlab/deepcinac/tree/master/demos/data

You will also need a model file (.json extension) and a file containing the weights of network (.h5 extensions). 
You can download some there: https://gitlab.com/cossartlab/deepcinac/tree/master/model

"""

# ------------------------- #
#     Setting file names
# ------------------------- #

# root path, just used to avoid copying the path everywhere
root_path = '/media/julien/Not_today/hne_not_today/data/'

# path to calcium imaging data
data_path = root_path  # os.path.join(root_path, "data")
movie_file_name = os.path.join(data_path, "p1_artificial_1.tif")

# string used to identify the recording from which you want to predict activity
identifier = "art_movie_1"

# Path to your model data. It's possible to have more than one model, and use
# each for different cell of the same recording (for exemple, one
# network could be specialized for interneurons and the other one for pyramidal
# cells)
# weights_file_name = os.path.join(root_path, "transient_classifier_weights_19_meso_v4.h5")
# json_file_name = os.path.join(root_path, "transient_classifier_model_architecture_meso_v4.json")

# model_path = os.path.join(root_path, "transient_classifier_model/meso_v12_epoch_12")

# json_file_name = os.path.join(model_path, "transient_classifier_model_architecture.json")
# weights_file_name = os.path.join(model_path, "transient_classifier_weights_12_BO.h5")


model_path = os.path.join(root_path, "transient_classifier_model")
# meso_v2_epoch_19
# weights_file_name = os.path.join(model_path, "cinac_weights_v2_epoch_19.h5")
# json_file_name = os.path.join(model_path, "cinac_model_v2.json")


# weights_file_name = os.path.join(model_path, "chen_v2_10_hz", "chen_v2_weights_18.h5")
# json_file_name = os.path.join(model_path, "chen_v2_10_hz", "chen_v2_model.json")

# weights_file_name = os.path.join(model_path, "chen_v4", "chen_v4_weights_8.h5")
# json_file_name = os.path.join(model_path, "chen_v4", "chen_v4_model.json")

# weights_file_name = os.path.join(model_path, "meso_v24_epoch_6", "cinac_weights_v24_epoch_6.h5")
# json_file_name = os.path.join(model_path, "meso_v24_epoch_6", "cinac_model_v24.json")

weights_file_name = os.path.join(model_path, "meso_v15_epoch_23", "cinac_weights_v15_epoch_23.h5")
json_file_name = os.path.join(model_path, "meso_v15_epoch_23", "cinac_model_v15.json")

# weights_file_name = os.path.join(root_path, "transient_classifier_weights_art_mov_1_test_05-0.9908.h5")
# json_file_name = os.path.join(root_path, "transient_classifier_model_architecture_.json")

# classifier_id = "meso_v2_epoch_19"
# classifier_id = "chen_v4"
# classifier_id = "meso_v24_epoch_6"
classifier_id = "meso_v15_epoch_23"
# classifier_id = "meso_v12_epoch_12"

# path of the directory where the results will be save
# a directory will be created each time the prediction is run
# the directory name will be the date and time at which the analysis has been run
# the predictions will be in this directory.
results_path = os.path.join(root_path, "results")

# not mandatory, just to test if you GPU is accessible
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# to add in GT:
# p5: 191205_191210_0_191210_a001 (meso_v2_epoch_19)
# p6: 190921_190927_0_190927_a000 (meso_v13_epoch_23)
# p7:200103_200110_200110_a000 (meso_v2_epoch_19)
# p13: 191122_191205_191205_a000 (meso_v13_epoch_23)

evaluate_inferences = True

if evaluate_inferences:
    inferences_dir = "/media/julien/Not_today/hne_not_today/data/cinac_ground_truth/for_benchmarks/to_benchmark"
    # inferences_dir = "/media/julien/Not_today/hne_not_today/data/cinac_ground_truth/for_benchmarks/to_benchmark_chen"
    #
    # inferences_dir = "/media/julien/Not_today/hne_not_today/data/cinac_ground_truth/for_benchmarks/to_benchmark_ins"
    # inferences_dir = "/media/julien/Not_today/hne_not_today/data/cinac_ground_truth/for_benchmarks/to_benchmark_geco"
    # inferences_dir = "/media/julien/Not_today/hne_not_today/data/cinac_ground_truth/for_benchmarks/to_benchmark_laura"
    # inferences_dir = "/media/julien/Not_today/hne_not_today/data/cinac_ground_truth/for_benchmarks/to_benchmark_arnaud"
    # inferences_dir = "/media/julien/Not_today/hne_not_today/data/cinac_ground_truth/for_benchmarks_clean_online/figure_8C"
    results_path = "/media/julien/Not_today/hne_not_today/results_hne/"
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    results_path = results_path + f"{time_str}"
    os.mkdir(results_path)

    benchmark_neuronal_activity_inferences(inferences_dir=inferences_dir, results_path=results_path,
                                           colorfull_boxplots=False, white_background=True,
                                           # colorfull_boxplots=False, white_background=True,
                                           with_legend=True, put_metric_as_y_axis_label=True,
                                           alpha_scatter=0.2, save_formats="png",
                                           using_patch_for_legend=True, predictions_stat_by_metrics=False,
                                           plot_proportion_frames_in_transients=False,
                                           color_cell_as_boxplot=False, with_cells=True, with_cell_number=True)
else:
    predict_from_cinal_file = True

    if predict_from_cinal_file:
        cinac_dir_name = "/media/julien/Not_today/hne_not_today/data/cinac_ground_truth/for_benchmarks"
        results_path = "/media/julien/Not_today/hne_not_today/results_hne/"
        time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        results_path = results_path + f"{time_str}"
        os.mkdir(results_path)

        activity_predictions_from_cinac_files(cinac_dir_name, results_path,
                                              json_file_name, weights_file_name, classifier_id=classifier_id,
                                              output_file_formats="npy")
    else:

        # ############
        # Creating an instance of CinacRecording
        # this class will be use to link the calcium imaging movie and the ROIs.
        # ############

        cinac_recording = CinacRecording(identifier=identifier)

        # Creating and adding to cinac_recording the calcium imaging movie data
        cinac_movie = CinacTiffMovie(tiff_file_name=movie_file_name)

        # if you have the movie already loaded in memory (for exemple in an nwb file,
        # if not using external link), then if you could do instead:

        # cinac_movie = CinacTiffMovie(tiff_movie=tiff_movie)
        # tiff_movie being a 3d numpy array (n_frames*dim_y*dim_x)

        cinac_recording.set_movie(cinac_movie)

        """
        Adding the information regarding the ROIs to the CinacRecording instance.
        There are four options, de-comment the one you need
        """

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

        coord_file = os.path.join(root_path, "map_coords.mat")
        data = hdf5storage.loadmat(coord_file)
        coord = data["coord_python"][0]
        cinac_recording.set_rois_2d_array(coord=coord, from_matlab=True)

        # ------------------------------------------------
        # options 3: NWB (Neurodata Without Borders) data
        # ------------------------------------------------

        """
        Args:
             nwb_data: nwb object instance
                 name_module: Name of the module to find segmentation. 
                 Used this way: nwb_data.modules[name_module]
                 Ex: name_module = 'ophys'
             name_segmentation: Name of the segmentation in which find the plane segmentation.
                 Used this way:get_plane_segmentation(name_segmentation)
                 Ex: name_segmentation = 'segmentation_suite2p'
             name_seg_plane: Name of the segmentation plane in which to find the ROIs data
                 Used this way: mod[name_segmentation]get_plane_segmentation(name_seq_plane)
                 Ex: name_segmentation = 'my_plane_seg'
        
        'pixel_mask' data need to be available in the segmentation plane for it to work
        
        """
        # nwb_data = None
        # name_module = ""
        # name_segmentation = ""
        # name_seg_plane = ""
        # cinac_recording.set_rois_from_nwb(nwb_data=nwb_data, name_module=name_module,
        #                                   name_segmentation=name_segmentation, name_seg_plane=name_seg_plane)

        # -----------------------
        # options 4: Pixel masks
        # -----------------------

        """
        Args:
            pixel_masks: list of list of 2 integers representing 
                         for each cell all the pixels that belongs to the cell
        This method is actually called by set_rois_from_nwb() after extracting the 
        pixel_masks data from the nwb_file
        """

        # pixel_masks = None
        # cinac_recording.set_rois_using_pixel_mask(pixel_masks=pixel_masks)

        """
        Then we decide which network will be used for predicting the cells' activity.
        
        A dictionnary with key a tuple of 3 elements is used.
        
        The 3 elements are:
        
        (string) the model file name (.json extension)
        (string) the weights of the network file name (.h5 extension)
        (string) identifier for this configuration, will be used to name the output file
        The dictionnary will contain as value the cells to be predicted by the key configuration. 
        If the value is set to None, then all the cells will be predicted using this configuration.
        """
        # if using cell_type_predictions
        using_cell_type_predictions = False
        if using_cell_type_predictions:
            # TODO: test it on google colab
            cell_type_config_file = ""
            data = np.load(file_name, allow_pickle=True)
            cell_type_predictions = data["predictions"]
            cell_type_to_classifier = {"pyramidal": (json_file_name, weights_file_name, identifier)}
            default_classifier = (json_file_name, weights_file_name, identifier)
            model_files_dict = \
                select_activity_classifier_on_cell_type_outputs(cell_type_config_file=cell_type_config_file,
                                                                cell_type_predictions=cell_type_predictions,
                                                                cell_type_to_classifier=cell_type_to_classifier,
                                                                default_classifier=default_classifier)
            print(f"model_files_dict {model_files_dict}")
        else:
            model_files_dict = dict()
            # predicting 10 first cells with this model, weights and string identifying the network
            model_files_dict[(json_file_name, weights_file_name, identifier)] = np.arange(20)

        """
        We now create an instance of CinacPredictor and add the CinacRecording we have just created.
        
        It's possible to add more than one instance of CinacRecording, they will be predicted on the same run then.
        
        The argument removed_cells_mapping allows to remove cells from the segmentation. 
        This could be useful as the network take in consideration the adjacent cells to predict the activity, 
        thus if a cell was wrongly added to segmentation, this could lower the accuracy of the classifier.
        """

        cinac_predictor = CinacPredictor(verbose=1)

        """
        Args:
        
            removed_cells_mapping: integers array of length the original numbers of 
                cells (such as defined in CinacRecording)
                and as value either of positive int representing the new index of 
                the cell or -1 if the cell has been removed
        """

        cinac_predictor.add_recording(cinac_recording=cinac_recording,
                                      removed_cells_mapping=None,
                                      model_files_dict=model_files_dict)

        """
        Finally, we run the prediction.
        
        The output format could be either a matlab file(.mat) and/or numpy one (.npy).
        
        If matlab is chosen, the predictions will be available under the key "predictions".
        
        The predictions are a 2d float array (n_cells * n_frames) with value between 0 and 1, representing the prediction of our classifier for each frame. 1 means the cell is 100% sure active at that time, 0 is 100% sure not active.
        
        A cell is considered active during the rising time of the calcium transient.
        
        We use a threshold of 0.5 to binarize the predictions array and make it a raster.
        """

        # you could decomment this line to make sure the GPU is used
        # with tf.device('/device:GPU:0'):

        # predictions are saved in the results_path and return as a dict,
        predictions_dict = cinac_predictor.predict(results_path=results_path, output_file_formats="npy")

        # Do output_file_formats=["npy", "mat"] to get both extensions

        """
        Code to convert the predictions as binary raster
        """

        # dictionary with key identifier of the recording and value a binary 2d array
        binary_predictions_dict = dict()

        for identifier, predictions in predictions_dict.items():
            binary_predictions = np.zeros((len(predictions), len(predictions[0])),
                                          dtype="int8")
            binary_predictions[predictions > 0.5] = 1
            binary_predictions_dict[identifier] = binary_predictions
