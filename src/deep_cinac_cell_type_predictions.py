from deepcinac.cinac_predictor import *
from deepcinac.cinac_structures import create_cinac_recording_from_cinac_file_segment, CinacRecording, CinacTiffMovie
import tensorflow as tf
import numpy as np
import hdf5storage
import os
from deepcinac.utils.cinac_file_utils import CinacFileReader, read_cell_type_categories_yaml_file
from deepcinac.utils.display import plot_hist_distribution
import sklearn.metrics
from datetime import datetime


def find_weight_and_model_file_in_dir(dir_name):
    file_names = []
    for (dirpath, dirnames, local_filenames) in os.walk(dir_name):
        file_names = local_filenames
        break

    weights_files = [os.path.join(dirpath, f) for f in file_names if f.endswith(".h5")]
    cinac_files = [os.path.join(dirpath, f) for f in file_names if f.endswith(".json")]
    if len(weights_files) == 0:
        raise Exception(f"No weights file in {dir_name}")
    if len(cinac_files) == 0:
        raise Exception(f"No model file in {dir_name}")

    return weights_files[0], cinac_files[0]


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
root_path = '/media/julien/Not_today/hne_not_today/'

# path to calcium imaging data
data_path = os.path.join(root_path, "data")

# Path to your model data. It's possible to have more than one model, and use
# each for different cell of the same recording (for exemple, one
# network could be specialized for interneurons and the other one for pyramidal
# cells)
weights_file_name, json_file_name = find_weight_and_model_file_in_dir(dir_name=
                                                                      os.path.join(data_path,
                                                                                   "cinac_cell_type_ground_truth",
                                                                                   "cell_type_classifier_models",
                                                                                   "training_28_epoch_5"))
# training_17_epoch_3 multi_class_training_test colab_1_epoch_3
# cell_type_yaml_file = os.path.join(data_path, "cinac_cell_type_ground_truth", "cell_type_yaml_files",
#                                    "pyr_vs_ins_binary.yaml")
cell_type_yaml_file = os.path.join(data_path, "cinac_cell_type_ground_truth", "cell_type_yaml_files",
                                   "pyr_vs_ins_multi_class.yaml")
# cell_type_yaml_file = os.path.join(data_path, "cinac_cell_type_ground_truth", "cell_type_yaml_files",
#                                    "pyr_vs_ins_vs_noise_multi_class.yaml")

# "sunday_19_01_20_acc_87-5_epoch_4", "sunday_19_01_20_acc_90-38" sunday_19_01_20_epoch_2
# weights_file_name = os.path.join(root_path, "transient_classifier_full_model_02-0.9883.h5")
# json_file_name = os.path.join(root_path, "transient_classifier_model_architecture_.json")

# path of the directory where the results will be save
# a directory will be created each time the prediction is run
# the directory name will be the date and time at which the analysis has been run
# the predictions will be in this directory.
results_path = os.path.join(root_path, "results_hne")
time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
results_path = os.path.join(results_path, f"{time_str}/")
os.mkdir(results_path)

# not mandatory, just to test if you GPU is accessible
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

with_cinac_file = True

if with_cinac_file:
    cinac_dir_name = os.path.join(data_path, "cinac_cell_type_ground_truth/for_testing")

    cell_type_from_code_dict, cell_type_to_code_dict, multi_class_arg = \
        read_cell_type_categories_yaml_file(yaml_file=cell_type_yaml_file, using_multi_class=1)

    print(f"### cell_type_from_code_dict {cell_type_from_code_dict}")

    n_cell_categories = len(cell_type_from_code_dict)

    if n_cell_categories < 2:
        raise Exception(f"You need at least 2 cell_type categories, you provided {n_cell_categories}: "
                        f"{list(cell_type_from_code_dict.values())}")

    # if n_class_for_predictions is equal 1, it means we are building a binary classifier
    # otherwise a multi class one

    if multi_class_arg is not None:
        if multi_class_arg or (n_cell_categories > 2):
            n_class_for_predictions = n_cell_categories
        else:
            n_class_for_predictions = 1
    else:
        if n_cell_categories > 2:
            n_class_for_predictions = n_cell_categories
        else:
            n_class_for_predictions = 1

    cinac_path_w_file_names = []
    frames_to_keep_dict = None
    cinac_file_names = []
    text_file = None
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    # used for multiclass
    more_than_one_cell_type = 0
    none_of_cell_types = 0
    # for sklearn metrics
    # from https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
    y_true = []
    y_pred = []
    # each key is the code representing the cell_type (see in cell_type_from_code_dict), each value is an int
    # representing the number of cell by type
    n_cells_by_type_dict = dict()
    cell_type_predictions_dict = dict()
    n_pyr_cell = 0
    n_in_cell = 0
    pyr_predictions = []
    ins_predictions = []
    # look for filenames in the fisrst directory, if we don't break, it will go through all directories
    for (dirpath, dirnames, local_filenames) in os.walk(cinac_dir_name):
        cinac_path_w_file_names = [os.path.join(dirpath, f) for f in local_filenames if f.endswith(".cinac")]
        cinac_file_names = [f for f in local_filenames if f.endswith(".cinac")]
        break
    for file_index, cinac_file_name in enumerate(cinac_path_w_file_names):
        cinac_file_reader = CinacFileReader(file_name=cinac_file_name)
        # cinac_movie = get_cinac_movie_from_cinac_file_reader(cinac_file_reader)
        segments_list = cinac_file_reader.get_all_segments()
        # identifier = os.path.basename(cinac_file_name)
        identifier = cinac_file_names[file_index]
        for segment in segments_list:
            cinac_recording = create_cinac_recording_from_cinac_file_segment(identifier=identifier,
                                                                             cinac_file_reader=cinac_file_reader,
                                                                             segment=segment)

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

            model_files_dict = dict()
            # we just one to predict the cell of interest, which is the first cell
            model_files_dict[(json_file_name, weights_file_name, identifier)] = np.arange(1)

            """
            We now create an instance of CinacPredictor and add the CinacRecording we have just created.
        
            It's possible to add more than one instance of CinacRecording, they will be predicted on the same run then.
        
            The argument removed_cells_mapping allows to remove cells from the segmentation. 
            This could be useful as the network take in consideration the adjacent cells to predict the activity, 
            thus if a cell was wrongly added to segmentation, this could lower the accuracy of the classifier.
            """

            cinac_predictor = CinacPredictor()

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
            predictions_dict = cinac_predictor.predict(results_path=results_path, output_file_formats="npy",
                                                       overlap_value=0, cell_type_classifier_mode=True,
                                                       n_segments_to_use_for_prediction=4,
                                                       cell_type_pred_fct=np.mean,
                                                       create_dir_for_results=False)
            # mean of the different predictions is already in the dict
            prediction_value = predictions_dict[list(predictions_dict.keys())[0]][0]

            right_prediction = False

            if cinac_recording.cell_type.strip().lower() not in cell_type_to_code_dict:
                print(f"cinac_recording.cell_type ({cinac_recording.cell_type}) not in cell_type_to_code_dict")
                continue
            cell_type_code = cell_type_to_code_dict[cinac_recording.cell_type.strip().lower()]
            if cell_type_code not in n_cells_by_type_dict:
                n_cells_by_type_dict[cell_type_code] = 0
                cell_type_predictions_dict[cell_type_code] = []
            n_cells_by_type_dict[cell_type_code] += 1
            if n_class_for_predictions == 1:
                cell_type_predictions_dict[cell_type_code].append(prediction_value)
                if cell_type_code == 0:
                    if prediction_value >= 0.5:
                        fp += 1
                    else:
                        tn += 1
                        right_prediction = True
                else:
                    if prediction_value >= 0.5:
                        tp += 1
                        right_prediction = True
                    else:
                        fn += 1
            else:
                if len(prediction_value) != n_class_for_predictions:
                    print("len(prediction_value) != n_class_for_predictions")
                    continue

                cell_type_predictions_dict[cell_type_code].append(prediction_value[cell_type_code])

                if len(np.where(prediction_value < 0.5)[0]) == len(prediction_value):
                    # means all values are under the 0.5 threshold, and the cell doesn't belong to any cell type
                    none_of_cell_types += 1
                if len(np.where(prediction_value >= 0.5)[0]) >= 2:
                    # means at least 2 values are over the 0.5 threshold,
                    # and the cell belong to more than one cell type
                    more_than_one_cell_type += 1

                # we want to use sklearn metrics for multiclass metrics
                y_true.append(cell_type_from_code_dict[cell_type_code])
                predicted_cell_type_code = np.argmax(prediction_value)
                y_pred.append(cell_type_from_code_dict[predicted_cell_type_code])
                if cell_type_code == predicted_cell_type_code:
                    right_prediction = True
                # if np.argmax(prediction_value) == cell_type_code:
                #     tp += 1
                # else:
                #     fp += 1
            print(f"Cell {segment[0]} [{cinac_recording.cell_type.strip().lower()}] from {identifier}, predictions: "
                  f"{prediction_value} {right_prediction}")
    if n_class_for_predictions == 1:
        # metrics:
        if (tp + fn) > 0:
            sensitivity = tp / (tp + fn)
        else:
            sensitivity = 1

        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 1

        if (tp + tn + fp + fn) > 0:
            accuracy = (tp + tn) / (tp + tn + fp + fn)
        else:
            accuracy = 1

        if (tp + fp) > 0:
            ppv = tp / (tp + fp)
        else:
            ppv = 1

        if (tn + fn) > 0:
            npv = tn / (tn + fn)
        else:
            npv = 1

    print("")
    print("-" * 100)
    metrics_title_str = "METRICS for "
    for cell_type_code, n_cells_by_type in n_cells_by_type_dict.items():
        cell_type_str = cell_type_from_code_dict[cell_type_code]
        metrics_title_str = metrics_title_str + f"{n_cells_by_type} {cell_type_str}, "
    print(f"{metrics_title_str[:-2]}")
    # print(f"METRICS for {n_pyr_cell} pyramidal cells and {n_in_cell} interneurons")
    if n_class_for_predictions == 1:
        print("-" * 100)
        print(f"Accuracy {accuracy}")
        print(f"Sensitivity {sensitivity}")
        print(f"Specificity {specificity}")
        print(f"PPV {ppv}")
        print(f"NPV {npv}")
        print("-" * 100)
    else:
        print(f"N cell classified has none of the cell types: {none_of_cell_types}")
        print(f"N cell classified has more than one cell type: {more_than_one_cell_type}")
        # Print the confusion matrix
        print("--- confusion matrix ---")
        print(sklearn.metrics.confusion_matrix(y_true, y_pred))

        # Print the precision and recall, among other metrics
        print(sklearn.metrics.classification_report(y_true, y_pred, digits=3))

    data_hist_dict = dict()
    for cell_type_code, cell_type_predictions in cell_type_predictions_dict.items():
        cell_type_str = cell_type_from_code_dict[cell_type_code]
        data_hist_dict[cell_type_str] = cell_type_predictions
        print(f"For {cell_type_str}, mean {np.round(np.mean(cell_type_predictions), 2)}, "
              f"std {np.round(np.std(cell_type_predictions), 2)}, "
              f"min {np.round(np.min(cell_type_predictions), 2)}, "
              f"max {np.round(np.max(cell_type_predictions), 2)}")
        plot_hist_distribution(distribution_data=[p*100 for p in cell_type_predictions],
                               description=f"hist_prediction_distribution_{cell_type_str}",
                               path_results=results_path,
                               tight_x_range=False,
                               twice_more_bins=True,
                               xlabel=f"{cell_type_str}",
                               save_formats="png")
    print("-" * 100)

else:
    movie_file_name = os.path.join(data_path, "p1_artificial_1.tif")
    # string used to identify the recording from which you want to predict activity
    identifier = "art_movie_1"
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

    # coord_file = os.path.join(root_path, "map_coords.mat")
    # data = hdf5storage.loadmat(coord_file)
    # coord = data["coord_python"][0]
    # cinac_recording.set_rois_2d_array(coord=coord, from_matlab=True)

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

    cinac_predictor = CinacPredictor()

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
    predictions_dict = cinac_predictor.predict(results_path=results_path, output_file_formats="npy",
                                               overlap_value=0, cell_type_classifier_mode=True)

    print(f"predictions_dict {predictions_dict}")
