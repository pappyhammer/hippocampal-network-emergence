from mouse_session import MouseSession
import numpy as np


def load_lexi_mouse_sessions(ms_str_to_load, param, load_traces,
                        for_cell_classifier=False, for_transient_classifier=False):
    """

    :param ms_str_to_load:  list of string, string being the id of a session
    :param param: instance of HNEParameters
    :param load_traces: if traces should be loaded in memory
    :param load_abf: loading abf and processing it
    :param for_cell_classifier: if sessions are loaded for the cell classifier
    :param for_transient_classifier: if sessions are loaded for the transient classifier
    :return: a dict of instances of MouseSession with keys being the string id
    """
    # for_cell_classifier is True means we don't remove the cell that has been marked as fake cells
    ms_str_to_ms_dict = dict()

    # exemple

    if "p12_171110_a000_ms" in ms_str_to_load:
        try_suite_2p = False
        p12_171110_a000_ms = MouseSession(age=12, session_id="17_11_10_a000", nb_ms_by_frame=100, param=param,
                                          weight=7)
        # calculated with 99th percentile on raster dur
        # p12_171110_a000_ms.activity_threshold = 10
        # p12_171110_a000_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
        # p12_171110_a000_ms.set_inter_neurons([106, 144])
        # duration of those interneurons: 18.29  14.4
        # variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
        #                      "spike_nums": "filt_Bin100ms_spikedigital",
        #                      "spike_durations": "LOC3"}

        p12_171110_a000_ms.load_tif_movie(path="p12/p12_17_11_10_a000/")

        variables_mapping = {"global_roi": "global_roi"}
        p12_171110_a000_ms.load_data_from_file(file_name_to_load=
                                                "p12/p12_17_11_10_a000/p12_17_11_10_a000_global_roi.mat",
                                                variables_mapping=variables_mapping)
        # caiman version
        # variables_mapping = {"spike_nums_dur": "corrected_rasterdur"} # rasterdur before
        # p12_171110_a000_ms.load_data_from_file(file_name_to_load=
        #                                          "p12/p12_17_11_10_a000/p12_17_11_10_a000_RasterDur.mat",
        #                                          variables_mapping=variables_mapping)

        variables_mapping = {"xshifts": "xshifts",
                             "yshifts": "yshifts"}
        p12_171110_a000_ms.load_data_from_file(file_name_to_load=
                                                "p12/p12_17_11_10_a000/p12_17_11_10_a000_params.mat",
                                                variables_mapping=variables_mapping)

        if not try_suite_2p:
            if for_cell_classifier or for_transient_classifier:
                variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
                                     "peak_nums": "LocPeakMatrix_Python",
                                     "cells_to_remove": "cells_to_remove",
                                     "inter_neurons_from_gui": "inter_neurons",
                                     "doubtful_frames_nums": "doubtful_frames_nums"}
                p12_171110_a000_ms.load_data_from_file(file_name_to_load=
                                                       "p12/p12_17_11_10_a000/p12_17_11_10_a000_GUI_fusion_validation.mat",
                                                       variables_mapping=variables_mapping,
                                                       from_gui=True)
                # keeping JD gui selection for test, then using: p12_17_11_10_a000_GUI_JD.mat
                p12_171110_a000_ms.build_spike_nums_dur()
                if for_cell_classifier:
                    p12_171110_a000_ms.load_cells_to_remove_from_txt(file_name="p12/p12_17_11_10_a000/"
                                                                           "p12_17_11_10_a000_cell_to_suppress_ground_truth.txt")
            else:
                pass
                # prediction based on rnn trained on 50 cells, BO
                # variables_mapping = {"predictions": "predictions"}
                # p12_171110_a000_ms.load_raster_dur_from_predictions(
                #     file_name="p12/p12_17_11_10_a000/" +
                #               "P12_17_11_10_a000_predictions_2019_03_14.20-19-48.mat",
                #     prediction_threshold=0.5, variables_mapping=variables_mapping)

                # if p12_171110_a000_ms.cell_cnn_predictions is not None:
                #     print(f"Using cnn predictions from {p12_171110_a000_ms.description}")
                #     # not taking into consideration cells that are not predicted as true from the cell classifier
                #     cells_predicted_as_false = np.where(p12_171110_a000_ms.cell_cnn_predictions < 0.5)[0]
                #     if p12_171110_a000_ms.cells_to_remove is None:
                #         p12_171110_a000_ms.cells_to_remove = cells_predicted_as_false
                #     else:
                #         p12_171110_a000_ms.cells_to_remove = np.concatenate((p12_171110_a000_ms.cells_to_remove,
                #                                                                cells_predicted_as_false))
                # p12_171110_a000_ms.load_cells_to_remove_from_txt(file_name="p12/p12_17_11_10_a000/"
                #                                                            "p12_17_11_10_a000_cell_to_suppress_ground_truth.txt")
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p12_171110_a000_ms.load_data_from_file(
                    file_name_to_load="p12/p12_17_11_10_a000/p12_17_11_10_a000_Traces.mat",
                    variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p12_171110_a000_ms.load_data_from_file(
                    file_name_to_load="p12/p12_17_11_10_a000/p12_17_11_10_a000_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p12_171110_a000_ms.load_data_from_file(
                file_name_to_load="p12/p12_17_11_10_a000/p12_17_11_10_a000_CellDetect.mat",
                variables_mapping=variables_mapping)
        p12_171110_a000_ms.set_avg_cell_map_tif(file_name="p12/p12_17_11_10_a000/AVG_p12_17_11_10_a000.tif")

        if not try_suite_2p:
            if not for_cell_classifier:
                p12_171110_a000_ms.clean_data_using_cells_to_remove()

        p12_171110_a000_ms.load_suite2p_data(data_path="p12/p12_17_11_10_a000/suite2p/", with_coord=try_suite_2p)

        p12_171110_a000_ms.clean_raster_at_concatenation()

        # p12_171110_a000_ms.load_caiman_results(path_data="p12/p12_17_11_10_a000/")

        ms_str_to_ms_dict["p12_171110_a000_ms"] = p12_171110_a000_ms

    return ms_str_to_ms_dict