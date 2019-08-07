from mouse_session import MouseSession
import numpy as np
from pattern_discovery.tools.signal import smooth_convolve


def smooth_traces(traces):
    # smoothing the trace
    windows = ['hanning', 'hamming', 'bartlett', 'blackman']
    i_w = 1
    window_length = 11
    for i in np.arange(traces.shape[0]):
        smooth_signal = smooth_convolve(x=traces[i], window_len=window_length,
                                        window=windows[i_w])
        beg = (window_length - 1) // 2
        traces[i] = smooth_signal[beg:-beg]


def load_mouse_sessions(ms_str_to_load, param, load_traces, load_abf=True, load_movie=True,
                        for_cell_classifier=False, for_transient_classifier=False):
    # for_cell_classifier is True means we don't remove the cell that has been marked as fake cells
    ms_str_to_ms_dict = dict()

    if "artificial_ms_1" in ms_str_to_load:
        artificial_ms = MouseSession(age=1, session_id="artificial_1", param=param)
        variables_mapping = {"coord": "coord_python"}
        artificial_ms.load_data_from_file(file_name_to_load="artificial_movies/1/map_coords.mat",
                                          variables_mapping=variables_mapping)
        artificial_ms.load_tif_movie(path="artificial_movies/1/")

        if for_cell_classifier or for_transient_classifier:
            variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
                                 "peak_nums": "LocPeakMatrix_Python"}
            artificial_ms.load_data_from_file(file_name_to_load=
                                              "artificial_movies/1/gui_data.mat",
                                              variables_mapping=variables_mapping,
                                              from_gui=True)

            artificial_ms.build_spike_nums_dur()
            artificial_ms.load_tiff_movie_in_memory()
            # artificial_ms.normalize_movie()
            artificial_ms.raw_traces = artificial_ms.build_raw_traces_from_movie()
            traces = np.copy(artificial_ms.raw_traces)
            smooth_traces(traces)
            artificial_ms.traces = traces
            artificial_ms.smooth_traces = traces

        ms_str_to_ms_dict["artificial_ms_1"] = artificial_ms

    if "artificial_ms_2" in ms_str_to_load:
        artificial_ms = MouseSession(age=2, session_id="artificial_2", param=param)
        variables_mapping = {"coord": "coord_python"}
        artificial_ms.load_data_from_file(file_name_to_load="artificial_movies/2_suite2p_contours/map_coords.mat",
                                          variables_mapping=variables_mapping)
        artificial_ms.load_tif_movie(path="artificial_movies/2_suite2p_contours/")

        if for_cell_classifier or for_transient_classifier:
            variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
                                 "peak_nums": "LocPeakMatrix_Python"}
            artificial_ms.load_data_from_file(file_name_to_load=
                                              "artificial_movies/2_suite2p_contours/gui_data.mat",
                                              variables_mapping=variables_mapping,
                                              from_gui=True)

            artificial_ms.build_spike_nums_dur()
            artificial_ms.load_tiff_movie_in_memory()
            # artificial_ms.normalize_movie()
            artificial_ms.raw_traces = artificial_ms.build_raw_traces_from_movie()
            traces = np.copy(artificial_ms.raw_traces)
            smooth_traces(traces)
            artificial_ms.traces = traces
            artificial_ms.smooth_traces = traces

        ms_str_to_ms_dict["artificial_ms_2"] = artificial_ms
    if "artificial_ms_3" in ms_str_to_load:
        artificial_ms = MouseSession(age=3, session_id="artificial_3", param=param)
        variables_mapping = {"coord": "coord_python"}
        artificial_ms.load_data_from_file(file_name_to_load="artificial_movies/3_suite2p_contours/map_coords.mat",
                                          variables_mapping=variables_mapping)
        artificial_ms.load_tif_movie(path="artificial_movies/3_suite2p_contours/")

        if for_cell_classifier or for_transient_classifier:
            variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
                                 "peak_nums": "LocPeakMatrix_Python"}
            artificial_ms.load_data_from_file(file_name_to_load=
                                              "artificial_movies/3_suite2p_contours/gui_data.mat",
                                              variables_mapping=variables_mapping,
                                              from_gui=True)

            artificial_ms.build_spike_nums_dur()
            artificial_ms.load_tiff_movie_in_memory()
            # artificial_ms.normalize_movie()
            artificial_ms.raw_traces = artificial_ms.build_raw_traces_from_movie()
            traces = np.copy(artificial_ms.raw_traces)
            smooth_traces(traces)
            artificial_ms.traces = traces
            artificial_ms.smooth_traces = traces

        ms_str_to_ms_dict["artificial_ms_3"] = artificial_ms

    # GAD-CRE
    if "p5_19_03_20_a000_ms" in ms_str_to_load:
        p5_19_03_20_a000_ms = MouseSession(age=5, session_id="19_03_20_a000",
                                           sampling_rate=8, param=param, weight=3.75)
        p5_19_03_20_a000_ms.use_suite_2p = False

        if not p5_19_03_20_a000_ms.use_suite_2p:
            variables_mapping = {"coord": "ContoursAll"}
            p5_19_03_20_a000_ms.load_data_from_file(
                file_name_to_load="p5/p5_19_03_20_a000/p5_19_03_20_a000_CellDetect_fiji.mat",
                variables_mapping=variables_mapping, from_fiji=True)

        ms_str_to_ms_dict["p5_19_03_20_a000_ms"] = p5_19_03_20_a000_ms

    if "p5_19_03_25_a000_ms" in ms_str_to_load:
        p5_19_03_25_a000_ms = MouseSession(age=5, session_id="19_03_25_a000",
                                           sampling_rate=8, param=param, weight=3.75)
        p5_19_03_25_a000_ms.use_suite_2p = True
        # p5_19_03_25_a000_ms.z_shift_periods = [(1666, 1728), (4111, 4192), (6723, 6850), (
        #         8454, 8494), (8988, 9089)]
        # for threshold prediction at 0.5
        # p5_19_03_25_a000_ms.activity_threshold = 12  # 1.6%

        # prediction based on rnn trained on 50 cells, BO,
        # variables_mapping = {"predictions": "predictions"}
        # p5_19_03_25_a000_ms.load_raster_dur_from_predictions(
        #     file_name="p5/p5_19_03_25_a000/predictions/" +
        #               ".mat",
        #     prediction_threshold=0.5, variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p5_19_03_25_a000_ms"] = p5_19_03_25_a000_ms

    if "p5_19_03_25_a001_ms" in ms_str_to_load:
        p5_19_03_25_a001_ms = MouseSession(age=5, session_id="19_03_25_a001",
                                           sampling_rate=8, param=param, weight=3.75)
        p5_19_03_25_a001_ms.use_suite_2p = True
        # calculated with 95th percentile on raster dur
        p5_19_03_25_a001_ms.activity_threshold = 13
        p5_19_03_25_a001_ms.z_shift_periods = [(295, 350),
                                               (10252, 10308)]
        # prediction based on rnn trained on 50 cells, BO,
        # variables_mapping = {"predictions": "predictions"}
        # p5_19_03_25_a001_ms.load_raster_dur_from_predictions(
        #     file_name="p5/p5_19_03_25_a001/predictions/" +
        #               "P5_19_03_25_a001_predictions__2019_05_08.01-58-24_GT_epoch11_all_cells.mat",
        #     prediction_threshold=0.5, variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p5_19_03_25_a001_ms"] = p5_19_03_25_a001_ms

    # GAD CRE
    # if "p5_19_03_20_a000_ms" in ms_str_to_load:
    #     p5_19_03_20_a000_ms = MouseSession(age=5, session_id="19_03_20_a000", sampling_rate=8,
    #                                        param=param, weight=3.9)
    #
    #     variables_mapping = {"global_roi": "global_roi"}
    #     p5_19_03_20_a000_ms.load_data_from_file(file_name_to_load=
    #                                             "p5/p5_19_03_20_a000/p5_19_03_20_a000_global_roi.mat",
    #                                             variables_mapping=variables_mapping)
    #     variables_mapping = {"xshifts": "xshifts",
    #                          "yshifts": "yshifts"}
    #     p5_19_03_20_a000_ms.load_data_from_file(file_name_to_load=
    #                                             "p5/p5_19_03_20_a000/MichelMotC_p5_19_03_20_a000_params.mat",
    #                                             variables_mapping=variables_mapping)
    #     if load_movie:
    #         p5_19_03_20_a000_ms.load_tif_movie(path="p5/p5_19_03_20_a000")

        # ms_str_to_ms_dict["p5_19_03_20_a000_ms"] = p5_19_03_20_a000_ms

    if "p6_18_02_07_a001_ms" in ms_str_to_load:
        p6_18_02_07_a001_ms = MouseSession(age=6, session_id="18_02_07_a001", param=param,
                                           sampling_rate=10, weight=4.35)
        p6_18_02_07_a001_ms.use_suite_2p = True

        # p6_18_02_07_a001_ms.z_shift_periods = [(456, 488),
        #                                        (1555, 1624),
        #                                        (2365, 2400),
        #                                        (2695, 2780),
        #                                        (2840, 2904),
        #                                        (3119, 3275),
        #                                        (4098, 4268),
        #                                        (4791, 5000),
        #                                        (5380, 5570),
        #                                        (5644, 5822),
        #                                        (6386, 6417),
        #                                        (6428, 6481),
        #                                        (6539, 6664),
        #                                        (7138, 7416),
        #                                        (7722, 7823),
        #                                        (8692, 8799),
        #                                        (8880, 9025)]

        # calculated with 99th percentile on raster dur
        # p6_18_02_07_a001_ms.activity_threshold = 15

        if not p6_18_02_07_a001_ms.use_suite_2p:
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p6_18_02_07_a001_ms.load_data_from_file(file_name_to_load=
                                                    "p6/p6_18_02_07_a001/caiman/p6_18_02_07_001_Corrected_RasterDur.mat",
                                                    variables_mapping=variables_mapping)
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p6_18_02_07_a001_ms.load_data_from_file(
                    file_name_to_load="p6/p6_18_02_07_a001/caiman/p6_18_02_07_a001_Traces.mat",
                    variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p6_18_02_07_a001_ms.load_data_from_file(
                    file_name_to_load="p6/p6_18_02_07_a001/caiman/p6_18_02_07_a001_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p6_18_02_07_a001_ms.load_data_from_file(
                file_name_to_load="p6/p6_18_02_07_a001/caiman/p6_18_02_07_a001_CellDetect.mat",
                variables_mapping=variables_mapping)

        # variables_mapping = {"predictions": "predictions"}
        # p6_18_02_07_a001_ms.load_raster_dur_from_predictions(
        #     file_name="p6/p6_18_02_07_a001/predictions/" +
        #               "P6_18_02_07_a001_predictions__2019_05_07.17-26-16_GT_epoch_11_all_cells.mat",
        #     prediction_threshold=0.5, variables_mapping=variables_mapping)

        if not p6_18_02_07_a001_ms.use_suite_2p:
            if not for_cell_classifier:
                p6_18_02_07_a001_ms.clean_data_using_cells_to_remove()

        ms_str_to_ms_dict["p6_18_02_07_a001_ms"] = p6_18_02_07_a001_ms

    if "p6_18_02_07_a002_ms" in ms_str_to_load:
        p6_18_02_07_a002_ms = MouseSession(age=6, session_id="18_02_07_a002", sampling_rate=10, param=param,
                                           weight=4.35)
        p6_18_02_07_a002_ms.use_suite_2p = True
        # calculated with 99th percentile on raster dur
        # p6_18_02_07_a002_ms.activity_threshold =

        if not p6_18_02_07_a002_ms.use_suite_2p:
            variables_mapping = {"spike_nums_dur": "rasterdur"}
            p6_18_02_07_a002_ms.load_data_from_file(file_name_to_load=
                                                    "p6/p6_18_02_07_a002/p6_18_02_07_a002_RasterDur_2nd_dec.mat",
                                                    variables_mapping=variables_mapping)
            p6_18_02_07_a002_ms.spike_struct.build_spike_nums_and_peak_nums()

            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p6_18_02_07_a002_ms.load_data_from_file(file_name_to_load="p6/p6_18_02_07_a002/p6_18_02_07_a002_Traces.mat",
                                                        variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p6_18_02_07_a002_ms.load_data_from_file(
                    file_name_to_load="p6/p6_18_02_07_a002/p6_18_02_07_a002_raw_Traces.mat",
                    variables_mapping=variables_mapping)

            variables_mapping = {"coord": "ContoursAll"}
            p6_18_02_07_a002_ms.load_data_from_file(file_name_to_load="p6/p6_18_02_07_a002/p6_18_02_07_a002_CellDetect.mat",
                                                    variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p6_18_02_07_a002_ms"] = p6_18_02_07_a002_ms

    # GAD-Cre
    if "p6_19_02_18_a000_ms" in ms_str_to_load:
        p6_19_02_18_a000_ms = MouseSession(age=6, session_id="19_02_18_a000", sampling_rate=8, param=param,
                                           weight=None)
        p6_19_02_18_a000_ms.use_suite_2p = False
        if for_transient_classifier:
            p6_19_02_18_a000_ms.use_suite_2p = False
        # calculated with 99th percentile on raster dur
        # p6_19_02_18_a000_ms.activity_threshold = 2

        if not p6_19_02_18_a000_ms.use_suite_2p:
            if for_transient_classifier:
                variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
                                     "peak_nums": "LocPeakMatrix_Python",
                                     "cells_to_remove": "cells_to_remove",
                                     "inter_neurons_from_gui": "inter_neurons",
                                     "doubtful_frames_nums": "doubtful_frames_nums"}
                p6_19_02_18_a000_ms.load_data_from_file(file_name_to_load=
                                                         "p6/p6_19_02_18_a000/p6_19_02_18_a000_ground_truth_cell_0_1_2_3.mat",
                                                         variables_mapping=variables_mapping,
                                                         from_gui=True)
                p6_19_02_18_a000_ms.build_spike_nums_dur()

            variables_mapping = {"coord": "ContoursAll"}
            p6_19_02_18_a000_ms.load_data_from_file(file_name_to_load="p6/p6_19_02_18_a000/p6_19_02_18_a000_CellDetect_fiji.mat",
                                                    variables_mapping=variables_mapping, from_fiji=True)

            p6_19_02_18_a000_ms.load_tif_movie(path="p6/p6_19_02_18_a000/")
            # p6_19_02_18_a000_ms.load_caiman_results(path_data="p6/p6_19_02_18_a000/")
            # p6_19_02_18_a000_ms.spike_struct.spike_nums = p6_19_02_18_a000_ms.caiman_spike_nums
            # p6_19_02_18_a000_ms.spike_struct.spike_nums_dur = p6_19_02_18_a000_ms.caiman_spike_nums_dur
            # p6_19_02_18_a000_ms.spike_struct.n_cells = len(p6_19_02_18_a000_ms.caiman_spike_nums_dur)
            # p6_19_02_18_a000_ms.spike_struct.labels = np.arange(p6_19_02_18_a000_ms.spike_struct.n_cells)

        ms_str_to_ms_dict["p6_19_02_18_a000_ms"] = p6_19_02_18_a000_ms

    if "p7_171012_a000_ms" in ms_str_to_load:
        p7_171012_a000_ms = MouseSession(age=7, session_id="17_10_12_a000", sampling_rate=10, param=param,
                                         weight=None)
        p7_171012_a000_ms.use_suite_2p = True
        if for_transient_classifier or for_cell_classifier:
            p7_171012_a000_ms.use_suite_2p = False

        # calculated with 99th percentile on raster dur
        # p7_171012_a000_ms.activity_threshold =

        if not p7_171012_a000_ms.use_suite_2p:
            variables_mapping = {"coord": "ContoursAll"}
            p7_171012_a000_ms.load_data_from_file(
                file_name_to_load="p7/p7_17_10_12_a000/p7_17_10_12_a000_CellDetect.mat",
                variables_mapping=variables_mapping)
            if for_cell_classifier or for_transient_classifier:
                variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
                                     "peak_nums": "LocPeakMatrix_Python",
                                     "cells_to_remove": "cells_to_remove",
                                     "inter_neurons_from_gui": "inter_neurons",
                                     "doubtful_frames_nums": "doubtful_frames_nums"}
                p7_171012_a000_ms.load_data_from_file(file_name_to_load=
                                                      "p7/p7_17_10_12_a000/p7_17_10_12_a000_fusion_validation.mat",
                                                      variables_mapping=variables_mapping,
                                                      from_gui=True)
                # p7_17_10_12_a000_fusion_validation.mat

                p7_171012_a000_ms.build_spike_nums_dur()
                if for_cell_classifier:
                    p7_171012_a000_ms. \
                        load_cells_to_remove_from_txt(file_name="p7/p7_17_10_12_a000/"
                                                                "p7_17_10_12_a000_cell_to_suppress_ground_truth.txt")
            # else:
            #     variables_mapping = {"predictions": "predictions"}
            #     p7_171012_a000_ms.load_raster_dur_from_predictions(
            #         file_name="p7/p7_17_10_12_a000/predictions/" +
            #                   "P7_17_10_12_a000_predictions__2019_04_30.23-32-43_epoch_11_no_overlap_no_trans.mat",
            #         prediction_threshold=0.5, variables_mapping=variables_mapping)
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p7_171012_a000_ms.load_data_from_file(
                    file_name_to_load="p7/p7_17_10_12_a000/p7_17_10_12_a000_Traces.mat",
                    variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p7_171012_a000_ms.load_data_from_file(
                    file_name_to_load="p7/p7_17_10_12_a000/p7_17_10_12_a000_raw_Traces.mat",
                    variables_mapping=variables_mapping)

        if not p7_171012_a000_ms.use_suite_2p:
            if not for_cell_classifier:
                p7_171012_a000_ms.clean_data_using_cells_to_remove()

        ms_str_to_ms_dict["p7_171012_a000_ms"] = p7_171012_a000_ms

    if "p7_17_10_18_a002_ms" in ms_str_to_load:
        p7_17_10_18_a002_ms = MouseSession(age=7, session_id="17_10_18_a002", sampling_rate=10, param=param,
                                           weight=None)
        p7_17_10_18_a002_ms.use_suite_2p = True
        # calculated with 99th percentile on raster dur
        # p7_17_10_18_a002_ms.activity_threshold =

        if not p7_17_10_18_a002_ms.use_suite_2p:
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p7_17_10_18_a002_ms.load_data_from_file(file_name_to_load=
                                                    "p7/p7_17_10_18_a002/p7_17_10_18_a002_Corrected_RasterDur.mat",
                                                    variables_mapping=variables_mapping)
            if load_traces:
                variables_mapping = {"raw_traces": "raw_traces"}
                p7_17_10_18_a002_ms.load_data_from_file(
                    file_name_to_load="p7/p7_17_10_18_a002/p7_17_10_18_a002_raw_traces.mat",
                    variables_mapping=variables_mapping)
                variables_mapping = {"traces": "C_df"}
                p7_17_10_18_a002_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_18_a002/p7_17_10_18_a002_Traces.mat",
                                                        variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p7_17_10_18_a002_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_18_a002/p7_17_10_18_a002_CellDetect.mat",
                                                    variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p7_17_10_18_a002_ms"] = p7_17_10_18_a002_ms

    if "p7_17_10_18_a004_ms" in ms_str_to_load:
        p7_17_10_18_a004_ms = MouseSession(age=7, session_id="17_10_18_a004", sampling_rate=10, param=param,
                                           weight=None)
        p7_17_10_18_a004_ms.use_suite_2p = True

        # calculated with 99th percentile on raster dur
        # p7_17_10_18_a004_ms.activity_threshold = 13
        if not p7_17_10_18_a004_ms.use_suite_2p:
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p7_17_10_18_a004_ms.load_data_from_file(file_name_to_load=
                                                    "p7/p7_17_10_18_a004/p7_17_10_18_a004_Corrected_RasterDur.mat",
                                                    variables_mapping=variables_mapping)
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p7_17_10_18_a004_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_18_a004/p7_17_10_18_a004_Traces.mat",
                                                        variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p7_17_10_18_a004_ms.load_data_from_file(
                    file_name_to_load="p7/p7_17_10_18_a004/p7_17_10_18_a004_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p7_17_10_18_a004_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_18_a004/p7_17_10_18_a004_CellDetect.mat",
                                                    variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p7_17_10_18_a004_ms"] = p7_17_10_18_a004_ms

    if "p7_18_02_08_a000_ms" in ms_str_to_load:
        p7_18_02_08_a000_ms = MouseSession(age=7, session_id="18_02_08_a000", sampling_rate=10, param=param,
                                           weight=3.85)
        p7_18_02_08_a000_ms.use_suite_2p = True
        # calculated with 99th percentile on raster dur
        # p7_18_02_08_a000_ms.activity_threshold =

        if not p7_18_02_08_a000_ms.use_suite_2p:
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p7_18_02_08_a000_ms.load_data_from_file(file_name_to_load=
                                                    "p7/p7_18_02_08_a000/p7_18_02_18_a000_Corrected_RasterDur.mat",
                                                    variables_mapping=variables_mapping)
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p7_18_02_08_a000_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a000/p7_18_02_08_a000_Traces.mat",
                                                        variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p7_18_02_08_a000_ms.load_data_from_file(
                    file_name_to_load="p7/p7_18_02_08_a000/p7_18_02_08_a000_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p7_18_02_08_a000_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a000/p7_18_02_08_a000_CellDetect.mat",
                                                    variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p7_18_02_08_a000_ms"] = p7_18_02_08_a000_ms

    if "p7_18_02_08_a001_ms" in ms_str_to_load:
        p7_18_02_08_a001_ms = MouseSession(age=7, session_id="18_02_08_a001", sampling_rate=10, param=param,
                                           weight=3.85)
        p7_18_02_08_a001_ms.use_suite_2p = True
        # calculated with 99th percentile on raster dur
        # p7_18_02_08_a001_ms.activity_threshold =

        if not p7_18_02_08_a001_ms.use_suite_2p:
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p7_18_02_08_a001_ms.load_data_from_file(file_name_to_load=
                                                    "p7/p7_18_02_08_a001/p7_18_02_18_a001_Corrected_RasterDur.mat",
                                                    variables_mapping=variables_mapping)
            p7_18_02_08_a001_ms.set_avg_cell_map_tif(file_name="p7/p7_18_02_08_a001/AVG_p7_18_02_08_a001.tif")
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p7_18_02_08_a001_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a001/p7_18_02_08_a001_Traces.mat",
                                                        variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p7_18_02_08_a001_ms.load_data_from_file(
                    file_name_to_load="p7/p7_18_02_08_a001/p7_18_02_08_a001_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            # variables_mapping = {"coord": "ContoursAll"}
            # p7_18_02_08_a001_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a001/p7_18_02_08_a001_CellDetect.mat",
            #                                         variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p7_18_02_08_a001_ms"] = p7_18_02_08_a001_ms

    if "p7_18_02_08_a002_ms" in ms_str_to_load:
        p7_18_02_08_a002_ms = MouseSession(age=7, session_id="18_02_08_a002", sampling_rate=10, param=param,
                                           weight=3.85)
        p7_18_02_08_a002_ms.use_suite_2p = True
        # calculated with 99th percentile on raster dur
        # p7_18_02_08_a002_ms.activity_threshold =

        if not p7_18_02_08_a002_ms.use_suite_2p:
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p7_18_02_08_a002_ms.load_data_from_file(file_name_to_load=
                                                    "p7/p7_18_02_08_a002/p7_18_02_08_a002_Corrected_RasterDur.mat",
                                                    variables_mapping=variables_mapping)
            p7_18_02_08_a002_ms.set_avg_cell_map_tif(file_name="p7/p7_18_02_08_a002/AVG_p7_18_02_08_a002.tif")
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p7_18_02_08_a002_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a002/p7_18_02_08_a002_Traces.mat",
                                                        variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p7_18_02_08_a002_ms.load_data_from_file(
                    file_name_to_load="p7/p7_18_02_08_a002/p7_18_02_08_a002_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p7_18_02_08_a002_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a002/p7_18_02_08_a002_CellDetect.mat",
                                                    variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p7_18_02_08_a002_ms"] = p7_18_02_08_a002_ms

    if "p7_18_02_08_a003_ms" in ms_str_to_load:
        p7_18_02_08_a003_ms = MouseSession(age=7, session_id="18_02_08_a003", sampling_rate=10, param=param,
                                           weight=3.85)

        p7_18_02_08_a003_ms.use_suite_2p = True

        # calculated with 99th percentile on raster dur
        # p7_18_02_08_a003_ms.activity_threshold =
        if not p7_18_02_08_a003_ms.use_suite_2p:
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p7_18_02_08_a003_ms.load_data_from_file(file_name_to_load=
                                                    "p7/p7_18_02_08_a003/p7_18_02_08_a003_Corrected_RasterDur.mat",
                                                    variables_mapping=variables_mapping)
            p7_18_02_08_a003_ms.set_avg_cell_map_tif(file_name="p7/p7_18_02_08_a003/AVG_p7_18_02_08_a003.tif")
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p7_18_02_08_a003_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a003/p7_18_02_08_a003_Traces.mat",
                                                        variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p7_18_02_08_a003_ms.load_data_from_file(
                    file_name_to_load="p7/p7_18_02_08_a003/p7_18_02_08_a003_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p7_18_02_08_a003_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a003/p7_18_02_08_a003_CellDetect.mat",
                                                    variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p7_18_02_08_a003_ms"] = p7_18_02_08_a003_ms

    if "p7_19_03_05_a000_ms" in ms_str_to_load:
        p7_19_03_05_a000_ms = MouseSession(age=7, session_id="19_03_05_a000",
                                           sampling_rate=8, param=param, weight=7)
        p7_19_03_05_a000_ms.use_suite_2p = True
        # for threshold prediction at 0.5
        # p7_19_03_05_a000_ms.activity_threshold =

        # prediction based on rnn ...
        # variables_mapping = {"predictions": "predictions"}
        # p7_19_03_05_a000_ms.load_raster_dur_from_predictions(
        #     file_name="p7/p7_19_03_05_a000/predictions/" +
        #               "P7_19_03_05_a000_predictions__2019_05_07.21-11-05_GT_epoch_11_all_cells",
        #     prediction_threshold=0.5, variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p7_19_03_05_a000_ms"] = p7_19_03_05_a000_ms

    if "p7_19_03_27_a000_ms" in ms_str_to_load:
        p7_19_03_27_a000_ms = MouseSession(age=7, session_id="19_03_27_a000",
                                           sampling_rate=8, param=param, weight=4.85)
        p7_19_03_27_a000_ms.use_suite_2p = True
        # for threshold prediction at 0.5
        # p7_19_03_27_a000_ms.activity_threshold =

        ms_str_to_ms_dict["p7_19_03_27_a000_ms"] = p7_19_03_27_a000_ms

    if "p7_19_03_27_a001_ms" in ms_str_to_load:
        p7_19_03_27_a001_ms = MouseSession(age=7, session_id="19_03_27_a001",
                                           sampling_rate=8, param=param, weight=4.85)
        p7_19_03_27_a001_ms.use_suite_2p = True
        # for threshold prediction at 0.5
        # p7_19_03_27_a000_ms.activity_threshold =

        p7_19_03_27_a001_ms.load_suite2p_data(data_path="p7/p7_19_03_27_a001/suite2p/", with_coord=True)

        # if load_movie:
        p7_19_03_27_a001_ms.load_tif_movie(path="p7/p7_19_03_27_a001")
        raw_traces_loaded = p7_19_03_27_a001_ms.load_raw_traces_from_npy(path="p7/p7_19_03_27_a001/")
        if not raw_traces_loaded:
            p7_19_03_27_a001_ms.load_tiff_movie_in_memory()
            p7_19_03_27_a001_ms.raw_traces = p7_19_03_27_a001_ms.build_raw_traces_from_movie()
            p7_19_03_27_a001_ms.save_raw_traces(path="p7/p7_19_03_27_a001/")

        # p7_19_03_27_a001_ms.clean_raster_at_concatenation()

        # p7_19_03_27_a001_ms.spike_struct.build_spike_nums_and_peak_nums()

        ms_str_to_ms_dict["p7_19_03_27_a001_ms"] = p7_19_03_27_a001_ms

    if "p7_19_03_27_a002_ms" in ms_str_to_load:
        p7_19_03_27_a002_ms = MouseSession(age=7, session_id="19_03_27_a002",
                                           sampling_rate=8, param=param, weight=4.85)
        p7_19_03_27_a002_ms.use_suite_2p = True
        # for threshold prediction at 0.5
        # p7_19_03_27_a002_ms.activity_threshold =

        ms_str_to_ms_dict["p7_19_03_27_a002_ms"] = p7_19_03_27_a002_ms

    if "p8_18_02_09_a000_ms" in ms_str_to_load:
        p8_18_02_09_a000_ms = MouseSession(age=8, session_id="18_02_09_a000", sampling_rate=10, param=param,
                                           weight=None)
        p8_18_02_09_a000_ms.use_suite_2p = True
        # calculated with 99th percentile on raster dur
        # p8_18_02_09_a000_ms.activity_threshold =

        if not p8_18_02_09_a000_ms.use_suite_2p:
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p8_18_02_09_a000_ms.load_data_from_file(file_name_to_load=
                                                    "p8/p8_18_02_09_a000/p8_18_02_09_a000_Corrected_RasterDur.mat",
                                                    variables_mapping=variables_mapping)
            p8_18_02_09_a000_ms.set_avg_cell_map_tif(file_name="p8/p8_18_02_09_a000/AVG_p8_18_02_09_a000.tif")
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p8_18_02_09_a000_ms.load_data_from_file(file_name_to_load="p8/p8_18_02_09_a000/p8_18_02_09_a000_Traces.mat",
                                                        variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p8_18_02_09_a000_ms.load_data_from_file(
                    file_name_to_load="p8/p8_18_02_09_a000/p8_18_02_09_a000_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p8_18_02_09_a000_ms.load_data_from_file(file_name_to_load="p8/p8_18_02_09_a000/p8_18_02_09_a000_CellDetect.mat",
                                                    variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p8_18_02_09_a000_ms"] = p8_18_02_09_a000_ms

    if "p8_18_02_09_a001_ms" in ms_str_to_load:
        p8_18_02_09_a001_ms = MouseSession(age=8, session_id="18_02_09_a001", sampling_rate=10, param=param,
                                           weight=None)
        p8_18_02_09_a001_ms.use_suite_2p = True
        # calculated with 99th percentile on raster dur
        # p8_18_02_09_a001_ms.activity_threshold =

        if not p8_18_02_09_a001_ms.use_suite_2p:
            # duration of those interneurons:
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p8_18_02_09_a001_ms.load_data_from_file(file_name_to_load=
                                                    "p8/p8_18_02_09_a001/p8_18_02_09_a001_Corrected_RasterDur.mat",
                                                    variables_mapping=variables_mapping)

            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p8_18_02_09_a001_ms.load_data_from_file(file_name_to_load="p8/p8_18_02_09_a001/p8_18_02_09_a001_Traces.mat",
                                                        variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p8_18_02_09_a001_ms.load_data_from_file(
                    file_name_to_load="p8/p8_18_02_09_a001/p8_18_02_09_a001_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            # variables_mapping = {"coord": "ContoursAll"}
            # p8_18_02_09_a001_ms.load_data_from_file(file_name_to_load="p8/p8_18_02_09_a001/p8_18_02_09_a001_CellDetect.mat",
            #                                         variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p8_18_02_09_a001_ms"] = p8_18_02_09_a001_ms

    if "p8_18_10_17_a000_ms" in ms_str_to_load:
        p8_18_10_17_a000_ms = MouseSession(age=8, session_id="18_10_17_a000", sampling_rate=10, param=param,
                                           weight=6)
        p8_18_10_17_a000_ms.use_suite_2p = True
        # calculated with 99th percentile on raster dur
        # p8_18_10_17_a000_ms.activity_threshold =

        if not p8_18_10_17_a000_ms.use_suite_2p:
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p8_18_10_17_a000_ms.load_data_from_file(
                file_name_to_load="p8/p8_18_10_17_a000/P8_18_10_17_a000_Corrected_RasterDur.mat",
                variables_mapping=variables_mapping)
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p8_18_10_17_a000_ms.load_data_from_file(file_name_to_load="p8/p8_18_10_17_a000/p8_18_10_17_a000_Traces.mat",
                                                        variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p8_18_10_17_a000_ms.load_data_from_file(
                    file_name_to_load="p8/p8_18_10_17_a000/p8_18_10_17_a000_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            p8_18_10_17_a000_ms.set_avg_cell_map_tif(file_name="p8/p8_18_10_17_a000/AVG_p8_18_10_17_a000.tif")
            variables_mapping = {"coord": "ContoursAll"}
            p8_18_10_17_a000_ms.load_data_from_file(file_name_to_load="p8/p8_18_10_17_a000/p8_18_10_17_a000_CellDetect.mat",
                                                    variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p8_18_10_17_a000_ms"] = p8_18_10_17_a000_ms

    if "p8_18_10_17_a001_ms" in ms_str_to_load:
        p8_18_10_17_a001_ms = MouseSession(age=8, session_id="18_10_17_a001", sampling_rate=10, param=param,
                                           weight=6)
        p8_18_10_17_a001_ms.use_suite_2p = True
        # calculated with 99th percentile on raster dur
        # p8_18_10_17_a001_ms.activity_threshold =

        if not p8_18_10_17_a001_ms.use_suite_2p:
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p8_18_10_17_a001_ms.load_data_from_file(file_name_to_load=
                                                    "p8/p8_18_10_17_a001/p8_18_10_17_a001_Corrected_RasterDur.mat",
                                                    variables_mapping=variables_mapping)
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p8_18_10_17_a001_ms.load_data_from_file(file_name_to_load="p8/p8_18_10_17_a001/p8_18_10_17_a001_Traces.mat",
                                                        variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p8_18_10_17_a001_ms.load_data_from_file(
                    file_name_to_load="p8/p8_18_10_17_a001/p8_18_10_17_a001_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            # variables_mapping = {"coord": "ContoursAll"}
            # p8_18_10_17_a001_ms.load_data_from_file(file_name_to_load="p8/p8_18_10_17_a001/p8_18_10_17_a001_CellDetect.mat",
            #                                         variables_mapping=variables_mapping)

        if load_abf:
            p8_18_10_17_a001_ms.load_abf_file(path_abf_data="p8/p8_18_10_17_a001/", sampling_rate=10000)

        ms_str_to_ms_dict["p8_18_10_17_a001_ms"] = p8_18_10_17_a001_ms

    if "p8_18_10_24_a005_ms" in ms_str_to_load:
        # 6.4
        p8_18_10_24_a005_ms = MouseSession(age=8, session_id="18_10_24_a005", sampling_rate=10, param=param,
                                           weight=6.4)

        # if True will  use the coord from suite2p, if False, will just load the info concerning suite2p in
        # if the dict suit2p_data in mouse_session
        p8_18_10_24_a005_ms.use_suite_2p = True
        if for_transient_classifier:
            p8_18_10_24_a005_ms.use_suite_2p = False

        # calculated with 99th percentile on raster dur
        # p8_18_10_24_a005_ms.activity_threshold =

        if not p8_18_10_24_a005_ms.use_suite_2p:
            if for_cell_classifier or for_transient_classifier:
                variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
                                     "peak_nums": "LocPeakMatrix_Python",
                                     "cells_to_remove": "cells_to_remove",
                                     "inter_neurons_from_gui": "inter_neurons",
                                     "doubtful_frames_nums": "doubtful_frames_nums"}
                p8_18_10_24_a005_ms.load_data_from_file(file_name_to_load=
                                                        "p8/p8_18_10_24_a005/p8_18_10_24_a005_fusion_validation.mat",
                                                        variables_mapping=variables_mapping,
                                                        from_gui=True)
                # p8_18_10_24_a005_GUI_Transiant MP.mat
                p8_18_10_24_a005_ms.build_spike_nums_dur()
            else:
                variables_mapping = {"predictions": "predictions"}
                p8_18_10_24_a005_ms.load_raster_dur_from_predictions(
                    file_name="p8/p8_18_10_24_a005/predictions/" +
                              "P8_18_10_24_a005_predictions__2019_05_02.12-55-41_GT_epoch_11_no_trans_no_over_all_cells.mat",
                    prediction_threshold=0.5, variables_mapping=variables_mapping)
                p8_18_10_24_a005_ms.clean_raster_at_concatenation()
                p8_18_10_24_a005_ms.spike_struct.build_spike_nums_and_peak_nums()

            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p8_18_10_24_a005_ms.load_data_from_file(
                    file_name_to_load="p8/p8_18_10_24_a005/p8_18_10_24_a005_Traces.mat",
                    variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p8_18_10_24_a005_ms.load_data_from_file(
                    file_name_to_load="p8/p8_18_10_24_a005/p8_18_10_24_a005_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p8_18_10_24_a005_ms.load_data_from_file(
                file_name_to_load="p8/p8_18_10_24_a005/p8_18_10_24_a005_CellDetect.mat",
                variables_mapping=variables_mapping)

        if not p8_18_10_24_a005_ms.use_suite_2p:
            if not for_cell_classifier:
                p8_18_10_24_a005_ms.clean_data_using_cells_to_remove()

        ms_str_to_ms_dict["p8_18_10_24_a005_ms"] = p8_18_10_24_a005_ms

    # Oriens movie
    if "p8_18_10_24_a006_ms" in ms_str_to_load:
        p8_18_10_24_a006_ms = MouseSession(age=8, session_id="18_10_24_a006",
                                           sampling_rate=10, param=param, weight=6.4)

        # oriens field, using Caiman
        p8_18_10_24_a006_ms.use_suite_2p = True
        if for_transient_classifier:
            p8_18_10_24_a006_ms.use_suite_2p = False

        if not p8_18_10_24_a006_ms.use_suite_2p:
            if for_cell_classifier or for_transient_classifier:
                variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
                                     "peak_nums": "LocPeakMatrix_Python",
                                     "cells_to_remove": "cells_to_remove",
                                     "inter_neurons_from_gui": "inter_neurons",
                                     "doubtful_frames_nums": "doubtful_frames_nums"}
                p8_18_10_24_a006_ms.load_data_from_file(file_name_to_load=
                                                        "p8/p8_18_10_24_a006/p8_18_10_24_a006_GUI_transients_RD.mat",
                                                        variables_mapping=variables_mapping,
                                                        from_gui=True)
                # used for training before: p8_18_10_24_a006_GUI_transients_RD.mat p8_18_10_24_a006_fusion_validation.mat
                p8_18_10_24_a006_ms.build_spike_nums_dur()

            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p8_18_10_24_a006_ms.load_data_from_file(file_name_to_load="p8/p8_18_10_24_a006/p8_18_10_24_a006_Traces.mat",
                                                        variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p8_18_10_24_a006_ms.load_data_from_file(
                    file_name_to_load="p8/p8_18_10_24_a006/p8_18_10_24_a006_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p8_18_10_24_a006_ms.load_data_from_file(file_name_to_load="p8/p8_18_10_24_a006/p8_18_10_24_a006_CellDetect.mat",
                                                    variables_mapping=variables_mapping)

            if not for_cell_classifier:
                p8_18_10_24_a006_ms.clean_data_using_cells_to_remove()

        ms_str_to_ms_dict["p8_18_10_24_a006_ms"] = p8_18_10_24_a006_ms

    if "p8_19_03_19_a000_ms" in ms_str_to_load:
        p8_19_03_19_a000_ms = MouseSession(age=8, session_id="19_03_19_a000",
                                           sampling_rate=8, param=param, weight=7.8)

        p8_19_03_19_a000_ms.use_suite_2p = True

        ms_str_to_ms_dict["p8_19_03_19_a000_ms"] = p8_19_03_19_a000_ms

    if "p9_17_12_06_a001_ms" in ms_str_to_load:
        p9_17_12_06_a001_ms = MouseSession(age=9, session_id="17_12_06_a001", sampling_rate=10, param=param,
                                           weight=5.6)
        p9_17_12_06_a001_ms.use_suite_2p = True
        # calculated with 99th percentile on raster dur
        # p9_17_12_06_a001_ms.activity_threshold =

        if not p9_17_12_06_a001_ms.use_suite_2p:
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p9_17_12_06_a001_ms.load_data_from_file(file_name_to_load=
                                                    "p9/p9_17_12_06_a001/p9_17_12_06_a001_Corrected_RasterDur.mat",
                                                    variables_mapping=variables_mapping)
            p9_17_12_06_a001_ms.set_avg_cell_map_tif(file_name="p9/p9_17_12_06_a001/AVG_p9_17_12_06_a001.tif")
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p9_17_12_06_a001_ms.load_data_from_file(file_name_to_load="p9/p9_17_12_06_a001/p9_17_12_06_a001_Traces.mat",
                                                        variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p9_17_12_06_a001_ms.load_data_from_file(
                    file_name_to_load="p9/p9_17_12_06_a001/p9_17_12_06_a001_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p9_17_12_06_a001_ms.load_data_from_file(file_name_to_load="p9/p9_17_12_06_a001/p9_17_12_06_a001_CellDetect.mat",
                                                    variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p9_17_12_06_a001_ms"] = p9_17_12_06_a001_ms

    if "p9_17_12_20_a001_ms" in ms_str_to_load:
        p9_17_12_20_a001_ms = MouseSession(age=9, session_id="17_12_20_a001", sampling_rate=10, param=param,
                                           weight=5.05)

        p9_17_12_20_a001_ms.use_suite_2p = True

        # calculated with 99th percentile on raster dur
        # p9_17_12_20_a001_ms.activity_threshold =

        if not p9_17_12_20_a001_ms.use_suite_2p:
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p9_17_12_20_a001_ms.load_data_from_file(file_name_to_load=
                                                    "p9/p9_17_12_20_a001/p9_17_12_20_a001_Corrected_RasterDur.mat",
                                                    variables_mapping=variables_mapping)
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p9_17_12_20_a001_ms.load_data_from_file(file_name_to_load="p9/p9_17_12_20_a001/p9_17_12_20_a001_Traces.mat",
                                                        variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p9_17_12_20_a001_ms.load_data_from_file(
                    file_name_to_load="p9/p9_17_12_20_a001/p9_17_12_20_a001_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p9_17_12_20_a001_ms.load_data_from_file(file_name_to_load="p9/p9_17_12_20_a001/p9_17_12_20_a001_CellDetect.mat",
                                                    variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p9_17_12_20_a001_ms"] = p9_17_12_20_a001_ms

    if "p9_18_09_27_a003_ms" in ms_str_to_load:
        p9_18_09_27_a003_ms = MouseSession(age=9, session_id="18_09_27_a003", sampling_rate=10, param=param,
                                           weight=6.65)

        p9_18_09_27_a003_ms.use_suite_2p = True
        if for_cell_classifier:
            p9_18_09_27_a003_ms.use_suite_2p = False

        # calculated with 95th percentile on raster dur
        p9_18_09_27_a003_ms.activity_threshold = 13
        # p9_18_09_27_a003_ms.set_low_activity_threshold(threshold=, percentile_value=1)

        if not p9_18_09_27_a003_ms.use_suite_2p:
            # variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
            #                      "spike_nums": "filt_Bin100ms_spikedigital",
            #                      "spike_durations": "LOC3"}
            # p9_18_09_27_a003_ms.load_data_from_file(file_name_to_load=
            #                                         "p9/p9_18_09_27_a003/p9_18_09_27_a003_Corrected_RasterDur.mat",
            #                                         variables_mapping=variables_mapping)
            variables_mapping = {"traces": "C_df"}
            p9_18_09_27_a003_ms.load_data_from_file(file_name_to_load=
                                                    "p9/p9_18_09_27_a003/p9_18_09_27_a003_Traces.mat",
                                                    variables_mapping=variables_mapping)
            variables_mapping = {"raw_traces": "raw_traces"}
            p9_18_09_27_a003_ms.load_data_from_file(
                file_name_to_load="p9/p9_18_09_27_a003/p9_18_09_27_a003_raw_Traces.mat",
                variables_mapping=variables_mapping)
            if load_movie:
                p9_18_09_27_a003_ms.load_tif_movie(path="p9/p9_18_09_27_a003/")
            variables_mapping = {"coord": "ContoursAll"}
            p9_18_09_27_a003_ms.load_data_from_file(file_name_to_load="p9/p9_18_09_27_a003/p9_18_09_27_a003_CellDetect.mat",
                                                    variables_mapping=variables_mapping)

            if for_cell_classifier or for_transient_classifier:
                variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
                                     "peak_nums": "LocPeakMatrix_Python",
                                     "cells_to_remove": "cells_to_remove",
                                     "inter_neurons_from_gui": "inter_neurons"}
                p9_18_09_27_a003_ms.load_data_from_file(file_name_to_load=
                                                        "p9/p9_18_09_27_a003/p9_18_09_27_a003_raw_TransientMP.mat",
                                                        variables_mapping=variables_mapping,
                                                        from_gui=True)

                p9_18_09_27_a003_ms.build_spike_nums_dur()
                if for_cell_classifier:
                    p9_18_09_27_a003_ms.load_cells_to_remove_from_txt(file_name="p9/p9_18_09_27_a003/"
                                                                                "p9_18_09_27_a003_cell_to_suppress_ground_truth.txt")

        if load_abf:
            p9_18_09_27_a003_ms.load_abf_file(path_abf_data="p9/p9_18_09_27_a003/",
                                               sampling_rate=10000,
                                              offset=0.1)

        ms_str_to_ms_dict["p9_18_09_27_a003_ms"] = p9_18_09_27_a003_ms

    if "p9_19_02_20_a000_ms" in ms_str_to_load:
        p9_19_02_20_a000_ms = MouseSession(age=9, session_id="19_02_20_a000",
                                           sampling_rate=8, param=param,
                                           weight=5.2)
        p9_19_02_20_a000_ms.use_suite_2p = True
        # for threshold prediction at 0.5
        # p9_19_02_20_a000_ms.activity_threshold =

        ms_str_to_ms_dict["p9_19_02_20_a000_ms"] = p9_19_02_20_a000_ms

    if "p9_19_02_20_a001_ms" in ms_str_to_load:
        p9_19_02_20_a001_ms = MouseSession(age=9, session_id="19_02_20_a001",
                                           sampling_rate=8, param=param,
                                           weight=5.2)

        p9_19_02_20_a001_ms.use_suite_2p = True

        # for threshold prediction at 0.5
        # p9_19_02_20_a001_ms.activity_threshold =

        ms_str_to_ms_dict["p9_19_02_20_a001_ms"] = p9_19_02_20_a001_ms

    if "p9_19_02_20_a002_ms" in ms_str_to_load:
        p9_19_02_20_a002_ms = MouseSession(age=9, session_id="19_02_20_a002",
                                           sampling_rate=8, param=param,
                                           weight=5.2)

        p9_19_02_20_a002_ms.use_suite_2p = True

        ms_str_to_ms_dict["p9_19_02_20_a002_ms"] = p9_19_02_20_a002_ms

    if "p9_19_02_20_a003_ms" in ms_str_to_load:
        p9_19_02_20_a003_ms = MouseSession(age=9, session_id="19_02_20_a003",
                                           sampling_rate=8, param=param,
                                           weight=5.2)

        p9_19_02_20_a003_ms.use_suite_2p = True

        # for threshold prediction at 0.5
        # p9_19_02_20_a003_ms.activity_threshold =

        # prediction based on rnn ...
        # variables_mapping = {"predictions": "predictions"}
        # p9_19_02_20_a003_ms.load_raster_dur_from_predictions(
        #     file_name="p9/p9_19_02_20_a003/predictions/" +
        #               "P9_19_02_20_a003_predictions__2019_05_07.22-06-15_GT_epoch_11_all_cells.mat",
        #     prediction_threshold=0.5, variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p9_19_02_20_a003_ms"] = p9_19_02_20_a003_ms

    if "p9_19_03_14_a000_ms" in ms_str_to_load:
        p9_19_03_14_a000_ms = MouseSession(age=9, session_id="19_03_14_a000",
                                           sampling_rate=8, param=param,
                                           weight=5.75)

        p9_19_03_14_a000_ms.use_suite_2p = True

        ms_str_to_ms_dict["p9_19_03_14_a000_ms"] = p9_19_03_14_a000_ms

    if "p9_19_03_14_a001_ms" in ms_str_to_load:
        p9_19_03_14_a001_ms = MouseSession(age=9, session_id="19_03_14_a001",
                                           sampling_rate=8, param=param,
                                           weight=5.75)

        p9_19_03_14_a001_ms.use_suite_2p = True

        ms_str_to_ms_dict["p9_19_03_14_a001_ms"] = p9_19_03_14_a001_ms

    if "p9_19_03_22_a000_ms" in ms_str_to_load:
        p9_19_03_22_a000_ms = MouseSession(age=9, session_id="19_03_22_a000",
                                           sampling_rate=8, param=param,
                                           weight=7.1)

        p9_19_03_22_a000_ms.use_suite_2p = True

        ms_str_to_ms_dict["p9_19_03_22_a000_ms"] = p9_19_03_22_a000_ms

    if "p9_19_03_22_a001_ms" in ms_str_to_load:
        p9_19_03_22_a001_ms = MouseSession(age=9, session_id="19_03_22_a001",
                                           sampling_rate=8, param=param,
                                           weight=7.1)

        p9_19_03_22_a001_ms.use_suite_2p = True

        # prediction based on rnn trained on 50 cells, BO,
        # variables_mapping = {"predictions": "predictions"}
        # p9_19_03_22_a001_ms.load_raster_dur_from_predictions(
        #     file_name="p9/p9_19_03_22_a001/" +
        #               "P9_19_03_22_a001_predictions_2019_04_11.23-54-06_all_cells_rnn_26_02_19_17-20-11.mat",
        #     prediction_threshold=0.5, variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p9_19_03_22_a001_ms"] = p9_19_03_22_a001_ms

    if "p10_17_11_16_a003_ms" in ms_str_to_load:
        p10_17_11_16_a003_ms = MouseSession(age=10, session_id="17_11_16_a003", sampling_rate=10, param=param,
                                            weight=6.1)
        p10_17_11_16_a003_ms.use_suite_2p = True
        # calculated with 99th percentile on raster dur
        # p10_17_11_16_a003_ms.activity_threshold =
        if not p10_17_11_16_a003_ms.use_suite_2p:
            # duration of those interneurons: 28
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p10_17_11_16_a003_ms.load_data_from_file(file_name_to_load=
                                                     "p10/p10_17_11_16_a003/p10_17_11_16_a003_Corrected_RasterDur.mat",
                                                     variables_mapping=variables_mapping)

            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p10_17_11_16_a003_ms.load_data_from_file(
                    file_name_to_load="p10/p10_17_11_16_a003/p10_17_11_16_a003_Traces.mat",
                    variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p10_17_11_16_a003_ms.load_data_from_file(
                    file_name_to_load="p10/p10_17_11_16_a003/p10_17_11_16_a003_raw_Traces.mat",
                    variables_mapping=variables_mapping)

            variables_mapping = {"coord": "ContoursAll"}
            p10_17_11_16_a003_ms.load_data_from_file(
                file_name_to_load="p10/p10_17_11_16_a003/p10_17_11_16_a003_CellDetect.mat",
                variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p10_17_11_16_a003_ms"] = p10_17_11_16_a003_ms

    if "p10_19_02_21_a002_ms" in ms_str_to_load:
        p10_19_02_21_a002_ms = MouseSession(age=10, session_id="19_02_21_a002",
                                            sampling_rate=8, param=param,
                                           weight=6.55)

        p10_19_02_21_a002_ms.use_suite_2p = True

        ms_str_to_ms_dict["p10_19_02_21_a002_ms"] = p10_19_02_21_a002_ms
    if "p10_19_02_21_a003_ms" in ms_str_to_load:
        p10_19_02_21_a003_ms = MouseSession(age=10, session_id="19_02_21_a003",
                                            sampling_rate=8, param=param,
                                           weight=6.55)

        p10_19_02_21_a003_ms.use_suite_2p = True

        ms_str_to_ms_dict["p10_19_02_21_a003_ms"] = p10_19_02_21_a003_ms

    if "p10_19_02_21_a005_ms" in ms_str_to_load:
        p10_19_02_21_a005_ms = MouseSession(age=10, session_id="19_02_21_a005",
                                            sampling_rate=8, param=param,
                                           weight=6.55)

        p10_19_02_21_a005_ms.use_suite_2p = True

        ms_str_to_ms_dict["p10_19_02_21_a005_ms"] = p10_19_02_21_a005_ms

    if "p10_19_03_08_a000_ms" in ms_str_to_load:
        p10_19_03_08_a000_ms = MouseSession(age=10, session_id="19_03_08_a000",
                                            sampling_rate=8, param=param,
                                            weight=8.6)

        p10_19_03_08_a000_ms.use_suite_2p = True

        ms_str_to_ms_dict["p10_19_03_08_a000_ms"] = p10_19_03_08_a000_ms

    if "p10_19_03_08_a001_ms" in ms_str_to_load:
        p10_19_03_08_a001_ms = MouseSession(age=10, session_id="19_03_08_a001",
                                            sampling_rate=8, param=param,
                                            weight=8.6)

        p10_19_03_08_a001_ms.use_suite_2p = True

        ms_str_to_ms_dict["p10_19_03_08_a001_ms"] = p10_19_03_08_a001_ms

    if "p11_17_11_24_a000_ms" in ms_str_to_load:
        p11_17_11_24_a000_ms = MouseSession(age=11, session_id="17_11_24_a000",
                                            sampling_rate=10, param=param,
                                            weight=6.7)

        p11_17_11_24_a000_ms.use_suite_2p = True
        if for_transient_classifier:
            p11_17_11_24_a000_ms.use_suite_2p = False

        # calculated with 99th percentile on raster dur
        # p11_17_11_24_a000_ms.activity_threshold = 11

        # variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
        #                      "spike_nums": "filt_Bin100ms_spikedigital",
        #                      "spike_durations": "LOC3"}
        # p11_17_11_24_a000_ms.load_data_from_file(file_name_to_load=
        #                                          "p11/p11_17_11_24_a000/p11_17_11_24_a000_Corrected_RasterDur.mat",
        #
        #                                          variables_mapping=variables_mapping)

        if not p11_17_11_24_a000_ms.use_suite_2p:
            if for_cell_classifier or for_transient_classifier:
                variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
                                     "peak_nums": "LocPeakMatrix_Python",
                                     "cells_to_remove": "cells_to_remove",
                                     "inter_neurons_from_gui": "inter_neurons",
                                     "doubtful_frames_nums": "doubtful_frames_nums"}
                p11_17_11_24_a000_ms.load_data_from_file(file_name_to_load=
                                                         "p11/p11_17_11_24_a000/p11_17_11_24_a000_fusion_validation.mat",
                                                         variables_mapping=variables_mapping,
                                                         from_gui=True)
                p11_17_11_24_a000_ms.build_spike_nums_dur()
            else:
                variables_mapping = {"predictions": "predictions"}
                p11_17_11_24_a000_ms.load_raster_dur_from_predictions(
                    file_name="p11/p11_17_11_24_a000/predictions/" +
                              "P11_17_11_24_a000_predictions__2019_05_03.17-19-11_GT_epoch_11_no_trans_no_over.mat",
                    prediction_threshold=0.5, variables_mapping=variables_mapping)
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p11_17_11_24_a000_ms.load_data_from_file(
                    file_name_to_load="p11/p11_17_11_24_a000/p11_17_11_24_a000_Traces.mat",
                    variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p11_17_11_24_a000_ms.load_data_from_file(
                    file_name_to_load="p11/p11_17_11_24_a000/p11_17_11_24_a000_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p11_17_11_24_a000_ms.load_data_from_file(
                file_name_to_load="p11/p11_17_11_24_a000/p11_17_11_24_a000_CellDetect.mat",
                variables_mapping=variables_mapping)

            if not for_cell_classifier:
                p11_17_11_24_a000_ms.clean_data_using_cells_to_remove()

        ms_str_to_ms_dict["p11_17_11_24_a000_ms"] = p11_17_11_24_a000_ms

    if "p11_17_11_24_a001_ms" in ms_str_to_load:
        p11_17_11_24_a001_ms = MouseSession(age=11, session_id="17_11_24_a001", sampling_rate=10, param=param,
                                            weight=6.7)
        p11_17_11_24_a001_ms.use_suite_2p = True

        # calculated with 99th percentile on raster dur
        # p11_17_11_24_a001_ms.activity_threshold =

        if not p11_17_11_24_a001_ms.use_suite_2p:
            # duration of those interneurons:
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p11_17_11_24_a001_ms.load_data_from_file(file_name_to_load=
                                                     "p11/p11_17_11_24_a001/p11_17_11_24_a001_Corrected_RasterDur.mat",
                                                     variables_mapping=variables_mapping)
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p11_17_11_24_a001_ms.load_data_from_file(
                    file_name_to_load="p11/p11_17_11_24_a001/p11_17_11_24_a001_Traces.mat",
                    variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p11_17_11_24_a001_ms.load_data_from_file(
                    file_name_to_load="p11/p11_17_11_24_a001/p11_17_11_24_a001_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p11_17_11_24_a001_ms.load_data_from_file(
                file_name_to_load="p11/p11_17_11_24_a001/p11_17_11_24_a001_CellDetect.mat",
                variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p11_17_11_24_a001_ms"] = p11_17_11_24_a001_ms

    if "p11_19_02_15_a000_ms" in ms_str_to_load:
        p11_19_02_15_a000_ms = MouseSession(age=11, session_id="19_02_15_a000",
                                            sampling_rate=8, param=param,
                                            weight=5.9)

        p11_19_02_15_a000_ms.use_suite_2p = True

        ms_str_to_ms_dict["p11_19_02_15_a000_ms"] = p11_19_02_15_a000_ms

    if "p11_19_02_22_a000_ms" in ms_str_to_load:
        p11_19_02_22_a000_ms = MouseSession(age=11, session_id="19_02_22_a000",
                                            sampling_rate=8, param=param,
                                            weight=7.95)

        p11_19_02_22_a000_ms.use_suite_2p = True

        ms_str_to_ms_dict["p11_19_02_22_a000_ms"] = p11_19_02_22_a000_ms

    # GAD-Cre Eleonora
    if "p11_19_04_30_a001_ms" in ms_str_to_load:
        p11_19_04_30_a001_ms = MouseSession(age=11, session_id="19_04_30_a001",
                                            sampling_rate=8, param=param)

        p11_19_04_30_a001_ms.use_suite_2p = False
        if for_transient_classifier:
            p11_19_04_30_a001_ms.use_suite_2p = False

        if not p11_19_04_30_a001_ms.use_suite_2p:
            if for_transient_classifier:
                variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
                                     "peak_nums": "LocPeakMatrix_Python",
                                     "cells_to_remove": "cells_to_remove",
                                     "inter_neurons_from_gui": "inter_neurons",
                                     "doubtful_frames_nums": "doubtful_frames_nums"}
                p11_19_04_30_a001_ms.load_data_from_file(file_name_to_load=
                                                         "p11/p11_19_04_30_a001/p11_19_04_30_a001_gound_truth.mat",
                                                         variables_mapping=variables_mapping,
                                                         from_gui=True)
                p11_19_04_30_a001_ms.build_spike_nums_dur()

            variables_mapping = {"coord": "ContoursAll"}
            p11_19_04_30_a001_ms.load_data_from_file(
                file_name_to_load="p11/p11_19_04_30_a001/p11_19_04_30_a001_CellDetect_fiji.mat",
                variables_mapping=variables_mapping)
            p11_19_04_30_a001_ms.load_tif_movie(path="p11/p11_19_04_30_a001/")
            # p11_19_04_30_a001_ms.load_caiman_results(path_data="p11/p11_19_04_30_a001/")
        ms_str_to_ms_dict["p11_19_04_30_a001_ms"] = p11_19_04_30_a001_ms

    if "p12_171110_a000_ms" in ms_str_to_load:
        p12_171110_a000_ms = MouseSession(age=12, session_id="17_11_10_a000",
                                          sampling_rate=10, param=param,
                                          weight=7)

        p12_171110_a000_ms.use_suite_2p = True
        if for_transient_classifier or for_cell_classifier:
            p12_171110_a000_ms.use_suite_2p = False

        # calculated with 99th percentile on raster dur
        p12_171110_a000_ms.activity_threshold = 13

        # caiman version
        # variables_mapping = {"spike_nums_dur": "corrected_rasterdur"} # rasterdur before
        # p12_171110_a000_ms.load_data_from_file(file_name_to_load=
        #                                          "p12/p12_17_11_10_a000/p12_17_11_10_a000_RasterDur.mat",
        #                                          variables_mapping=variables_mapping)

        if not p12_171110_a000_ms.use_suite_2p:
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
                # variables_mapping = {"spike_nums_dur": "spike_nums_dur_predicted"}
                # not the best prediction, but done on all CNN validated cells
                # p12_171110_a000_ms.\
                #     load_data_from_file(file_name_to_load=
                #                         "p12/p12_17_11_10_a000/P12_17_11_10_a000_predictions_2019_02_06.22-48-11_all_cnn_cells_trained_2_p12_cells.mat",
                #                         variables_mapping=variables_mapping)
                # trained on 50 cells + artificial data, 3 inputs, with overlap 0.9 and 3 transformations
                # rnn trained on 26/02/2019 17-20-11 on 391 cells

                # prediction based on rnn trained on 50 cells, BO
                # variables_mapping = {"predictions": "predictions"}
                # p12_171110_a000_ms.load_raster_dur_from_predictions(
                #     file_name="p12/p12_17_11_10_a000/predictions/" +
                #               "P12_17_11_10_a000_predictions_2019_03_14.20-19-48.mat",
                #     prediction_threshold=0.5, variables_mapping=variables_mapping)

                # prediction GT epoch 11
                # p12_171110_a000_ms.load_raster_dur_from_predictions(
                #     file_name="p12/p12_17_11_10_a000/predictions/" +
                #               "P12_17_11_10_a000_predictions__2019_05_02.14-57-14_GT_epoch11_no_trans_no_over_all_cells.mat",
                #     prediction_threshold=0.5, variables_mapping=variables_mapping)

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

        if not p12_171110_a000_ms.use_suite_2p:
            if not for_cell_classifier:
                p12_171110_a000_ms.clean_data_using_cells_to_remove()

        # p12_171110_a000_ms.load_caiman_results(path_data="p12/p12_17_11_10_a000/")

        ms_str_to_ms_dict["p12_171110_a000_ms"] = p12_171110_a000_ms

    if "p12_17_11_10_a002_ms" in ms_str_to_load:
        p12_17_11_10_a002_ms = MouseSession(age=12, session_id="17_11_10_a002",
                                            sampling_rate=10, param=param,
                                            weight=7)

        p12_17_11_10_a002_ms.use_suite_2p = True

        # calculated with 99th percentile on raster dur
        # p12_17_11_10_a002_ms.activity_threshold =

        if not p12_17_11_10_a002_ms.use_suite_2p:
            # variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
            #                      "spike_nums": "filt_Bin100ms_spikedigital",
            #                      "spike_durations": "LOC3"}
            # p12_17_11_10_a002_ms.load_data_from_file(file_name_to_load=
            #                                          "p12/p12_17_11_10_a002/p12_17_11_10_a002_Corrected_RasterDur.mat",
            #                                          variables_mapping=variables_mapping)
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p12_17_11_10_a002_ms.load_data_from_file(
                    file_name_to_load="p12/p12_17_11_10_a002/p12_17_11_10_a002_Traces.mat",
                    variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p12_17_11_10_a002_ms.load_data_from_file(
                    file_name_to_load="p12/p12_17_11_10_a002/p12_17_11_10_a002_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p12_17_11_10_a002_ms.load_data_from_file(
                file_name_to_load="p12/p12_17_11_10_a002/p12_17_11_10_a002_CellDetect.mat",
                variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p12_17_11_10_a002_ms"] = p12_17_11_10_a002_ms

    # GAD-cre
    if "p12_19_02_08_a000_ms" in ms_str_to_load:
        p12_19_02_08_a000_ms = MouseSession(age=12, session_id="19_02_08_a000",
                                            sampling_rate=8, param=param,
                                            weight=7.55)

        p12_19_02_08_a000_ms.use_suite_2p = False

        if not p12_19_02_08_a000_ms.use_suite_2p:
            variables_mapping = {"coord": "ContoursAll"}
            p12_19_02_08_a000_ms.load_data_from_file(
                file_name_to_load="p12/p12_19_02_08_a000/p12_19_02_08_a000_CellDetect_fiji.mat",
                variables_mapping=variables_mapping, from_fiji=True)

        ms_str_to_ms_dict["p12_19_02_08_a000_ms"] = p12_19_02_08_a000_ms

    if "p13_18_10_29_a000_ms" in ms_str_to_load:
        p13_18_10_29_a000_ms = MouseSession(age=13, session_id="18_10_29_a000",
                                            sampling_rate=10, param=param,
                                            weight=9.4)

        p13_18_10_29_a000_ms.use_suite_2p = True

        # calculated with 99th percentile on raster dur
        # p13_18_10_29_a000_ms.activity_threshold =

        if not p13_18_10_29_a000_ms.use_suite_2p:
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p13_18_10_29_a000_ms.load_data_from_file(file_name_to_load=
                                                     "p13/p13_18_10_29_a000/p13_18_10_29_a000_Corrected_RasterDur.mat",
                                                     variables_mapping=variables_mapping)
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p13_18_10_29_a000_ms.load_data_from_file(
                    file_name_to_load="p13/p13_18_10_29_a000/p13_18_10_29_a000_Traces.mat",
                    variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p13_18_10_29_a000_ms.load_data_from_file(
                    file_name_to_load="p13/p13_18_10_29_a000/p13_2018_10_29_a000_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            # variables_mapping = {"coord": "ContoursAll"}
            # p13_18_10_29_a000_ms.load_data_from_file(file_name_to_load=
            #                                          "p13/p13_18_10_29_a000/p13_18_10_29_a000_CellDetect.mat",
            #                                          variables_mapping=variables_mapping)

        if load_abf:
            p13_18_10_29_a000_ms.load_abf_file(abf_file_name="p13/p13_18_10_29_a000/p13_18_10_29_a000.abf",
                                               threshold_piezo=None,  sampling_rate=10000)

        ms_str_to_ms_dict["p13_18_10_29_a000_ms"] = p13_18_10_29_a000_ms

    if "p13_18_10_29_a001_ms" in ms_str_to_load:
        p13_18_10_29_a001_ms = MouseSession(age=13, session_id="18_10_29_a001", sampling_rate=10, param=param,
                                            weight=9.4)

        p13_18_10_29_a001_ms.use_suite_2p = True
        if for_transient_classifier:
            p13_18_10_29_a001_ms.use_suite_2p = False

        # calculated with 99th percentile on raster dur
        # p13_18_10_29_a001_ms.activity_threshold =

        if not p13_18_10_29_a001_ms.use_suite_2p:
            if for_cell_classifier or for_transient_classifier:
                variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
                                     "peak_nums": "LocPeakMatrix_Python",
                                     "cells_to_remove": "cells_to_remove",
                                     "inter_neurons_from_gui": "inter_neurons",
                                     "doubtful_frames_nums": "doubtful_frames_nums"}
                p13_18_10_29_a001_ms.load_data_from_file(file_name_to_load=
                                                         "p13/p13_18_10_29_a001/p13_18_10_29_a001_GUI_transients_RD.mat",
                                                         variables_mapping=variables_mapping,
                                                         from_gui=True)
                # p13_18_10_29_a001_GUI_transients_RD.mat p13_18_10_29_a001_fusion_validation.mat

                p13_18_10_29_a001_ms.build_spike_nums_dur()
            # else:
            #     variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
            #                      "spike_nums": "filt_Bin100ms_spikedigital",
            #                      "spike_durations": "LOC3"}
            #     p13_18_10_29_a001_ms.load_data_from_file(file_name_to_load=
            #                                              "p13/p13_18_10_29_a001/p13_18_10_29_a001_Corrected_RasterDur.mat",
            #                                              variables_mapping=variables_mapping)
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p13_18_10_29_a001_ms.load_data_from_file(
                    file_name_to_load="p13/p13_18_10_29_a001/p13_18_10_29_a001_Traces.mat",
                    variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p13_18_10_29_a001_ms.load_data_from_file(
                    file_name_to_load="p13/p13_18_10_29_a001/p13_18_10_29_a001_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p13_18_10_29_a001_ms.load_data_from_file(file_name_to_load=
                                                     "p13/p13_18_10_29_a001/p13_18_10_29_a001_CellDetect.mat",
                                                     variables_mapping=variables_mapping)

            if not for_cell_classifier:
                p13_18_10_29_a001_ms.clean_data_using_cells_to_remove()

            if load_abf:
                p13_18_10_29_a001_ms.load_abf_file(abf_file_name="p13/p13_18_10_29_a001/p13_18_10_29_a001.abf",
                                                   threshold_piezo=None, sampling_rate=10000)
        ms_str_to_ms_dict["p13_18_10_29_a001_ms"] = p13_18_10_29_a001_ms

    # oriens
    if "p13_18_10_29_a002_ms" in ms_str_to_load:
        p13_18_10_29_a002_ms = MouseSession(age=13, session_id="18_10_29_a002",
                                            sampling_rate=10, param=param,
                                            weight=9.4)

        p13_18_10_29_a002_ms.use_suite_2p = True

        ms_str_to_ms_dict["p13_18_10_29_a002_ms"] = p13_18_10_29_a002_ms

    if "p13_19_03_11_a000_ms" in ms_str_to_load:
        p13_19_03_11_a000_ms = MouseSession(age=13, session_id="19_03_11_a000",
                                            sampling_rate=8, param=param,
                                            weight=None)

        p13_19_03_11_a000_ms.use_suite_2p = True

        ms_str_to_ms_dict["p13_19_03_11_a000_ms"] = p13_19_03_11_a000_ms

    if "p14_18_10_23_a000_ms" in ms_str_to_load:
        p14_18_10_23_a000_ms = MouseSession(age=14, session_id="18_10_23_a000",
                                            sampling_rate=10, param=param,
                                            weight=10.35)
        p14_18_10_23_a000_ms.use_suite_2p = True
        # calculated with 99th percentile on raster dur
        # p14_18_10_23_a000_ms.activity_threshold =

        if not p14_18_10_23_a000_ms.use_suite_2p:
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p14_18_10_23_a000_ms.load_data_from_file(file_name_to_load=
                                                     "p14/p14_18_10_23_a000/p14_18_10_23_a000_Corrected_RasterDur.mat",
                                                     variables_mapping=variables_mapping)
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p14_18_10_23_a000_ms.load_data_from_file(
                    file_name_to_load="p14/p14_18_10_23_a000/p14_18_10_23_a000_Traces.mat",
                    variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p14_18_10_23_a000_ms.load_data_from_file(
                    file_name_to_load="p14/p14_18_10_23_a000/p14_18_10_23_a000_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p14_18_10_23_a000_ms.load_data_from_file(
                file_name_to_load="p14/p14_18_10_23_a000/p14_18_10_23_a000_CellDetect.mat",
                variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p14_18_10_23_a000_ms"] = p14_18_10_23_a000_ms

    if "p14_18_10_23_a001_ms" in ms_str_to_load:
        # only interneurons in p14_18_10_23_a001_ms
        p14_18_10_23_a001_ms = MouseSession(age=14, session_id="18_10_23_a001", sampling_rate=10, param=param,
                                            weight=10.35)

        p14_18_10_23_a001_ms.use_suite_2p = True

        # calculated with 99th percentile on raster dur
        # p14_18_10_23_a001_ms.activity_threshold =

        if not p14_18_10_23_a001_ms.use_suite_2p:
            variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p14_18_10_23_a001_ms.load_data_from_file(file_name_to_load=
                                                     "p14/p14_18_10_23_a001/p14_18_10_23_a001_Corrected_RasterDur.mat",
                                                     variables_mapping=variables_mapping)
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p14_18_10_23_a001_ms.load_data_from_file(
                    file_name_to_load="p14/p14_18_10_23_a001/p14_18_10_23_a001_Traces.mat",
                    variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p14_18_10_23_a001_ms.load_data_from_file(
                    file_name_to_load="p14/p14_18_10_23_a001/p14_18_10_23_a001_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p14_18_10_23_a001_ms.load_data_from_file(
                file_name_to_load="p14/p14_18_10_23_a001/p14_18_10_23_a001_CellDetect.mat",
                variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p14_18_10_23_a001_ms"] = p14_18_10_23_a001_ms

    if "p14_18_10_30_a001_ms" in ms_str_to_load:
        p14_18_10_30_a001_ms = MouseSession(age=14, session_id="18_10_30_a001", sampling_rate=10, param=param,
                                            weight=8.9)

        p14_18_10_30_a001_ms.use_suite_2p = True

        # calculated with 99th percentile on raster dur
        # p14_18_10_30_a001_ms.activity_threshold =

        if not p14_18_10_30_a001_ms.use_suite_2p:
            variables_mapping = {"spike_nums_dur": "rasterdur",
                                 "spike_nums": "filt_Bin100ms_spikedigital",
                                 "spike_durations": "LOC3"}
            p14_18_10_30_a001_ms.load_data_from_file(file_name_to_load=
                                                     "p14/p14_18_10_30_a001/p14_18_10_30_a001_RasterDur.mat",
                                                     variables_mapping=variables_mapping)
            if load_traces:
                variables_mapping = {"traces": "C_df"}
                p14_18_10_30_a001_ms.load_data_from_file(
                    file_name_to_load="p14/p14_18_10_30_a001/p14_18_10_30_a001_Traces.mat",
                    variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "raw_traces"}
                p14_18_10_30_a001_ms.load_data_from_file(
                    file_name_to_load="p14/p14_18_10_30_a001/p14_18_10_30_a001_raw_Traces.mat",
                    variables_mapping=variables_mapping)
            variables_mapping = {"coord": "ContoursAll"}
            p14_18_10_30_a001_ms.load_data_from_file(
                file_name_to_load="p14/p14_18_10_30_a001/p14_18_10_30_a001_CellDetect.mat",
                variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p14_18_10_30_a001_ms"] = p14_18_10_30_a001_ms

    if "p16_18_11_01_a002_ms" in ms_str_to_load:
        p16_18_11_01_a002_ms = MouseSession(age=16, session_id="18_11_01_a002",
                                            sampling_rate=10, param=param,
                                            weight=8.9)

        p16_18_11_01_a002_ms.use_suite_2p = True

        ms_str_to_ms_dict["p16_18_11_01_a002_ms"] = p16_18_11_01_a002_ms

    if "p19_19_04_08_a000_ms" in ms_str_to_load:
        p19_19_04_08_a000_ms = MouseSession(age=19, session_id="19_04_08_a000",
                                            sampling_rate=8, param=param,
                                            weight=11.7)

        p19_19_04_08_a000_ms.use_suite_2p = True

        ms_str_to_ms_dict["p19_19_04_08_a000_ms"] = p19_19_04_08_a000_ms

    if "p19_19_04_08_a001_ms" in ms_str_to_load:
        p19_19_04_08_a001_ms = MouseSession(age=19, session_id="19_04_08_a001",
                                            sampling_rate=8, param=param,
                                            weight=11.7)
        p19_19_04_08_a001_ms.activity_threshold = 18
        p19_19_04_08_a001_ms.use_suite_2p = True

        ms_str_to_ms_dict["p19_19_04_08_a001_ms"] = p19_19_04_08_a001_ms

    if "p21_19_04_10_a000_ms" in ms_str_to_load:
        p21_19_04_10_a000_ms = MouseSession(age=21, session_id="19_04_10_a000",
                                            sampling_rate=8, param=param,
                                            weight=13.4)
        p21_19_04_10_a000_ms.use_suite_2p = True

        ms_str_to_ms_dict["p21_19_04_10_a000_ms"] = p21_19_04_10_a000_ms

    if "p21_19_04_10_a001_ms" in ms_str_to_load:
        p21_19_04_10_a001_ms = MouseSession(age=21, session_id="19_04_10_a001",
                                            sampling_rate=8, param=param,
                                            weight=13.4)
        p21_19_04_10_a001_ms.use_suite_2p = True

        ms_str_to_ms_dict["p21_19_04_10_a001_ms"] = p21_19_04_10_a001_ms

    if "p21_19_04_10_a000_j3_ms" in ms_str_to_load:
        p21_19_04_10_a000_j3_ms = MouseSession(age=21, session_id="19_04_10_a000_j3",
                                               sampling_rate=8, param=param,
                                            weight=11.8)
        p21_19_04_10_a000_j3_ms.use_suite_2p = True

        ms_str_to_ms_dict["p21_19_04_10_a000_j3_ms"] = p21_19_04_10_a000_j3_ms

    if "p21_19_04_10_a001_j3_ms" in ms_str_to_load:
        p21_19_04_10_a001_j3_ms = MouseSession(age=21, session_id="19_04_10_a001_j3",
                                               sampling_rate=8, param=param,
                                            weight=11.8)
        p21_19_04_10_a001_j3_ms.use_suite_2p = True

        ms_str_to_ms_dict["p21_19_04_10_a001_j3_ms"] = p21_19_04_10_a001_j3_ms

    # arnaud_ms = MouseSession(age=24, session_id="arnaud", nb_ms_by_frame=50, param=param)
    # arnaud_ms.activity_threshold = 13
    # arnaud_ms.set_inter_neurons([])
    # variables_mapping = {"spike_nums": "spikenums"}
    # arnaud_ms.load_data_from_file(file_name_to_load="spikenumsarnaud.mat", variables_mapping=variables_mapping)

    if "p41_19_04_30_a000_ms" in ms_str_to_load:
        p41_19_04_30_a000_ms = MouseSession(age=41, session_id="19_04_30_a000",
                                            sampling_rate=8, param=param,
                                            weight=25)

        p41_19_04_30_a000_ms.use_suite_2p = True

        # for threshold prediction at 0.5
        # p41_19_04_30_a000_ms.activity_threshold = 8

        # prediction based on rnn on GT epoch 11, no trans, no overlap
        # variables_mapping = {"predictions": "predictions"}
        # p41_19_04_30_a000_ms.load_raster_dur_from_predictions(
        #     file_name="p41/p41_19_04_30_a000/" +
        #               "predictions/P41_19_04_30_a000_predictions__2019_05_03.20-11-45_GT_epoch_11_no_trans_no_over.mat",
        #     prediction_threshold=0.5, variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p41_19_04_30_a000_ms"] = p41_19_04_30_a000_ms

    if "p60_arnaud_ms" in ms_str_to_load:
        p60_arnaud_ms = MouseSession(age=60, session_id="arnaud_a_529", sampling_rate=10, param=param)
        p60_arnaud_ms.activity_threshold = 9
        p60_arnaud_ms.set_inter_neurons([])
        # duration of those interneurons:
        variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p60_arnaud_ms.load_data_from_file(file_name_to_load=
                                          "p60/a529/Arnaud_RasterDur.mat",
                                          variables_mapping=variables_mapping)

        # variables_mapping = {"traces": "C_df"}
        # p60_arnaud_ms.load_data_from_file(file_name_to_load="p60/a529/Arnaud_a_529_corr_Traces.mat",
        #                                          variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p60_arnaud_ms.load_data_from_file(file_name_to_load="p60/a529/Arnaud_a_529_corr_CellDetect.mat",
                                          variables_mapping=variables_mapping)
        ms_str_to_ms_dict["p60_arnaud_ms"] = p60_arnaud_ms

    if "p60_a529_2015_02_25_ms" in ms_str_to_load:
        p60_a529_2015_02_25_ms = MouseSession(age=60, session_id="a529_2015_02_25",
                                              sampling_rate=10, param=param)
        p60_a529_2015_02_25_ms.activity_threshold = 10
        p60_a529_2015_02_25_ms.set_inter_neurons([])
        # duration of those interneurons:
        variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p60_a529_2015_02_25_ms.load_data_from_file(file_name_to_load=
                                                   "p60/a529_2015_02_25/a529_2015_02_25_RasterDur.mat",
                                                   variables_mapping=variables_mapping)
        variables_mapping = {"raw_traces": "raw_traces"}
        p60_a529_2015_02_25_ms.load_data_from_file(file_name_to_load=
                                                   "p60/a529_2015_02_25/MotCorre_529_15_02_25_raw_Traces.mat",
                                                   variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p60_a529_2015_02_25_ms.load_data_from_file(
            file_name_to_load="p60/a529_2015_02_25/MotCorre_529_15_02_25_CellDetect.mat",
            variables_mapping=variables_mapping)
        p60_a529_2015_02_25_ms.set_avg_cell_map_tif(file_name="p60/a529_2015_02_25/AVG_a529_2015_02_25.tif")
        if load_movie:
            p60_a529_2015_02_25_ms.load_tif_movie(path="p60/a529_2015_02_25/")
        ms_str_to_ms_dict["p60_a529_2015_02_25_ms"] = p60_a529_2015_02_25_ms

    if "p60_a529_2015_02_25_v_arnaud_ms" in ms_str_to_load:
        p60_a529_2015_02_25_v_arnaud_ms = MouseSession(age=60, session_id="a529_2015_02_25_v_arnaud",
                                                       nb_ms_by_frame=100, param=param)
        # p60_a529_2015_02_25_v_arnaud_ms.activity_threshold = 5
        p60_a529_2015_02_25_v_arnaud_ms.set_inter_neurons([])
        # duration of those interneurons:
        variables_mapping = {"traces": "Tr1b",
                             "spike_nums": "Raster"}
        p60_a529_2015_02_25_v_arnaud_ms.load_data_from_file(file_name_to_load=
                                                            "p60/a529_2015_02_25_v_arnaud/a529-20150225_Raster_all_cells.mat.mat",
                                                            variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p60_a529_2015_02_25_v_arnaud_ms"] = p60_a529_2015_02_25_v_arnaud_ms

    # marco's mice
    if "p60_20160506_gadcre01_01_ms" in ms_str_to_load:
        p60_20160506_gadcre01_01_ms = MouseSession(age=60, session_id="20160506_gadcre01_01",
                                              sampling_rate=10, param=param)
        p60_20160506_gadcre01_01_ms.use_suite_2p = False

        if not p60_20160506_gadcre01_01_ms.use_suite_2p:
            if load_traces:
                variables_mapping = {"traces": "TracePyrdf"}
                p60_20160506_gadcre01_01_ms.load_data_from_file(
                    file_name_to_load="p60/p60_20160506_gadcre01_01/p60_20160506_gadcre01_01_Traces.mat",
                    variables_mapping=variables_mapping)
                variables_mapping = {"raw_traces": "TracePyr"}
                p60_20160506_gadcre01_01_ms.load_data_from_file(
                    file_name_to_load="p60/p60_20160506_gadcre01_01/p60_20160506_gadcre01_01_Traces.mat",
                    variables_mapping=variables_mapping)

            p60_20160506_gadcre01_01_ms.load_caiman_results(path_data="p60/p60_20160506_gadcre01_01/")
            p60_20160506_gadcre01_01_ms.spike_struct.spike_nums = p60_20160506_gadcre01_01_ms.caiman_spike_nums
            p60_20160506_gadcre01_01_ms.spike_struct.spike_nums_dur = p60_20160506_gadcre01_01_ms.caiman_spike_nums_dur
            p60_20160506_gadcre01_01_ms.spike_struct.n_cells = len(p60_20160506_gadcre01_01_ms.caiman_spike_nums_dur)
            p60_20160506_gadcre01_01_ms.spike_struct.labels = np.arange(p60_20160506_gadcre01_01_ms.spike_struct.n_cells)

            variables_mapping = {"coord": "ContoursPyr"}
            p60_20160506_gadcre01_01_ms.load_data_from_file(
                file_name_to_load="p60/p60_20160506_gadcre01_01/p60_20160506_gadcre01_01_CellDetect.mat",
                variables_mapping=variables_mapping)
        ms_str_to_ms_dict["p60_20160506_gadcre01_01_ms"] = p60_20160506_gadcre01_01_ms

    if "richard_015_D74_P2_ms" in ms_str_to_load:
        # from 46517 to the end : all awake, no sleep, but no information about moving or not.
        richard_015_D74_P2_ms = MouseSession(age=60, session_id="richard_015_D74_P2", param=param)
        richard_015_D74_P2_ms.use_suite_2p = True
        if not richard_015_D74_P2_ms.use_suite_2p:
            richard_015_D74_P2_ms.activity_threshold = 19
            variables_mapping = {"spike_nums_dur": "Spike_Times_Onset_to_Peak"}
            richard_015_D74_P2_ms.load_data_from_file(file_name_to_load=
                                                      "richard_data/015/Cue/015_D74_P2/Spike_Times_Onset_to_Peak.mat",
                                                      variables_mapping=variables_mapping)
            richard_015_D74_P2_ms.load_richard_data(path_data="richard_data/015/Cue/015_D74_P2/")
        ms_str_to_ms_dict["richard_015_D74_P2_ms"] = richard_015_D74_P2_ms

    if "richard_015_D89_P2_ms" in ms_str_to_load:
        richard_015_D89_P2_ms = MouseSession(age=60, session_id="richard_015_D89_P2", param=param)
        richard_015_D89_P2_ms.activity_threshold = 22
        variables_mapping = {"spike_nums_dur": "Spike_Times_Onset_to_Peak"}
        richard_015_D89_P2_ms.load_data_from_file(file_name_to_load=
                                                  "richard_data/015/Cue/015_D89_P2/Spike_Times_Onset_to_Peak.mat",
                                                  variables_mapping=variables_mapping)
        richard_015_D89_P2_ms.load_richard_data(path_data="richard_data/015/Cue/015_D89_P2/")

        ms_str_to_ms_dict["richard_015_D89_P2_ms"] = richard_015_D89_P2_ms

    if "richard_015_D66_P2_ms" in ms_str_to_load:
        richard_015_D66_P2_ms = MouseSession(age=60, session_id="richard_015_D66_P2", param=param)
        variables_mapping = {"spike_nums_dur": "Spike_Times_Onset_to_Peak"}
        richard_015_D66_P2_ms.activity_threshold = 22
        richard_015_D66_P2_ms.load_data_from_file(file_name_to_load=
                                                  "richard_data/015/Nocue/015_D66_P2/Spike_Times_Onset_to_Peak.mat",
                                                  variables_mapping=variables_mapping)
        richard_015_D66_P2_ms.load_richard_data(path_data="richard_data/015/Nocue/015_D66_P2/")

        ms_str_to_ms_dict["richard_015_D66_P2_ms"] = richard_015_D66_P2_ms

    if "richard_015_D75_P2_ms" in ms_str_to_load:
        richard_015_D75_P2_ms = MouseSession(age=60, session_id="richard_015_D75_P2", param=param)
        richard_015_D75_P2_ms.activity_threshold = 18
        variables_mapping = {"spike_nums_dur": "Spike_Times_Onset_to_Peak"}
        richard_015_D75_P2_ms.load_data_from_file(file_name_to_load=
                                                  "richard_data/015/Nocue/015_D75_P2/Spike_Times_Onset_to_Peak.mat",
                                                  variables_mapping=variables_mapping)
        richard_015_D75_P2_ms.load_richard_data(path_data="richard_data/015/Nocue/015_D75_P2/")

        ms_str_to_ms_dict["richard_015_D75_P2_ms"] = richard_015_D75_P2_ms

    if "richard_018_D32_P2_ms" in ms_str_to_load:
        richard_018_D32_P2_ms = MouseSession(age=60, session_id="richard_018_D32_P2", param=param)
        richard_018_D32_P2_ms.activity_threshold = 18
        variables_mapping = {"spike_nums_dur": "Spike_Times_Onset_to_Peak"}
        richard_018_D32_P2_ms.load_data_from_file(file_name_to_load=
                                                  "richard_data/018/Cue/018_D32_P2/Spike_Times_Onset_to_Peak.mat",
                                                  variables_mapping=variables_mapping)
        richard_018_D32_P2_ms.load_richard_data(path_data="richard_data/018/Cue/018_D32_P2/")

        ms_str_to_ms_dict["richard_018_D32_P2_ms"] = richard_018_D32_P2_ms

    if "richard_018_D28_P2_ms" in ms_str_to_load:
        richard_018_D28_P2_ms = MouseSession(age=60, session_id="richard_018_D28_P2", param=param)
        use_suite_2p = False
        if not use_suite_2p:
            richard_018_D28_P2_ms.activity_threshold = 22
            variables_mapping = {"spike_nums_dur": "Spike_Times_Onset_to_Peak"}
            richard_018_D28_P2_ms.load_data_from_file(file_name_to_load=
                                                      "richard_data/018/Nocue/018_D28_P2/Spike_Times_Onset_to_Peak.mat",
                                                      variables_mapping=variables_mapping)
            richard_018_D28_P2_ms.load_richard_data(path_data="richard_data/018/Nocue/018_D28_P2/")

        ms_str_to_ms_dict["richard_018_D28_P2_ms"] = richard_018_D28_P2_ms

    if "richard_028_D1_P1_ms" in ms_str_to_load:
        richard_028_D1_P1_ms = MouseSession(age=60, session_id="richard_028_D1_P1",param=param)
        richard_028_D1_P1_ms.activity_threshold = 44
        variables_mapping = {"spike_nums_dur": "Spike_Times_Onset_to_Peak"}
        richard_028_D1_P1_ms.load_data_from_file(file_name_to_load=
                                                 "richard_data/028/Cue/028_D1_P1/Spike_Times_Onset_to_Peak.mat",
                                                 variables_mapping=variables_mapping)
        richard_028_D1_P1_ms.load_richard_data(path_data="richard_data/028/Cue/028_D1_P1/")

        ms_str_to_ms_dict["richard_028_D1_P1_ms"] = richard_028_D1_P1_ms

    if "richard_028_D2_P1_ms" in ms_str_to_load:
        richard_028_D2_P1_ms = MouseSession(age=60, session_id="richard_028_D2_P1", param=param)

        richard_028_D2_P1_ms.use_suite_2p = False
        if not richard_028_D2_P1_ms.use_suite_2p:
            richard_028_D2_P1_ms.activity_threshold = 30
            variables_mapping = {"spike_nums_dur": "Spike_Times_Onset_to_Peak"}
            richard_028_D2_P1_ms.load_data_from_file(file_name_to_load=
                                                     "richard_data/028/Nocue/028_D2_P1/Spike_Times_Onset_to_Peak.mat",
                                                     variables_mapping=variables_mapping)
            richard_028_D2_P1_ms.load_richard_data(path_data="richard_data/028/Nocue/028_D2_P1/")

        ms_str_to_ms_dict["richard_028_D2_P1_ms"] = richard_028_D2_P1_ms

    # common action for all mouse sessions
    for ms in ms_str_to_ms_dict.values():

        # just load the file_name in memory not the all movie in memory
        ms.load_tif_movie(path=f"p{ms.age}/{ms.description.lower()}/")

        ms.load_suite2p_data(data_path=f"p{ms.age}/{ms.description.lower()}/suite2p/", with_coord=ms.use_suite_2p)

        if load_traces:
            if ms.use_suite_2p or (ms.raw_traces is None):
                raw_traces_loaded = ms.load_raw_traces_from_npy(path=f"p{ms.age}/{ms.description.lower()}/")
                if not raw_traces_loaded:
                    ms.load_tiff_movie_in_memory()
                    ms.raw_traces = ms.build_raw_traces_from_movie()
                    ms.save_raw_traces(path=f"p{ms.age}/{ms.description.lower()}/")
        # TODO: option to load predictions, clean the raster produced to eliminate overlap base on co-activation
        # of overlaping cell and correlation value of source_profile, then save this file for later use
        # {self.description_}spike_nums_dur_predicted_and_cleaned_{key_prediction}.npy
        # LOADING PREDICTIONS if not loaded (if no spike_nums loaded, could be caiman one)
        if ms.spike_struct.spike_nums_dur is None:
            prediction_threshold = 0.5
            # key that should be on the prediction file_name to be loaded
            # prediction_key = "meso_v1_epoch_9" # "meso_v1_epoch_9"
            prediction_key = "gad_cre_v1_epoch_15"
            variables_mapping = {"predictions": "predictions"}
            ms.load_raster_dur_from_predictions(
                path_name=f"p{ms.age}/{ms.description.lower()}/predictions/",
                prediction_key=prediction_key,
                prediction_threshold=prediction_threshold,
                variables_mapping=variables_mapping, use_filtered_version=False)

        # next 2 functions won't do anything is spike_nums_dur is None
        ms.clean_raster_at_concatenation()

        ms.spike_struct.build_spike_nums_and_peak_nums()

        variables_mapping = {"shift_twitch": "shift_twitch",
                             "shift_long": "shift_long",
                             "shift_unclassified": "shift_unclassified"}
        ms.load_data_from_period_selection_gui(path_to_load=f"p{ms.age}/{ms.description.lower()}/",
                                                                variables_mapping=variables_mapping)

        ms.load_speed_from_file(path_to_load=f"p{ms.age}/{ms.description.lower()}/")

        ms.load_seq_pca_results(path=f"{param.path_data}/PCA_sequences_Robin/")

        ms.load_lfp_data(path_to_load=f"p{ms.age}/{ms.description.lower()}/")

        ms.load_graph_data(path_to_load=f"p{ms.age}/{ms.description.lower()}/")

        ms.load_raw_motion_translation_shift_data(path_to_load=f"p{ms.age}/{ms.description.lower()}/")

        if load_abf and (not ms.abf_loaded):
            # if sampling_rate is not 50000, load specific data for a session
            # loading abf
            if ms.age >= 10:
                # default LFP channel is 4
                # if LFP in channel 1, load specific data for a session
                ms.load_abf_file(path_abf_data=f"p{ms.age}/{ms.description.lower()}/", run_channel=2)
            else:
                # default LFP is channel 3
                # if channel 1 for LFP, load specific data for a session
                ms.load_abf_file(path_abf_data=f"p{ms.age}/{ms.description.lower()}/")

        # TODO: charge those files automatically if they exists

        variables_mapping = {"global_roi": "global_roi"}
        # p5_19_03_25_a001_ms.load_data_from_file(file_name_to_load=
        #                                         "p5/p5_19_03_25_a001/p5_19_03_25_a001_global_roi.mat",
        #                                         variables_mapping=variables_mapping)
        # variables_mapping = {"xshifts": "xshifts",
        #                      "yshifts": "yshifts"}
        # p5_19_03_25_a001_ms.load_data_from_file(file_name_to_load=
        #                                         "p5/p5_19_03_25_a001/MichelMotC_p5_19_03_25_a001_params.mat",
        #                                         variables_mapping=variables_mapping)


        # variables_mapping = {"xshifts": "xoff",
        #                      "yshifts": "yoff"}
        # p6_18_02_07_a001_ms.load_data_from_file(file_name_to_load=
        #                                         "p6/p6_18_02_07_a001/p6_18_02_07_a001_ops_params.npy",
        #                                         variables_mapping=variables_mapping)


    return ms_str_to_ms_dict
