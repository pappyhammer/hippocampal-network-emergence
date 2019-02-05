from mouse_session import MouseSession


def load_mouse_sessions(ms_str_to_load, param, load_traces, load_abf=True, load_movie=True,
                        for_cell_classifier=False, for_transient_classifier=False):
    # for_cell_classifier is True means we don't remove the cell that has been marked as fake cells
    ms_str_to_ms_dict = dict()

    if "p6_18_02_07_a001_ms" in ms_str_to_load:
        p6_18_02_07_a001_ms = MouseSession(age=6, session_id="18_02_07_a001", nb_ms_by_frame=100, param=param,
                                           weight=4.35)
        # calculated with 99th percentile on raster dur
        p6_18_02_07_a001_ms.activity_threshold = 15
        # p6_18_02_07_a001_ms.set_low_activity_threshold(threshold=3, percentile_value=1)
        # p6_18_02_07_a001_ms.set_low_activity_threshold(threshold=5, percentile_value=5)
        p6_18_02_07_a001_ms.set_inter_neurons([28, 36, 54, 75])
        # duration of those interneurons: [ 18.58 17.78   19.  17.67]
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p6_18_02_07_a001_ms.load_data_from_file(file_name_to_load=
                                                "p6/p6_18_02_07_a001/p6_18_02_07_001_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p6_18_02_07_a001_ms.load_data_from_file(file_name_to_load="p6/p6_18_02_07_a001/p6_18_02_07_a001_Traces.mat",
                                                    variables_mapping=variables_mapping)
            variables_mapping = {"raw_traces": "raw_traces"}
            p6_18_02_07_a001_ms.load_data_from_file(
                file_name_to_load="p6/p6_18_02_07_a001/p6_18_02_07_a001_raw_Traces.mat",
                variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p6_18_02_07_a001_ms.load_data_from_file(file_name_to_load="p6/p6_18_02_07_a001/p6_18_02_07_a001_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        p6_18_02_07_a001_ms.set_avg_cell_map_tif(file_name="p6/p6_18_02_07_a001/AVG_p6_18_02_07_a001.tif")
        if load_abf:
            p6_18_02_07_a001_ms.load_abf_file(abf_file_name="p6/p6_18_02_07_a001/p6_18_02_07_001.abf",
                                              threshold_piezo=25, just_load_npz_file=False)  # 7
        if load_movie:
            p6_18_02_07_a001_ms.load_tif_movie(path="p6/p6_18_02_07_a001/")
        ms_str_to_ms_dict["p6_18_02_07_a001_ms"] = p6_18_02_07_a001_ms
        # p6_18_02_07_a001_ms.plot_cell_assemblies_on_map()

    if "p6_18_02_07_a002_ms" in ms_str_to_load:
        p6_18_02_07_a002_ms = MouseSession(age=6, session_id="18_02_07_a002", nb_ms_by_frame=100, param=param,
                                           weight=4.35)
        # calculated with 99th percentile on raster dur
        p6_18_02_07_a002_ms.activity_threshold = 10
        # p6_18_02_07_a002_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        # p6_18_02_07_a002_ms.set_low_activity_threshold(threshold=1, percentile_value=5)
        p6_18_02_07_a002_ms.set_inter_neurons([40, 90])
        # duration of those interneurons: 16.27  23.33
        # variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
        #                      "spike_nums": "filt_Bin100ms_spikedigital",
        #                      "spike_durations": "LOC3"}
        variables_mapping = {"spike_nums_dur": "rasterdur"}
        p6_18_02_07_a002_ms.load_data_from_file(file_name_to_load=
                                                "p6/p6_18_02_07_a002/p6_18_02_07_a002_RasterDur_2nd_dec.mat",
                                                variables_mapping=variables_mapping)
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
        p6_18_02_07_a002_ms.set_avg_cell_map_tif(file_name="p6/p6_18_02_07_a002/AVG_p6_18_02_07_a002.tif")
        if load_abf:
            p6_18_02_07_a002_ms.load_abf_file(abf_file_name="p6/p6_18_02_07_a002/p6_18_02_07_002.abf",
                                              threshold_piezo=25)
        if load_movie:
            p6_18_02_07_a002_ms.load_tif_movie(path="p6/p6_18_02_07_a002/")
        ms_str_to_ms_dict["p6_18_02_07_a002_ms"] = p6_18_02_07_a002_ms

    if "p7_171012_a000_ms" in ms_str_to_load:
        p7_171012_a000_ms = MouseSession(age=7, session_id="17_10_12_a000", nb_ms_by_frame=100, param=param,
                                         weight=None)
        # calculated with 99th percentile on raster dur
        # p7_171012_a000_ms.activity_threshold = 19
        # p7_171012_a000_ms.set_low_activity_threshold(threshold=6, percentile_value=1)
        # p7_171012_a000_ms.set_low_activity_threshold(threshold=7, percentile_value=5)
        # p7_171012_a000_ms.set_inter_neurons([305, 360, 398, 412])
        # p7_171012_a000_ms.set_inter_neurons([])
        # duration of those interneurons: 13.23  12.48  10.8   11.88
        # variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
        #                      "spike_nums": "filt_Bin100ms_spikedigital",
        #                      "spike_durations": "LOC3"}
        # p7_171012_a000_ms.load_data_from_file(
        #     file_name_to_load="p7/p7_17_10_12_a000/p7_17_10_12_a000_Corrected_RasterDur.mat",
        #     variables_mapping=variables_mapping)
        p7_171012_a000_ms.set_avg_cell_map_tif(file_name="p7/p7_17_10_12_a000/AVG_p7_17_10_12_a000.tif")
        variables_mapping = {"coord": "ContoursAll"}
        p7_171012_a000_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_12_a000/p7_17_10_12_a000_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        if for_cell_classifier or for_transient_classifier:
            variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
                                 "peak_nums": "LocPeakMatrix_Python",
                                 "cells_to_remove": "cells_to_remove",
                                 "inter_neurons_from_gui": "inter_neurons"}
            p7_171012_a000_ms.load_data_from_file(file_name_to_load=
                                                   "p7/p7_17_10_12_a000/p7_17_10_12_a000_GUI_transients_RD.mat",
                                                   variables_mapping=variables_mapping,
                                                   from_gui=True)

            p7_171012_a000_ms.build_spike_nums_dur()
            if for_cell_classifier:
                p7_171012_a000_ms.\
                    load_cells_to_remove_from_txt(file_name="p7/p7_17_10_12_a000/"
                                                            "p7_17_10_12_a000_cell_to_suppress_ground_truth.txt")
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p7_171012_a000_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_12_a000/p7_17_10_12_a000_Traces.mat",
                                                  variables_mapping=variables_mapping)
            variables_mapping = {"raw_traces": "raw_traces"}
            p7_171012_a000_ms.load_data_from_file(
                file_name_to_load="p7/p7_17_10_12_a000/p7_17_10_12_a000_raw_Traces.mat",
                variables_mapping=variables_mapping)
        # variables_mapping = {"coord": "ContoursAll"} ContoursSoma ContoursIntNeur
        # p7_171012_a000_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_12_a000/p7_17_10_12_a000_CellDetect.mat",
        #                                          variables_mapping=variables_mapping)
        if load_movie:
            p7_171012_a000_ms.load_tif_movie(path="p7/p7_17_10_12_a000/")
        ms_str_to_ms_dict["p7_171012_a000_ms"] = p7_171012_a000_ms

    if "p7_17_10_18_a002_ms" in ms_str_to_load:
        p7_17_10_18_a002_ms = MouseSession(age=7, session_id="17_10_18_a002", nb_ms_by_frame=100, param=param,
                                           weight=None)
        # calculated with 99th percentile on raster dur
        p7_17_10_18_a002_ms.activity_threshold = 14
        # p7_17_10_18_a002_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
        # p7_17_10_18_a002_ms.set_low_activity_threshold(threshold=4, percentile_value=5)
        p7_17_10_18_a002_ms.set_inter_neurons([51])
        # duration of those interneurons: 14.13
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p7_17_10_18_a002_ms.load_data_from_file(file_name_to_load=
                                                "p7/p7_17_10_18_a002/p7_17_10_18_a002_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        p7_17_10_18_a002_ms.set_avg_cell_map_tif(file_name="p7/p7_17_10_18_a002/AVG_p7_17_10_18_a002.tif")
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
        if load_movie:
            p7_17_10_18_a002_ms.load_tif_movie(path="p7/p7_17_10_18_a002/")
        ms_str_to_ms_dict["p7_17_10_18_a002_ms"] = p7_17_10_18_a002_ms

    if "p7_17_10_18_a004_ms" in ms_str_to_load:
        p7_17_10_18_a004_ms = MouseSession(age=7, session_id="17_10_18_a004", nb_ms_by_frame=100, param=param,
                                           weight=None)
        # calculated with 99th percentile on raster dur
        p7_17_10_18_a004_ms.activity_threshold = 13
        # p7_17_10_18_a004_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
        # p7_17_10_18_a004_ms.set_low_activity_threshold(threshold=3, percentile_value=5)
        p7_17_10_18_a004_ms.set_inter_neurons([298])
        # duration of those interneurons: 15.35
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p7_17_10_18_a004_ms.load_data_from_file(file_name_to_load=
                                                "p7/p7_17_10_18_a004/p7_17_10_18_a004_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        p7_17_10_18_a004_ms.set_avg_cell_map_tif(file_name="p7/p7_17_10_18_a004/AVG_p7_17_10_18_a004.tif")
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
        if load_movie:
            p7_17_10_18_a004_ms.load_tif_movie(path="p7/p7_17_10_18_a004/")
        ms_str_to_ms_dict["p7_17_10_18_a004_ms"] = p7_17_10_18_a004_ms

    if "p7_18_02_08_a000_ms" in ms_str_to_load:
        p7_18_02_08_a000_ms = MouseSession(age=7, session_id="18_02_08_a000", nb_ms_by_frame=100, param=param,
                                           weight=3.85)
        # calculated with 99th percentile on raster dur
        p7_18_02_08_a000_ms.activity_threshold = 10
        # p7_18_02_08_a000_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
        # p7_18_02_08_a000_ms.set_low_activity_threshold(threshold=2, percentile_value=5)
        p7_18_02_08_a000_ms.set_inter_neurons([56, 95, 178])
        # duration of those interneurons: 12.88  13.94  13.04
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p7_18_02_08_a000_ms.load_data_from_file(file_name_to_load=
                                                "p7/p7_18_02_08_a000/p7_18_02_18_a000_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        p7_18_02_08_a000_ms.set_avg_cell_map_tif(file_name="p7/p7_18_02_08_a000/AVG_p7_18_02_08_a000.tif")
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
        if load_movie:
            p7_18_02_08_a000_ms.load_tif_movie(path="p7/p7_18_02_08_a000/")

        if load_abf:
            p7_18_02_08_a000_ms.load_abf_file(abf_file_name="p7/p7_18_02_08_a000/p7_18_02_08_a000.abf",
                                              threshold_ratio=2, just_load_npz_file=True)  # threshold_piezo=4,
        ms_str_to_ms_dict["p7_18_02_08_a000_ms"] = p7_18_02_08_a000_ms

    if "p7_18_02_08_a001_ms" in ms_str_to_load:
        p7_18_02_08_a001_ms = MouseSession(age=7, session_id="18_02_08_a001", nb_ms_by_frame=100, param=param,
                                           weight=3.85)
        # calculated with 99th percentile on raster dur
        p7_18_02_08_a001_ms.activity_threshold = 12
        # p7_18_02_08_a001_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
        # p7_18_02_08_a001_ms.set_low_activity_threshold(threshold=3, percentile_value=5)
        p7_18_02_08_a001_ms.set_inter_neurons([151])
        # duration of those interneurons: 22.11
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
        variables_mapping = {"coord": "ContoursAll"}
        p7_18_02_08_a001_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a001/p7_18_02_08_a001_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        if load_abf:
            p7_18_02_08_a001_ms.load_abf_file(abf_file_name="p7/p7_18_02_08_a001/p7_18_02_08_a001.abf",
                                              threshold_piezo=4)
        if load_movie:
            p7_18_02_08_a001_ms.load_tif_movie(path="p7/p7_18_02_08_a001/")
        ms_str_to_ms_dict["p7_18_02_08_a001_ms"] = p7_18_02_08_a001_ms

    if "p7_18_02_08_a002_ms" in ms_str_to_load:
        p7_18_02_08_a002_ms = MouseSession(age=7, session_id="18_02_08_a002", nb_ms_by_frame=100, param=param,
                                           weight=3.85)
        # calculated with 99th percentile on raster dur
        p7_18_02_08_a002_ms.activity_threshold = 9
        # p7_18_02_08_a002_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
        # p7_18_02_08_a002_ms.set_low_activity_threshold(threshold=1, percentile_value=5)
        p7_18_02_08_a002_ms.set_inter_neurons([207])
        # duration of those interneurons: 22.3
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
        if load_movie:
            p7_18_02_08_a002_ms.load_tif_movie(path="p7/p7_18_02_08_a002/")
        if load_abf:
            p7_18_02_08_a002_ms.load_abf_file(abf_file_name="p7/p7_18_02_08_a002/p7_18_02_08_a002.abf",
                                              threshold_piezo=2.5)
        ms_str_to_ms_dict["p7_18_02_08_a002_ms"] = p7_18_02_08_a002_ms

    if "p7_18_02_08_a003_ms" in ms_str_to_load:
        p7_18_02_08_a003_ms = MouseSession(age=7, session_id="18_02_08_a003", nb_ms_by_frame=100, param=param,
                                           weight=3.85)
        # calculated with 99th percentile on raster dur
        p7_18_02_08_a003_ms.activity_threshold = 7
        # p7_18_02_08_a003_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        # p7_18_02_08_a003_ms.set_low_activity_threshold(threshold=0, percentile_value=5)
        p7_18_02_08_a003_ms.set_inter_neurons([171])
        # duration of those interneurons: 14.92
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
        if load_movie:
            p7_18_02_08_a003_ms.load_tif_movie(path="p7/p7_18_02_08_a003/")
        if load_abf:
            p7_18_02_08_a003_ms.load_abf_file(abf_file_name="p7/p7_18_02_08_a003/p7_18_02_08_a003.abf",
                                              threshold_piezo=9)  # used to be 2.5
        ms_str_to_ms_dict["p7_18_02_08_a003_ms"] = p7_18_02_08_a003_ms

    if "p8_18_02_09_a000_ms" in ms_str_to_load:
        p8_18_02_09_a000_ms = MouseSession(age=8, session_id="18_02_09_a000", nb_ms_by_frame=100, param=param,
                                           weight=None)
        # calculated with 99th percentile on raster dur
        p8_18_02_09_a000_ms.activity_threshold = 8
        # p8_18_02_09_a000_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        # p8_18_02_09_a000_ms.set_low_activity_threshold(threshold=1, percentile_value=5)
        p8_18_02_09_a000_ms.set_inter_neurons([64, 91])
        # duration of those interneurons: 12.48  11.47
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
        if load_abf:
            p8_18_02_09_a000_ms.load_abf_file(abf_file_name="p8/p8_18_02_09_a000/p8_18_02_09_a000.abf",
                                              threshold_piezo=2)  # used to be 1.5
        if load_movie:
            p8_18_02_09_a000_ms.load_tif_movie(path="p8/p8_18_02_09_a000/")
        ms_str_to_ms_dict["p8_18_02_09_a000_ms"] = p8_18_02_09_a000_ms

    if "p8_18_02_09_a001_ms" in ms_str_to_load:
        p8_18_02_09_a001_ms = MouseSession(age=8, session_id="18_02_09_a001", nb_ms_by_frame=100, param=param,
                                           weight=None)
        # calculated with 99th percentile on raster dur
        p8_18_02_09_a001_ms.activity_threshold = 10
        # p8_18_02_09_a001_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
        # p8_18_02_09_a001_ms.set_low_activity_threshold(threshold=2, percentile_value=5)
        p8_18_02_09_a001_ms.set_inter_neurons([])
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
        variables_mapping = {"coord": "ContoursAll"}
        p8_18_02_09_a001_ms.load_data_from_file(file_name_to_load="p8/p8_18_02_09_a001/p8_18_02_09_a001_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        p8_18_02_09_a001_ms.set_avg_cell_map_tif(file_name="p8/p8_18_02_09_a001/AVG_p8_18_02_09_a001.tif")
        if load_abf:
            p8_18_02_09_a001_ms.load_abf_file(abf_file_name="p8/p8_18_02_09_a001/p8_18_02_09_a001.abf",
                                              threshold_piezo=3)  # 1.5 before then 2
        if load_movie:
            p8_18_02_09_a001_ms.load_tif_movie(path="p8/p8_18_02_09_a001/")
        ms_str_to_ms_dict["p8_18_02_09_a001_ms"] = p8_18_02_09_a001_ms

    if "p8_18_10_17_a000_ms" in ms_str_to_load:
        p8_18_10_17_a000_ms = MouseSession(age=8, session_id="18_10_17_a000", nb_ms_by_frame=100, param=param,
                                           weight=6)
        # calculated with 99th percentile on raster dur
        p8_18_10_17_a000_ms.activity_threshold = 11
        # p8_18_10_17_a000_ms.set_low_activity_threshold(threshold=, percentile_value=1)
        # p8_18_10_17_a000_ms.set_low_activity_threshold(threshold=, percentile_value=5)
        p8_18_10_17_a000_ms.set_inter_neurons([27, 70])
        # duration of those interneurons: 23.8, 43
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
        if load_movie:
            p8_18_10_17_a000_ms.load_tif_movie(path="p8/p8_18_10_17_a000/")
        ms_str_to_ms_dict["p8_18_10_17_a000_ms"] = p8_18_10_17_a000_ms

    if "p8_18_10_17_a001_ms" in ms_str_to_load:
        p8_18_10_17_a001_ms = MouseSession(age=8, session_id="18_10_17_a001", nb_ms_by_frame=100, param=param,
                                           weight=6)
        # calculated with 99th percentile on raster dur
        p8_18_10_17_a001_ms.activity_threshold = 9
        # p8_18_10_17_a001_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        # p8_18_10_17_a001_ms.set_low_activity_threshold(threshold=1, percentile_value=5)
        p8_18_10_17_a001_ms.set_inter_neurons([117, 135, 217, 271])
        # duration of those interneurons: 32.33, 171, 144.5, 48.8
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
        variables_mapping = {"coord": "ContoursAll"}
        p8_18_10_17_a001_ms.load_data_from_file(file_name_to_load="p8/p8_18_10_17_a001/p8_18_10_17_a001_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        p8_18_10_17_a001_ms.set_avg_cell_map_tif(file_name="p8/p8_18_10_17_a001/AVG_p8_18_10_17_a001.tif")
        # CORRUPTED ABF ??
        if load_abf:
            p8_18_10_17_a001_ms.load_abf_file(abf_file_name="p8/p8_18_10_17_a001/p8_18_10_17_a001.abf",
                                              threshold_piezo=0.4, piezo_channel=2, sampling_rate=10000)
        if load_movie:
            p8_18_10_17_a001_ms.load_tif_movie(path="p8/p8_18_10_17_a001/")
        ms_str_to_ms_dict["p8_18_10_17_a001_ms"] = p8_18_10_17_a001_ms

    if "p8_18_10_24_a005_ms" in ms_str_to_load:
        # 6.4
        p8_18_10_24_a005_ms = MouseSession(age=8, session_id="18_10_24_a005", nb_ms_by_frame=100, param=param,
                                           weight=6.4)
        # calculated with 99th percentile on raster dur
        p8_18_10_24_a005_ms.activity_threshold = 9
        # p8_18_10_24_a005_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        # p8_18_10_24_a005_ms.set_low_activity_threshold(threshold=1, percentile_value=5)
        # p8_18_10_24_a005_ms.set_inter_neurons([33, 112, 206])
        # duration of those interneurons: 18.92, 27.33, 20.55
        # variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
        #                      "spike_nums": "filt_Bin100ms_spikedigital",
        #                      "spike_durations": "LOC3"}
        # p8_18_10_24_a005_ms.load_data_from_file(file_name_to_load=
        #                                         "p8/p8_18_10_24_a005/p8_18_10_24_a005_Corrected_RasterDur.mat",
        #                                         variables_mapping=variables_mapping)
        p8_18_10_24_a005_ms.set_avg_cell_map_tif(file_name="p8/p8_18_10_24_a005/AVG_p8_18_10_24_a005.tif")

        if for_cell_classifier or for_transient_classifier:
            variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
                                 "peak_nums": "LocPeakMatrix_Python",
                                 "cells_to_remove": "cells_to_remove",
                                 "inter_neurons_from_gui": "inter_neurons"}
            p8_18_10_24_a005_ms.load_data_from_file(file_name_to_load=
                                                   "p8/p8_18_10_24_a005/p8_18_10_24_a005_GUI_Transiant MP.mat",
                                                   variables_mapping=variables_mapping,
                                                   from_gui=True)
            p8_18_10_24_a005_ms.build_spike_nums_dur()

        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p8_18_10_24_a005_ms.load_data_from_file(file_name_to_load="p8/p8_18_10_24_a005/p8_18_10_24_a005_Traces.mat",
                                                    variables_mapping=variables_mapping)
            variables_mapping = {"raw_traces": "raw_traces"}
            p8_18_10_24_a005_ms.load_data_from_file(
                file_name_to_load="p8/p8_18_10_24_a005/p8_18_10_24_a005_raw_Traces.mat",
                variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p8_18_10_24_a005_ms.load_data_from_file(file_name_to_load="p8/p8_18_10_24_a005/p8_18_10_24_a005_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        if load_abf:
            p8_18_10_24_a005_ms.load_abf_file(abf_file_name="p8/p8_18_10_24_a005/p8_18_10_24_a005.abf",
                                              threshold_piezo=0.5)  # used to be 0.4
        if load_movie:
            p8_18_10_24_a005_ms.load_tif_movie(path="p8/p8_18_10_24_a005/")
        ms_str_to_ms_dict["p8_18_10_24_a005_ms"] = p8_18_10_24_a005_ms

    # p9_17_11_29_a002 low participation comparing to other, dead shortly after the recording
    # p9_17_11_29_a002_ms = MouseSession(age=9, session_id="17_11_29_a002", nb_ms_by_frame=100, param=param,
    #                                    weight=5.7)
    # # calculated with 99th percentile on raster dur
    # p9_17_11_29_a002_ms.activity_threshold = 10
    # p9_17_11_29_a002_ms.set_inter_neurons([170])
    # # limit ??
    # # duration of those interneurons: 21
    # variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
    #                      "spike_nums": "filt_Bin100ms_spikedigital",
    #                      "spike_durations": "LOC3"}
    # p9_17_11_29_a002_ms.load_data_from_file(file_name_to_load="p9/p9_17_11_29_a002/p9_17_11_29_a002_RasterDur.mat",
    #                                         variables_mapping=variables_mapping)

    # p9_17_11_29_a003_ms = MouseSession(age=9, session_id="17_11_29_a003", nb_ms_by_frame=100, param=param,
    #                                    weight=5.7)
    # # calculated with 99th percentile on raster dur
    # p9_17_11_29_a003_ms.activity_threshold = 7
    # p9_17_11_29_a003_ms.set_inter_neurons([1, 13, 54])
    # # duration of those interneurons: 21.1 22.75  23
    # variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
    #                      "spike_nums": "filt_Bin100ms_spikedigital",
    #                      "spike_durations": "LOC3"}
    # p9_17_11_29_a003_ms.load_data_from_file(file_name_to_load="p9/p9_17_11_29_a003/p9_17_11_29_a003_RasterDur.mat",
    #                                         variables_mapping=variables_mapping)
    if "p9_17_12_06_a001_ms" in ms_str_to_load:
        p9_17_12_06_a001_ms = MouseSession(age=9, session_id="17_12_06_a001", nb_ms_by_frame=100, param=param,
                                           weight=5.6)
        # calculated with 99th percentile on raster dur
        p9_17_12_06_a001_ms.activity_threshold = 8
        # p9_17_12_06_a001_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        p9_17_12_06_a001_ms.set_inter_neurons([72])
        # duration of those interneurons:15.88
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
        if load_abf:
            p9_17_12_06_a001_ms.load_abf_file(abf_file_name="p9/p9_17_12_06_a001/p9_17_12_06_a001.abf",
                                              threshold_piezo=1.5)
        if load_movie:
            p9_17_12_06_a001_ms.load_tif_movie(path="p9/p9_17_12_06_a001/")
        ms_str_to_ms_dict["p9_17_12_06_a001_ms"] = p9_17_12_06_a001_ms

    if "p9_17_12_20_a001_ms" in ms_str_to_load:
        p9_17_12_20_a001_ms = MouseSession(age=9, session_id="17_12_20_a001", nb_ms_by_frame=100, param=param,
                                           weight=5.05)
        # calculated with 99th percentile on raster dur
        p9_17_12_20_a001_ms.activity_threshold = 8
        # p9_17_12_20_a001_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        p9_17_12_20_a001_ms.set_inter_neurons([32])
        # duration of those interneurons: 10.35
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
        p9_17_12_20_a001_ms.set_avg_cell_map_tif(file_name="p9/p9_17_12_20_a001/AVG_p9_17_12_20_a001.tif")
        if load_abf:
            p9_17_12_20_a001_ms.load_abf_file(abf_file_name="p9/p9_17_12_20_a001/p9_17_12_20_a001.abf",
                                              threshold_piezo=3)  # used to be 2
        if load_movie:
            p9_17_12_20_a001_ms.load_tif_movie(path="p9/p9_17_12_20_a001/")
        ms_str_to_ms_dict["p9_17_12_20_a001_ms"] = p9_17_12_20_a001_ms

    if "p9_18_09_27_a003_ms" in ms_str_to_load:
        p9_18_09_27_a003_ms = MouseSession(age=9, session_id="18_09_27_a003", nb_ms_by_frame=100, param=param,
                                           weight=6.65)
        # calculated with 99th percentile on raster dur
        p9_18_09_27_a003_ms.activity_threshold = 9
        # p9_18_09_27_a003_ms.set_low_activity_threshold(threshold=, percentile_value=1)
        p9_18_09_27_a003_ms.set_inter_neurons([2, 9, 67, 206])
        # duration of those interneurons: 59.1, 32, 28, 35.15
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
        p9_18_09_27_a003_ms.set_avg_cell_map_tif(file_name="p9/p9_18_09_27_a003/AVG_p9_18_09_27_a003.tif")
        if load_abf:
            p9_18_09_27_a003_ms.load_abf_file(abf_file_name="p9/p9_18_09_27_a003/p9_18_09_27_a003.abf",
                                              threshold_piezo=0.06, piezo_channel=2, sampling_rate=10000,
                                              offset=0.1)
        ms_str_to_ms_dict["p9_18_09_27_a003_ms"] = p9_18_09_27_a003_ms

    if "p10_17_11_16_a003_ms" in ms_str_to_load:
        p10_17_11_16_a003_ms = MouseSession(age=10, session_id="17_11_16_a003", nb_ms_by_frame=100, param=param,
                                            weight=6.1)
        # calculated with 99th percentile on raster dur
        p10_17_11_16_a003_ms.activity_threshold = 6
        # p10_17_11_16_a003_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        p10_17_11_16_a003_ms.set_inter_neurons([8])
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
        p10_17_11_16_a003_ms.set_avg_cell_map_tif(file_name="p10/p10_17_11_16_a003/AVG_p10_17_11_16_a003.tif")
        variables_mapping = {"coord": "ContoursAll"}
        p10_17_11_16_a003_ms.load_data_from_file(
            file_name_to_load="p10/p10_17_11_16_a003/p10_17_11_16_a003_CellDetect.mat",
            variables_mapping=variables_mapping)
        if load_movie:
            p10_17_11_16_a003_ms.load_tif_movie(path="p10/p10_17_11_16_a003/")
        ms_str_to_ms_dict["p10_17_11_16_a003_ms"] = p10_17_11_16_a003_ms

    if "p11_17_11_24_a000_ms" in ms_str_to_load:
        p11_17_11_24_a000_ms = MouseSession(age=11, session_id="17_11_24_a000", nb_ms_by_frame=100, param=param,
                                            weight=6.7)
        # calculated with 99th percentile on raster dur
        # p11_17_11_24_a000_ms.activity_threshold = 11
        # p11_17_11_24_a000_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
        # p11_17_11_24_a000_ms.set_inter_neurons([193])
        # duration of those interneurons: 19.09
        # variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
        #                      "spike_nums": "filt_Bin100ms_spikedigital",
        #                      "spike_durations": "LOC3"}
        # p11_17_11_24_a000_ms.load_data_from_file(file_name_to_load=
        #                                          "p11/p11_17_11_24_a000/p11_17_11_24_a000_Corrected_RasterDur.mat",
        #
        #                                          variables_mapping=variables_mapping)
        if for_cell_classifier or for_transient_classifier:
            variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
                                 "peak_nums": "LocPeakMatrix_Python",
                                 "cells_to_remove": "cells_to_remove",
                                 "inter_neurons_from_gui": "inter_neurons"}
            p11_17_11_24_a000_ms.load_data_from_file(file_name_to_load=
                                                    "p11/p11_17_11_24_a000/p11_17_11_24_a000_GUI_transients_RD.mat",
                                                    variables_mapping=variables_mapping,
                                                    from_gui=True)
            p11_17_11_24_a000_ms.build_spike_nums_dur()
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
        p11_17_11_24_a000_ms.set_avg_cell_map_tif(file_name="p11/p11_17_11_24_a000/AVG_p11_17_11_24_a000.tif")
        if load_movie:
            p11_17_11_24_a000_ms.load_tif_movie(path="p11/p11_17_11_24_a000/")
        # p11_17_11_24_a000_ms.plot_cell_assemblies_on_map()
        ms_str_to_ms_dict["p11_17_11_24_a000_ms"] = p11_17_11_24_a000_ms

    if "p11_17_11_24_a001_ms" in ms_str_to_load:
        p11_17_11_24_a001_ms = MouseSession(age=11, session_id="17_11_24_a001", nb_ms_by_frame=100, param=param,
                                            weight=6.7)
        # calculated with 99th percentile on raster dur
        p11_17_11_24_a001_ms.activity_threshold = 10
        # p11_17_11_24_a001_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
        p11_17_11_24_a001_ms.set_inter_neurons([])
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
        p11_17_11_24_a001_ms.set_avg_cell_map_tif(file_name="p11/p11_17_11_24_a001/AVG_p11_17_11_24_a001.tif")
        if load_movie:
            p11_17_11_24_a001_ms.load_tif_movie(path="p11/p11_17_11_24_a001/")
        ms_str_to_ms_dict["p11_17_11_24_a001_ms"] = p11_17_11_24_a001_ms

    if "p12_171110_a000_ms" in ms_str_to_load:
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
        if for_cell_classifier or for_transient_classifier:
            variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
                                 "peak_nums": "LocPeakMatrix_Python",
                                 "cells_to_remove": "cells_to_remove",
                                 "inter_neurons_from_gui": "inter_neurons"}
            p12_171110_a000_ms.load_data_from_file(file_name_to_load=
                                                   "p12/p12_17_11_10_a000/p12_17_11_10_a000_GUI_JDMP.mat",
                                                   variables_mapping=variables_mapping,
                                                   from_gui=True)
            p12_171110_a000_ms.build_spike_nums_dur()
            if for_cell_classifier:
                p12_171110_a000_ms.load_cells_to_remove_from_txt(file_name="p12/p12_17_11_10_a000/"
                                                                       "p12_17_11_10_a000_cell_to_suppress_ground_truth.txt")
        # else:
        #     variables_mapping = {"spike_nums_dur": "spike_nums_dur_predicted"}
        #     p12_171110_a000_ms.\
        #         load_data_from_file(file_name_to_load=
        #                             "p12/p12_17_11_10_a000/P12_17_11_10_a000_predictions_2019_01_26.19-22-21.mat",
        #                             variables_mapping=variables_mapping)
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
        if load_movie:
            p12_171110_a000_ms.load_tif_movie(path="p12/p12_17_11_10_a000/")

        if (not for_cell_classifier) and (not for_transient_classifier):
            p12_171110_a000_ms.clean_data_using_cells_to_remove()

        ms_str_to_ms_dict["p12_171110_a000_ms"] = p12_171110_a000_ms

    if "p12_17_11_10_a002_ms" in ms_str_to_load:
        p12_17_11_10_a002_ms = MouseSession(age=12, session_id="17_11_10_a002", nb_ms_by_frame=100, param=param,
                                            weight=7)
        # calculated with 99th percentile on raster dur
        p12_17_11_10_a002_ms.activity_threshold = 11
        # p12_17_11_10_a002_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
        p12_17_11_10_a002_ms.set_inter_neurons([150, 252])
        # duration of those interneurons: 16.17, 24.8
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
        p12_17_11_10_a002_ms.set_avg_cell_map_tif(file_name="p12/p12_17_11_10_a002/AVG_p12_17_11_10_a002.tif")
        if load_movie:
            p12_17_11_10_a002_ms.load_tif_movie(path="p12/p12_17_11_10_a002/")

        ms_str_to_ms_dict["p12_17_11_10_a002_ms"] = p12_17_11_10_a002_ms

    if "p13_18_10_29_a000_ms" in ms_str_to_load:
        p13_18_10_29_a000_ms = MouseSession(age=13, session_id="18_10_29_a000", nb_ms_by_frame=100, param=param,
                                            weight=9.4)
        # calculated with 99th percentile on raster dur
        p13_18_10_29_a000_ms.activity_threshold = 13
        # p13_18_10_29_a000_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
        p13_18_10_29_a000_ms.set_inter_neurons([5, 26, 27, 35, 38])
        # duration of those interneurons: 13.57, 16.8, 22.4, 12, 14.19
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
        variables_mapping = {"coord": "ContoursAll"}
        p13_18_10_29_a000_ms.load_data_from_file(file_name_to_load=
                                                 "p13/p13_18_10_29_a000/p13_18_10_29_a000_CellDetect.mat",
                                                 variables_mapping=variables_mapping)
        p13_18_10_29_a000_ms.set_avg_cell_map_tif(file_name="p13/p13_18_10_29_a000/AVG_p13_18_10_29_a000.tif")
        if load_abf:
            p13_18_10_29_a000_ms.load_abf_file(abf_file_name="p13/p13_18_10_29_a000/p13_18_10_29_a000.abf",
                                               threshold_piezo=None, with_run=True, sampling_rate=10000)
        if load_movie:
            p13_18_10_29_a000_ms.load_tif_movie(path="p13/p13_18_10_29_a000/")
        ms_str_to_ms_dict["p13_18_10_29_a000_ms"] = p13_18_10_29_a000_ms

    if "p13_18_10_29_a001_ms" in ms_str_to_load:
        p13_18_10_29_a001_ms = MouseSession(age=13, session_id="18_10_29_a001", nb_ms_by_frame=100, param=param,
                                            weight=9.4)
        # calculated with 99th percentile on raster dur
        # p13_18_10_29_a001_ms.activity_threshold = 11
        # p13_18_10_29_a001_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
        # p13_18_10_29_a001_ms.set_inter_neurons([68])
        # duration of those interneurons: 13.31

        if for_cell_classifier or for_transient_classifier:
            variables_mapping = {"spike_nums": "Bin100ms_spikedigital_Python",
                                 "peak_nums": "LocPeakMatrix_Python",
                                 "cells_to_remove": "cells_to_remove",
                                 "inter_neurons_from_gui": "inter_neurons"}
            p13_18_10_29_a001_ms.load_data_from_file(file_name_to_load=
                                                  "p13/p13_18_10_29_a001/p13_18_10_29_a001_GUI_transients_RD.mat",
                                                  variables_mapping=variables_mapping,
                                                  from_gui=True)

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
                file_name_to_load="p13/p13_18_10_29_a001/p13_2018_10_29_a001_raw_Traces.mat",
                variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p13_18_10_29_a001_ms.load_data_from_file(file_name_to_load=
                                                 "p13/p13_18_10_29_a001/p13_18_10_29_a001_CellDetect.mat",
                                                 variables_mapping=variables_mapping)
        p13_18_10_29_a001_ms.set_avg_cell_map_tif(file_name="p13/p13_18_10_29_a001/AVG_p13_18_10_29_a001.tif")
        if load_abf:
            p13_18_10_29_a001_ms.load_abf_file(abf_file_name="p13/p13_18_10_29_a001/p13_18_10_29_a001.abf",
                                               threshold_piezo=None, with_run=True, sampling_rate=10000)
        if load_movie:
            p13_18_10_29_a001_ms.load_tif_movie(path="p13/p13_18_10_29_a001/")
        ms_str_to_ms_dict["p13_18_10_29_a001_ms"] = p13_18_10_29_a001_ms

    if "p14_18_10_23_a000_ms" in ms_str_to_load:
        p14_18_10_23_a000_ms = MouseSession(age=14, session_id="18_10_23_a000", nb_ms_by_frame=100, param=param,
                                            weight=10.35)
        # calculated with 99th percentile on raster dur
        p14_18_10_23_a000_ms.activity_threshold = 8
        # p14_18_10_23_a000_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        p14_18_10_23_a000_ms.set_inter_neurons([0])
        # duration of those interneurons: 24.33
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
        p14_18_10_23_a000_ms.set_avg_cell_map_tif(file_name="p14/p14_18_10_23_a000/AVG_p14_18_10_23_a000.tif")
        if load_movie:
            p14_18_10_23_a000_ms.load_tif_movie(path="p14/p14_18_10_23_a000/")
        ms_str_to_ms_dict["p14_18_10_23_a000_ms"] = p14_18_10_23_a000_ms

    if "p14_18_10_23_a001_ms" in ms_str_to_load:
        # only interneurons in p14_18_10_23_a001_ms
        p14_18_10_23_a001_ms = MouseSession(age=14, session_id="18_10_23_a001", nb_ms_by_frame=100, param=param,
                                            weight=10.35)
        # calculated with 99th percentile on raster dur
        p14_18_10_23_a001_ms.activity_threshold = 8
        # p14_18_10_23_a001_ms.set_inter_neurons(np.arange(31))
        p14_18_10_23_a001_ms.set_inter_neurons([])
        # duration of those interneurons: 24.33
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
        p14_18_10_23_a001_ms.set_avg_cell_map_tif(file_name="p14/p14_18_10_23_a001/AVG_p14_18_10_23_a001.tif")
        if load_movie:
            p14_18_10_23_a001_ms.load_tif_movie(path="p14/p14_18_10_23_a001/")
        ms_str_to_ms_dict["p14_18_10_23_a001_ms"] = p14_18_10_23_a001_ms

    if "p14_18_10_30_a001_ms" in ms_str_to_load:
        p14_18_10_30_a001_ms = MouseSession(age=14, session_id="18_10_30_a001", nb_ms_by_frame=100, param=param,
                                            weight=8.9)
        # calculated with 99th percentile on raster dur
        p14_18_10_30_a001_ms.activity_threshold = 11
        # p14_18_10_30_a001_ms.set_low_activity_threshold(threshold=, percentile_value=1)
        p14_18_10_30_a001_ms.set_inter_neurons([0])
        # duration of those interneurons: 24.33
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
        p14_18_10_30_a001_ms.set_avg_cell_map_tif(file_name="p14/p14_18_10_30_a001/AVG_p14_18_10_30_a001.tif")
        if load_movie:
            p14_18_10_30_a001_ms.load_tif_movie(path="p14/p14_18_10_30_a001/")
        ms_str_to_ms_dict["p14_18_10_30_a001_ms"] = p14_18_10_30_a001_ms

    # arnaud_ms = MouseSession(age=24, session_id="arnaud", nb_ms_by_frame=50, param=param)
    # arnaud_ms.activity_threshold = 13
    # arnaud_ms.set_inter_neurons([])
    # variables_mapping = {"spike_nums": "spikenums"}
    # arnaud_ms.load_data_from_file(file_name_to_load="spikenumsarnaud.mat", variables_mapping=variables_mapping)

    if "p60_arnaud_ms" in ms_str_to_load:
        p60_arnaud_ms = MouseSession(age=60, session_id="arnaud_a_529", nb_ms_by_frame=100, param=param)
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
        p60_a529_2015_02_25_ms = MouseSession(age=60, session_id="a529_2015_02_25", nb_ms_by_frame=100, param=param)
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

    return ms_str_to_ms_dict
