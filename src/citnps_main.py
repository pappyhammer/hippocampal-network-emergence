from citnps.citnps_main_window import run_citnps
import os
from datetime import datetime
import numpy as np
from citnps.utils.video.video_reader import OpenCvVideoReader
from pynwb import NWBHDF5IO
import math
import yaml

class AnalysisOfMvt:

    def __init__(self, data_path, results_path, identifier, nwb_file, left_cam_id, right_cam_id,
                 config_citnps_yaml_file, mandatory_piezo_threshold=None,
                 bodyparts_to_fusion=None,
                 cicada_file=None):
        """

        Args:
            data_path: directory where the npz files containing the piezo signal is located
            results_path:
            identifier:
            nwb_file:
            left_cam_id:
            right_cam_id:
            config_citnps_yaml_file: yaml file output of citnps analysis. Useful to get the threshold for the
            piezo signal.
            bodyparts_to_fusion: None or list of str, represent to bodypart to fusion if multiple angle exists
            (like tail: tail left and right will be merged, if one of both move, then tail is moving)
            cicada_file:
        """
        self.data_path = data_path
        # each key is a the side, each value is a dict with 1 key the bodypart and value is the movement value (1d array)
        # evauated for this bodypart at a given timestamps
        self.mvt_data_dict = dict()
        self.results_path = results_path
        self.cicada_file = cicada_file
        self.nwb_file = nwb_file
        self.n_frames = 0
        self.identifier = identifier
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id

        self.mandatory_piezo_threshold = mandatory_piezo_threshold

        # each key is a bodypart + "_" + side, each value is a float representing the piezo signal threshold
        self.threshold_z_score_mvt = dict()
        if os.path.isfile(config_citnps_yaml_file):
            with open(config_citnps_yaml_file, 'r') as stream:
                citnps_config = yaml.load(stream, Loader=yaml.FullLoader)
                for bodypart, bodypart_config_dict in citnps_config.items():
                    if mandatory_piezo_threshold is not None:
                        self.threshold_z_score_mvt[bodypart] = mandatory_piezo_threshold
                    else:
                        self.threshold_z_score_mvt[bodypart] = bodypart_config_dict["piezo_threshold"]
        else:
            raise Exception("No config_citnps_yaml_file given")


        # self.threshold_z_score_mvt = 0.75

        self.n_body_parts = 0

        self.bodyparts_to_fusion = bodyparts_to_fusion if bodyparts_to_fusion is not None else list()

        self.time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")

        self.bodyparts = ['forelimb_left', 'forelimb_right', "hindlimb_left", "hindlimb_right", "tail"]

        # checking with have all values in config
        # for bodypart in self.bodyparts:
        #     if bodypart not in self.threshold_z_score_mvt:
        #         raise Exception(f"{bodypart} is missing in the config_citnps_yaml_file")

        # take as key the index of the body_part and return its name
        self.body_parts_name_dict = dict()
        # take the name of a bodypart and return it's index, in body_parts_pos_to_concatenate
        self.bodyparts_indices = dict()
        # matches indices in self.bodyparts_indices
        # shape is (n_body_parts, 2 (x, y), n_frames)
        # values could be equal to np.nan
        self.limbs_body_parts_pos_array = None

        # skeleton array shape: # n_bodyparts, (length, orientation, likelihood), n_frames

        self.load_mvt_by_bodypart_files()

        self.behavior_by_frame_left, self.behavior_left_time_stamps, self.gt_behavior_left = encode_frames_from_cicada(
            cicada_file=self.cicada_file,
            nwb_file=self.nwb_file,
            cam_id=self.left_cam_id,
            side_to_exclude="right",
            no_behavior_str="still")

        self.behavior_by_frame_right, self.behavior_right_time_stamps, self.gt_behavior_right = \
            encode_frames_from_cicada(
                cicada_file=self.cicada_file,
                nwb_file=self.nwb_file,
                cam_id=self.right_cam_id,
                side_to_exclude="left",
                no_behavior_str="still")

        if self.cicada_file is None:
            self.gt_behavior = None
        else:
            self.gt_behavior = np.load(self.cicada_file)

        # array (n_body_parts, n_frames), binary, if 1 there is mvt, if 0 no mvt
        self.binary_mvt_matrix = None

        self.fusion_both_side_timestamps()

        # self.evaluate_speed()
        # self.fusion_skeleton_data()

        # # using skeleton and pos values, we decide which part of the body is moving and during which frames
        self.build_binary_mvt_matrix()

        self.classify_behavior()

    def load_mvt_by_bodypart_files(self):
        mvt_bodypart_files = [f for f in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, f))
                     and (not f.startswith(".")) and f.endswith(".npz")]
        for mvt_bodypart_file in mvt_bodypart_files:
            mvt_data_dict = np.load(os.path.join(self.data_path, mvt_bodypart_file))
            for key, mvt_data in mvt_data_dict.items():
                if key == "timestamps":
                    continue
                # print(f"key {key}")
                # key will be bodypart name
                # index of the understand, key is like "hindlimb_left"
                index_ = key.index("_")
                side = key[index_+1:]
                bodypart = key[:index_]
                if side not in self.mvt_data_dict:
                    self.mvt_data_dict[side] = {bodypart: mvt_data}
                else:
                    self.mvt_data_dict[side].update({bodypart: mvt_data})

    def fusion_both_side_timestamps(self):
        # first frame will be the furthest timestamps
        if self.behavior_left_time_stamps[0] > self.behavior_right_time_stamps[0]:
            first_time_stamp = self.behavior_left_time_stamps[0]
        else:
            first_time_stamp = self.behavior_right_time_stamps[0]

        if self.behavior_left_time_stamps[-1] < self.behavior_right_time_stamps[-1]:
            last_time_stamp = self.behavior_left_time_stamps[-1]
        else:
            last_time_stamp = self.behavior_right_time_stamps[-1]

        # now we cut time_stamps and frames, so both side are synchronized
        # first left side
        if self.behavior_left_time_stamps[0] != first_time_stamp:
            frame_index = find_nearest(self.behavior_left_time_stamps, first_time_stamp)
            self.behavior_left_time_stamps = self.behavior_left_time_stamps[frame_index:]
            for bodypart, mvt_data in self.mvt_data_dict["left"].items():
                self.mvt_data_dict["left"][bodypart] = mvt_data[frame_index:]
            # self.body_parts_pos_array_left = self.body_parts_pos_array_left[:, :, frame_index:]
            # self.skeleton_parts_pos_array_left = self.skeleton_parts_pos_array_left[:, :, frame_index:]

        if self.behavior_left_time_stamps[-1] != last_time_stamp:
            frame_index = find_nearest(self.behavior_left_time_stamps, last_time_stamp)
            self.behavior_left_time_stamps = self.behavior_left_time_stamps[:frame_index + 1]
            for bodypart, mvt_data in self.mvt_data_dict["left"].items():
                self.mvt_data_dict["left"][bodypart] = mvt_data[:frame_index + 1]
            # self.body_parts_pos_array_left = self.body_parts_pos_array_left[:, :, :frame_index + 1]
            # self.skeleton_parts_pos_array_left = self.skeleton_parts_pos_array_left[:, :, :frame_index + 1]

        if self.behavior_right_time_stamps[0] != first_time_stamp:
            frame_index = find_nearest(self.behavior_right_time_stamps, first_time_stamp)
            self.behavior_right_time_stamps = self.behavior_right_time_stamps[frame_index:]
            for bodypart, mvt_data in self.mvt_data_dict["right"].items():
                self.mvt_data_dict["right"][bodypart] = mvt_data[frame_index:]
            # self.body_parts_pos_array_right = self.body_parts_pos_array_right[:, :, frame_index:]
            # self.skeleton_parts_pos_array_right = self.skeleton_parts_pos_array_right[:, :, frame_index:]

        if self.behavior_right_time_stamps[-1] != last_time_stamp:
            frame_index = find_nearest(self.behavior_right_time_stamps, last_time_stamp)
            self.behavior_right_time_stamps = self.behavior_right_time_stamps[:frame_index + 1]
            for bodypart, mvt_data in self.mvt_data_dict["right"].items():
                self.mvt_data_dict["right"][bodypart] = mvt_data[:frame_index + 1]
            # self.body_parts_pos_array_right = self.body_parts_pos_array_right[:, :, :frame_index + 1]
            # self.skeleton_parts_pos_array_right = self.skeleton_parts_pos_array_right[:, :, :frame_index + 1]

        self.n_frames = len(self.behavior_left_time_stamps)
        # for side in self.mvt_data_dict.keys():
        #     for bodypart, mvt_data in self.mvt_data_dict[side].items():
        #         print(f"{side} -> {bodypart} -> {mvt_data.shape}")

    def build_binary_mvt_matrix(self):
        sides = list(self.mvt_data_dict.keys())
        # we hypothesized that bodypart are in both sides, animal is symetric
        start_side = sides[0]
        sides = sides[1:] if len(sides) > 1 else []

        # first counting how many bodyparts
        for bodypart in self.mvt_data_dict[start_side].keys():
            self.n_body_parts += 1

            if bodypart in self.bodyparts_to_fusion:
                continue

            self.n_body_parts += len(sides)

        self.binary_mvt_matrix = np.zeros((self.n_body_parts, self.n_frames), dtype="int8")

        # take as key the index of the body_part and return its name
        # self.body_parts_name_dict = dict()
        # # take the name of a bodypart and return it's index, in body_parts_pos_to_concatenate
        # self.bodyparts_indices = dict()

        distance_min_peaks = 2

        bodypart_index = 0
        for bodypart, mvt_data in self.mvt_data_dict[start_side].items():
            # each peak represent a movement
            # peaks, properties = signal.find_peaks(x=mvt_data, height=self.threshold_z_score_mvt,
            #                                       distance=distance_min_peaks)
            # self.binary_mvt_matrix[bodypart_index, peaks] = 1
            self.binary_mvt_matrix[bodypart_index, mvt_data > self.threshold_z_score_mvt[bodypart + f"_{start_side}"]] = 1
            if bodypart in self.bodyparts_to_fusion:
                for side in sides:
                    other_side_mvt_data = self.mvt_data_dict[side][bodypart]
                    # peaks, properties = signal.find_peaks(x=other_side_mvt_data, height=self.threshold_z_score_mvt,
                    #                                       distance=distance_min_peaks)
                    # self.binary_mvt_matrix[bodypart_index, peaks] = 1
                    self.binary_mvt_matrix[bodypart_index, other_side_mvt_data >
                                           self.threshold_z_score_mvt[bodypart + f"_{side}"]] = 1

                self.body_parts_name_dict[bodypart_index] = bodypart
                self.bodyparts_indices[bodypart] = bodypart_index
                bodypart_index += 1
            else:
                self.body_parts_name_dict[bodypart_index] = bodypart + "_" + start_side
                self.bodyparts_indices[bodypart + "_" + start_side] = bodypart_index
                bodypart_index += 1
                for side in sides:
                    # peaks, properties = signal.find_peaks(x=other_side_mvt_data, height=self.threshold_z_score_mvt,
                    #                                       distance=distance_min_peaks)
                    # self.binary_mvt_matrix[bodypart_index, peaks] = 1
                    other_side_mvt_data = self.mvt_data_dict[side][bodypart]
                    self.binary_mvt_matrix[bodypart_index, other_side_mvt_data >
                                           self.threshold_z_score_mvt[bodypart + f"_{side}"]] = 1
                    self.body_parts_name_dict[bodypart_index] = bodypart + "_" + side
                    self.bodyparts_indices[bodypart + "_" + side] = bodypart_index
                    bodypart_index += 1

        # self.binary_mvt_matrix[np.where(self.distance_matrix > self.distance_threshold)] = 1

    def classify_behavior(self):

        # variables to adjust
        # still_vs_mvt[np.where(np.sum(mvt_matrix, axis=0) >= 2)] = 1
        # gap_to_fill = 3
        # (n_moving_bodyparts >= len(bodyparts_triplets) - 2)

        # ### Just any kind of mvt for more than 1 frame of duration ###
        # mvt is 1
        # still_vs_mvt = np.zeros(self.n_frames, dtype="int16")

        # ### now we want to distinguish the different type of mvt and to make it stronger to outlier
        # we will keep a mvt only if 2 of 3 parts of each component of the body are in mvt

        # key being the bodypart
        still_vs_mvt_dict = dict()
        still_vs_mvt_matrix = np.zeros(self.binary_mvt_matrix.shape, dtype="int8")
        # first key key to put in npz and value a list of periods
        periods_by_type_of_mvt = dict()
        periods_by_type_of_mvt["complex_mvt"] = np.zeros(self.n_frames, dtype="int8")
        periods_by_type_of_mvt["startle"] = np.zeros(self.n_frames, dtype="int8")

        for bodypart_index, bodypart in enumerate(self.bodyparts):
            periods_by_type_of_mvt[f"twitch_{bodypart}"] = np.zeros(self.n_frames, dtype="int8")
            periods_by_type_of_mvt[f"mvt_{bodypart}"] = np.zeros(self.n_frames, dtype="int8")

            still_vs_mvt = np.copy(self.binary_mvt_matrix[self.bodyparts_indices[bodypart]])

            invert_still_vs_mvt = 1 - still_vs_mvt
            still_periods = get_continous_time_periods(invert_still_vs_mvt)
            gap_to_fill = 2
            # feeling the gap of 2 frames without mvt
            for still_period in still_periods:
                if (still_period[1] - still_period[0] + 1) <= gap_to_fill:
                    still_vs_mvt[still_period[0]:still_period[1] + 1] = 1
            # TODO: See to remove mvt of less than 2 frames here
            still_vs_mvt_dict[bodypart] = still_vs_mvt
            still_vs_mvt_matrix[bodypart_index] = still_vs_mvt
            print(f"{bodypart} np.sum(still_vs_mvt) {np.sum(still_vs_mvt)}")

        # now we want to identify startle and complex_mvt
        # we create dictionnaries that contains each part with tuple representing start and end of movement.
        # we will remove those that have been identified as part of a startle or complex_mvt
        # key is the bodyaprt, then dict with id for the period, and tuple of int representing the period
        periods_id_by_bodypart_dict = dict()
        # periods_array_with_id
        # key is bodypart, value is an array of n_frames with value of each the id of the period, starting from 1
        periods_array_with_id_dict = dict()
        # list with an unique id all periods (bodypart family, first_frame, last_frame)
        all_periods = set()

        # ranking period according to their duration, behavior is recording at 20 Hz
        # 1000 ms / 50 ms
        twitch_duration_threshold = 20
        # startle_threshold = 24

        for key_bodypart, still_vs_mvt in still_vs_mvt_dict.items():
            periods = get_continous_time_periods(still_vs_mvt)
            periods_id_by_bodypart_dict[key_bodypart] = dict()
            periods_array_with_id = np.zeros(self.n_frames, dtype="int16")
            for period_index, period in enumerate(periods):
                periods_array_with_id[period[0]:period[1] + 1] = period_index + 1
                periods_id_by_bodypart_dict[key_bodypart][period_index + 1] = period
                all_periods.add((key_bodypart, period[0], period[1]))
                # print(f"all_periods.add {(key_bodypart, period[0], period[1])}")
            periods_array_with_id_dict[key_bodypart] = periods_array_with_id

        # print(f"periods_array_with_id_dict {periods_array_with_id_dict}")

        # now we want to explore all periods in periods_id_by_bodypart_dict to check which ones should be associated
        # to startle or complex mvt
        while len(all_periods) > 0:
            # get period id and remove it from the set in the same time
            period_id = all_periods.pop()
            key_bodypart = period_id[0]
            first_frame = period_id[1]
            last_frame = period_id[2]
            mvt_duration = last_frame - first_frame + 1
            # we want at least 2 frames (100 ms) to consider it as mvt
            if mvt_duration < 2:
                continue
            # we get the names of other bodyparts to explore
            bodyparts_names = list(self.bodyparts)
            bodyparts_names.remove(key_bodypart)
            # KeyError: ('hindlimb', 22245, 22300)
            # key is id (bodypart, first_frame, last_frame) and value is the duration
            other_mvts_dict = dict()
            # complex_mvt or startle are defined as if all or all - 2 bodypart are active in the same time
            for bodypart_to_explore in bodyparts_names:
                periods_array_with_id = periods_array_with_id_dict[bodypart_to_explore]
                # we look if a concomittent mvt happened
                if np.sum(periods_array_with_id[first_frame:last_frame + 1]) > 0:
                    # another mvt happened in the same time
                    period_ids = list(np.unique(periods_array_with_id[first_frame:last_frame + 1]))
                    if 0 in period_ids:
                        period_ids.remove(0)
                    if len(period_ids) > 1:
                        # we keep the one with the most frames
                        max_n_frames = 0
                        period_id = None
                        for i in period_ids:
                            frames_count = len(np.where(periods_array_with_id[first_frame:last_frame + 1] == i)[0])
                            if frames_count > max_n_frames:
                                period_id = i
                                max_n_frames = frames_count
                    else:
                        period_id = period_ids[0]
                    period = periods_id_by_bodypart_dict[bodypart_to_explore][period_id]
                    other_mvts_dict[(bodypart_to_explore, period[0], period[1])] = period[1] - period[0] + 1
            # now let's count how many other part are moving in the same time
            n_moving_bodyparts = len(other_mvts_dict)
            if (n_moving_bodyparts >= (len(self.bodyparts) - 3)) or (
                    n_moving_bodyparts > 0 and np.max(list(other_mvts_dict.values())) > 40):
                # then we are in the category startle or complex_mvt
                # if one of the other mvt part has a long mvt, then we rank it complex_mvt
                # complex mvt here is defined as more than 2 sec
                # we take the first frame of all mvts, and the last last_frame of all
                first_first_frame = first_frame
                last_last_frame = last_frame
                for key_other_mvt in other_mvts_dict.keys():
                    if key_other_mvt[1] < first_first_frame:
                        first_first_frame = key_other_mvt[1]
                    if key_other_mvt[2] > last_last_frame:
                        last_last_frame = key_other_mvt[2]
                    # we remove other mvt from the set meanwhile
                    if key_other_mvt in all_periods:
                        all_periods.remove(key_other_mvt)

                if np.max(list(other_mvts_dict.values())) > twitch_duration_threshold:
                    # using array, we can extend previous complet_mvt period
                    periods_by_type_of_mvt["complex_mvt"][first_first_frame:last_last_frame + 1] = 1
                else:
                    periods_by_type_of_mvt["startle"][first_first_frame:last_last_frame + 1] = 1
            else:
                # we are in the category twitch or mvt of one body part
                if mvt_duration <= twitch_duration_threshold:
                    # we classify it as a twitch
                    periods_by_type_of_mvt[f"twitch_{key_bodypart}"][first_frame:last_frame + 1] = 1
                else:
                    # else a mvt
                    periods_by_type_of_mvt[f"mvt_{key_bodypart}"][first_frame:last_frame + 1] = 1

        # we're gonna a loop until no more changes are done
        modif_done = True
        # str just use for display purpose
        modif_reason = ""
        n_loops_of_modif = 0
        while modif_done:
            modif_done = False
            n_loops_of_modif += 1
            print(f"n_loops_of_modif {n_loops_of_modif} {modif_reason}")
            modif_reason = ""
            # now we want to fusion periods of complex_mvt that would be close and also merge them with close by other mvt
            complex_mvt_periods = get_continous_time_periods(periods_by_type_of_mvt["complex_mvt"])
            # gap between two complex mvt to fill, 20 frames == 1.5sec
            max_gap_by_complex_mvt = 30
            for complex_mvt_period_index, complex_mvt_period in enumerate(complex_mvt_periods[:-1]):
                last_frame = complex_mvt_period[1]
                next_frame = complex_mvt_periods[complex_mvt_period_index + 1][0]
                if next_frame - last_frame <= max_gap_by_complex_mvt:
                    # we fill the gap
                    periods_by_type_of_mvt["complex_mvt"][last_frame + 1:next_frame] = 1
                    modif_done = True
                    modif_reason = modif_reason + " max_gap_by_complex_mvt"
                    # Removing other mvt happening during the GAP
                    first_frame_index = complex_mvt_period[0]
                    last_frame_index = complex_mvt_periods[complex_mvt_period_index][1]
                    # a simple way, we remove all positive frame from first to last_frame_index in other mvt
                    for mvt_key, mvt_array in periods_by_type_of_mvt.items():
                        if mvt_key == "complex_mvt":
                            continue
                        # TODO: See to make sure we don't cut some periods when doing so
                        #  to do so, we would need to encode each period with an index, to make sure it did diseappear
                        mvt_array[first_frame_index:last_frame_index + 1] = 0

            # now we want to fusion startles and very close twitches from startles together
            max_gap_startle = 15  # 750 msec

            startle_mvt_periods = get_continous_time_periods(periods_by_type_of_mvt["startle"])
            # first we check if startles didn't become complex_mvt
            for startle_mvt_period_index, startle_mvt_period in enumerate(startle_mvt_periods):
                if startle_mvt_period[1] - startle_mvt_period[0] + 1 > twitch_duration_threshold:
                    modif_done = True
                    modif_reason = modif_reason + " startle_extended"
                    periods_by_type_of_mvt["complex_mvt"][startle_mvt_period[0]:startle_mvt_period[1] + 1] = 1
                    periods_by_type_of_mvt["startle"][startle_mvt_period[0]:startle_mvt_period[1] + 1] = 0

            startle_mvt_periods = get_continous_time_periods(periods_by_type_of_mvt["startle"])
            for startle_mvt_period_index, startle_mvt_period in enumerate(startle_mvt_periods[:-1]):
                last_frame = startle_mvt_period[1]
                next_frame = startle_mvt_periods[startle_mvt_period_index + 1][0]
                if next_frame - last_frame <= max_gap_startle:
                    modif_done = True
                    modif_reason = modif_reason + " max_gap_startle"
                    first_frame_index = startle_mvt_period[0]
                    last_frame_index = startle_mvt_periods[startle_mvt_period_index][1]
                    duration_new_startle = last_frame_index - first_frame_index + 1
                    # then either it becomes a complex_mvt or become a longer startle
                    if duration_new_startle > twitch_duration_threshold:
                        periods_by_type_of_mvt["complex_mvt"][first_frame_index:last_frame_index + 1] = 1
                        # We remove it from startle
                        periods_by_type_of_mvt["startle"][first_frame_index:last_frame_index + 1] = 0
                    else:
                        periods_by_type_of_mvt["startle"][last_frame + 1:next_frame] = 1
                        # a simple way, we remove all positive frame from first to last_frame_index in other mvt
                        for mvt_key, mvt_array in periods_by_type_of_mvt.items():
                            if mvt_key == "startle":
                                continue
                            # TODO: See to make sure we don't cut some periods when doing so
                            #  to do so, we would need to encode each period with an index, to make sure it did diseappear
                            mvt_array[first_frame_index:last_frame_index + 1] = 0
            for mvt_key, mvt_array in periods_by_type_of_mvt.items():
                if mvt_key in ["complex_mvt"]:  # , "startle"
                    continue
                is_mvt_startle = False
                if mvt_key == "startle":
                    is_mvt_startle = True
                gap_for_fusion = {"startle": 10, "complex_mvt": 20}  # 500 msec & 1 sec
                mvt_periods = get_continous_time_periods(mvt_array)
                # then we look at all periods in mvt_complex and startle to see if we can fusion
                for mvt_period in mvt_periods:
                    first_frame_mvt = mvt_period[0]
                    last_frame_mvt = mvt_period[1]
                    gap_value = min(first_frame_mvt, gap_for_fusion["complex_mvt"])
                    if gap_value > 0:
                        if np.sum(periods_by_type_of_mvt["complex_mvt"]
                                  [first_frame_mvt - gap_for_fusion["complex_mvt"]:first_frame_mvt]) > 0:
                            # then we change it to complex_mvt
                            periods_by_type_of_mvt["complex_mvt"][first_frame_mvt:last_frame_mvt + 1] = 1
                            mvt_array[first_frame_mvt:last_frame_mvt + 1] = 0
                            modif_done = True
                            modif_reason = modif_reason + " gap fusion complex mvt"
                            continue
                    # if a startle was detected just before a complex_mvt, we should consider it
                    # as part of the complex_mvt and not as a startle
                    if is_mvt_startle:
                        # we use the fusion gap of startle here still
                        gap_value = min(self.n_frames - last_frame_mvt, gap_for_fusion["startle"])
                        if gap_value > 0:
                            if np.sum(periods_by_type_of_mvt["complex_mvt"]
                                      [last_frame_mvt:last_frame_mvt + gap_for_fusion["startle"]]) > 0:
                                # then we change it to startle
                                periods_by_type_of_mvt["complex_mvt"][first_frame_mvt:last_frame_mvt + 1] = 1
                                mvt_array[first_frame_mvt:last_frame_mvt + 1] = 0
                                modif_done = True
                                modif_reason = modif_reason + " gap fusion startle before complex mvt"
                                continue
                    if not is_mvt_startle:
                        # for startle mvt, we only look to fusion them with complex mvt
                        # for startle we look before and after
                        gap_value = min(first_frame_mvt, gap_for_fusion["startle"])
                        if gap_value > 0:
                            if np.sum(periods_by_type_of_mvt["startle"]
                                      [first_frame_mvt - gap_for_fusion["startle"]:first_frame_mvt]) > 0:
                                # then we change it to startle
                                periods_by_type_of_mvt["startle"][first_frame_mvt:last_frame_mvt + 1] = 1
                                mvt_array[first_frame_mvt:last_frame_mvt + 1] = 0
                                modif_done = True
                                modif_reason = modif_reason + " gap fusion startle before"
                                continue
                        # gap_value = min(self.n_frames - last_frame_mvt, gap_for_fusion["startle"])
                        # if gap_value > 0:
                        #     if np.sum(periods_by_type_of_mvt["startle"]
                        #               [last_frame_mvt:last_frame_mvt + gap_for_fusion["startle"]]) > 0:
                        #         # then we change it to startle
                        #         periods_by_type_of_mvt["startle"][first_frame_mvt:last_frame_mvt + 1] = 1
                        #         mvt_array[first_frame_mvt:last_frame_mvt + 1] = 0
                        #         modif_done = True
                        #         modif_reason = modif_reason + " gap fusion startle after"
                        #         continue
        print(f"n_loops_of_modif final:  {n_loops_of_modif}")

        for action_str in ["cicada", "citnps_tag"]:
            # will be saved in the npz
            # each key is the tag, and value is a 2d array (2x n_intervals) with start and finish in sec on lines,
            # each column is an interval
            # the cicada version will have the original as key for the behavior
            # the citnps_tag version will have added in front the original key, "citnps_" so it could be distinguish
            # from a potential manual ground truth
            behaviors_encoding_dict = dict()
            for key_behavior, binary_array in periods_by_type_of_mvt.items():
                if action_str == "cicada":
                    key_to_use = key_behavior
                else:
                    key_to_use = "citnps_" + key_behavior
                periods = get_continous_time_periods(binary_array)
                if "left" in key_behavior:
                    behaviors_encoding_dict[key_to_use] = \
                        encode_period_with_timestamps(periods=periods,
                                                      timestamps=self.behavior_left_time_stamps)
                else:
                    behaviors_encoding_dict[key_to_use] = \
                        encode_period_with_timestamps(periods=periods,
                                                      timestamps=self.behavior_right_time_stamps)
            if self.gt_behavior is not None and (action_str == "cicada"):
                evaluate_behavior_predictions(ground_truth_labels=self.gt_behavior,
                                              other_labels=behaviors_encoding_dict,
                                              n_frames=self.n_frames,
                                              behavior_timestamps=self.behavior_right_time_stamps)
            to_add_to_filename = ""
            if self.mandatory_piezo_threshold is not None:
                to_add_to_filename = f"_piezo_threshold_{np.round(self.mandatory_piezo_threshold, 1)}.npz"
            np.savez(os.path.join(self.results_path, f"citnps_{self.identifier}_{action_str}{to_add_to_filename}.npz"),
                     **behaviors_encoding_dict)

        # extracting just when mouvement happens
        still_vs_mvt_vector = np.sum(still_vs_mvt_matrix, axis=0)
        still_vs_mvt_vector[still_vs_mvt_vector > 1] = 1
        periods = get_continous_time_periods(still_vs_mvt_vector)
        # print(f"sum periods {len(periods)}")
        behaviors_encoding_dict = dict()
        behaviors_encoding_dict["citnps_mvt"] = \
            encode_period_with_timestamps(periods=periods,
                                          timestamps=self.behavior_right_time_stamps)
        file_name = f"citnps_{self.identifier}_1_category_mvt.npz"
        if self.mandatory_piezo_threshold is not None:
            file_name = f"citnps_{self.identifier}_1_category_mvt_piezo_threshold_{np.round(self.mandatory_piezo_threshold, 1)}.npz"
        np.savez(os.path.join(self.results_path, file_name),
                 **behaviors_encoding_dict)

def encode_frames_from_cicada(cicada_file, nwb_file, cam_id, side_to_exclude=None, no_behavior_str="still"):
    """
    Return a list of len n_frames with str representing each frame behavior
    Args:
        cicada_file:
        side_to_exclude: not apply for npz content, all of it is return as the third argument

    Returns:

    """

    behavior_time_stamps = get_behaviors_movie_time_stamps(nwb_file=nwb_file, cam_id=cam_id)
    if cicada_file is None:
        return None, behavior_time_stamps, None

    npz_content = np.load(cicada_file)

    tags = []
    tags_index = dict()
    n_tags = 0
    for tag_name, value in npz_content.items():
        if side_to_exclude in tag_name:
            continue
        if value.shape[1] > 0:
            tags.append(tag_name)
            tags_index[tag_name] = n_tags
            n_tags += 1

    print(f"encode_frames_from_cicada() {cam_id} tags {tags}")

    interval_id = 0
    interval_info_dict = dict()
    intervals_array = np.zeros((len(tags), len(behavior_time_stamps)), dtype="int16")

    # print(f"behavior_time_stamps {behavior_time_stamps}")
    for tag_name, value in npz_content.items():
        if side_to_exclude in tag_name:
            continue

        for i in range(value.shape[1]):
            # print(f"value[0, i] {value[0, i]}, {value[1, i]+1}")
            # TODO: Transform the seconds value in the frame number of the given cam
            first_frame = match_frame_to_timestamp(behavior_time_stamps, value[0, i])
            last_frame = match_frame_to_timestamp(behavior_time_stamps, value[1, i])
            intervals_array[tags_index[tag_name], first_frame:last_frame] = interval_id
            # duration in sec
            interval_info_dict[interval_id] = value[1, i] - value[0, i] + 1
            interval_id += 1

    tags_by_frame = []
    for frame_index in range(len(behavior_time_stamps)):
        tags_indices = np.where(intervals_array[:, frame_index] > 0)[0]
        if len(tags_indices) == 0:
            tags_by_frame.append(no_behavior_str)
        elif len(tags_indices) == 1:
            tags_by_frame.append(tags[tags_indices[0]])
        else:
            # then we select the shorter one
            for i, tag_index in enumerate(tags_indices):
                if i == 0:
                    shorter_tag = tags[tag_index]
                    interval_id = intervals_array[tag_index, frame_index]
                    shorter_duration = interval_info_dict[interval_id]
                else:
                    interval_id = intervals_array[tag_index, frame_index]
                    duration = interval_info_dict[interval_id]
                    if duration < shorter_duration:
                        shorter_duration = duration
                        shorter_tag = tags[tag_index]
            tags_by_frame.append(shorter_tag)
    return tags_by_frame, behavior_time_stamps, npz_content



def match_frame_to_timestamp(frame_timestamps, timestamp):
    """
    Find which frame match the given timestamp
    Args:
        frame_timestamps:
        timestamp:

    Returns:

    """
    index = find_nearest(frame_timestamps, timestamp)
    # index = bisect_right(frame_timestamps, timestamp) - 1
    return index



def encode_period_with_timestamps(periods, timestamps):
    """
    Return an array that will contain the periods, used to save npz for CICADA
    Args:
        periods:
        timestamps:

    Returns:

    """
    mvt_encoding = np.zeros((2, len(periods)))
    for index, mvt_period in enumerate(periods):
        # Removing mvt period that last only 1 frame
        if mvt_period[1] - mvt_period[0] < 2:
            continue
        mvt_encoding[0, index] = timestamps[mvt_period[0]]
        mvt_encoding[1, index] = timestamps[mvt_period[1]]

    return mvt_encoding



def evaluate_behavior_predictions(ground_truth_labels, other_labels, n_frames, behavior_timestamps):
    """

    Args:
        ground_truth_labels: dict with key behavior and value 2d array (2*frames)
        other_labels: dict with key behavior and value 2d array (2*frames)

    Returns:

    """

    # first we determine until which timestamps the the ground truth has been labeled
    last_time_stamp_to_consider = 0
    for key_behavior in other_labels.keys():
        if key_behavior not in ground_truth_labels:
            continue
        bahavior_periods = ground_truth_labels[key_behavior]
        for index in range(bahavior_periods.shape[1]):
            first_time_stamp = bahavior_periods[0, index]
            last_time_stamp = bahavior_periods[1, index]
            if last_time_stamp > last_time_stamp_to_consider:
                last_time_stamp_to_consider = last_time_stamp

    for key_behavior, other_behavior_activity_array in other_labels.items():
        if key_behavior not in ground_truth_labels:
            print(f"{key_behavior} not defined in ground truth")
            continue

        print(f"## METRICS for {key_behavior}")
        other_behavior_activity_array = transform_second_behavior_matrix_to_frame(behavior_periods=
                                                                                  other_behavior_activity_array,
                                                                                  behavior_timestamps=
                                                                                  behavior_timestamps,
                                                                                  last_time_stamp_to_consider=
                                                                                  last_time_stamp_to_consider)

        other_binary_activity = get_binary_vector_from_2d_behavior_array(other_behavior_activity_array, n_frames)

        gt_behavior_activity_array = ground_truth_labels[key_behavior]
        gt_behavior_activity_array = transform_second_behavior_matrix_to_frame(behavior_periods=
                                                                               gt_behavior_activity_array,
                                                                               behavior_timestamps=behavior_timestamps)
        gt_binary_activity = get_binary_vector_from_2d_behavior_array(gt_behavior_activity_array, n_frames)

        # now we count tp, fp
        tp = 0
        fp = 0
        # tn should not exists
        tn = 0
        fn = 0
        for active_period_index in range(gt_behavior_activity_array.shape[1]):
            first_frame = gt_behavior_activity_array[0, active_period_index]
            last_frame = gt_behavior_activity_array[1, active_period_index]
            if np.sum(other_binary_activity[first_frame:last_frame + 1]) > 0:
                tp += 1
            else:
                fn += 1

        for period_index in range(other_behavior_activity_array.shape[1]):
            first_frame = other_behavior_activity_array[0, period_index]
            last_frame = other_behavior_activity_array[1, period_index]
            if np.sum(gt_binary_activity[first_frame:last_frame + 1]) == 0:
                fp += 1

        print(f"tp {tp}, fp {fp}, fn {fn}")
        print(f"Over {gt_behavior_activity_array.shape[1]} active periods, {tp} were identified and {fn} were missed")
        print(f"{fn} were wrongly assigned as present")

        if (tp + fn) > 0:
            sensitivity = tp / (tp + fn)
        else:
            sensitivity = 1

        if (tp + fp) > 0:
            ppv = tp / (tp + fp)
        else:
            ppv = 1

        print(f"SENSITIVITY: {np.round(sensitivity * 100, 2)}%, PPV: {np.round(ppv * 100, 2)}%")

        print(f"")


def transform_second_behavior_matrix_to_frame(behavior_periods, behavior_timestamps,
                                              last_time_stamp_to_consider=None):
    """

    Args:
        behavior_periods: (2xn_periods) array
        behavior_timestamps: give the timestamps (in sec) for each frame
        last_time_stamp_to_consider: (float) if not None, any period of activity that start after this
        timestamp will not be taken into consideration
    Returns:

    """
    n_periods = behavior_periods.shape[1]
    if last_time_stamp_to_consider is not None:
        # we count how many periods to consider
        n_periods = 0
        for index in range(behavior_periods.shape[1]):
            first_time_stamp = behavior_periods[0, index]
            if first_time_stamp <= last_time_stamp_to_consider:
                n_periods += 1

    behavior_in_frames = np.zeros((2, n_periods), dtype="int16")
    real_index = 0
    for index in range(behavior_periods.shape[1]):
        if last_time_stamp_to_consider is not None:
            first_time_stamp = behavior_periods[0, index]
            if first_time_stamp > last_time_stamp_to_consider:
                # periods being ordered
                break
        first_frame = match_frame_to_timestamp(behavior_timestamps, behavior_periods[0, index])
        last_frame = match_frame_to_timestamp(behavior_timestamps, behavior_periods[1, index])
        behavior_in_frames[0, index] = first_frame
        behavior_in_frames[1, index] = last_frame
    return behavior_in_frames


def get_binary_vector_from_2d_behavior_array(behavior_2d_array, n_frames):
    result = np.zeros(n_frames, dtype="int16")
    # print(f"behavior_2d_array {behavior_2d_array}")
    for interval_index in range(behavior_2d_array.shape[1]):
        result[behavior_2d_array[0, interval_index]: behavior_2d_array[1, interval_index] + 1] = 1
    return result


def get_continous_time_periods(binary_array):
    """
    take a binary array and return a list of tuples representing the first and last position(included) of continuous
    positive period
    This code was copied from another project or from a forum, but i've lost the reference.
    :param binary_array:
    :return:
    """
    binary_array = np.copy(binary_array).astype("int8")
    n_times = len(binary_array)
    d_times = np.diff(binary_array)
    # show the +1 and -1 edges
    pos = np.where(d_times == 1)[0] + 1
    neg = np.where(d_times == -1)[0] + 1

    if (pos.size == 0) and (neg.size == 0):
        if len(np.nonzero(binary_array)[0]) > 0:
            return [(0, n_times - 1)]
        else:
            return []
    elif pos.size == 0:
        # i.e., starts on an spike, then stops
        return [(0, neg[0])]
    elif neg.size == 0:
        # starts, then ends on a spike.
        return [(pos[0], n_times - 1)]
    else:
        if pos[0] > neg[0]:
            # we start with a spike
            pos = np.insert(pos, 0, 0)
        if neg[-1] < pos[-1]:
            #  we end with aspike
            neg = np.append(neg, n_times - 1)
        # NOTE: by this time, length(pos)==length(neg), necessarily
        # h = np.matrix([pos, neg])
        h = np.zeros((2, len(pos)), dtype="int32")
        h[0] = pos
        h[1] = neg
        if np.any(h):
            result = []
            for i in np.arange(h.shape[1]):
                if h[1, i] == n_times - 1:
                    result.append((h[0, i], h[1, i]))
                else:
                    result.append((h[0, i], h[1, i] - 1))
            return result
    return []


def find_nearest(array, value, is_sorted=True):
    """
    Return the index of the nearest content in array of value.
    from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    return -1 or len(array) if the value is out of range for sorted array
    Args:
        array:
        value:
        is_sorted:

    Returns:

    """
    if len(array) == 0:
        return -1

    if is_sorted:
        if value < array[0]:
            return -1
        elif value > array[-1]:
            return len(array)
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
            return idx - 1
        else:
            return idx
    else:
        array = np.asarray(array)
        idx = (np.abs(array - value)).idxmin()
        return idx


#####################################################
################### NWB functions ###################
#####################################################

def get_behaviors_movie_time_stamps(nwb_file, cam_id):
    """

    Args:
        nwb_file:
        cam_id: '22983298' or '23109588'

    Returns:

    """
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()

    for name, acquisition_data in nwb_data.acquisition.items():
        if name.startswith(f"cam_{cam_id}"):
            return np.array(acquisition_data.timestamps)

    io.close()

    return None


def get_intervals_names(nwb_file):
    """
    Return a list representing the intervals contains in this data
    Returns:

    """
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()

    if nwb_data.intervals is None:
        io.close()
        return []

    intervals = []
    for name_interval in nwb_data.intervals.keys():
        intervals.append(name_interval)

    io.close()
    return intervals

def get_ci_movie_time_stamps(nwb_file):
    """
    return a np.array with the timestamps of each frame of the CI movie
    return None if non available
    Returns:

    """
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()

    if "ci_frames" not in nwb_data.acquisition:
        return None
    ci_frames_timestamps = np.array(nwb_data.acquisition["ci_frames"].timestamps)

    io.close()

    return ci_frames_timestamps


def get_interval_times(nwb_file, interval_name):
    """
    Return an interval times (start and stop in seconds) as a numpy array of 2*n_times.
    Args:
        interval_name: Name of the interval to retrieve

    Returns: None if the interval doesn't exists or a 2d array

    """
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()

    if interval_name not in nwb_data.intervals:
        io.close()
        return None

    df = nwb_data.intervals[interval_name].to_dataframe()

    # TODO: See to make it more modulable in case someone will use another name
    if ("start_time" not in df) or \
            ("stop_time" not in df):
        return None

    # time series
    start_time_ts = df["start_time"]
    stop_time_ts = df["stop_time"]

    # it shouldn't be the case
    if len(start_time_ts) != len(stop_time_ts):
        print(f"len(start_time_ts) {len(start_time_ts)} != {len(stop_time_ts)} len(stop_time_ts)")
        return None

    data = np.zeros((2, len(start_time_ts)))
    data[0] = np.array(start_time_ts)
    data[1] = np.array(stop_time_ts)

    io.close()

    return data

#####################################################
# ############### END NWB functions #################
#####################################################

def find_files_with_ext(dir_to_explore, extension, keyword=None, go_deep=False):
    subfiles = []
    for (dirpath, dirnames, filenames) in os.walk(dir_to_explore):
        subfiles.extend([os.path.join(dirpath, filename) for filename in
                         filenames if filename.endswith(extension) and (not filename.startswith("."))])
        if not go_deep:
            break

    if keyword is not None:
        subfiles = [f for f in subfiles if (keyword in os.path.basename(f))]

    return subfiles


def find_dirs_with_keyword(dir_to_explore, keyword, go_deep=False):
    selected_dirnames = []
    for (dirpath, dirnames, filenames) in os.walk(dir_to_explore):
        selected_dirnames.extend([os.path.join(dirpath, dir_name) for dir_name in dirnames
                                  if (keyword in dir_name) and (not dir_name.startswith("."))])
        if not go_deep:
            break

    return selected_dirnames

def citnps_main(extract_piezo_signal):
    # -------------------------------------------- #
    # -------------------------------------------- #
    # age = 5
    # animal_id = "191127_191202"
    # session_id = "191202_a001"
    # animal_id = "200306_200311"
    # session_id = "200311_a000"
    age = 7
    animal_id = "200206_200213"
    session_id = "200213_a000"
    # directory in hne_not_today
    type_of_recording = "red_ins"
    # if CITNPS has already be run for this session
    # yaml_config_file = "/media/julien/Not_today/hne_not_today/data/red_ins/p5/191127_191202/191202_a001/behavior_191127_191202_191202_a001/p5_191127_191202_191202_a001_citnps_config.yaml"
    yaml_config_file = "/media/julien/Not_today/hne_not_today/data/red_ins/p7/200206_200213/200213_a000/behavior_200206_200213_200213_a000/p7_200206_200213_200213_a000_citnps_config.yaml"
    # yaml_config_file = None

    # if not None, force the threshold for all bodypart to this value
    mandatory_piezo_threshold = 0.5

    # Note that a NWB file should exist for this session
    # -------------------------------------------- #
    # -------------------------------------------- #

    left_cam_id = "23109588"
    right_cam_id = "22983298"

    movie_queue_size = 2000
    citnps_movies_dict = dict()

    root_path = "/media/julien/Not_today/hne_not_today/"

    nwb_path = os.path.join(root_path, "data", "nwb_files")

    # a directory with timestamps will be created in results directory when using the GUI
    results_path = os.path.join(root_path, "results_hne")

    data_id = f"p{age}_{animal_id}_{session_id}"

    nwb_files = find_files_with_ext(nwb_path, "nwb")
    nwb_files = [f for f in nwb_files if f"{animal_id}_{session_id}" in f]
    if len(nwb_files) != 1:
        raise Exception(f"Should be only on nwb file with {animal_id}_{session_id}, got {nwb_files} "
                        f"in path {nwb_path}")
    nwb_file = nwb_files[0]

    data_path = os.path.join(root_path, "data", type_of_recording, f"p{age}", animal_id, session_id)

    piezo_signals_data_path = os.path.join(data_path, f"signal_{animal_id}_{session_id}")

    behavior_data_path = find_dirs_with_keyword(data_path, keyword="behavior")
    if len(behavior_data_path) != 1:
        print(f"Should be a single behavior directory, found: {behavior_data_path}")
        return
    behavior_data_path = behavior_data_path[0]

    config_citnps_yaml_file = find_files_with_ext(behavior_data_path, extension="yaml")

    config_citnps_yaml_file = [f for f in config_citnps_yaml_file if "citnps_config" in f]
    if len(config_citnps_yaml_file) == 1:
        config_citnps_yaml_file = config_citnps_yaml_file[0]
    else:
        config_citnps_yaml_file = None

    avi_files = find_files_with_ext(behavior_data_path, extension="avi")

    for avi_file in avi_files:
        if "22983298" in avi_file:
            citnps_movies_dict["right"] = avi_file
        else:
            citnps_movies_dict["left"] = avi_file

    right_movie = OpenCvVideoReader(citnps_movies_dict["right"], queueSize=2000)
    n_frames_right = right_movie.length

    left_movie = OpenCvVideoReader(citnps_movies_dict["left"], queueSize=2000)
    n_frames_left = left_movie.length

    behavior_time_stamps_dict = dict()

    # 672
    right_behavior_time_stamps = get_behaviors_movie_time_stamps(nwb_file, cam_id=right_cam_id)
    if len(right_behavior_time_stamps) != n_frames_right:
        if len(right_behavior_time_stamps) > n_frames_right:
            # removing the last frames
            print(f"Removing frames on right timestamps {len(right_behavior_time_stamps)} vs {n_frames_right}")
            right_behavior_time_stamps = right_behavior_time_stamps[:n_frames_right - len(right_behavior_time_stamps)]
        else:
            raise Exception(f"Wrong number of frames on the right side "
                            f"{len(right_behavior_time_stamps)} vs {n_frames_right}")

    left_behavior_time_stamps = get_behaviors_movie_time_stamps(nwb_file, cam_id=left_cam_id)
    if len(left_behavior_time_stamps) != n_frames_left:
        if len(left_behavior_time_stamps) > n_frames_left:
            # removing the last frames
            print(f"Removing frames on left timestamps {len(left_behavior_time_stamps)} vs {n_frames_left}")
            left_behavior_time_stamps = left_behavior_time_stamps[:n_frames_left - len(left_behavior_time_stamps)]
        else:
            raise Exception(f"Wrong number of frames on the left side "
                            f"{len(left_behavior_time_stamps)} vs {n_frames_left}")

    behavior_time_stamps_dict["right"] = right_behavior_time_stamps
    behavior_time_stamps_dict["left"] = left_behavior_time_stamps
    ci_movie_time_stamps = get_ci_movie_time_stamps(nwb_file)

    intervals_names = get_intervals_names(nwb_file)
    if "ci_recording_on_pause" not in intervals_names:
        raise Exception("ci_recording_on_pause epochs not in the NWB")

    pause_epochs = get_interval_times(nwb_file, "ci_recording_on_pause")

    # bodyparts_dict = {"left": list(bodyparts), "right": list(bodyparts)}
    bodyparts_dict = dict()
    bodyparts_without_sides = ["forelimb", "hindlimb", "tail"]
    for side in citnps_movies_dict.keys():
        bodyparts = []
        for bodypart in bodyparts_without_sides:
            bodyparts.append(bodypart + "_" + side)
        bodyparts_dict[side] = bodyparts

    # indicate when their is a change in luminosity (laser on/off) (epochs in frames).
    # Each key is the id of a movie, 1d array of len n_transitions, each value (int) represent a
    # frame when the luminosity transition happens. Allows to normalize the signal according to luminosity.
    luminosity_change_dict = dict()
    luminosity_change_dict["right"] = np.zeros(pause_epochs.shape[1] * 2 + 2, dtype="int64")
    luminosity_change_dict["left"] = np.zeros(pause_epochs.shape[1] * 2 + 2, dtype="int64")

    first_ci_frame_dict = dict()
    last_ci_frame_dict = dict()

    shift_with_movie = 2

    # finding which frame from the behavior matches the CI activation (for z-score)
    first_ci_frame_right = int(find_nearest(right_behavior_time_stamps, ci_movie_time_stamps[0]))
    # print(f"pause_epochs {pause_epochs}")
    print(
        f"ci_movie_time_stamps[0] {ci_movie_time_stamps[0]}")
    print(f"right_behavior_time_stamps[first_ci_frame_right] around {right_behavior_time_stamps[first_ci_frame_right-3:first_ci_frame_right+3]}")
    print(f"right_behavior_time_stamps[first_ci_frame_right] {right_behavior_time_stamps[first_ci_frame_right]}")
    print(f"right_behavior_time_stamps[first_ci_frame_right] ok {right_behavior_time_stamps[first_ci_frame_right-2]}")
    first_ci_frame_dict["right"] = first_ci_frame_right - shift_with_movie
    last_ci_frame_right = int(find_nearest(right_behavior_time_stamps, ci_movie_time_stamps[-1]))
    last_ci_frame_dict["right"] = last_ci_frame_right - shift_with_movie

    # print(f"first_ci_frame_right {first_ci_frame_right}")
    first_ci_frame_left = int(find_nearest(left_behavior_time_stamps, ci_movie_time_stamps[0]))
    first_ci_frame_dict["left"] = first_ci_frame_left - shift_with_movie
    last_ci_frame_left = int(find_nearest(left_behavior_time_stamps, ci_movie_time_stamps[-1]))
    last_ci_frame_dict["left"] = last_ci_frame_left - shift_with_movie

    for side in ["right", "left"]:
        luminosity_change_dict[side][0] = first_ci_frame_dict[side]
        for i in range(pause_epochs.shape[1]):
            luminosity_change_dict[side][(i*2) + 1] = int(find_nearest(behavior_time_stamps_dict[side],
                                                                       pause_epochs[0, i])) # - shift_with_movie
            luminosity_change_dict[side][(i * 2) + 2] = int(find_nearest(behavior_time_stamps_dict[side],
                                                                         pause_epochs[1, i])) - shift_with_movie
        luminosity_change_dict[side][-1] = last_ci_frame_dict[side] + 1

    # print(f"citnps_main luminosity_change_dict {luminosity_change_dict}")

    left_movie.close_reader()
    right_movie.close_reader()

    if not extract_piezo_signal:
        time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        results_path = os.path.join(results_path, time_str)
        os.mkdir(results_path)

        # Ground truth file (manual labeling)
        cicada_file = find_files_with_ext(behavior_data_path, extension="npz")
        if len(cicada_file) > 1:
            print(f"Should be a single cicada file, found: {cicada_file}")
            return
        if len(cicada_file) == 0:
            cicada_file = None
        else:
            cicada_file = cicada_file[0]

        AnalysisOfMvt(data_path=piezo_signals_data_path, results_path=results_path, identifier=data_id,
                      mandatory_piezo_threshold=mandatory_piezo_threshold,
                      nwb_file=nwb_file, left_cam_id=left_cam_id, right_cam_id=right_cam_id,
                      config_citnps_yaml_file=config_citnps_yaml_file,
                      bodyparts_to_fusion=["tail"], cicada_file=cicada_file)
        return

    run_citnps(movies_dict=citnps_movies_dict, from_cicada=False, data_id=data_id,
               luminosity_change_dict=luminosity_change_dict,
               movie_queue_size=movie_queue_size, bodyparts_dict=bodyparts_dict,
               results_path=results_path,
               behavior_time_stamps_dict=behavior_time_stamps_dict,
               yaml_config_file=yaml_config_file)


if __name__ == "__main__":
    # if extract_piezo_signal == True, then we extract the "piezo" signal from the movie
    # if False, then we analyze this "piezo" signal, to identify the behavior epochs
    citnps_main(extract_piezo_signal=False)
