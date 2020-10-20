from datetime import datetime
from pynwb import NWBHDF5IO
from abc import abstractmethod
from cv2 import VideoCapture
from threading import Thread
import cv2
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from skimage.segmentation import random_walker
import skimage.draw as draw
import sys
from random import randrange
import yaml
from scipy import signal
import time

# from: https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
# import the Queue class from Python 3
if sys.version_info >= (3, 0):
	from queue import Queue
# otherwise, import the Queue class for Python 2.7
else:
	raise Exception("Works only with Python 3")

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


class AnalysisOfMvt:

    def __init__(self, data_path, results_path, identifier, nwb_file, left_cam_id, right_cam_id,
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

        self.threshold_z_score_mvt = 0.75

        self.n_body_parts = 0

        self.bodyparts_to_fusion = bodyparts_to_fusion if bodyparts_to_fusion is not None else list()

        self.time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")

        self.bodyparts = ['forelimb_left', 'forelimb_right', "hindlimb_left", "hindlimb_right", "tail"]

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
        # self.threshold_z_score_mvt
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
            self.binary_mvt_matrix[bodypart_index, mvt_data > self.threshold_z_score_mvt] = 1
            if bodypart in self.bodyparts_to_fusion:
                for side in sides:
                    other_side_mvt_data = self.mvt_data_dict[side][bodypart]
                    # peaks, properties = signal.find_peaks(x=other_side_mvt_data, height=self.threshold_z_score_mvt,
                    #                                       distance=distance_min_peaks)
                    # self.binary_mvt_matrix[bodypart_index, peaks] = 1
                    self.binary_mvt_matrix[bodypart_index, other_side_mvt_data > self.threshold_z_score_mvt] = 1

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
                    self.binary_mvt_matrix[bodypart_index, other_side_mvt_data > self.threshold_z_score_mvt] = 1
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
        # first key key to put in npz and value a list of periods
        periods_by_type_of_mvt = dict()
        periods_by_type_of_mvt["complex_mvt"] = np.zeros(self.n_frames, dtype="int8")
        periods_by_type_of_mvt["startle"] = np.zeros(self.n_frames, dtype="int8")

        for bodypart in self.bodyparts:
            periods_by_type_of_mvt[f"twitch_{bodypart}"] = np.zeros(self.n_frames, dtype="int8")
            periods_by_type_of_mvt[f"mvt_{bodypart}"] = np.zeros(self.n_frames, dtype="int8")

            still_vs_mvt = np.copy(self.binary_mvt_matrix[self.bodyparts_indices[bodypart]])

            invert_still_vs_mvt = 1 - still_vs_mvt
            still_periods = get_continous_time_periods(invert_still_vs_mvt)
            gap_to_fill = 3
            # feeling the gap of 2 frames without mvt
            for still_period in still_periods:
                if (still_period[1] - still_period[0] + 1) <= gap_to_fill:
                    still_vs_mvt[still_period[0]:still_period[1] + 1] = 1
            # TODO: See to remove mvt of less than 2 frames here
            still_vs_mvt_dict[bodypart] = still_vs_mvt
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

        for action_str in ["cicada", "evaluate"]:
            # will be saved in the npz
            # each key is the tag, and value is a 2d array (2x n_intervals) with start and finish in sec on lines,
            # each column is an interval
            behaviors_encoding_dict = dict()
            for key_behavior, binary_array in periods_by_type_of_mvt.items():
                if action_str == "evaluate":
                    key_to_use = key_behavior
                else:
                    key_to_use = "auto_" + key_behavior
                periods = get_continous_time_periods(binary_array)
                if "left" in key_behavior:
                    behaviors_encoding_dict[key_to_use] = \
                        encode_period_with_timestamps(periods=periods,
                                                      timestamps=self.behavior_left_time_stamps)
                else:
                    behaviors_encoding_dict[key_to_use] = \
                        encode_period_with_timestamps(periods=periods,
                                                      timestamps=self.behavior_right_time_stamps)
            if self.gt_behavior is not None and (action_str == "evaluate"):
                evaluate_behavior_predictions(ground_truth_labels=self.gt_behavior,
                                              other_labels=behaviors_encoding_dict,
                                              n_frames=self.n_frames,
                                              behavior_timestamps=self.behavior_right_time_stamps)

            np.savez(os.path.join(self.results_path, f"test_{self.identifier}_{action_str}.npz"),
                     **behaviors_encoding_dict)



class DraggableRectangle:
    lock = None  # only one can be animated at a time
    def __init__(self, rect):
        self.rect = rect
        self.press = None
        self.background = None

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes: return
        if DraggableRectangle.lock is not None: return
        contains, attrd = self.rect.contains(event)
        if not contains: return
        # print('event contains', self.rect.xy)
        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata
        DraggableRectangle.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.rect.figure.canvas
        axes = self.rect.axes
        self.rect.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.rect.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.rect)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if DraggableRectangle.lock is not self:
            return
        if event.inaxes != self.rect.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)

        canvas = self.rect.figure.canvas
        axes = self.rect.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.rect)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        'on release we reset the press data'
        if DraggableRectangle.lock is not self:
            return

        self.press = None
        DraggableRectangle.lock = None

        # turn off the rect animation property and reset the background
        self.rect.set_animated(False)
        self.background = None

        # redraw the full figure
        self.rect.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

class ImageAreaSelection:

    def __init__(self, video_reader, initial_frame, n_frames, size_rect, title, binary_thresholds,
                 x_bottom_left, y_bottom_left):
        """
            Build a plot with a image a rectangle of size_rect that can be moved over the image.
            Call bottom_left coin coordinates
        Args:
            video_reader:
            n_frames:
            initial_frame:
            size_rect:
            title:
            binary_thresholds: list of 1 to 2 int, between 0 and 255, in ascending order, representing the different
            level of gray
            x_bottom_left:
            y_bottom_left:
        """
        self.video_reader = video_reader
        self.n_frames = n_frames
        self.initial_frame = initial_frame
        self.binary_thresholds = binary_thresholds
        self.size_rect = list(size_rect)
        self.title = title
        self.fig = None
        self.last_frames = []
        self.imshow_obj = None
        self.current_frame = initial_frame
        self.binary_mode = False
        self.rect = None
        self.step_rect_dim = 5
        # bottom left corner coordinates
        self.x_bottom_left = x_bottom_left
        self.y_bottom_left = y_bottom_left

    def select_area(self):
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.title(self.title)
        ax = self.fig.add_subplot(111)
        img_frame = self.video_reader.get_frame(self.current_frame)
        ax.set_ylim(img_frame.shape[0])
        ax.set_xlim(img_frame.shape[1])
        # ax.imshow(np.fliplr(img), origin='lower')
        if self.binary_mode:
            img_frame = binarize_frame(img_frame, binary_thresholds=self.binary_thresholds)
        self.imshow_obj = ax.imshow(np.flipud(img_frame), cmap='gray')
        # ax.imshow(img)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        # rects = ax.bar(range(10), 20*np.random.rand(10))
        left = self.x_bottom_left
        bottom = self.y_bottom_left
        width = self.size_rect[0]
        height = self.size_rect[1]
        self.rect = plt.Rectangle((left, bottom), width, height, fill=False, color="red", lw=1)
        # rect.set_transform(ax.transAxes)
        # rect.set_clip_on(False)
        ax.add_patch(self.rect)
        drs = []
        dr = DraggableRectangle(self.rect)
        dr.connect()
        drs.append(dr)

        plt.show()
        return [int(value) for value in dr.rect.xy]

    def print_keyboard_shortcuts(self):
        print("GUI keyboard shortcuts")
        print(f"r -> display a random frame")
        print(f"e -> Last frame displayed")
        print(f"m -> Display the frame just before in time")
        print(f"p -> Display the frame just after in time")
        print(f"b -> In/out binary mode")
        print(f"c -> Change binary thresholds")
        print(f"left -> Reduce width of the rectangle")
        print(f"right -> Increase width of the rectangle")
        print(f"down -> Reduce height of the rectangle")
        print(f"up -> Increase height of the rectangle")
        print(" ")

    def press(self, event):
        # print('press', event.key)
        sys.stdout.flush()
        if event.key == 'r':
            self.last_frames.append(self.current_frame)
            self.display_new_frame(self.pick_random_frame())
        elif event.key == 'p':
            # like plus one
            self.last_frames.append(self.current_frame)
            self.display_new_frame(self.current_frame+1)
        elif event.key == 'm':
            # like minus one
            self.last_frames.append(self.current_frame)
            self.display_new_frame(self.current_frame-1)
        elif event.key == 'e':
            # last img displayed
            if len(self.last_frames) > 0:
                last_frame = self.last_frames[-1]
                self.last_frames = self.last_frames[:-1]
                self.display_new_frame(last_frame)
        elif event.key == 'b':
            self.binary_mode = not self.binary_mode
            self.display_new_frame(self.current_frame)
        elif event.key == "c":
            self.change_binary_thresholds()
        elif event.key == 'left':
            self.change_width_rect(pixels_diff=self.step_rect_dim * -1)
        elif event.key == 'right':
            self.change_width_rect(pixels_diff=self.step_rect_dim)
        elif event.key == 'up':
            self.change_height_rect(pixels_diff=self.step_rect_dim)
        elif event.key == 'down':
            self.change_height_rect(pixels_diff=self.step_rect_dim * -1)

    def change_binary_thresholds(self):
        print(" ")
        print(f"Actual binary thresholds: {self.binary_thresholds}")
        input_tresholds = input("Binary thresholds:")
        split_values = input_tresholds.split(",")
        new_thresholds = []
        for split_value in split_values:
            try:
                thr = int(split_value)
                if thr < 1 or thr > 254:
                    print(f"{thr} out of bounds 1-254")
                    return
                if (len(new_thresholds) > 0) and (thr <= new_thresholds[-1]):
                    print(f"{split_values}: should be in ascending order")
                    return
                new_thresholds.append(thr)
            except ValueError:
                print(f"{split_value} is not an integer")
                return

        if len(new_thresholds) == 0 or len(new_thresholds) > 2:
            print(f"{new_thresholds} len should be 1 or 2")
            return

        self.binary_thresholds = new_thresholds
        print(f"New binary thresholds: {self.binary_thresholds}")
        if self.binary_mode:
            # we update the image
            self.display_new_frame(frame = self.current_frame)

    def change_width_rect(self, pixels_diff):
        width_rect = self.rect.get_width()
        self.size_rect[0] = max(20, self.size_rect[0] + pixels_diff)
        self.rect.set_width(w=max(20, width_rect + pixels_diff))
        self.fig.canvas.draw()

    def change_height_rect(self, pixels_diff):
        height_rect = self.rect.get_height()
        self.size_rect[1] = max(20, self.size_rect[1] + pixels_diff)
        self.rect.set_height(h=max(20, height_rect + pixels_diff))
        self.fig.canvas.draw()

    def display_new_frame(self, frame):
        self.current_frame = frame
        img_frame = self.video_reader.get_frame(frame)
        if self.binary_mode:
            img_frame = binarize_frame(img_frame,
                                       binary_thresholds=self.binary_thresholds)
        self.imshow_obj.set_data(np.flipud(img_frame))
        self.fig.canvas.draw()

    def pick_random_frame(self):
        return randrange(self.n_frames)

    def get_size_rect(self):
        return self.size_rect

    def get_binary_thresholds(self):
        return self.binary_thresholds
#
# def plot_img_with_rect(img, title="", size_rect=(1600, 1000)):
#     """
#     Build a plot with a image a rectangle of size_rect that can be moved over the image.
#     It returns the bottom_left coin coordinates
#     Args:
#         img:
#         title:
#         size_rect:
#
#     Returns:
#
#     """
#     fig = plt.figure()
#     plt.title(title)
#     ax = fig.add_subplot(111)
#     ax.set_ylim(img.shape[0])
#     ax.set_xlim(img.shape[1])
#     # ax.imshow(np.fliplr(img), origin='lower')
#     ax.imshow(np.flipud(img))
#     # ax.imshow(img)
#     plt.gca().invert_xaxis()
#     plt.gca().invert_yaxis()
#     # rects = ax.bar(range(10), 20*np.random.rand(10))
#     left = 10
#     bottom = 10
#     width = size_rect[0]
#     height = size_rect[1]
#     rect = plt.Rectangle((left, bottom), width, height, fill=False, color="red", lw=1)
#     # rect.set_transform(ax.transAxes)
#     # rect.set_clip_on(False)
#     ax.add_patch(rect)
#     drs = []
#     dr = DraggableRectangle(rect)
#     dr.connect()
#     drs.append(dr)
#
#     plt.show()
#     return [int(value) for value in dr.rect.xy]

def norm01(data):
    min_value = np.min(data)
    max_value = np.max(data)

    difference = max_value - min_value

    data -= min_value

    if difference > 0:
        data = data / difference

    return data

class VideoReaderWrapper:
    """
        An abstract class that should be inherited in order to create a specific video format wrapper.
        A class can be created using either different packages or aim at specific format.

    """

    def __init__(self):
        self._length = None
        self._width = None
        self._height = None
        self._fps = None

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def fps(self):
        return self._fps

    @abstractmethod
    def close_reader(self):
        pass

class OpenCvVideoReader(VideoReaderWrapper):
    """
    Use OpenCv to read video
    see https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
    """

    def __init__(self, video_file_name, queueSize=128):
        """

        Args:
            video_file_name:
            queueSize : The maximum number of frames to store in the queue.
            This value defaults to 128 frames, but you depending on (1) the frame dimensions
            of your video and (2) the amount of memory you can spare, you may want to raise/lower this value.
        """
        VideoReaderWrapper.__init__(self)

        self.video_file_name = video_file_name
        self.basename_video_file_name = os.path.basename(video_file_name)

        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.video_capture = VideoCapture(self.video_file_name)
        self.stopped = False

        if self.video_capture.isOpened() == False:
            raise Exception(f"Error opening video file {self.video_file_name}")


        # initialize the queue used to store frames read from
        # the video file
        self.video_queue = Queue(maxsize=queueSize)

        # length in frames
        self._length = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # in pixels
        self._width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # frame per seconds
        self._fps = self.video_capture.get(cv2.CAP_PROP_FPS)

        # for the thread
        self.next_frame_to_read = 0
        self.last_frame_to_read = self._length - 1

        print(f"OpenCvVideoReader init for {self.basename_video_file_name}: "
              f"self.width {self.width}, self.height {self.height}, n frames {self._length}")

    def start(self, first_frame_to_read, last_frame_to_read):
        # start a thread to read frames from the file video stream
        self.next_frame_to_read = first_frame_to_read
        self.last_frame_to_read = min(self._length - 1, last_frame_to_read)
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
            # otherwise, ensure the queue has room in it
            if not self.video_queue.full():
                if self.next_frame_to_read > self.last_frame_to_read:
                    self.stop()
                    return
                # read the next frame from the file
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.next_frame_to_read)
                (grabbed, frame) = self.video_capture.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
                # add the frame to the queue
                self.video_queue.put(frame)
                self.next_frame_to_read += 1

    def read(self):
        # return next frame in the queue
        return self.video_queue.get()

    def all_frames_on_queue(self):
        """
        Return True if all frames have been processed
        Returns:

        """
        if self.next_frame_to_read > self.last_frame_to_read:
            return True

    def more(self):
        return self.video_queue.qsize() > 0

    def get_frame(self, frame_index):
        if (frame_index >= self._length) or (frame_index < 0):
            return None

        # The first argument of cap.set(), number 2 defines that parameter for setting the frame selection.
        # Number 2 defines flag CAP_PROP_POS_FRAMES which is
        # a 0-based index of the frame to be decoded/captured next.
        # The second argument defines the frame number in range 0.0-1.0
        # frame_no = frame_index / self._length
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read the next frame from the video. If you set frame 749 above then the code will return the last frame.
        # 'res' is boolean result of operation, one may use it to check if frame was successfully read.
        res, frame = self.video_capture.read()

        if res:
            return frame
        else:
            return None

    def close_reader(self):
        # When everything done, release the capture
        self.video_capture.release()
        cv2.destroyAllWindows()


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


def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax


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

def find_files_with_ext(dir_to_explore, extension):
    subfiles = []
    for (dirpath, dirnames, filenames) in os.walk(dir_to_explore):
        subfiles = [os.path.join(dirpath, filename) for filename in filenames]
        break
    return [f for f in subfiles if f.endswith(extension)]


def circle_points(resolution, center, radius):
    """
    Generate points which define a circle on an image.Centre refers to the centre of the circle
    """
    # from: https://towardsdatascience.com/image-segmentation-using-pythons-scikit-image-module-533a61ecc980
    radians = np.linspace(0, 2 * np.pi, resolution)
    c = center[1] + radius * np.cos(radians)  # polar co-ordinates
    r = center[0] + radius * np.sin(radians)

    return np.array([c, r]).T

def randomw_walker(img_test):
    # TOO SLOW
    print(f"img_test.shape {img_test.shape}")
    # into grey_scale
    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)

    # img_test = img_test.astype(np.uint8)
    img_test = ((img_test / img_test.max(axis=0).max(axis=0)) * 255).astype(np.uint8)

    # plt.imshow(img_test, 'gray')
    # plt.show()
    # return
    # The range of the binary image spans over (-1, 1).
    # We choose the hottest and the coldest pixels as markers.
    markers = np.zeros(img_test.shape, dtype=np.uint)
    # markers[data < -0.95] = 1
    # markers[data > 0.95] = 2

    # markers[data < -0.95] = 1
    # markers[500, 650] = 2

    # image_labels = np.zeros(image_gray.shape, dtype=np.uint8)

    # Exclude last point because a closed path should not have duplicate points
    points = circle_points(200, [500, 620], 300)[:-1]
    # points = circle_points(200, [80, 250], 80)[:-1]
    indices = draw.circle_perimeter(500, 620, 35)  # from here
    markers[indices] = 1
    markers[points[:, 1].astype(np.int), points[:, 0].astype(np.int)] = 2

    # Run random walker algorithm
    # labels = random_walker(img_test, markers, beta=10, mode='bf')
    image_segmented = random_walker(img_test, markers)

    fig, ax = image_show(img_test)
    ax.imshow(image_segmented == 1, alpha=0.3)
    plt.show()

    # Plot results
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
    #                                     sharex=True, sharey=True)
    # ax1.imshow(img_test, cmap='gray')
    # ax1.axis('off')
    # ax1.set_title('Noisy data')
    # ax2.imshow(markers, cmap='magma')
    # ax2.axis('off')
    # ax2.set_title('Markers')
    # ax3.imshow(labels, cmap='gray')
    # ax3.axis('off')
    # ax3.set_title('Segmentation')
    #
    # fig.tight_layout()
    # plt.show()
    return


def binarize_frame(img_frame, binary_thresholds):
    """

    Args:
        img_frame:
        binary_thresholds: list of 1 to 2 int, between 0 and 255, in ascending order, representing the different
            level of gray

    Returns:

    """
    # into grey_scale
    img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)

    # plt.imshow(img_frame, 'gray')
    # plt.show()

    # img_frame = img_frame.astype(np.uint8)
    img_frame = ((img_frame / img_frame.max(axis=0).max(axis=0)) * 255).astype(np.uint8)

    # bluring the image a bit
    img_frame = cv2.medianBlur(img_frame, 5)

    # Applying Histogram Equalization
    # img_frame = cv2.equalizeHist(img_frame)

    if len(binary_thresholds) == 1:
        # used to be 75
        ret, th1 = cv2.threshold(img_frame, binary_thresholds[0], 255, cv2.THRESH_BINARY)
    else:
        if len(binary_thresholds) != 2:
            raise Exception("Only 1 or 2 binary_thresholds are supported yet")
        # used to be [65, 150]
        th1 = np.zeros_like(img_frame)
        th1[img_frame < binary_thresholds[0]] = 0
        th1[np.logical_and(img_frame >= binary_thresholds[0], img_frame < binary_thresholds[1])] = 127
        th1[img_frame >= binary_thresholds[1]] = 255
    # print(f"np.unique(th1) {np.unique(th1)}")
    # Otsu's thresholding
    # ret2, th1 = cv2.threshold(img_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    th1 = np.array(th1).astype(float)
    # th3 = cv2.adaptiveThreshold(img_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                             cv2.THRESH_BINARY, 11, 2)

    """
                th2 = cv2.adaptiveThreshold(img_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
                th3 = cv2.adaptiveThreshold(img_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

                titles = ['Original Image', 'Global Thresholding (v = 127)',
                          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
                images = [img_frame, th1, th2, th3]

                for i in range(4):
                    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
                    plt.title(titles[i])
                    plt.xticks([]), plt.yticks([])
                plt.show()
    """

    return th1

def ploting_power_spectrum(filtered_signal, raw_signal, title, times_to_mark, show_it, results_path, file_name):
    """
        Plot the values and the power spectrum of 2 signals
        :param filtered_signal (array): signal that has been filterd
        :param raw_signal (array): raw signal, in which no filter has been applied
        :param title (str): plot title
        :param times_to_mark (list or np.array): None or float values, for each value, a dashed vertical line will be ploted
        at the x coordinate given.
        :param show_it (bool): if True, plot the figure
        :param results_path (str): path where toe save the figure
        :param file_name (str): file_name in which to save the figure
        :return:
    """
    amplitude_thresholds = [0.5, 1]
    sampling_rate = 20
    fourier_transform = np.fft.rfft(filtered_signal)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, sampling_rate / 2, len(power_spectrum))

    fourier_transform = np.fft.rfft(raw_signal)
    abs_fourier_transform = np.abs(fourier_transform)
    original_power_spectrum = np.square(abs_fourier_transform)
    original_frequency = np.linspace(0, sampling_rate / 2, len(original_power_spectrum))

    min_value = min(np.min(filtered_signal), np.min(raw_signal))
    max_value = max(np.max(filtered_signal), np.max(raw_signal))

    if show_it:
        linewidth = 1
    else:
        # for pdf to zoom
        linewidth = 0.1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
    ax1.plot(filtered_signal, c="blue", zorder=10, linewidth=linewidth)
    ax1.plot(raw_signal, c="red", zorder=5, linewidth=linewidth)
    ax2.plot(frequency, power_spectrum, c="blue", zorder=10, linewidth=linewidth)
    ax2.plot(original_frequency, original_power_spectrum, c="red", zorder=5, linewidth=linewidth)
    if times_to_mark is not None:
        for time_to_mark in times_to_mark:
            ax1.vlines(time_to_mark, min_value,
                       max_value, color="black", linewidth=linewidth*1.5,
                       linestyles="dashed", zorder=1)
    for amplitude_threshold in amplitude_thresholds:
        ax1.hlines(amplitude_threshold, 0, len(raw_signal)-1,
                   color="black", linewidth=linewidth,
                   linestyles="dashed")
    plt.title(title)
    if show_it:
        plt.show()

    save_formats = ["png", "pdf"]
    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{results_path}/{file_name}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())

    plt.close()

def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = signal.butter(order, [low, high], btype='bandstop')
    # y = signal.lfilter(i, u, data)
    # TODO: see if there is also a shift here
    # to correct shift (https://stackoverflow.com/questions/45098384/butterworth-filter-x-shift)
    y = signal.filtfilt(i, u, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    # produce a shift in the signal
    # y = signal.lfilter(b, a, data)
    # to correct shift (https://stackoverflow.com/questions/45098384/butterworth-filter-x-shift)
    y = signal.filtfilt(b, a, data)
    return y

def z_score_normalization_by_luminosity_state(mvt_signal, luminosity_states, n_frames, flatten_transitions):
    """
    Normalize the signal using z_score on chunks based on frames indices in luminosity_states determining the change
    is luminosity
    Args:
        mvt_signal:
        luminosity_states:
        n_frames: int, number of frames, starting at zero. Could be less than len(mvt_signal) in case we don't
        want to z-score it all
        flatten_transitions (bool): if True, we put frames around the luminosity change at the mean of the following
        segment

    Returns:

    """
    # plt.plot(mvt_signal)
    # plt.title(f"Before normalization")
    # plt.show()

    mvt_signal = np.copy(mvt_signal)
    last_chunk_frame = 0
    for luminosity_state_index, luminosity_change_frame in enumerate(luminosity_states):
        # print(f"luminosity_change_frame {luminosity_change_frame}, last_chunk_frame {last_chunk_frame}")
        if flatten_transitions:
            if last_chunk_frame == 0:
                # print(f"mean_value : {np.mean(mvt_signal[1:min(luminosity_change_frame, n_frames)])}")
                mvt_signal[0] = np.mean(mvt_signal[1:min(luminosity_change_frame, n_frames)])
            else:
                last_frame = min(luminosity_change_frame, n_frames)
                # print(f"mean_value : {np.mean(mvt_signal[last_chunk_frame+3:last_frame])}")
                # flattening the frame aroudn the change in luninosity
                mvt_signal[last_chunk_frame:last_chunk_frame+3] = np.mean(mvt_signal[last_chunk_frame+3:last_frame])

                if luminosity_state_index == 1:
                    mvt_signal[last_chunk_frame-2:last_chunk_frame] = \
                        np.mean(mvt_signal[:last_chunk_frame-2])
                else:
                    mvt_signal[last_chunk_frame - 2:last_chunk_frame] = \
                        np.mean(mvt_signal[luminosity_states[luminosity_state_index - 2]:last_chunk_frame - 2])

        if luminosity_change_frame > n_frames:
            mvt_signal[last_chunk_frame:n_frames] = stats.zscore(mvt_signal[last_chunk_frame:n_frames])
            last_chunk_frame = n_frames
            # plt.plot(mvt_signal)
            # plt.title(f"state {luminosity_state_index}, before break")
            # plt.show()
            break

        mvt_signal[last_chunk_frame:luminosity_change_frame] = \
            stats.zscore(mvt_signal[last_chunk_frame:luminosity_change_frame])
        last_chunk_frame = luminosity_change_frame
        # plt.plot(mvt_signal)
        # plt.title(f"state {luminosity_state_index}")
        # plt.show()

    if last_chunk_frame < n_frames:
        if flatten_transitions and (n_frames - last_chunk_frame > 5):
            mvt_signal[last_chunk_frame:last_chunk_frame + 3] = np.mean(mvt_signal[last_chunk_frame + 3:n_frames])
            mvt_signal[last_chunk_frame-2:last_chunk_frame] = \
                np.mean(mvt_signal[luminosity_states[-2]:last_chunk_frame - 2])
        mvt_signal[last_chunk_frame:n_frames] = stats.zscore(mvt_signal[last_chunk_frame:n_frames])
        # plt.plot(mvt_signal)
        # plt.title(f"Last chunk")
        # plt.show()

    # plt.plot(mvt_signal)
    # plt.title(f"Before return")
    # plt.show()

    return mvt_signal

def main_behavior_analysis(extract_mvt):
    root_path = "/media/julien/Not_today/hne_not_today/"
    # root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    data_path = os.path.join(root_path, "data/tada_data/for_training/")

    age = 5
    animal_id = "191127_191202"
    session_id = "191202_a001"

    data_path_for_mvt_outcome = os.path.join(root_path, "data", "red_ins", f"p{age}", animal_id, session_id,
                                             f"signal_{animal_id}_{session_id}")

    results_path = os.path.join(root_path, "data/tada_data/results_tada/")
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    #
    results_path = os.path.join(results_path, time_str)
    os.mkdir(results_path)

    data_id = f"p{age}_{animal_id}_{session_id}"

    data_path = os.path.join(data_path, data_id)

    npz_data_path = os.path.join(data_path, "behaviors_timestamps")

    nwb_file = find_files_with_ext(data_path, "nwb")[0]

    cicada_file = find_files_with_ext(npz_data_path, "npz")[0]

    npz_base_name = os.path.basename(cicada_file)
    new_npz_base_name = npz_base_name[:-4] + "_in_frames" + ".npz"
    new_npz_file = os.path.join(data_path, new_npz_base_name)

    if not extract_mvt:
        AnalysisOfMvt(data_path=data_path_for_mvt_outcome, results_path=results_path, identifier=data_id,
                      nwb_file=nwb_file, left_cam_id="23109588", right_cam_id="22983298",
                      bodyparts_to_fusion= ["tail"], cicada_file=cicada_file)
        return

    avi_files = find_files_with_ext(data_path, "avi")

    for avi_file in avi_files:
        if "22983298" in avi_file:
            right_movie_file = avi_file
        else:
            left_movie_file = avi_file

    right_movie = OpenCvVideoReader(right_movie_file, queueSize=2000)
    n_frames_right = right_movie.length

    left_movie = OpenCvVideoReader(left_movie_file, queueSize=2000)
    n_frames_left = left_movie.length

    # 672
    right_behavior_time_stamps = get_behaviors_movie_time_stamps(nwb_file, cam_id="22983298")
    if len(right_behavior_time_stamps) != n_frames_right:
        if len(right_behavior_time_stamps) > n_frames_right:
            # removing the last frames
            print(f"Removing frames on right timestamps {len(right_behavior_time_stamps)} vs {n_frames_right}")
            right_behavior_time_stamps = right_behavior_time_stamps[:n_frames_right - len(right_behavior_time_stamps)]
        else:
            raise Exception(f"Wrong number of frames on the right side "
                            f"{len(right_behavior_time_stamps)} vs {n_frames_right}")

    left_behavior_time_stamps = get_behaviors_movie_time_stamps(nwb_file, cam_id="23109588")
    if len(left_behavior_time_stamps) != n_frames_left:
        if len(left_behavior_time_stamps) > n_frames_left:
            # removing the last frames
            print(f"Removing frames on left timestamps {len(left_behavior_time_stamps)} vs {n_frames_left}")
            left_behavior_time_stamps = left_behavior_time_stamps[:n_frames_left - len(left_behavior_time_stamps)]
        else:
            raise Exception(f"Wrong number of frames on the left side "
                            f"{len(left_behavior_time_stamps)} vs {n_frames_left}")

    ci_movie_time_stamps = get_ci_movie_time_stamps(nwb_file)

    intervals_names = get_intervals_names(nwb_file)
    if "ci_recording_on_pause" not in intervals_names:
        raise Exception("ci_recording_on_pause epochs not in the NWB")

    pause_epochs = get_interval_times(nwb_file, "ci_recording_on_pause")
    print(f"pause_epochs {pause_epochs.shape}")

    n_frames_dict = dict()
    movie_dict = dict()
    behavior_time_stamps_dict = dict()

    # for looping over both sides
    n_frames_dict["right"] = n_frames_right
    movie_dict["right"] = right_movie
    behavior_time_stamps_dict["right"] = right_behavior_time_stamps

    n_frames_dict["left"] = n_frames_left
    movie_dict["left"] = left_movie
    behavior_time_stamps_dict["left"] = left_behavior_time_stamps

    if len(left_behavior_time_stamps) != n_frames_left:
        print(f"len(left_behavior_time_stamps) {len(left_behavior_time_stamps)} vs {n_frames_left} n_frames")

    if len(right_behavior_time_stamps) != n_frames_right:
        print(f"len(right_behavior_time_stamps) {len(right_behavior_time_stamps)} vs {n_frames_right} n_frames")

    first_ci_frame_dict = dict()
    last_ci_frame_dict = dict()
    # indicate when their is a change in luminosity (laser on/off) (epochs in frames), 1d array n_transitions:
    luminosity_change_dict = dict()
    luminosity_change_dict["right"] = np.zeros(pause_epochs.shape[1]*2+2, dtype="int64")
    luminosity_change_dict["left"] = np.zeros(pause_epochs.shape[1]*2+2, dtype="int64")

    # finding which frame from the behavior matches the CI activation (for z-score)
    first_ci_frame_right = int(find_nearest(right_behavior_time_stamps, ci_movie_time_stamps[0]))
    first_ci_frame_dict["right"] = first_ci_frame_right
    last_ci_frame_right = int(find_nearest(right_behavior_time_stamps, ci_movie_time_stamps[-1]))
    last_ci_frame_dict["right"] = last_ci_frame_right

    # print(f"first_ci_frame_right {first_ci_frame_right}")
    first_ci_frame_left = int(find_nearest(left_behavior_time_stamps, ci_movie_time_stamps[0]))
    first_ci_frame_dict["left"] = first_ci_frame_left
    last_ci_frame_left = int(find_nearest(left_behavior_time_stamps, ci_movie_time_stamps[-1]))
    last_ci_frame_dict["left"] = last_ci_frame_left

    for side in ["right", "left"]:
        luminosity_change_dict[side][0] = first_ci_frame_dict[side]
        for i in range(pause_epochs.shape[1]):
            luminosity_change_dict[side][(i*2) + 1] = int(find_nearest(behavior_time_stamps_dict[side],
                                                                   pause_epochs[0, i]))
            luminosity_change_dict[side][(i*2) + 2] = int(find_nearest(behavior_time_stamps_dict[side],
                                                                   pause_epochs[1, i]+1))
        luminosity_change_dict[side][-1] = last_ci_frame_dict[side] + 1

    bodyparts = ["forelimb",  "hindlimb", "tail"]
    # bodyparts = ["forelimb"]
    size_rect_legs = [350, 250]
    size_rect_tail = [400, 250]
    default_binary_thresholds = [65, 150]
    # kzy is a string f"{side}_{bodypart}"
    bodyparts_config_yaml_file = os.path.join(data_path, "gui_config", f'{data_id}_bodyparts_config_mvt_gui.yaml')
    if os.path.isfile(bodyparts_config_yaml_file):
        print(f"Loading gui config {data_id}_bodyparts_config_mvt_gui.yaml")
        with open(bodyparts_config_yaml_file, 'r') as stream:
            bodyparts_config = yaml.load(stream, Loader=yaml.FullLoader)
    else:
        bodyparts_config = dict()

    actual_binary_thresholds = default_binary_thresholds
    for bodypart in bodyparts:
        for side in ["right", "left"]:
            # to make a copy and avoid problem when dumping config in the yaml
            actual_binary_thresholds = list(actual_binary_thresholds)
            img_frame = movie_dict[side].get_frame(0)

            if f"{side}_{bodypart}" in bodyparts_config:
                x_bottom_left, y_bottom_left, size_rect, \
                actual_binary_thresholds = bodyparts_config[f"{side}_{bodypart}"]
                y_bottom_left = img_frame.shape[0] - y_bottom_left
            else:
                if bodypart == "tail":
                    size_rect = size_rect_tail
                else:
                    size_rect = size_rect_legs
                x_bottom_left, y_bottom_left = (50, 10)
            image_area_selection = ImageAreaSelection(video_reader=movie_dict[side],
                                                      initial_frame=0, n_frames=n_frames_dict[side],
                                                      size_rect=size_rect, title=f"{side} {bodypart}",
                                                      x_bottom_left=x_bottom_left,
                                                      y_bottom_left=y_bottom_left,
                                                      binary_thresholds=actual_binary_thresholds)
            image_area_selection.print_keyboard_shortcuts()
            x_bottom_left, y_bottom_left = image_area_selection.select_area()
            # x_bottom_left, y_bottom_left = plot_img_with_rect(img_frame, title="", size_rect=size_rect)
            y_bottom_left = img_frame.shape[0] - y_bottom_left
            size_rect = image_area_selection.get_size_rect()
            # print(f"size_rect {size_rect}")
            binary_thresholds = image_area_selection.get_binary_thresholds()
            bodyparts_config[f"{side}_{bodypart}"] = [x_bottom_left, y_bottom_left, size_rect, binary_thresholds]
            actual_binary_thresholds = binary_thresholds
    # TODO: for new toolbox, make a new function that will take as argument he sides/movie to explore and the bodypart
    #  to analyse and return a dict with the signal
    for side in ["right", "left"]:
        print(f"luminosity_change {side} side: {luminosity_change_dict[side]}")
        # ----------------------------------------------------
        # ----------------------------------------------------
        # ----------------------------------------------------
        # frames = np.arange(n_frames_dict[side])
        frames = np.arange(1500)

        # ----------------------------------------------------
        # ----------------------------------------------------
        # ----------------------------------------------------
        # key is a string representing the bodypart,
        # value is a 1d array representing the diff between each binarized frame
        mvt_by_bodypart = dict()
        last_frame_by_bodypart = dict()

        movie_dict[side].start(first_frame_to_read=frames[0], last_frame_to_read=frames[-1])
        # giving him a bit of advance
        time.sleep(5.0)
        # for frame in frames:
        # going though the frames
        frame = frames[0]
        while True:
            # all_frames_on_queue
            if not movie_dict[side].more():
                if not movie_dict[side].all_frames_on_queue():
                    # then it means all frames are not in queue, then we make a break to let the thread fill a bit
                    time.sleep(1)
                else:
                    break
            if frame % 2000 == 0:
                print(f"Processing frame {frame}")
            # img_frame = movie_dict[side].get_frame(frame)
            img_frame = movie_dict[side].read()
            # plt.imshow(img_frame, 'gray')
            # plt.show()

            for bodypart in bodyparts:
                # diff of pixels between 2 consecutive binary frames
                if bodypart not in mvt_by_bodypart:
                    mvt_by_bodypart[bodypart] = np.zeros(n_frames_dict[side], dtype="float")
                    last_frame_by_bodypart[bodypart] = None

                bodypart_config = bodyparts_config[f"{side}_{bodypart}"]
                if frame == 0:
                    print(f"bodypart_config {side}_{bodypart} : {bodypart_config}")
                x_bottom_left, y_bottom_left, size_rect, binary_thresholds = bodypart_config

                img_frame_cropped = img_frame[y_bottom_left - size_rect[1]:y_bottom_left,
                            x_bottom_left:x_bottom_left + size_rect[0]]
                # start_time = time.time()
                img_frame_cropped_bin = binarize_frame(img_frame_cropped, binary_thresholds)
                # stop_time = time.time()
                # print(f"Time to binarize one frame: "
                #       f"{np.round(stop_time - start_time, 5)} s")
                # plt.imshow(img_frame_cropped_bin, 'gray')
                # plt.show()

                # diff between two frames, to build the movement 1d array
                if last_frame_by_bodypart[bodypart] is not None:
                    mvt_by_bodypart[bodypart][frame] = np.sum(np.abs(np.subtract(last_frame_by_bodypart[bodypart],
                                                                                 img_frame_cropped_bin)))
                    # mvt[frame] = np.sum(last_frame == img_frame)
                last_frame_by_bodypart[bodypart] = img_frame_cropped_bin
            frame += 1
        # now normalizing the signal and filtering it
        for bodypart in bodyparts:
            # normalization of the signal using z-score
            mvt = z_score_normalization_by_luminosity_state(mvt_signal=mvt_by_bodypart[bodypart],
                                                            luminosity_states=luminosity_change_dict[side],
                                                            n_frames=frames[-1]+1,
                                                            flatten_transitions=True)
            original_mvt = np.copy(mvt)
            use_low_band_pass_filter = True
            if use_low_band_pass_filter:
                # remove frequency higher than 2 Hz
                mvt = butter_lowpass_filter(data=mvt, cutoff=1.5, fs=20, order=10)
            else:
                # bandstop filter
                # issue with bandstop, session without laser have a ten-fold lower amplitude
                mvt = butter_bandstop_filter(data=mvt, lowcut=2.8, highcut=3.5, fs=20, order=6)
                # mvt = butter_bandstop_filter(data=mvt, lowcut=5.3, highcut=5.5, fs=20, order=6)
                # mvt = butter_bandstop_filter(data=mvt, lowcut=8, highcut=8.6, fs=20, order=6)

            # second normalization after filtering
            mvt = z_score_normalization_by_luminosity_state(mvt_signal=mvt,
                                                            luminosity_states=luminosity_change_dict[side],
                                                            n_frames=frames[-1]+1,
                                                            flatten_transitions=False)

            # noise at 3.05hz
            ploting_power_spectrum(filtered_signal=mvt, raw_signal=original_mvt, title=f"{side}_{bodypart}",
                                   times_to_mark=luminosity_change_dict[side], show_it=True,
                                   results_path=results_path,
                                   file_name=f"{data_id}_{side}_{bodypart}_mvt_power_spectrum")

            arg_dict = {f"{bodypart}_{side}": mvt,
                        "timestamps": behavior_time_stamps_dict[side]}
            np.savez(os.path.join(results_path, f"{data_id}_{side}_{bodypart}_mvt.npz"), **arg_dict)

    for key in movie_dict.keys():
        movie_dict[key].close_reader()

    with open(os.path.join(results_path, f'{data_id}_bodyparts_config_mvt_gui.yaml'), 'w') as outfile:
        yaml.dump(bodyparts_config, outfile, default_flow_style=False)


if __name__ == "__main__":
    extract_mvt = True
    # extract_mvt = False
    main_behavior_analysis(extract_mvt=extract_mvt)

