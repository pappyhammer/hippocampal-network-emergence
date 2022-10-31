import numpy as np
import math
import pandas as pd
import pyabf
import matplotlib.pyplot as plt
import os
from tkinter import filedialog as fd
import pickle


def compute_position(abf_data_rate, run_data, direction_data, different_speed, belt_length):
    n_bins = len(run_data)
    distance = np.zeros(n_bins, dtype=float)

    nb_period_by_wheel = 256
    wheel_diam_cm = 2 * math.pi * 5

    cm_by_period = wheel_diam_cm / nb_period_by_wheel

    binary_mvt_data = np.zeros(len(run_data), dtype="int8")
    binary_mvt_data[run_data >= 4] = 1
    d_times = np.diff(binary_mvt_data)
    pos_times = np.where(d_times == 1)[0] + 1

    correlation_time_ms = 25
    window_size = int((correlation_time_ms / 1000) * abf_data_rate)

    max_corr_values = []
    corr_delay = []

    for index, pos_bin in enumerate(pos_times[1:]):
        # Find the mouse direction: based on phase shift between speed channel et direction channel
        window_start = max(0, (pos_bin - window_size))
        window_stop = min((pos_bin + window_size), n_bins)
        position_ch1 = run_data[window_start: window_stop]
        position_ch2 = direction_data[window_start: window_stop]
        xcorr_vect = np.correlate(position_ch1, position_ch2, mode="full")
        xcorr_vect_center = (len(xcorr_vect) - 1) / 2
        max_corr = np.argmax(xcorr_vect)
        max_corr_values.append(np.max(xcorr_vect))
        corr_delay.append((max_corr - xcorr_vect_center) / abf_data_rate)
        if different_speed[pos_bin]:
            # Mean it is likely that there is signal in only one channel (so it is not ok to increment position)
            distance[pos_times[index - 1]:pos_bin] = distance[pos_times[index - 1] - 1]
            continue
        if xcorr_vect_center >= max_corr:
                direction = -1
        else:
                direction = 1

        # Increment mouse position:
        #  add cm_by_period each time we find the onset of a period and reset distance to 0 at belt length
        distance[pos_times[index - 1]:pos_bin] = distance[pos_times[index - 1] - 1] + direction * cm_by_period
        if distance[pos_times[index - 1] - 1] + direction * cm_by_period > belt_length:
            distance[pos_times[index - 1]:pos_bin] = distance[pos_times[index - 1]:pos_bin] - belt_length

    return distance


def detect_run_periods(abf_data_rate, run_data, min_speed):
    """
    Using the data from the abf regarding the speed of the animal on the treadmill, return the speed in cm/s
    at each timestamps as well as period when the animal is moving (using min_speed threshold)

    Args:
        run_data (list): Data from the subject run
        min_speed (float): Minimum speed
        abf_data_rate

    Returns:
        mvt_periods (list): List of movements periods
        speed_during_movement_periods (list) : List of subject speed during movements
        speed_by_time (list) : List of subject speed by time

    """
    nb_period_by_wheel = 256
    wheel_diam_cm = 2 * math.pi * 5

    cm_by_period = wheel_diam_cm / nb_period_by_wheel
    binary_mvt_data = np.zeros(len(run_data), dtype="int8")
    speed_by_time = np.zeros(len(run_data))
    is_running = np.zeros(len(run_data), dtype="int8")

    binary_mvt_data[run_data >= 4] = 1
    d_times = np.diff(binary_mvt_data)
    pos_times = np.where(d_times == 1)[0] + 1

    for index, pos in enumerate(pos_times[1:]):
        run_duration = pos - pos_times[index - 1]
        run_duration_s = run_duration / abf_data_rate
        # in cm/s
        speed = cm_by_period / run_duration_s
        if speed >= min_speed:
            speed_by_time[pos_times[index - 1]:pos] = speed
            is_running[pos_times[index - 1]:pos] = 1

    # Do a smoothing to avoid speed peaks on short timescale
    smoothing_window = int(0.5 * abf_data_rate)
    speed_by_time_df = pd.DataFrame({'speed_by_time': speed_by_time})
    smooth_df = speed_by_time_df.rolling(smoothing_window, min_periods=None, center=True,
                                         win_type=None, on=None, axis=0, closed=None).mean()
    speed_by_time = smooth_df['speed_by_time'].values

    return speed_by_time


def main():
    abf_file = fd.askopenfilename(title="Please select ABF file:",
                                  filetypes=(("AFB files", "*.abf"), ("all files", "*.*")))
    print(f"ABF file: {abf_file}")

    abf = pyabf.ABF(abf_file)

    run_channel = 1
    direction_channel = 2
    belt_length = 192

    if os.path.isdir(os.path.join(os.path.dirname(abf_file), "abf_analysis")) is False:
        os.mkdir(os.path.join(os.path.dirname(abf_file), "abf_analysis"))
    saving_path = os.path.join(os.path.dirname(abf_file), "abf_analysis")

    print(f'Speed analysis from channel 1')
    abf.setSweep(sweepNumber=0, channel=run_channel)
    run_data_ch1 = abf.sweepY
    speed_ch_1 = detect_run_periods(abf_data_rate=abf.dataRate, run_data=run_data_ch1, min_speed=0.5)
    print(f'Done')

    print(f'Speed analysis from channel 2')
    abf.setSweep(sweepNumber=0, channel=direction_channel)
    run_data_ch2 = abf.sweepY
    speed_ch_2 = detect_run_periods(abf_data_rate=abf.dataRate, run_data=run_data_ch2, min_speed=0.5)
    print(f'Done')

    print(f'Compare speed')
    # speed_difference = speed_ch_1 - speed_ch_2
    # abs_difference = np.abs(speed_difference)
    # plt.hist(abs_difference, bins=2000, density=True)
    # plt.xlabel("Speed difference")
    # plt.ylabel("Density")
    # plt.show()

    same_speed = np.isclose(a=speed_ch_2, b=speed_ch_1, rtol=0.1, atol=0.8, equal_nan=True)
    different_speed = np.invert(same_speed)

    print(f"Found {np.round(len(np.where(different_speed)[0]) / len(different_speed) * 100, 2)} %"
          f" of ABF data points with different speed")
    speed_ch_1[np.where(different_speed)[0]] = 0
    print(f'Done')

    print(f'Compute position')
    distance = compute_position(abf_data_rate=abf.dataRate, run_data=run_data_ch1, direction_data=run_data_ch2,
                                different_speed=different_speed, belt_length=belt_length)
    print(f'Done')

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
                             gridspec_kw={'width_ratios': [1]},
                             figsize=(15, 15), dpi=300)

    print(f'Plot speed')
    axes[0].plot(np.arange(0, len(speed_ch_1)) / abf.dataRate, speed_ch_1)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (cm/s)")

    print(f'Plot position')
    axes[1].plot(np.arange(0, len(distance)) / abf.dataRate, distance)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Position (cm)")

    print(f"Save in PDF format:")
    fig.savefig(os.path.join(saving_path, f'speed_and_position.pdf'))
    print(f"Save in interactive python format:")
    with open(os.path.join(saving_path, f'speed_and_position.pickle'), 'wb') as f:
        pickle.dump(fig, f, pickle.HIGHEST_PROTOCOL)


main()
