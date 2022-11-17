"""Analyzing the .abf files."""

# MIT License

# Copyright (c) 2021-2022, Hervé Rouault <herve.rouault@univ-amu.fr>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy import ndimage, signal
import math
import pandas as pd
import pyabf
import matplotlib.pyplot as plt
import os
from tkinter import filedialog as fd


def compute_position_and_speed(abf_filename, run_channel, direction_channel, lap_channel, wheel_thr, lap_thr,
                               belt_length, wheel_size, belt_model):
    print(f"Extracting the positions...")
    abf = pyabf.ABF(abf_filename)

    # Decoding the position
    if direction_channel is None:
        abf.setSweep(sweepNumber=0, channel=run_channel)
        wheel_chan = abf.sweepY
        time = abf.sweepX
        transi = position_ticks(chan1=wheel_chan, chan2=None, belt_model=belt_model, thr=wheel_thr)
    else:
        abf.setSweep(sweepNumber=0, channel=run_channel)
        wheel_chan1 = abf.sweepY
        time = abf.sweepX
        abf.setSweep(sweepNumber=0, channel=direction_channel)
        wheel_chan2 = abf.sweepY
        transi = position_ticks(chan1=wheel_chan1, chan2=wheel_chan2, belt_model=belt_model, thr=wheel_thr)
    pos = np.cumsum(transi)

    # laps extraction
    if lap_channel is not None:
        abf.setSweep(sweepNumber=0, channel=lap_channel)
        laps_chan = abf.sweepY

        # Filtering the signal
        sos1 = signal.butter(2, 0.001, output="sos")
        laps_filt = signal.sosfiltfilt(sos1, laps_chan)

        trans_lap = np.zeros(laps_filt.size - 1, dtype=int)
        trans_lap += np.logical_and(
            laps_filt[:-1] < lap_thr, laps_filt[1:] >= lap_thr
        ).astype(int)
        lap_inds = np.nonzero(trans_lap)[0]
        print(f"Found {lap_inds.size} lap(s), @ time indexes: {np.round(lap_inds / abf.dataRate, 2)} sec")
    else:
        print(f"No lap signal")
        lap_inds = None

    wheel_n_ticks = 1024
    wheel_diameter = wheel_size
    ticks_per_cm = wheel_n_ticks / np.pi / wheel_diameter

    positions_real = np.copy(pos)
    if lap_inds is not None:
        for i in lap_inds:
            positions_real[i:] -= positions_real[i]
        off0 = belt_length * ticks_per_cm - positions_real[lap_inds[0] - 1]
        positions_real[: lap_inds[0]] += int(off0)
        track_length = belt_length * ticks_per_cm
        positions_real = np.mod(positions_real, track_length)
    else:
        track_length = belt_length * ticks_per_cm
        positions_real = np.mod(positions_real, track_length)

    position_cm = positions_real / ticks_per_cm

    print(f"Derivating & Smoothing the speed...")
    dt = np.diff(time)[0]
    velocity_real = np.diff(pos) / dt / ticks_per_cm
    gauss_smooth_width = 64 * 500
    input_ = np.fft.fft(velocity_real)
    velocity_cm_s = ndimage.fourier_gaussian(input_, sigma=gauss_smooth_width)
    velocity_cm_s = np.fft.ifft(velocity_cm_s)
    velocity_cm_s = velocity_cm_s.real

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
                             gridspec_kw={'width_ratios': [1]},
                             figsize=(15, 15), dpi=300)

    print(f'Plot speed')
    axes[0].plot(np.arange(0, len(velocity_cm_s)) / abf.dataRate, velocity_cm_s)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (cm/s)")

    print(f'Plot position')
    if lap_inds is not None:
        for xc in lap_inds:
            axes[1].axvline(x=xc / abf.dataRate, color="red")
    axes[1].plot(np.arange(0, len(position_cm)) / abf.dataRate, position_cm)

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Position (cm)")

    if os.path.isdir(os.path.join(os.path.dirname(abf_filename), "abf_analysis")) is False:
        os.mkdir(os.path.join(os.path.dirname(abf_filename), "abf_analysis"))
    saving_path = os.path.join(os.path.dirname(abf_filename), "abf_analysis")

    print(f"Saving distance & speed figure")
    fig.tight_layout()
    fig.savefig(os.path.join(saving_path, f'speed_and_position.pdf'))

    return time[:-1], pos, lap_inds


def position_ticks(chan1, chan2, belt_model, thr=1):

    """Computes the position transitions.

    Args:
        chan1 (npt.NDArray[np.float64]): First position channel
        chan2 (npt.NDArray[np.float64] | None, optional): Second channel if
            present
        thr (float): Crossing threshold value

    Returns:
        npt.NDArray[np.int64]: The transition array

    """

    sos = signal.butter(2, 0.2, output="sos")
    if chan2 is None:
        chan1_filt = signal.sosfiltfilt(sos, chan1)
        transi = np.zeros(chan1.size - 1, dtype=int)
        transi_up = np.logical_and(chan1_filt[:-1] > thr, chan1_filt[1:] < thr)
        transi += transi_up.astype(int)
        transi_down = np.logical_and(
            chan1_filt[:-1] < thr, chan1_filt[1:] > thr
        )
        transi += transi_down.astype(int)
    else:
        chan1_filt = signal.sosfiltfilt(sos, chan1)
        chan2_filt = signal.sosfiltfilt(sos, chan2)
        if belt_model == "L&N":
            transi = double_sign_transi(
                chan1_filt, chan2_filt, thr
            ) - double_sign_transi(chan2_filt, chan1_filt, thr)
        else:
            transi = double_sign_transi(
                chan2_filt, chan1_filt, thr
            ) - double_sign_transi(chan1_filt, chan2_filt, thr)

    return transi


def double_sign_transi(chan1, chan2, thr=1):

    """Extract signed ticks from two signals.

    Args:
        chan1 (npt.NDArray[np.float64]): First analog signal
        chan2 (npt.NDArray[np.float64]): Second analog signal
        thr (float): Value for the transition detections

    Returns:
        npt.NDArray[np.int64]: The signed transitions
    """
    # First: finding the transitions on signal 1
    transi = np.zeros(chan1.size - 1, dtype=int)
    transi_up = np.logical_and(chan1[:-1] < thr, chan1[1:] > thr)
    transi += transi_up.astype(int)
    transi_down = np.logical_and(chan1[:-1] > thr, chan1[1:] < thr)
    transi -= transi_down.astype(int)

    # Second: using the signal 2 to find the polarity of the transitions
    sign = np.zeros(chan2.shape, dtype=int)
    sign += (chan2 > thr).astype(int)
    sign -= (chan2 < thr).astype(int)
    transi *= sign[1:]

    return transi


def run_stats(time, positions_real, positions_real_cum, velocity_real, run_thr=1,
              run_subsample=500):

    """Compute the run epochs statistics.

    Args:
        time (npt.NDArray[np.float64]): Time in seconds.
        positions_real (npt.NDArray[np.float64]): Absolute position of the
            animal.
        positions_real_cum (npt.NDArray[np.float64]): Cumulated position of the
            animal.
        velocity_real (npt.NDArray[np.float64]): Velocity of the animal.

    Returns:
        npt.NDArray[np.float64: stop positions
        npt.NDArray[np.float64: run time-widths
        npt.NDArray[np.float64: run position width

    """
    runs = velocity_real > run_thr
    sub_sampl = run_subsample

    # start and stop indices
    run_starts = np.nonzero(np.logical_and(~runs[:-1], runs[1:]))[0]
    run_ends = np.nonzero(np.logical_and(runs[:-1], ~runs[1:]))[0]
    if run_ends[0] < run_starts[0]:
        run_ends = run_ends[1:]
    if run_ends.size > run_starts.size:
        run_ends = run_ends[: run_starts.size]
    if run_ends.size < run_starts.size:
        run_starts = run_starts[: run_ends.size]

    run_timewidth = time[run_ends * sub_sampl] - time[run_starts * sub_sampl]
    run_poswidth = (
        positions_real_cum[run_ends * sub_sampl]
        - positions_real_cum[run_starts * sub_sampl]
    )

    pos_stops = positions_real[run_starts * sub_sampl]

    return pos_stops, run_timewidth, run_poswidth


def check_reliability(time, positions, lap_inds, plot_path, belt_length, wheel_size):

    """Check the sanity of the extracted positions with plots.

    Plot also some statistics of the run and immobility periods.

    Args:
        time (npt.NDArray[np.float64]): Time in seconds.
        positions (npt.NDArray[np.int64]): Cumulated position of the animal.
        lap_inds (npt.NDArray[np.int64] | None): Indices for the whole laps.
        conf_dict (dict[str, Any]): Configuration dictionary
        plot_path (Path | None, optional): path when saving plots, None
            otherwise.

    """
    n_bins = 20
    hist_pos, edges = np.histogram(positions, bins=n_bins)
    fig = plt.figure(figsize=(6, 9))
    gspec = fig.add_gridspec(3, 2)
    ax_pos = fig.add_subplot(gspec[0, :])
    ax_stop = fig.add_subplot(gspec[1, 0])
    ax_vel = fig.add_subplot(gspec[1, 1])
    ax_runw = fig.add_subplot(gspec[2, 0])
    ax_runt = fig.add_subplot(gspec[2, 1])

    track_length_cm = belt_length
    wheel_n_ticks = 1024
    wheel_diameter = wheel_size
    ticks_per_cm = wheel_n_ticks / np.pi / wheel_diameter

    sub_sampl = 500
    sel_sub = slice(None, None, sub_sampl)
    positions_real = np.copy(positions)
    ax_pos.set_title("Position")
    if lap_inds is not None:
        for i in lap_inds:
            positions_real[i:] -= positions_real[i]
        off0 = track_length_cm * ticks_per_cm - positions_real[lap_inds[0] - 1]
        positions_real[: lap_inds[0]] += int(off0)
        track_length = track_length_cm * ticks_per_cm
        positions_real = np.mod(positions_real, track_length)
    else:
        track_length = track_length_cm * ticks_per_cm
        positions_real = np.mod(positions_real, track_length)

    ax_pos.plot(time[sel_sub], positions_real[sel_sub] / ticks_per_cm)
    ax_pos.set_xlabel("Time (s)")
    ax_pos.set_ylabel("Position (cm)")
    if lap_inds is not None:
        ax_pos.vlines(
            time[lap_inds],
            ymin=-5,
            ymax=np.max(positions_real) / ticks_per_cm + 5,
            colors="red",
        )

    velocity = np.diff(positions[sel_sub])
    velocity_real = velocity / np.diff(time[sel_sub])
    gauss_smooth_width = 64
    input_ = np.fft.fft(velocity_real)
    velocity_real = ndimage.fourier_gaussian(input_, sigma=gauss_smooth_width)
    velocity_real = np.fft.ifft(velocity_real)
    velocity_real = velocity_real.real

    velocity_real /= ticks_per_cm

    pos_stops, run_timewidth, run_poswidth = run_stats(
        time,
        positions_real / ticks_per_cm,
        positions / ticks_per_cm,
        velocity_real
    )

    fig2, ax2 = plt.subplots()
    ax2.scatter(run_timewidth, pos_stops)
    ax2.set_title(
        "Scatter plot of the stop position as a function of stop duration"
    )
    ax2.set_xlabel("Stop duration (s)")
    ax2.set_ylabel("Stop position (cm)")
    if plot_path is not None:
        print(f"Saving Run Stops Statistics figure")
        plot_path2 = os.path.join(os.path.dirname(plot_path), "postion_stops_scatter.pdf")
        fig2.savefig(plot_path2)

    ax_stop.set_title("Distribution of the stop positions")
    # When using the lap signal, the stop positions can be negative
    if lap_inds is not None:
        bins_stop = np.linspace(np.min(pos_stops), np.max(pos_stops), n_bins)
    else:
        bins_stop = np.linspace(0, track_length_cm, n_bins)

    if lap_inds is not None:
        ax_stop.hist(
            pos_stops,
            bins=bins_stop,
        )
    else:
        ax_stop.hist(
            np.mod(pos_stops, track_length) / ticks_per_cm,
            bins=bins_stop,
        )

    ax_stop.set_xlabel("Positions (cm)")
    ax_stop.set_ylabel("Counts")
    ax_vel.set_title("Distribution of the non-zero velocity")
    ax_vel.hist(velocity_real[velocity_real > 1.0], bins=n_bins)
    ax_vel.set_xlabel("Velocity (cm / s)")
    ax_vel.set_ylabel("Counts")
    ax_runw.set_title("Run width distribution")
    ax_runw.set_xlabel("Distance (cm)")
    ax_runw.set_ylabel("Counts")
    ax_runw.hist(run_poswidth, bins=n_bins)
    ax_runt.set_title("Run time distribution")
    ax_runt.set_xlabel("Time (s)")
    ax_runt.set_ylabel("Counts")
    ax_runt.hist(run_timewidth)
    fig.tight_layout()
    if plot_path is not None:
        print(f"Saving Run Statistics figure")
        fig.savefig(plot_path)

    fig, ax = plt.subplots()
    ax.plot(time[:-sub_sampl:sub_sampl], velocity_real)
    ax.plot(time[:-sub_sampl:sub_sampl], velocity_real > 1.0)


def main():
    abf_file = fd.askopenfilename(title="Please select ABF file:",
                                  filetypes=(("AFB files", "*.abf"), ("all files", "*.*")))
    print(f"ABF file: {abf_file}")
    save_path = os.path.join(os.path.dirname(abf_file), "abf_analysis")
    plot_path = os.path.join(save_path, "trajectory_stats.pdf")

    run_channel = 1  # 'int' or None
    direction_channel = 2  # 'int' or None
    absolute_position_channel = 3  # 'int' or None
    belt_length = 192
    wheel_size = 10  # VV = 3.25, L&N = 10
    belt_model = "L&N"  # "VV" ou "L&N"

    time, posis, lap_inds = compute_position_and_speed(abf_filename=abf_file, run_channel=run_channel,
                                                       direction_channel=direction_channel,
                                                       lap_channel=absolute_position_channel, wheel_thr=4, lap_thr=4,
                                                       belt_length=belt_length, wheel_size=wheel_size,
                                                       belt_model=belt_model)

    print(f" ")
    print(f"---------- Check reliability ----------")

    check_reliability(time=time, positions=posis, lap_inds=lap_inds, plot_path=plot_path,
                      belt_length=belt_length, wheel_size=wheel_size)


main()
