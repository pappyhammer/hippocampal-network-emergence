from math import pi, sin
import matplotlib.pyplot as plt
import struct
import numpy as np
from array import array

"""
Code used to generate a sine wave that changes linearly between 2 frequencies. 
Then the sine wave is saved in a template file for Heka PatchMaster software. 

The parameters are:
- min and max frequency
- Amplitude (in mV) and amplitude (in mV) where to starts
- Duration in seconds
- Sampling rate
- 
"""

# from https://stackoverflow.com/questions/19771328/sine-wave-that-exponentialy-changes-between-frequencies-f1-and-f2-at-given-time

def save_in_file(file_name, data):
    f = open(file_name, 'wb')
    # '<' for little endian
    # https://docs.python.org/3/library/struct.html#format-characters
    float_array = struct.pack('<' + ('f' * len(data)), *data)
    # https://docs.python.org/3/library/array.html
    # float_array = array('f', data)
    f.write(float_array)
    f.close()


def sweep(f_start, f_end, interval, n_steps, amplitude):
    time_values = []
    unknown_values = []
    y_values = []
    for i in range(n_steps):
        delta = i / float(n_steps)
        t = interval * delta
        phase = 2 * pi * t * (f_start + (f_end - f_start) * delta / 2)
        # print t, phase * 180 / pi, 3 * sin(phase)
        time_values.append(t)
        unknown_values.append(phase * 180 / pi)

        # we want integers so we can convert it as bytes
        # we add amplitudeso we don't have negative values
        y_values.append(amplitude * sin(phase))
        # return t, phase * 180 / pi, 3 * sin(phase)
    return time_values, unknown_values, y_values


def main():
    sampling_rate = 10000
    duration_sec = 30
    n_steps = duration_sec * sampling_rate
    # amplitude in mV
    amplitude = 0.05
    # pivot point
    # then max value is pivot_value + amplitude, and min is pivot_value - amplitude
    pivot_value = 0
    min_freq = 0
    max_freq = 15
    time_values, unknown_values, y_values = sweep(min_freq, max_freq, interval=duration_sec,
                                                  n_steps=n_steps, amplitude=amplitude)
    y_values = [y - pivot_value for y in y_values]
    file_name = "/media/julien/Not_today/hne_not_today/results_hne/laurent/zap_fct_current.tpl"
    save_in_file(file_name, y_values)
    plt.plot(time_values, y_values)
    plt.show()


if __name__ == "__main__":
    main()
