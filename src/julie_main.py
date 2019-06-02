import numpy as np
from neo.io import NeuralynxIO
import os
from matplotlib import pyplot as plt
import quantities as pq
from elephant.spectral import welch_psd

from ephyviewer import mkQApp, MainViewer, TraceViewer, TimeFreqViewer
from ephyviewer import InMemoryAnalogSignalSource
import ephyviewer

root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
path_data = os.path.join(root_path, "data/julie_data")

animals = {}


reader = NeuralynxIO(dirname=os.path.join(path_data, "04232019 kcnq2_1", "2019-04-24_10-14-07 (vu)"))
# , "CSC3.ncs"
print(f"supported_objects: {reader.supported_objects}")
print(f"readable_objects: {reader.readable_objects}")

reader.parse_header()
print(reader)

for k, v in reader.header.items():
    print(k, v)

print()
print('#'*50)
print()

n_channels = len(reader.header["signal_channels"])
n_events = len(reader.header["event_channels"])
print(f"n_channels {n_channels}, n_events {n_events}")
nb_event_channel = reader.event_channels_count()
print('nb_event_channel', nb_event_channel)

raw_sigs = reader.get_analogsignal_chunk(channel_indexes = np.arange(n_channels))
#  channel_names=['CSC3', 'CSC4']
# read bloc
bl = reader.read()
print(f"len(bl) {len(bl)}")
seg = reader.read_segment()
print(type(seg))
sampling_rate = reader.get_signal_sampling_rate()
signal_size = reader.get_signal_size(block_index=0, seg_index=0)
print(f"signal_size {signal_size}: {signal_size/sampling_rate}s")
print(f"sampling_rate {sampling_rate}")
# 'NeuralynxIO' object has no attribute 'event_timestamps'

print("SEG: " + str(seg.file_origin))
# fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1)
# # ax2 = fig.add_subplot(2, 1, 2)
# ax1.set_title(seg.file_origin)
# ax1.set_ylabel('arbitrary units')
# mint = 0 * pq.s
# maxt = np.inf * pq.s
# for i, asig in enumerate(seg.analogsignals):
#     times = asig.times.rescale('s').magnitude
#     asig = asig.magnitude
#     ax1.plot(times, asig)
# #
# # trains = [st.rescale('s').magnitude for st in seg.spiketrains]
# # colors = plt.cm.jet(np.linspace(0, 1, len(seg.spiketrains)))
# # ax2.eventplot(trains, colors=colors)
#
# plt.show()

print(f"seg.analogsignals[0].duration {seg.analogsignals[0].duration}")
print(f"seg.analogsignals[0].shape {seg.analogsignals[0].shape}")

freqs, psd = welch_psd(seg.analogsignals[0][:, 0], freq_res=0.1) # num_seg=100)

print(f"freqs {freqs.shape} {freqs.units}, psd: {psd.shape} {psd.units}")

# Look in data_analysis
# apply_wavelet(patient, data_bin_for_fft, wp, channel_str, use_baseline=True, tenlog10=True)
# called by apply_wavelet_to_all_data
# bosc_frequency_band_episodes_finder to find band of frequencies using the BOSC method
# used in the function: apply_wavelet_throught_sessions

# ephyviewer
use_ephyviewer = False
if use_ephyviewer:
    # you must first create a main Qt application (for event loop)
    app = mkQApp()

    t_start = 700 * sampling_rate

    # Create the main window that can contain several viewers
    win = MainViewer(debug=True, show_auto_scale=True)

    # Create a datasource for the viewer
    # here we use InMemoryAnalogSignalSource but
    # you can alose use your custum datasource by inheritance
    source = InMemoryAnalogSignalSource(np.array(seg.analogsignals), sampling_rate, t_start)

    # create a viewer for signal with TraceViewer
    view1 = TraceViewer(source=source, name='trace')
    view1.params['scale_mode'] = 'same_for_all'
    view1.auto_scale()

    # create a time freq viewer conencted to the same source
    view2 = TimeFreqViewer(source=source, name='tfr')

    view2.params['show_axis'] = False
    view2.params['timefreq', 'deltafreq'] = 1
    view2.by_channel_params['ch3', 'visible'] = True

    # add them to mainwindow
    win.add_view(view1)
    win.add_view(view2)

    # show main window and run Qapp
    win.show()
    app.exec_()