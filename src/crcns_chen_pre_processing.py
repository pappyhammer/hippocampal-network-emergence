import numpy as np
import hdf5storage
import os
from read_roi import read_roi_file, read_roi_zip
import math
import scipy.io as sio


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

def main():
    root_path = '/media/julien/Not_today/hne_not_today/data/crcns_data/chen_2013/'

    # identifier = "20120416_cell1_001"
    identifier = "20120416_cell1_002"
    # identifier = "20120417_cell3_001"
    # identifier = "20120417_cell3_003"
    # identifier = "20120515_cell1_003"
    # identifier = "20120417_cell4_001"
    # identifier = "20120627_cell3_001"
    # identifier = "20120515_cell1_004"
    # identifier = "20120417_cell4_002"
    # identifier = "20120417_cell3_002"
    # identifier = "20120627_cell3_002"
    # identifier = "20120417_cell1_002"
    # identifier = "20120417_cell4_003"
    # identifier = "20120417_cell5_002"
    # identifier = "20120515_cell1_003"
    # identifier = "20120515_cell1_005"
    # identifier = "20120515_cell1_006"
    # identifier = "20120627_cell4_002"
    # identifier = "20120627_cell4_004"
    # identifier = "20120627_cell4_005"

    # 20120417_cell4_001
    two_files_version = False
    # data_path = os.path.join(root_path, "for_test", identifier)
    data_path = os.path.join(root_path, identifier)

    downsampling_factor = 6

    # test = hdf5storage.loadmat(os.path.join(data_path, 'cell1_002.mat'))
    # print(f"test {test}")

    if two_files_version:
        frame_times_file = os.path.join(data_path, f"{identifier}_frame_times.mat")
        spike_times_file = os.path.join(data_path, f"{identifier}_spike_times.mat")
        frame_times = hdf5storage.loadmat(frame_times_file)['t_frame']
        spike_times = hdf5storage.loadmat(spike_times_file)['spike_time']
    else:
        info = hdf5storage.loadmat(os.path.join(data_path, f'info_{identifier}.mat'))
        frame_times = info['t_frame']
        spike_times = info['spike_time']
    frame_times = np.ndarray.flatten(frame_times)
    print(f"n frames {len(frame_times)}")

    zip_file = os.path.join(data_path, f"{identifier}_25_25.zip")
    roi_file = os.path.join(data_path, f"{identifier}_25_25.roi")
    zip_data = None
    if os.path.isfile(zip_file):
        zip_data = read_roi_zip(zip_file)
    else:
        roi = read_roi_file(roi_file)

    # test = hdf5storage.loadmat(os.path.join(data_path, 'p12_17_11_10_a000_CellDetect.mat'))
    # print(f"test {test}")

    # print(f"n frames {spike_times}")
    # print(f"n spikes {len(spike_times)}")
    # print(f"frames {frame_times.shape}")
    # print(f"spikes {spike_times}")

    # print(f"roi {roi}")

    # all_contours = []
    if zip_data is not None:
        coords_caiman_style = np.empty((len(zip_data),), dtype=np.object)
        # for i in range(len(self.map_coords)):
        #     coords_caiman_style[i] = self.map_coords[i]
        for roi_index, roi in enumerate(zip_data.values()):
            n_points = len(roi['x'])
            contours = np.zeros((2, n_points), dtype="int16")
            contours[0] = roi['x']
            contours[1] = roi['y']
            coords_caiman_style[roi_index] = contours
            # all_contours.append(contours)
            # print(f"all_contours {all_contours}")
    else:
        coords_caiman_style = np.empty((1,), dtype=np.object)
        roi = roi[list(roi.keys())[0]]
        n_points = len(roi['x'])
        contours = np.zeros((2, n_points), dtype="int16")
        contours[0] = roi['x']
        contours[1] = roi['y']
        coords_caiman_style[0] = contours

    n_times = len(frame_times)
    if downsampling_factor > 1:
        if n_times % downsampling_factor != 0:
            raise Exception(f"Number of frames {n_times} not divisible by {downsampling_factor}")
        n_times = n_times // downsampling_factor

    spikes_num = np.zeros((len(coords_caiman_style), n_times), dtype="int8")
    for index_spike, spike_time in enumerate(spike_times):
        frame = find_nearest(array=frame_times, value=spike_time)
        frame = int(frame / downsampling_factor)
        if frame == n_times:
            frame -= 1
        spikes_num[0, frame] = 1

    np.save(os.path.join(data_path, f"{identifier}_contours.npy"), coords_caiman_style)
    # sio.savemat(os.path.join(data_path, f"{identifier}_contours.mat"), {"AllContours": coords_caiman_style})
    if downsampling_factor > 1:
        np.save(os.path.join(data_path, f"{identifier}_spikes_{int(60/downsampling_factor)}_hz.npy"), spikes_num)
    else:
        np.save(os.path.join(data_path, f"{identifier}_spikes.npy"), spikes_num)


if __name__ == "__main__":
    main()
