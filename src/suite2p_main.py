import numpy as np
import sys
# option to import from github folder
# sys.path.insert(0, 'C:/Users/carse/github/suite2p')
import suite2p
from suite2p.run_s2p import run_s2p

def main():
    # set your options for running
    # overwrites the run_s2p.default_ops
    ops = {
            'fast_disk': [], # used to store temporary binary file, defaults to save_path0 (set as a string NOT a list)
            'save_path0': [], # stores results, defaults to first item in data_path
            'delete_bin': False, # whether to delete binary file after processing
            # main settings
            'nplanes' : 1, # each tiff has these many planes in sequence
            'nchannels' : 1, # each tiff has these many channels per plane
            'functional_chan' : 1, # this channel is used to extract functional ROIs (1-based)
            'diameter':3, # this is the main parameter for cell detection, 2-dimensional if Y and X are different (e.g. [6 12])
            'tau':  2., # this is the main parameter for deconvolution, 2 for gcamp6s
            'fs': 10.,  # sampling rate (total across planes)
            # output settings
            'save_mat': False, # whether to save output as matlab files
            'combined': True, # combine multiple planes into a single result /single canvas for GUI
            # parallel settings
            'num_workers': 0, # 0 to select num_cores, -1 to disable parallelism, N to enforce value
            'num_workers_roi': -1, # 0 to select number of planes, -1 to disable parallelism, N to enforce value
            # registration settings
            'do_registration': True, # whether to register data
            'nimg_init': 200, # subsampled frames for finding reference image
            'batch_size': 200, # number of frames per batch
            'maxregshift': 0.1, # max allowed registration shift, as a fraction of frame max(width and height)
            'align_by_chan' : 1, # when multi-channel, you can align by non-functional channel (1-based)
            'reg_tif': False, # whether to save registered tiffs
            'subpixel' : 10, # precision of subpixel registration (1/subpixel steps)
            # cell detection settings
            'connected': True, # whether or not to keep ROIs fully connected (set to 0 for dendrites)
            'navg_frames_svd': 5000, # max number of binned frames for the SVD
            'nsvd_for_roi': 1000, # max number of SVD components to keep for ROI detection
            'max_iterations': 100, # maximum number of iterations to do cell detection
            'ratio_neuropil': 6., # ratio between neuropil basis size and cell radius
            'ratio_neuropil_to_cell': 3, # minimum ratio between neuropil radius and cell radius
            'tile_factor': 1., # use finer (>1) or coarser (<1) tiles for neuropil estimation during cell detection
            # TODO: Try to lower the treshold to get more ROI
            'threshold_scaling': 1., # adjust the automatically determined threshold by this scalar multiplier
            'max_overlap': 0.80, # cells with more overlap than this get removed during triage, before refinement
            'inner_neuropil_radius': 2, # number of pixels to keep between ROI and neuropil donut
            'outer_neuropil_radius': np.inf, # maximum neuropil radius
            'min_neuropil_pixels': 300, # minimum number of pixels in the neuropil
            # deconvolution settings
            'baseline': 'maximin', # baselining mode
            'win_baseline': 60., # window for maximin
            'sig_baseline': 10., # smoothing constant for gaussian filter
            'prctile_baseline': 8.,# optional (whether to use a percentile baseline)
            'neucoeff': .7,  # neuropil coefficient
          }

    # provide an h5 path in 'h5py' or a tiff path in 'data_path'
    # db overwrites any ops (allows for experiment specific settings)
    db = {
        'h5py': '/Users/pappyhammer/Documents/academique/these_inmed/suite2p/suite2p_tiffs/p8.h5',  # a single h5 file path
        'h5py_key': 'data',
        'look_one_level_down': False,  # whether to look in ALL subfolders when searching for tiffs
        # 'data_path': ['/home/julien/these_inmed/suite2p/suite2p_tiffs'],  # a list of folders with tiffs
        # 'data_path': ['/Users/pappyhammer/Documents/academique/these_inmed/suite2p/suite2p_tiffs'],
        # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)

        'subfolders': [],  # choose subfolders of 'data_path' to look in (optional)
        # 'fast_disk': '/home/julien/these_inmed/suite2p/suite2p_bin',  # string which specifies where the binary file will be stored (should be an SSD)
        'fast_disk': '/Users/pappyhammer/Documents/academique/these_inmed/suite2p/suite2p_bin'
    }

    # run one experiment
    opsEnd = run_s2p(ops=ops, db=db)

main()