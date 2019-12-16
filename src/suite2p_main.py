import numpy as np
# option to import from github folder
# sys.path.insert(0, 'C:/Users/carse/github/suite2p')
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
            'do_registration': False, # whether to register data
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
            # TODO: Try to lower the threshold to get more ROI
            'threshold_scaling': 0.7, # adjust the automatically determined threshold by this scalar multiplier
            'max_overlap': 0.70, # cells with more overlap than this get removed during triage, before refinement
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

    # new OPS (cop/past from https://github.com/MouseLand/suite2p on the 1st december 2019
    ops = {
        # file paths
        'look_one_level_down': False,  # whether to look in all subfolders when searching for tiffs
        'fast_disk': [],  # used to store temporary binary file, defaults to save_path0
        'delete_bin': False,  # whether to delete binary file after processing
        'mesoscan': False,  # for reading in scanimage mesoscope files
        'h5py': [],  # take h5py as input (deactivates data_path)
        'h5py_key': 'data',  # key in h5py where data array is stored
        'save_path0': [],  # stores results, defaults to first item in data_path
        'subfolders': [],
        # main settings
        'nplanes': 1,  # each tiff has these many planes in sequence
        'nchannels': 1,  # each tiff has these many channels per plane
        'functional_chan': 1,  # this channel is used to extract functional ROIs (1-based)
        'tau': 2.,  # this is the main parameter for deconvolution, 2 for gcamp6s
        'fs': 10.,  # sampling rate (PER PLANE - e.g. if you have 12 planes then this should be around 2.5)
        'force_sktiff': False,  # whether or not to use scikit-image for tiff reading
        # output settings
        'preclassify': 0,  # apply classifier before signal extraction with probability 0.5 (turn off with value 0)
        'save_mat': False,  # whether to save output as matlab files
        'combined': True,  # combine multiple planes into a single result /single canvas for GUI
        'aspect': 1.0,  # um/pixels in X / um/pixels in Y (for correct aspect ratio in GUI)
        # bidirectional phase offset
        'do_bidiphase': False,
        'bidiphase': 0,
        # registration settings
        'do_registration': 0,  # whether to register data (2 forces re-registration)
        'keep_movie_raw': False,
        'nimg_init': 300,  # subsampled frames for finding reference image
        'batch_size': 500,  # number of frames per batch
        'maxregshift': 0.1,  # max allowed registration shift, as a fraction of frame max(width and height)
        'align_by_chan': 1,  # when multi-channel, you can align by non-functional channel (1-based)
        'reg_tif': False,  # whether to save registered tiffs
        'reg_tif_chan2': False,  # whether to save channel 2 registered tiffs
        'subpixel': 10,  # precision of subpixel registration (1/subpixel steps)
        'smooth_sigma': 1.15,  # ~1 good for 2P recordings, recommend >5 for 1P recordings
        'th_badframes': 1.0,
        # this parameter determines which frames to exclude when determining cropping - set it smaller to exclude more frames
        'pad_fft': False,
        # non rigid registration settings
        'nonrigid': True,  # whether to use nonrigid registration
        'block_size': [128, 128],  # block size to register (** keep this a multiple of 2 **)
        'snr_thresh': 1.2,
        # if any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing
        'maxregshiftNR': 5,  # maximum pixel shift allowed for nonrigid, relative to rigid
        # 1P settings
        '1Preg': False,  # whether to perform high-pass filtering and tapering
        'spatial_hp': 50,  # window for spatial high-pass filtering before registration
        'pre_smooth': 2,  # whether to smooth before high-pass filtering before registration
        'spatial_taper': 50,
        # how much to ignore on edges (important for vignetted windows, for FFT padding do not set BELOW 3*ops['smooth_sigma'])
        # cell detection settings
        'roidetect': True,  # whether or not to run ROI extraction
        'spatial_scale': 0,  # 0: multi-scale; 1: 6 pixels, 2: 12 pixels, 3: 24 pixels, 4: 48 pixels
        'connected': True,  # whether or not to keep ROIs fully connected (set to 0 for dendrites)
        'nbinned': 5000,  # max number of binned frames for cell detection
        'max_iterations': 100,  # maximum number of iterations to do cell detection
        'threshold_scaling': 1,  # adjust the automatically determined threshold by this scalar multiplier used to be 5
        'max_overlap': 0.7,  # cells with more overlap than this get removed during triage, before refinement default: 0.75
        'high_pass': 100,  # running mean subtraction with window of size 'high_pass' (use low values for 1P)
        # ROI extraction parameters
        'inner_neuropil_radius': 2,  # number of pixels to keep between ROI and neuropil donut
        'min_neuropil_pixels': 300,  # minimum number of pixels in the neuropil
        'allow_overlap': True,  # pixels that are overlapping are thrown out (False) or added to both ROIs (True)
        # channel 2 detection settings (stat[n]['chan2'], stat[n]['not_chan2'])
        'chan2_thres': 0.65,  # minimum for detection of brightness on channel 2
        # deconvolution settings
        'baseline': 'maximin',  # baselining mode (can also choose 'prctile')
        'win_baseline': 60.,  # window for maximin
        'sig_baseline': 10.,  # smoothing constant for gaussian filter
        'prctile_baseline': 8.,  # optional (whether to use a percentile baseline)
        'neucoeff': .7,  # neuropil coefficient
        'xrange': np.array([0, 0]),
        'yrange': np.array([0, 0])
    }

    """
    ops to modify defaults):
    'preclassify': 0.5
    'threshold_scaling': 5 # select many more cells if lowered
    'max_overlap': 0.75
     'inner_neuropil_radius': 2,  # number of pixels to keep between ROI and neuropil donut
    'min_neuropil_pixels': 350,  # minimum number of pixels in the neuropil
     'allow_overlap': False,  # pixels that are overlapping are thrown out (False) or added to both ROIs (True)
    'neucoeff': .7,  # neuropil coefficient
    'allow_overlap'
    
    To run the code:
    conda activate suite2p
    To install it from sources: pip install -e .
    then python suite2p_main.py
    and: python -m suite2p
    conda deactivate
    """

    # provide an h5 path in 'h5py' or a tiff path in 'data_path'
    # db overwrites any ops (allows for experiment specific settings)
    # p8_19_09_29_1_a001.h5
    db = {
        # 'h5py': '/Users/pappyhammer/Documents/academique/these_inmed/suite2p/suite2p_tiffs/MichelMotC_p8_19_09_29_1_a000.h5',  # a single h5 file path
        'h5py': '/home/julien/these_inmed/suite2p/suite2p_tiffs/MotC_Yannick_version_p8_19_09_29_0_a001.h5',
        # 'h5py_key': 'data',
        'look_one_level_down': False,  # whether to look in ALL subfolders when searching for tiffs
        # 'data_path': ['/home/julien/these_inmed/suite2p/suite2p_tiffs'],  # a list of folders with tiffs
        # 'data_path': ['/Users/pappyhammer/Documents/academique/these_inmed/suite2p/suite2p_tiffs'],
        # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)

        'subfolders': [],  # choose subfolders of 'data_path' to look in (optional)
        'fast_disk': '/home/julien/these_inmed/suite2p/suite2p_bin',  # string which specifies where the binary file will be stored (should be an SSD)
        # 'fast_disk': '/Users/pappyhammer/Documents/academique/these_inmed/suite2p/suite2p_bin'
    }

    # run one experiment
    opsEnd = run_s2p(ops=ops, db=db)


main()