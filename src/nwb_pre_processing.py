import os
from cicada.preprocessing.cicada_preprocessing_main import find_dir_to_convert
from cicada.preprocessing.cicada_data_to_nwb import convert_data_to_nwb

def main():
    root_path = "/media/julien/Not_today/hne_not_today/"
    # root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    # dir_to_explore = os.path.join(root_path, "data/red_ins/p6/191205_191211_1")
    # dir_to_explore = os.path.join(root_path, "data/rem_data")
    # dir_to_explore = os.path.join(root_path, "data/chronic")
    # dir_to_explore = os.path.join(root_path, "data/artem_data")
    # dir_to_explore = os.path.join(root_path, "data/marco_data")
    # dir_to_explore = os.path.join(root_path, "data")
    dir_to_explore = os.path.join(root_path, "data/SWISS_data/p9/190211_190220_0")
    # dir_to_explore = os.path.join(root_path, "data/rem_data/200219_200226/")
    default_convert_to_nwb_yml_file = "/home/julien/these_inmed/hne_project/cicada_gitlab/cicada/src/cicada/preprocessing/pre_processing_default.yaml"
    # default_convert_to_nwb_yml_file = f"{dir_to_explore}/pre_processing_marco.yaml"
    session_dirs = find_dir_to_convert(dir_to_explore=dir_to_explore,
                                       keywords=[["session_data"], ["subject_data"]],
                                       extensions=("yaml", "yml"))
    print(f"session_dirs {session_dirs}")

    # nwb_files_dir = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/data/nwb_files/"
    # nwb_files_dir = "H:/Documents/Data/julien/data/nwb_files"
    nwb_files_dir = "/media/julien/Not_today/hne_not_today/data/nwb_files/"

    sessions_to_exclude = ["p5_19_09_02_a000",  # no data except abf
                           "p5_19_04_24_a001",  # GadCre no ROIs yet
                           "p6_19_03_21_a001",  # GadCre no ROIs yet
                           "p7_19_02_19_a000",  # GadCre no ROIs yet
                           "p7_17_10_18_a002",  # no abf
                           "p7_17_10_18_a004",  # no abf
                           "p8_19_03_19_a000",  # no abf
                           "p10_17_11_16_a003",  # no abf
                           "p10_19_03_04_a000",  # GadCre no ROIs yet
                           "p10_19_04_29_a000",  # GadCre no ROIs yet
                           "p10_19_04_29_a001",  # GadCre no ROIs yet
                           "p10_19_04_29_a002",  # GadCre no ROIs yet
                           "p10_19_04_29_a003",  # GadCre no ROIs yet
                           "p11_19_02_07_a001",  # GadCre no ROIs yet
                           "p11_19_02_07_a002",  # GadCre no ROIs yet
                           "p11_19_04_30_a000",  # GadCre no ROIs yet
                           "p11_19_04_30_a002",  # GadCre no ROIs yet
                           "p11_17_11_24_a000",  # no abf
                           "p11_17_11_24_a001",  # no abf
                           "p12_19_02_08_a000",  # no abf, GadCre with ROIs
                           "p11_19_02_15_0000",  # issue with abf, see why
                           "p8_18_10_17_a000",  # sampling rate abf < 50 kHz
                           "p8_18_10_17_a001",  # sampling rate abf < 50 kHz
                           "p9_18_09_27_a003",  # sampling rate abf < 50 kHz
                           "p13_18_10_29_a000",  # sampling rate abf < 50 kHz
                           "p13_18_10_29_a001",  # sampling rate abf < 50 kHz
                           "p13_18_10_29_a002",  # not data to use
                           "p14_18_10_23_a000", # no abf
                           "p14_18_10_23_a001" # no abf
                           ]
    print(f"Number of sessions_to_exclude: {len(sessions_to_exclude)}")

    for session_dir in session_dirs:
        # keeping only one session
        # if os.path.split(session_dir)[1] != "p12_17_11_10_a000":
        #     continue

        session_dir_name = os.path.split(session_dir)[1]
        # ["5", "6", "7"]

        """



        """
        # for marco
        # mb048
        # 20200211
        # 20200212
        # claire_52
        # claire_153
        # if session_dir_name not in ["claire_153"]:
        #     continue
        # if session_dir_name not in ["p60_a529_2015_02_25"]:
        #     continue
        # if session_dir_name not in ["p8_19_09_29_1_a000", "p8_19_09_29_1_a001"]:
        #     continue
        # if session_dir_name not in ["p8_19_09_29_0_a000", "p8_19_09_29_0_a001"]:
        #     continue
        # if session_dir_name not in ["171124_a001"]:
        #     continue
        # if session_dir_name not in ["181023_a000"]:
        #     continue
        # p5: 191210_0_191210_a001, p7: 200110_a000, p9:190930_a001
        # if session_dir_name not in ["191210_a001"]:
        #     continue
        # if session_dir_name not in ["191210_a001", "200110_a000", "190930_a001"]:
        #     continue
        # p8 190921_190929_0_190929_a000
        # if session_dir_name not in ["171124_a001"]:
        #     continue
        # p5 rem
        # if session_dir_name not in ["200311_a001", "200311_a000"]:
        #     continue
        # if session_dir_name not in ["190902_a000"]:
        #     continue
        # p5191127_191202_191202_a000 : 191202_a000
        # p5191127_191202_191202_a001 : 191202_a001
        # if session_dir_name not in ["190923_a001"]:
        #     continue
        # if session_dir_name not in ["171110_a002"]:
        #     continue
        # if session_dir_name not in ["180207_a002"]:
        #     continue
        # if session_dir_name not in ["simulated_data_v1"]:
        #     continue
        # artem_191119
        # if session_dir_name not in ["171018_a002"]:
        #     continue
        # p13 chronic 181016_181029_180929_a001 (then p14 & p15)
        # if session_dir_name not in ["181029_a001"]:
        #     continue
        # if session_dir_name not in ["181030_a001"]:
        #     continue
        # if session_dir_name not in ["200311_a001"]:
        #     continue
        # p13 chronic 181016_181029_180929_a002 (then p14 & p15, p16)
        # if session_dir_name not in ["181029_a002"]:
        #     continue
        # if session_dir_name not in ["181030_a002"]:
        #     continue
        # if session_dir_name not in ["181031_a002"]:
        #     continue
        # if session_dir_name not in ["191210_a001"]:
        #     continue
        # if session_dir_name not in ["181029_a002", "181030_a002", "181031_a002", "181101_a002"]:
        #     continue
        # 190929_a000: p8 190921_190929_0
        # if session_dir_name not in ["180209_a004"]:
        #     continue
        if session_dir_name not in ["190220_a001"]:
            continue
        # if session_dir_name not in ["200110_a001"]:
        #     continue
        # p6 190921_190927_1_190927_a000
        # if session_dir_name not in ["190927_a000"]:
        #     continue
        # if session_dir_name not in ["p12_17_11_10_a000"]:
        #     continue
        # if session_dir_name[1:3] not in ["41"] or session_dir_name in sessions_to_exclude:
        #     continue
        print(f"Loading data for {os.path.split(session_dir)[1]}")
        convert_data_to_nwb(data_to_convert_dir=session_dir,
                            default_convert_to_nwb_yml_file=default_convert_to_nwb_yml_file,
                            nwb_files_dir=nwb_files_dir)


if __name__ == "__main__":
    main()