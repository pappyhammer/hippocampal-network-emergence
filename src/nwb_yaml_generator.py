import yaml
import os
import pandas as pd
from datetime import datetime
import numpy as np

# ----------------------
#      columns indices
# ----------------------
EXT_AGE_COL = 0
EXT_SUBJECT_ID_COL = 1
EXT_IMAGING_DATE_COL = 2
EXT_SESSION_COL = 3
EXT_SESSION_ID_COL = 4
EXT_PLANE_LOC_COL = 5
EXT_PIEZO_CH_COL = 8
EXT_TREADMMILL_CH_COL = 9
EXT_BEHAVIOR_1_CH_COL = 10
EXT_BEHAVIOR_2_CH_COL = 11
EXT_LFP_CH_COL = 15


MAIN_SURGERY_DATA_COL = 0
MAIN_SUBJECT_ID_COL = 2
MAIN_RECORDING_DATE_COL = 9
MAIN_LINE_COL = 5
MAIN_AGE_INJECTION_COL = 7
MAIN_VIRUS_COL = 8
MAIN_WEIGHT_COL = 4
MAIN_IMAGING_FILMS_COL = 11
MAIN_IMAGING_NOTES_COL = 12
MAIN_SURGERY_NOTES_COL = 13


class SessionNwbYamlGenerator:
    def __init__(self, main_df, subject_ext_df, index_session_ext_df, subject_id, path_results):
        self.main_df = main_df
        self.subject_ext_df = subject_ext_df
        self.index_session_ext_df = index_session_ext_df
        self.session = subject_ext_df.iloc[index_session_ext_df, EXT_SESSION_COL]
        self.ext_session_id = subject_ext_df.iloc[index_session_ext_df, EXT_SESSION_ID_COL]
        self.imaging_date = subject_ext_df.iloc[index_session_ext_df, EXT_IMAGING_DATE_COL]
        self.imaging_date = datetime.strptime(self.imaging_date, '%y_%m_%d')
        # Oriens or pyramidale
        self.image_plane_location = str(subject_ext_df.iloc[index_session_ext_df, EXT_PLANE_LOC_COL])
        if self.image_plane_location == "nan":
            self.image_plane_location = ""
        self.age = int(subject_ext_df.iloc[index_session_ext_df, EXT_AGE_COL])

        self.subject_id = subject_id

        # from the subject id we get the date of birth
        birth_date_str = self.subject_id[:6]
        self.birth_date = datetime.strptime(birth_date_str, '%y%m%d')

        # from the session id we get the date of recording
        recording_date_str = self.ext_session_id[:6]
        self.recording_date = datetime.strptime(recording_date_str, '%y%m%d')
        # self.recording_date_main_format = self.recording_date.strftime("%d/%m/%Y")
        # print(f"self.subject_id {self.subject_id} {self.recording_date_main_format}")
        # print(f"self.main_df.iloc[:, MAIN_RECORDING_DATE_COL] {self.main_df.iloc[10, MAIN_RECORDING_DATE_COL]}")

        self.session_description = f"p{self.age}_{self.ext_session_id}"

        self.main_session_df = self.main_df.loc[(self.main_df.iloc[:, MAIN_SUBJECT_ID_COL] == self.subject_id) &
                                                (self.main_df.iloc[:, MAIN_RECORDING_DATE_COL] == self.recording_date)]
        if len(self.main_session_df) == 0:
            print(f"0 main_session_df: self.subject_id {self.subject_id}, self.recording_date {self.recording_date}")
            # print(f"len(session_df) {len(self.main_session_df)}")
        elif len(self.main_session_df) > 1:
            print(f"len(main_session_df) > 1  : self.subject_id {self.subject_id}, "
                  f"self.recording_date {self.recording_date}")

        # Surgery date
        self.surgery_date = self.main_session_df.iloc[0, MAIN_SURGERY_DATA_COL]

        # getting the weight of the animal, if it exists, otherwise None
        self.weight = None
        weight = self.main_session_df.iloc[0, MAIN_WEIGHT_COL]
        weight = str(weight).strip()
        if weight not in ["nan", "xxx"]:
            # - when it's chronic recording
            # TODO: see what to do with that
            if not (weight[0] == "-"):
                if weight[-1] == "g":
                    weight = weight[:-1].strip()
                self.weight = float(weight)

        # else:
        #     print(f"weight {weight}")

        # we create a directory that will contains the files
        self.path_results = os.path.join(path_results, self.session_description)
        if not os.path.exists(self.path_results):
            os.mkdir(self.path_results)

    def generate_yaml_files(self):
        self.generate_subject_yaml()
        self.generate_session_yaml()
        self.generate_abf_yaml()

    def generate_abf_yaml(self):
        """
        Generate the abf yaml file with informations about the channels
        Returns:

        """
        abf_dict = dict()

        """
        run_channel
        lfp_channel
        behaviors_channel
        """

        abf_dict["frames_channel"] = int(0)

        piezo_channel = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_PIEZO_CH_COL]).strip()
        if piezo_channel not in ["nan"]:
            abf_dict["piezo_channels"] = int(piezo_channel)

        # abf_dict["piezo_downsampling_hz"] = 50

        run_channel = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_TREADMMILL_CH_COL]).strip()
        if run_channel not in ["nan"]:
            abf_dict["run_channel"] = int(run_channel)

        lfp_channel = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_LFP_CH_COL]).strip()
        if lfp_channel not in ["nan"]:
            abf_dict["lfp_channel"] = int(lfp_channel)

        behavior_channels = []
        beh_1 = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_BEHAVIOR_1_CH_COL]).strip()
        if beh_1 not in ["nan"]:
            behavior_channels.append(int(beh_1))
        beh_2 = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_BEHAVIOR_2_CH_COL]).strip()
        if beh_2 not in ["nan"]:
            behavior_channels.append(int(beh_2))
        if len(behavior_channels) > 0:
            if len(behavior_channels) == 1:
                abf_dict["behavior_channels"] = behavior_channels[0]
            else:
                abf_dict["behavior_channels"] = behavior_channels

        with open(os.path.join(self.path_results, "nwb_abf_" + self.subject_id + ".yaml"), 'w') as outfile:
            yaml.dump(abf_dict, outfile, default_flow_style=False)

    def generate_session_yaml(self):
        """
        Generate the session yaml file
        Returns:

        """
        session_dict = dict()

        # session_description MANDATORY: (str) a description of the session where this data was generated
        session_dict["session_description"] = f"Recording of session {self.ext_session_id} from subject {self.subject_id}"

        # identifier (str) lab-specific ID for the session
        session_dict["identifier"] = self.session_description

        # session_start_time ANDATORY: Use to fill the session_start_time field,
        # you have to indicate the date and the time:
        # in this format: '%m/%d/%y %H:%M:%S' representing the start of the recording session

        session_dict["session_start_time"] = self.recording_date.strftime("%m/%d/%y %H:%M:%S")

        # device
        session_dict["device"] = "2Pdevice"

        # emission_lambda
        session_dict["emission_lambda"] = 510.0

        # excitation_lambda
        session_dict["excitation_lambda"] = 920.0

        # image_plane_location
        # TODO: See how to get it
        if self.image_plane_location != "":
            session_dict["image_plane_location"] = self.image_plane_location

        indicator_str = str(self.main_session_df.iloc[0, MAIN_VIRUS_COL])
        # indicator
        if indicator_str.strip() not in ["nan", "x"]:
            session_dict["indicator"] = indicator_str.strip()
        else:
            indicator_str = None

        # lab
        session_dict["lab"] = "Cossart Lab"

        # institution
        session_dict["lab"] = "INMED"

        # experimenter
        session_dict["experimenter"] = "Robin Dard"

        # experiment_description (str)  – general description of the experiment
        session_dict["experiment_description"] = "recording with head fixed"

        # keywords
        session_dict["keywords"] = ["pup", "calcium imaging"]

        # notes (str) Notes about the experiment
        notes_str = ""
        notes_1 = str(self.main_session_df.iloc[0, MAIN_IMAGING_FILMS_COL])
        notes_2 = str(self.main_session_df.iloc[0, MAIN_IMAGING_NOTES_COL])
        if (notes_1 != "nan") or (notes_2 != "nan"):
            notes_str = notes_str + "Notes on imaging: "
            if notes_1 != "nan":
                notes_str = notes_str + notes_1 + ". "
            if notes_2 != "nan":
                notes_str = notes_str + notes_2 + ". "
            # print(f"notes_str {notes_str}")
        if notes_str != "":
            session_dict["notes"] = ""

        # pharmacology (str): Description of drugs used, including how and when they were administered.
        # Anesthesia(s), painkiller(s), etc., plus dosage, concentration, etc.
        # session_dict["pharmacology"] = ""

        # protocol: (str) Experimental protocol, if applicable. E.g., include IACUC protocol
        # session_dict["protocol"] = ""
        # (str) – Publication information.PMID, DOI, URL, etc.
        # If multiple, concatenate together and describe which is which. such as PMID, DOI, URL, etc
        # session_dict["related_publications"]= ""

        # (str) – Narrative description about surgery/surgeries, including date(s) and who performed surgery.
        surgery_str = f"performed by Robin Dard on the {self.surgery_date.strftime('%m/%d/%y')}"
        if str(self.main_session_df.iloc[0, MAIN_SURGERY_NOTES_COL]) != "nan":
            surgery_str = surgery_str + ", notes: " + str(self.main_session_df.iloc[0, MAIN_SURGERY_NOTES_COL])
            # print(f"surgery_str {surgery_str}")
        session_dict["surgery"] = surgery_str

        # virus (str) – Information about virus(es) used in experiments,
        # including virus ID, source, date made, injection location, volume, etc.
        virus_str = ""
        if str(self.main_session_df.iloc[0, MAIN_AGE_INJECTION_COL]) not in ["x", "nan"]:
            age_injection = str(self.main_session_df.iloc[0, MAIN_AGE_INJECTION_COL])
            if age_injection.strip().lower()[0] == "e":
                if indicator_str is not None:
                    virus_str = f"Injection of {indicator_str} at {age_injection.strip()}"
                else:
                    virus_str = f"Injection at {age_injection.strip()}"
            elif age_injection.strip().lower()[0] == "p":
                if indicator_str is not None:
                    virus_str = f"Ventricular injection of {indicator_str} at {age_injection.strip()}"
                else:
                    virus_str = f"Ventricular i at {age_injection.strip()}"
        if virus_str != "":
            session_dict["virus"] = virus_str

        """
        # Description of slices, including information about preparation thickness, orientation, temperature and bath solution
        slices: None
        #  (str) – Script file used to create this NWB file.
        source_script: None
        # (str) – Name of the source_script file
        source_script_file_name: None
        #  (str) – Notes about data collection and analysis.
        data_collection: NOne
        # virus (str) – Information about virus(es) used in experiments,
        # including virus ID, source, date made, injection location, volume, etc.
        virus: Ventricular injection of GCaMP6s at p0
        # (str) – Notes about stimuli, such as how and where presented.
        stimulus_notes: None
        # an extension that contains lab-specific meta-data, should be a list of string
        lab_meta_data: None
        """

        with open(os.path.join(self.path_results, "nwb_session_data_" + self.session_description + ".yaml"), 'w') \
                as outfile:
            yaml.dump(session_dict, outfile, default_flow_style=False)

    def generate_subject_yaml(self):
        """
        Generate the subject yaml file
        Returns:

        """
        subject_dict = dict()

        # sex: is unknown

        # date of birth
        subject_dict["date_of_birth"] = self.birth_date.strftime("%m/%d/%Y")
        # age
        subject_dict["age"] = self.age
        # subject_id
        subject_dict["subject_id"] = self.subject_id
        # genotype
        subject_dict["genotype"] = self.main_session_df.iloc[0, MAIN_LINE_COL]
        # species
        subject_dict["species"] = "SWISS wild type"

        # Description: empty for now

        # weight
        if self.weight is not None:
            subject_dict["weight"] = self.weight

        with open(os.path.join(self.path_results, "nwb_subject_data_" + self.subject_id + ".yaml"), 'w') as outfile:
            yaml.dump(subject_dict, outfile, default_flow_style=False)

def main():
    # loading the root_path
    root_path = None
    with open("param_hne.txt", "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split('=')
            root_path = line_list[1]
    if root_path is None:
        raise Exception("Root path is None")

    path_data = root_path + "data/"
    path_results = root_path + "results_nwb_yaml_generator/"

    main_excel_file = os.path.join(path_data, "excel_files_for_nwb", "Pups_13_11_18_version_9th_sept.xlsx")
    external_info_excel_file = os.path.join(path_data, "excel_files_for_nwb", "pups_external_info.xlsx")

    main_df = pd.read_excel(main_excel_file, sheet_name=f"Experiments")
    ext_df = pd.read_excel(external_info_excel_file, sheet_name=f"SWISS")

    # print(f"main_df {main_df}")
    # print(f"ext_df {ext_df}")

    # ------------------------------------------------
    # First we clean the external info data frame
    # ------------------------------------------------
    # removing the first line
    ext_df = ext_df.iloc[1:, ]
    # then filling the NAN created by the merged cells
    for col_index in [EXT_AGE_COL, EXT_SUBJECT_ID_COL, EXT_IMAGING_DATE_COL]:
        ext_df.iloc[:, col_index] = pd.Series(ext_df.iloc[:, col_index]).fillna(method='ffill')

    # print(ext_df.iloc[:, EXT_SUBJECT_ID_COL])
    # print(ext_df.iloc[2, EXT_AGE_COL])

    # We want to identify all unique animals ID from ext_df
    animal_ids = list(set(ext_df.iloc[:, EXT_SUBJECT_ID_COL]))

    # ------------------------------------------------
    # First we clean the main data frame
    # ------------------------------------------------
    # removing the first line
    main_df = main_df.iloc[1:, ]
    # then filling the NAN created by the merged cells
    for col_index in [MAIN_SUBJECT_ID_COL, MAIN_SURGERY_DATA_COL, MAIN_AGE_INJECTION_COL]:
        main_df.iloc[:, col_index] = pd.Series(main_df.iloc[:, col_index]).fillna(method='ffill')

    print(f"{len(animal_ids)} animals{animal_ids}")

    n_sessions = 0
    for animal_id in animal_ids:
        subject_ext_df = ext_df.loc[ext_df.iloc[:, EXT_SUBJECT_ID_COL] == animal_id]
        for index in range(len(subject_ext_df)):
            n_sessions += 1

            session_generator = SessionNwbYamlGenerator(main_df=main_df, subject_ext_df=subject_ext_df,
                                                        index_session_ext_df=index,
                                                        subject_id=animal_id, path_results=path_results)
            session_generator.generate_yaml_files()

    print(f"n_sessions: {n_sessions}")

if __name__ == "__main__":
    main()