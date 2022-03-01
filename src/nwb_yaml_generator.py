import yaml
import os
import pandas as pd
from datetime import datetime
import numpy as np

# ----------------------
#      columns indices
# ----------------------
EXT_AGE_COL = 0
EXT_AGE_UNIT_COL = 1
EXT_SUBJECT_ID_COL = 2
EXT_EXPERIMENTER_COL = 3
EXT_IMAGING_DATE_COL = 4
EXT_SESSION_COL = 5
EXT_SESSION_ID_COL = 7
EXT_PLANE_LOC_COL = 8
EXT_NWB_NOTES_COL = 9
EXT_PIEZO_CH_COL = 12
EXT_TREADMMILL_CH_COL = 13
EXT_TREAD_DIRECTION_CH_COL = 14
EXT_BELT_LENGTH = 15
EXT_BELT_TYPE = 16
EXT_BEHAVIOR_1_CH_COL = 17
EXT_BEHAVIOR_2_CH_COL = 18
EXT_LFP_CH_COL = 19
EXT_NUCHAL_EMG = 20


MAIN_SURGERY_DATA_COL = 1
MAIN_MOUSE_DOB = 2
MAIN_SUBJECT_ID_COL = 3
MAIN_SUBJECT_SEX = 4
MAIN_WEIGHT_COL = 6
MAIN_LINE_COL = 7
MAIN_LINE_QUALITY = 8
MAIN_TAMOXIFEN_GAVAGE = 9
MAIN_AGE_INJECTION_COL = 10
MAIN_VIRUS_COL = 11
MAIN_VIRUS_INJECTION_SITE = 12
MAIN_VIRUS_EXPRESSION = 13
MAIN_RECORDING_DATE_COL = 14


class SessionNwbYamlGenerator:
    def __init__(self, main_df, subject_ext_df, index_session_ext_df, subject_id, path_results):
        self.main_df = main_df
        self.subject_ext_df = subject_ext_df
        self.index_session_ext_df = index_session_ext_df
        self.session = subject_ext_df.iloc[index_session_ext_df, EXT_SESSION_COL]
        self.ext_session_id = str(subject_ext_df.iloc[index_session_ext_df, EXT_SESSION_ID_COL])
        self.imaging_date = str(subject_ext_df.iloc[index_session_ext_df, EXT_IMAGING_DATE_COL])
        self.imaging_date = datetime.strptime(self.imaging_date, '%y_%m_%d')
        self.experimenter = subject_ext_df.iloc[index_session_ext_df, EXT_EXPERIMENTER_COL]
        if self.experimenter == 'nan':
            self.experimenter = 'RD'
        # Oriens or pyramidale
        self.image_plane_location = str(subject_ext_df.iloc[index_session_ext_df, EXT_PLANE_LOC_COL])
        if self.image_plane_location == "nan":
            self.image_plane_location = ""
        self.age = int(subject_ext_df.iloc[index_session_ext_df, EXT_AGE_COL])
        self.age_unit = str(subject_ext_df.iloc[index_session_ext_df, EXT_AGE_UNIT_COL]).capitalize()

        self.subject_id = subject_id

        # from the subject id we get the date of birth
        try:
            birth_date_str = self.subject_id[:6]
            self.birth_date = datetime.strptime(birth_date_str, '%y%m%d')
            get_dob_from_main = False
        except ValueError:
            get_dob_from_main = True

        # from the session id we get the date of recording
        # recording_date_str = self.ext_session_id[:6]
        # self.recording_date = datetime.strptime(recording_date_str, '%y%m%d')
        self.recording_date = self.imaging_date
        # print(f"recording_date_str: {recording_date_str}")
        # self.recording_date_main_format = self.recording_date.strftime("%d/%m/%Y")
        # print(f"self.subject_id {self.subject_id} {self.recording_date_main_format}")
        # print(f"self.main_df.iloc[:, MAIN_RECORDING_DATE_COL] {self.main_df.iloc[10, MAIN_RECORDING_DATE_COL]}")

        self.session_description = f"P{self.age}{self.age_unit}_{self.subject_id}_{self.ext_session_id}"

        self.main_session_df = self.main_df.loc[(self.main_df.iloc[:, MAIN_SUBJECT_ID_COL] == self.subject_id) &
                                                (self.main_df.iloc[:, MAIN_RECORDING_DATE_COL] == self.recording_date)]

        if get_dob_from_main:
            birth_date_str = str(self.main_session_df.iloc[0, MAIN_MOUSE_DOB])
            self.birth_date = datetime.strptime(birth_date_str, '%d/%m/%Y')

        if len(self.main_session_df) == 0:
            print(f"0 main_session_df: self.subject_id {self.subject_id}, self.recording_date {self.recording_date}")
            # print(f"len(session_df) {len(self.main_session_df)}")
        elif len(self.main_session_df) > 1:
            print(f"len(main_session_df) > 1  : self.subject_id {self.subject_id}, "
                  f"self.recording_date {self.recording_date}")

        # Surgery date
        self.surgery_date = self.main_session_df.iloc[0, MAIN_SURGERY_DATA_COL]
        # print(f"surgery date: {self.surgery_date}")
        # getting the weight of the animal, if it exists, otherwise None
        self.weight = None
        weight = self.main_session_df.iloc[0, MAIN_WEIGHT_COL]
        weight = str(weight).strip()
        if weight not in ["nan", "xxx", 'na']:
            # - when it's chronic recording
            # TODO: see what to do with that
            if not (weight[0] == "-"):
                if weight[-1] == "g":
                    weight = weight[:-1].strip()
                self.weight = float(weight)

        # else:
        #     print(f"weight {weight}")
        # we create a directory that will contains the files
        # session_dir_name = f"p{self.age}_{self.recording_date.strftime('%y_%m_%d')}_{self.ext_session_id[-4:]}"
        # self.path_results = os.path.join(path_results, f"p{self.age}", session_dir_name)
        path_data = "D:/Robin/data_hne/yaml_creation"
        path_results = os.path.join(path_data, "yaml_files")
        folder_name = self.subject_id + '_' + self.ext_session_id
        saving_directory = os.path.join(path_results, folder_name)
        os.mkdir(saving_directory)
        self.path_results = saving_directory
        if os.path.exists(self.path_results):
            self.generate_yaml_files()

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
            try:
                abf_dict["piezo_channels"] = int(piezo_channel)
            except ValueError:
                # means that the value is a float
                abf_dict["piezo_channels"] = int(float(piezo_channel))

        # abf_dict["piezo_downsampling_hz"] = 50

        run_channel = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_TREADMMILL_CH_COL]).strip()
        if run_channel not in ["nan"]:
            try:
                abf_dict["run_channel"] = int(run_channel)
            except ValueError:
                # means that the value is a float
                abf_dict["run_channel"] = int(float(run_channel))

        direction_channel = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_TREAD_DIRECTION_CH_COL]).strip()
        if direction_channel not in ["nan"]:
            try:
                abf_dict["direction_channel"] = int(direction_channel)
            except ValueError:
                # means that the value is a float
                abf_dict["direction_channel"] = int(float(direction_channel))

        belt_length = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_BELT_LENGTH]).strip()
        if belt_length not in ["nan"]:
            abf_dict["belt_length"] = int(float(belt_length))

        lfp_channel = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_LFP_CH_COL]).strip()
        if lfp_channel not in ["nan"]:
            abf_dict["lfp_channel"] = int(float(lfp_channel))

        behavior_channels = []
        beh_1 = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_BEHAVIOR_1_CH_COL]).strip()
        if beh_1 not in ["nan"]:
            behavior_channels.append(int(float(beh_1)))
        beh_2 = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_BEHAVIOR_2_CH_COL]).strip()
        if beh_2 not in ["nan"]:
            behavior_channels.append(int(float(beh_2)))
        if len(behavior_channels) > 0:
            if len(behavior_channels) == 1:
                abf_dict["behavior_channels"] = behavior_channels[0]
                if beh_1 not in ["nan"]:
                    abf_dict["behavior_adc_names"] = ["22983298"]
                else:
                    abf_dict["behavior_adc_names"] = ["23109588"]
            else:
                abf_dict["behavior_channels"] = behavior_channels
                abf_dict["behavior_adc_names"] = ["23109588", "22983298"]

        with open(os.path.join(self.path_results, "nwb_abf_" + self.subject_id + ".yaml"), 'w') as outfile:
            yaml.dump(abf_dict, outfile, default_flow_style=False, explicit_start=True)

    def generate_session_yaml(self):
        """
        Generate the session yaml file
        Returns:

        """
        session_dict = dict()

        # session_description MANDATORY: (str) a description of the session where this data was generated
        session_dict["session_description"] = f"Session: {self.ext_session_id}, from subject: {self.subject_id}"
        belt_type = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_BELT_TYPE]).strip()
        if belt_type not in ["nan"]:
            session_dict["session_description"] = f"Session: {self.ext_session_id}, from subject: {self.subject_id}, " \
                                                  f"on {belt_type.lower()} belt"

        # identifier (str) lab-specific ID for the session
        session_dict["identifier"] = self.session_description

        # identifier (str) lab-specific ID for the session
        session_dict["session_id"] = self.ext_session_id

        # session_start_time MANDATORY: Use to fill the session_start_time field,
        # you have to indicate the date and the time:
        # in this format: '%m/%d/%y %H:%M:%S' representing the start of the recording session

        session_dict["session_start_time"] = self.recording_date.strftime("%m/%d/%y %H:%M:%S")

        # device
        session_dict["device"] = "2P microscope"

        # emission_lambda
        session_dict["emission_lambda"] = 510.0

        # excitation_lambda
        session_dict["excitation_lambda"] = 920.0

        # image_plane_location
        if self.image_plane_location != "":
            session_dict["image_plane_location"] = "stratum " + self.image_plane_location

        indicator_str = str(self.main_session_df.iloc[0, MAIN_VIRUS_COL])
        # indicator
        if indicator_str.strip() not in ["nan", "x"]:
            session_dict["indicator"] = indicator_str.strip()
        else:
            # default
            indicator_str = "GCaMP6s"
            session_dict["indicator"] = indicator_str

        # viruses
        virus_remark = str(self.main_session_df.iloc[0, MAIN_VIRUS_EXPRESSION])
        if virus_remark.lower() in ['x', 'nan']:
            virus_comment = "Good"
        elif virus_remark.lower() in ['leak', 'leaked', 'leaky']:
            virus_comment = "Leaked"
        else:
            virus_comment = virus_remark.capitalize()

        virus_injection_site = str(self.main_session_df.iloc[0, MAIN_VIRUS_INJECTION_SITE])
        if virus_injection_site is None:
            vir_inj_site = "left lateral ventricle"
        else:
            vir_inj_site = virus_injection_site.strip()

        virus_id = "none"
        viral_volume = "NA"
        if indicator_str.strip() not in ["nan", "x"]:
            used_indicator = indicator_str.strip()
            # classic GCaMP6s injection in ventricle
            if used_indicator == "GCaMP6s":
                virus_id = "AAV1.Syn.GCaMP6s.WPRE.SV40"
                viral_volume = "2 uL"
            # only INs in Gadcre
            if used_indicator == "GCaMP6f flex":
                virus_id = "AAV1.Syn.Flex.GCaMP6f.WPRE.SV40"
                viral_volume = "2 uL"
            # manip redINs in GadCre animals + ctrl in sstcre no dreadd with cno
            if used_indicator in ["flex-Tomato + GCaMP6s", "flex-TdTomato + GCaMP6s"]:
                virus_id = "AAV1.Syn.GCaMP6s.WPRE.SV40 and AAV9.CAG.Flex.tdTomato"
                if vir_inj_site == "left lateral ventricle" or vir_inj_site == "ventricle":
                    viral_volume = "1.3 uL and 0.7 uL respectively"
                else:
                    viral_volume = "2 uL and 10 nL respectively"
            # manip ctrl in sstcre no dreadd (mCherry) with cno
            if used_indicator in ["flex-mCherry + GCaMP6s"]:
                virus_id = "AAV1.Syn.GCaMP6s.WPRE.SV40 and AAV9.CAG.Flex.mCherry"
                if vir_inj_site == "left lateral ventricle" or vir_inj_site == "ventricle":
                    viral_volume = "1.3 uL and 0.7 uL respectively"
                else:
                    viral_volume = "2 uL and 10 nL respectively"
            # manip flex axon GCaMP in gadCre animals robin
            if used_indicator == "flex-axon-GCaMP6s":
                virus_id = "AAV9-hSynapsin.Flex.axon-GCaMP6s"
                viral_volume = "2 uL"
            # manip sstcre dreadd
            if used_indicator == "flex-hM4DGi + GCaMP6s":
                virus_id = "AAV9-hSynapsin.Flex-hM4DGi and AAV1.Syn.GCaMP6s.WPRE.SV40"
                viral_volume = "10 nL and 2 uL respectively"
            # manip gad cre axon robin, axon vmt erwan, axon entorhinal ctx erwan
            if used_indicator == "flex-axon-GCaMP6s + TdTomato":
                virus_id = "AAV9-hSynapsin.Flex.axon-GCaMP6s and AAV9.CAG.tdTomato"
                if vir_inj_site == "left lateral ventricle" or vir_inj_site == "ventricle":
                    viral_volume = "1.3 uL and 0.7 uL respectively"
                else:
                    viral_volume = "10 nL and 2 uL respectively"
            # manip axon vmt erwan, axon entorhinal ctx erwan
            if used_indicator == "flex-axon-GCaMP6s + flex-TdTomato":
                virus_id = "AAV9-hSynapsin.Flex.axon-GCaMP6s and AAV9.CAG.Flex.tdTomato"
                viral_volume = "10 nL and 2 uL respectively"

        # lab
        session_dict["lab"] = "Cossart Lab"

        # institution
        session_dict["institution"] = "INMED - INSERMU1249"

        # experimenter
        session_dict["experimenter"] = self.experimenter

        # experiment_description (str)  – general description of the experiment
        session_dict["experiment_description"] = "In-vivo 2P calcium imaging on head fixed mouse pup"

        # keywords
        session_dict["keywords"] = ["pup", "calcium imaging"]

        # notes (str) Notes about the experiment
        notes = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_NWB_NOTES_COL])
        if notes == 'nan':
            notes_str = "No given note for this NWB"
        else:
            notes_str = notes

        session_dict["notes"] = notes_str

        # pharmacology (str): Description of drugs used, including how and when they were administered.
        # Anesthesia(s), painkiller(s), etc., plus dosage, concentration, etc.
        session_dict["pharmacology"] = 'Anesthesia: Isoflurane 1-3% in a 90% O2 / 10% air mix,  ' \
                                       'Painkillers: Buprenorphine [0.05-0.1] mg.kg-1'

        # protocol: (str) Experimental protocol, if applicable. E.g., include IACUC protocol
        # session_dict["protocol"] = ""
        # (str) – Publication information.PMID, DOI, URL, etc.
        # If multiple, concatenate together and describe which is which. such as PMID, DOI, URL, etc
        # session_dict["related_publications"]= ""

        # (str) – Narrative description about surgery/surgeries, including date(s) and who performed surgery.
        surgery_str = f"Performed by {self.experimenter} on {self.surgery_date.strftime('%m/%d/%y')}"
        # if str(self.main_session_df.iloc[0, MAIN_SURGERY_NOTES_COL]) != "nan":
        #     surgery_str = surgery_str + ", notes: " + str(self.main_session_df.iloc[0, MAIN_SURGERY_NOTES_COL])

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
                    virus_str = f"Age at injection: {age_injection.strip()}, Injection-site: {vir_inj_site}, " \
                                f"VirusID: {virus_id}, Volume: {viral_volume}, " \
                                f"Expression/Labelling: {virus_comment}, Source: Addgene"
                else:
                    virus_str = f"Age at injection: {age_injection.strip()}"
        if virus_str != "":
            session_dict["virus"] = virus_str

        # Supplementary behavioral monitoring
        monitoring = ""
        piezo_channel = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_PIEZO_CH_COL]).strip()
        if piezo_channel not in ["nan"]:
            if len(monitoring) >= 1:
                monitoring = monitoring + ", Piezzo recording"
            else:
                monitoring = monitoring + "Piezzo recording"
        run_channel = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_TREADMMILL_CH_COL]).strip()
        if run_channel not in ["nan"]:
            if len(monitoring) >= 1:
                monitoring = monitoring + ", Treadmill analysis"
            else:
                monitoring = monitoring + "Treadmill analysis"
        lfp_channel = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_LFP_CH_COL]).strip()
        if lfp_channel not in ["nan"]:
            if len(monitoring) >= 1:
                monitoring = monitoring + ", LFPs recording"
            else:
                monitoring = monitoring + "LFPs recording"
        beh_1 = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_BEHAVIOR_1_CH_COL]).strip()
        beh_2 = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_BEHAVIOR_2_CH_COL]).strip()
        if beh_1 not in ["nan"] or beh_2 not in ["nan"]:
            if len(monitoring) >= 1:
                monitoring = monitoring + ", Video recording"
            else:
                monitoring = monitoring + "Video recording"
        is_emg = str(self.subject_ext_df.iloc[self.index_session_ext_df, EXT_NUCHAL_EMG]).strip()
        if is_emg not in ["nan", "no", "na"]:
            if len(monitoring) >= 1:
                monitoring = monitoring + ", Nuchal EMG recording"
            else:
                monitoring = monitoring + "Nuchal EMG recording"
        if monitoring != "":
            session_dict["supplementary_behavioral_monitoring"] = monitoring
        else:
            session_dict["supplementary_behavioral_monitoring"] = "None"

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
            yaml.dump(session_dict, outfile, default_flow_style=False, explicit_start=True)

    def generate_subject_yaml(self):
        """
        Generate the subject yaml file
        Returns:

        """
        subject_dict = dict()

        # sex: is unknown
        sex = str(self.main_session_df.iloc[0, MAIN_SUBJECT_SEX])
        if sex not in ["NA", "nan", "x"]:
            subject_dict["sex"] = sex
        else:
            subject_dict["sex"] = "U"

        # date of birth
        subject_dict["date_of_birth"] = self.birth_date.strftime("%m/%d/%Y")
        # age
        subject_dict["age"] = f"P{self.age}{self.age_unit}"
        # subject_id
        subject_dict["subject_id"] = self.subject_id
        # genotype
        line = str(self.main_session_df.iloc[0, MAIN_LINE_COL])
        if line not in ["nan"]:
            if line.lower() == "swiss":
                line = 'Wild-type'
                subject_dict["genotype"] = "Wild-type"
            else:
                subject_dict["genotype"] = line
        line_remark = str(self.main_session_df.iloc[0, MAIN_LINE_QUALITY])
        if line_remark.lower() not in ['nan', 'x']:
            remark = line_remark.lower()
            subject_dict["genotype"] = line + f" {remark}"
        # Tamoxifen induction:
        tamox_age = str(self.main_session_df.iloc[0, MAIN_TAMOXIFEN_GAVAGE])
        if tamox_age not in ["x", "NA", "nan"]:
            subject_dict["genotype"] = line + f" + tamox. {tamox_age}"

        # species
        subject_dict["species"] = "Mus musculus"

        # strain
        subject_dict["strain"] = "SWISS"  # put "C57BL/6J" "CD1", ..... here

        # Description: empty for now

        # weight
        if self.weight is not None:
            subject_dict["weight"] = self.weight

        with open(os.path.join(self.path_results, "nwb_subject_data_" + self.subject_id + ".yaml"), 'w') as outfile:
            yaml.dump(subject_dict, outfile, default_flow_style=False, explicit_start=True)


def main():
    # loading the root_path
    # root_path = None
    # with open("param_hne.txt", "r", encoding='UTF-8') as file:
    #     for nb_line, line in enumerate(file):
    #         line_list = line.split('=')
    #         root_path = line_list[1]
    # if root_path is None:
    #     raise Exception("Root path is None")

    path_data = "D:/Robin/data_hne/yaml_creation"
    path_results = os.path.join(path_data, "yaml_files")

    main_excel_file = os.path.join(path_data, "pups_experiments_for_yaml_robin.xlsx")
    external_info_excel_file = os.path.join(path_data, "pups_info_for_yaml.xlsx")

    main_df = pd.read_excel(main_excel_file, sheet_name=f"Imaging Experiments")
    swiss_df = pd.read_excel(external_info_excel_file, sheet_name=f"SWISS")
    # gadcre_redins_gcamp_df = pd.read_excel(external_info_excel_file, sheet_name=f"GadCre-RedINs-GCAmP")
    # gadcre_gcamp_df = pd.read_excel(external_info_excel_file, sheet_name=f"GadCre-GCamP")
    # gadcre_axon_gcamp_df = pd.read_excel(external_info_excel_file, sheet_name="GadCre_flexAxonGCaMP")
    # sstcre_dreadd_df = pd.read_excel(external_info_excel_file, sheet_name="SstCre_hM4DGi_GCaMP")
    # sstcre_nodreadd_df = pd.read_excel(external_info_excel_file, sheet_name="SSTCre_nodreadd_cno")
    # sstcre_dreadd_salin_df = pd.read_excel(external_info_excel_file, sheet_name="SSTCre_dreadd_salin")
    # emx1cre_dreadd_df = pd.read_excel(external_info_excel_file, sheet_name="Emx1Cre")
    # vglut2cre_dreadd_df = pd.read_excel(external_info_excel_file, sheet_name="Vglut2Cre")
    # swiss_vpa_df = pd.read_excel(external_info_excel_file, sheet_name=f"SWISS_VPA")

    # removing the first lines
    # swiss_df = swiss_df.iloc[1:, ]
    # gadcre_redins_gcamp_df = gadcre_redins_gcamp_df.iloc[1:, ]
    # gadcre_gcamp_df = gadcre_gcamp_df.iloc[1:, ]
    # gadcre_axon_gcamp_df = gadcre_axon_gcamp_df.iloc[1:, ]

    # concatenating the 2 dataframe

    # SWISS
    frames = [swiss_df]
    ext_df = pd.concat(frames)

    # # GadCre red-INs
    # frames = [gadcre_redins_gcamp_df]
    # ext_df = pd.concat(frames)

    # GadCre-GCaMP
    # frames = [gadcre_gcamp_df]
    # ext_df = pd.concat(frames)

    # GadCre flex axon GCaMP
    # frames = [gadcre_axon_gcamp_df]
    # ext_df = pd.concat(frames)

    # SstCre_hM4DGi_GCaMP
    # frames = [sstcre_dreadd_df]
    # ext_df = pd.concat(frames)

    # SstCre_dreadd_salin
    # frames = [sstcre_dreadd_salin_df]
    # ext_df = pd.concat(frames)

    # SstCre_no_dreadd_cno_GCaMP
    # frames = [sstcre_nodreadd_df]
    # ext_df = pd.concat(frames)

    # Emx1Cre_hM4DGi_GCaMP Erwan
    # frames = [emx1cre_dreadd_df]
    # ext_df = pd.concat(frames)

    # Vglut2Cre_hM4DGi_GCaMP Erwan
    # frames = [vglut2cre_dreadd_df]
    # ext_df = pd.concat(frames)

    # SWISS VPA
    # frames = [swiss_vpa_df]
    # ext_df = pd.concat(frames)

    # print(f"main_df: {main_df}")
    # print(f"ext_df:")
    # print(f"{ext_df}")

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
    # print(f"ext_df {ext_df}")

    # We want to identify all unique animals ID from ext_df
    animal_ids = list(set(ext_df.iloc[:, EXT_SUBJECT_ID_COL]))

    # ------------------------------------------------
    # First we clean the main data frame
    # ------------------------------------------------
    # removing the first line
    # main_df = main_df.iloc[1:, ]
    # then filling the NAN created by the merged cells
    for col_index in [MAIN_SUBJECT_ID_COL, MAIN_SURGERY_DATA_COL, MAIN_AGE_INJECTION_COL]:
        main_df.iloc[:, col_index] = pd.Series(main_df.iloc[:, col_index]).fillna(method='ffill')

    print(f"{len(animal_ids)} animal(s): {animal_ids}")

    n_sessions = 0
    for animal_id in animal_ids:
        print(f"animal ID: {animal_id}")
        subject_ext_df = ext_df.loc[ext_df.iloc[:, EXT_SUBJECT_ID_COL] == animal_id]
        for index in range(len(subject_ext_df)):
            n_sessions += 1

            session_generator = SessionNwbYamlGenerator(main_df=main_df, subject_ext_df=subject_ext_df,
                                                        index_session_ext_df=index,
                                                        subject_id=animal_id, path_results=path_results)

    print(f"n_sessions: {n_sessions}")


if __name__ == "__main__":
    main()