import numpy as np
import pandas as pd
from datetime import datetime
import unidecode
# reg exp
import re
from sortedcontainers import SortedDict
import os


def add_key_to_map_dict(map_dict, key_to_add):
    """
    Take a dict that match a string to a code and key_to_add to the dict using the next int among the max values one
    :param map_dict:
    :param  key_to_add:
    :return:
    """
    if key_to_add in map_dict:
        return
    values = list(map_dict.values())
    new_value = np.max(values) + 1
    map_dict[key_to_add] = new_value


class Cleaner:
    def __init__(self, df_data):
        self.df_data = df_data
        self.df_clean = df_data
        self.n_lines = self.df_clean.shape[0]
        self.dates_columns = ["date_greffe", "date_naissance"]

    def create_patient_id_column(self):
        patient_ids_list = []
        patient_str_dict = {}
        patient_id = 0
        for index, value in enumerate(self.df_clean.loc[:, "nom"]):
            last_name = value
            first_name = self.df_clean.loc[index, "prenom"]
            birth_date = self.df_clean.loc[index, "date_naissance"]
            patient_str = first_name + last_name + birth_date
            if patient_str not in patient_str_dict:
                patient_str_dict[patient_str] = patient_id
                patient_ids_list.append(patient_id)
                patient_id += 1
            else:
                patient_ids_list.append(patient_str_dict[patient_str])

        self.df_clean.insert(loc=0, column="patient_id",
                             value=patient_ids_list,
                             allow_duplicates=False)

    def do_anonymization(self):
        # will earse first_name, name and birth_date columns
        # we will keep only the graft year
        self.df_clean.drop(columns=['prenom', 'nom', "date_naissance"], inplace=True)
        for index, value in enumerate(self.df_clean.loc[:, "date_greffe"]):
            date_time = datetime.strptime(value, '%d/%m/%Y')
            self.df_clean.at[index, "date_greffe"] = date_time.year

    def clean(self):
        # first we change the name of the columns
        # removing the space and the accentuation
        # removing_accentuation
        columns_name = list(self.df_clean.columns)
        for i, name in enumerate(columns_name):
            # removing space at the beginning and end
            new_name = name.strip()
            # replacing space by _
            new_name = new_name.replace(" ", "_")
            # removing accentuation
            new_name = unidecode.unidecode(new_name)
            columns_name[i] = new_name
        self.df_clean.columns = columns_name

        # setting indices
        self.df_clean = self.df_clean.set_index(np.arange(self.n_lines))

        # dealing with data format
        for date_name in self.dates_columns:
            self.df_clean[date_name] = pd.to_datetime(self.df_clean[date_name])
            for index, value in enumerate(self.df_clean.loc[:, date_name]):
                date_time = value
                if date_time.year > datetime.now().year:
                    # then we remove a century
                    self.df_clean.at[index, date_name] = datetime(date_time.year - 100, date_time.month, date_time.day)
            # print(f"self.df_clean[date_name].dt.year {self.df_clean[date_name].dt.year}")

        # add age column
        index_column = self.df_clean.columns.get_loc("date_naissance")
        self.df_clean.insert(loc=index_column + 1, column="age",
                             value=(self.df_clean.loc[:, "date_greffe"] - self.df_clean.loc[:,
                                                                          "date_naissance"]).astype('<m8[Y]'),
                             allow_duplicates=False)

        for date_name in self.dates_columns:
            self.df_clean[date_name] = self.df_clean[date_name].dt.strftime('%d/%m/%Y')
        self.create_patient_id_column()


class CleanerCoder(Cleaner):
    def __init__(self, df_data, keep_original):
        """

        :param df_data:
        :param keep_original: if True, keep the original column adding "_original" to the name, still adding the encoded
        column
        """
        super().__init__(df_data=df_data)

        self.keep_original = keep_original
        self.mapping_dict = dict()

        # key is a column name, value is a list of the 2 dict to pass as argument of the clean fct
        self.column_to_clean_with_reg_exp = dict()
        # value is a list with the mapping dict and special_cases if exists
        self.column_to_clean_dict = dict()

        # code for each column values
        self.sexe_mapping = {"NA": -1, "F": 0, "M": 1}
        self.mapping_dict["sexe"] = self.sexe_mapping
        self.column_to_clean_dict["sexe"] = [self.sexe_mapping]

        self.diabete_mapping = {"NA": -1, "non": 0, "oui": 1}
        self.mapping_dict["ATCD_Diabete"] = self.diabete_mapping
        self.column_to_clean_dict["ATCD_Diabete"] = [self.diabete_mapping, dict()]

        self.greffe_cornee_mapping = {"NA": -1, "non": 0, "oui": 1}
        self.mapping_dict["ATCD_greffe_cornee"] = self.greffe_cornee_mapping
        self.column_to_clean_dict["ATCD_greffe_cornee"] = [self.greffe_cornee_mapping, dict()]

        self.glaucome_mapping = {"NA": -1, "non": 0, "oui": 1}
        self.mapping_dict["ATCD_Glaucome"] = self.glaucome_mapping
        self.column_to_clean_dict["ATCD_Glaucome"] = [self.glaucome_mapping, dict()]

        statut_cristallin_to_map = ["phaque", "pke odg"]
        self.statut_cristallin_mapping = {"NA": -1}
        for code, statut_cristallin in enumerate(statut_cristallin_to_map):
            self.statut_cristallin_mapping[statut_cristallin] = code
        self.mapping_dict["statut_cristallin"] = self.statut_cristallin_mapping
        self.statut_cristallin_patterns = {}
        self.column_to_clean_with_reg_exp["statut_cristallin"] = [self.statut_cristallin_mapping,
                                                                  self.statut_cristallin_patterns]

        loc_ulcere_to_map = ["central", "superieur", "inferieur"]
        self.loc_ulcere_mapping = {"NA": -1}
        for code, loc_ulcere in enumerate(loc_ulcere_to_map):
            self.loc_ulcere_mapping[loc_ulcere] = code
        self.mapping_dict["localisation_ulcere"] = self.loc_ulcere_mapping
        self.loc_ulcere_patterns = {"cenral": "central",
                                    "inf": "inferieur",
                                    "sup": "superieur"}
        self.column_to_clean_with_reg_exp["localisation_ulcere"] = [self.loc_ulcere_mapping,
                                                                    self.loc_ulcere_patterns]
        type_gma_to_map = []
        self.type_gma_mapping = {"NA": -1}
        for code, type_gma in enumerate(type_gma_to_map):
            self.type_gma_mapping[type_gma] = code
        self.mapping_dict["Type_GMA"] = self.type_gma_mapping
        self.type_gma_patterns = {}
        self.column_to_clean_with_reg_exp["Type_GMA"] = [self.type_gma_mapping,
                                                         self.type_gma_patterns]

        antifongiques_to_map = []
        self.antifongiques_mapping = {"NA": -1}
        for code, antifongiques in enumerate(antifongiques_to_map):
            self.antifongiques_mapping[antifongiques] = code
        self.mapping_dict["antifongiques"] = self.antifongiques_mapping
        self.antifongiques_patterns = {}
        self.column_to_clean_with_reg_exp["antifongiques"] = [self.antifongiques_mapping,
                                                              self.antifongiques_patterns]

        ttt_compl_to_map = ["ciclo 2%", "cure entropion", "cure ectropion"]
        self.ttt_compl_mapping = {"NA": -1}
        for code, ttt_compl in enumerate(ttt_compl_to_map):
            self.ttt_compl_mapping[ttt_compl] = code
        self.mapping_dict["ttt_complementaire"] = self.ttt_compl_mapping
        self.ttt_compl_patterns = {".*ciclo.*": "ciclo 2%",
                                   ".*entropion.*": "cure entropion",
                                   ".*ectropion.*": "cure ectropion"}
        self.column_to_clean_with_reg_exp["ttt_complementaire"] = [self.ttt_compl_mapping,
                                                                   self.ttt_compl_patterns]

        ttt_compl_2_to_map = ["tarsorraphie  definitive"]
        self.ttt_compl_2_mapping = {"NA": -1}
        for code, ttt_compl_2 in enumerate(ttt_compl_2_to_map):
            self.ttt_compl_2_mapping[ttt_compl_2] = code
        self.mapping_dict["ttt_complementaire_2"] = self.ttt_compl_2_mapping
        self.ttt_compl_2_patterns = {".*tarsorraphie.*definitive.*": "tarsorraphie  definitive"}
        self.column_to_clean_with_reg_exp["ttt_complementaire_2"] = [self.ttt_compl_2_mapping,
                                                                     self.ttt_compl_2_patterns]

        sous_cat_fact_ass_to_map = []
        self.sous_cat_fact_ass_mapping = {"NA": -1}
        for code, sous_cat_fact_ass in enumerate(sous_cat_fact_ass_to_map):
            self.sous_cat_fact_ass_mapping[sous_cat_fact_ass] = code
        self.mapping_dict["sous_categorie_facteur_associe"] = self.sous_cat_fact_ass_mapping
        self.sous_cat_fact_ass_patterns = {}
        self.column_to_clean_with_reg_exp["sous_categorie_facteur_associe"] = [self.sous_cat_fact_ass_mapping,
                                                                               self.sous_cat_fact_ass_patterns]

        facteur_associe_2_to_map = []
        self.facteur_associe_2_mapping = {"NA": -1}
        for code, facteur_associe_2 in enumerate(facteur_associe_2_to_map):
            self.facteur_associe_2_mapping[facteur_associe_2] = code
        self.mapping_dict["facteur_associe_2"] = self.sous_cat_fact_ass_mapping
        self.facteur_associe_2_patterns = {}
        self.column_to_clean_with_reg_exp["facteur_associe_2"] = [self.facteur_associe_2_mapping,
                                                                  self.facteur_associe_2_patterns]

        complication_to_map = ["abces cornee"]
        self.complication_mapping = {"NA": -1}
        for code, complication in enumerate(complication_to_map):
            self.complication_mapping[complication] = code
        self.mapping_dict["complication"] = self.complication_mapping
        self.complication_patterns = {"abes cornee": "abces cornee"}
        self.column_to_clean_with_reg_exp["complication"] = [self.complication_mapping,
                                                             self.complication_patterns]

        prelevement_to_map = []
        self.prelevement_mapping = {"NA": -1}
        for code, prelevement in enumerate(prelevement_to_map):
            self.prelevement_mapping[prelevement] = code
        self.mapping_dict["prelevement"] = self.prelevement_mapping
        self.prelevement_patterns = {}
        self.column_to_clean_with_reg_exp["prelevement"] = [self.prelevement_mapping,
                                                             self.prelevement_patterns]

        germe_to_map = []
        self.germe_mapping = {"NA": -1}
        for code, germe in enumerate(germe_to_map):
            self.germe_mapping[germe] = code
        self.mapping_dict["germe"] = self.germe_mapping
        self.germe_patterns = {}
        self.column_to_clean_with_reg_exp["germe"] = [self.germe_mapping,
                                                            self.germe_patterns]

        # ATCD_HSV_2_episodes_au_moins
        atcd_hsv_to_map = ["oui"]
        self.atcd_hsv_mapping = {"NA": -1}
        for code, atcd_hsv in enumerate(atcd_hsv_to_map):
            self.atcd_hsv_mapping[atcd_hsv] = code
        self.mapping_dict["ATCD_HSV_2_episodes_au_moins"] = self.atcd_hsv_mapping
        self.atcd_hsv_patterns = {}
        self.column_to_clean_with_reg_exp["ATCD_HSV_2_episodes_au_moins"] = [self.atcd_hsv_mapping,
                                                      self.atcd_hsv_patterns]

        immunodepression_to_map = ["oui", "non"]
        self.immunodepression_mapping = {"NA": -1}
        for code, immunodepression in enumerate(immunodepression_to_map):
            self.immunodepression_mapping[immunodepression] = code
        self.mapping_dict["ATCD_immunodepression"] = self.immunodepression_mapping
        self.immunodepression_patterns = {}
        self.column_to_clean_with_reg_exp["ATCD_immunodepression"] = [self.immunodepression_mapping,
                                                                             self.immunodepression_patterns]

        dysthyroidie_to_map = ["oui", "non"]
        self.dysthyroidie_mapping = {"NA": -1}
        for code, dysthyroidie in enumerate(dysthyroidie_to_map):
            self.dysthyroidie_mapping[dysthyroidie] = code
        self.mapping_dict["ATCD_dysthyroidie"] = self.dysthyroidie_mapping
        self.dysthyroidie_patterns = {}
        self.column_to_clean_with_reg_exp["ATCD_dysthyroidie"] = [self.dysthyroidie_mapping,
                                                                 self.dysthyroidie_patterns]
        # cause_immunodepression
        cause_immunodepression_to_map = ["non"]
        self.cause_immunodepression_mapping = {"NA": -1}
        for code, cause_immunodepression in enumerate(cause_immunodepression_to_map):
            self.cause_immunodepression_mapping[cause_immunodepression] = code
        self.mapping_dict["cause_immunodepression"] = self.cause_immunodepression_mapping
        self.cause_immunodepression_patterns = {}
        self.column_to_clean_with_reg_exp["cause_immunodepression"] = [self.cause_immunodepression_mapping,
                                                             self.cause_immunodepression_patterns]

        etiologies_to_map = ["ulcere inflammatoire", "inflammatoire",
                             "ulcere mecanique", "keratite infectieuse",
                             "insuffisance limbique",
                             "anomalie statique palpebrale", "decompensation bulleuse epitheliale",
                             "destruction aigue surface", "neurotrophique", "Keratopathie en bandelette",
                             "reconstruction", "refection BF"]
        self.etiology_mapping = {"NA": -1}
        for code, etiology in enumerate(etiologies_to_map):
            self.etiology_mapping[etiology] = code
        self.mapping_dict["etiologie"] = self.etiology_mapping
        # each key is a tuple of reg_ex, and the value is a key to etiology_mapping
        # the key should be lower case
        self.etiology_patterns = {"ulcere.* inflammatoire": "ulcere inflammatoire",
                                  "inflammaoire": "inflammatoire",
                                  "keratite.*infectieuse": "keratite infectieuse",
                                  "insuffisance limbique": "insuffisance limbique",
                                  "anomalie statique.*rale": "anomalie statique palpebrale",
                                  "decompensation bulleuse.*": "decompensation bulleuse epitheliale",
                                  "destruction aigue surface": "destruction aigue surface",
                                  ("neurotrophique", "neurotrophiqe"): "neurotrophique",
                                  ("keratopathie.*bandelette", "keratite.*bandelette"): "Keratopathie en bandelette",
                                  "recon.*ction": "reconstruction", "refection.*bf": "refection BF"}
        self.column_to_clean_with_reg_exp["etiologie"] = [self.etiology_mapping, self.etiology_patterns]

        categorie_nk_to_map = ["infectieuse", "brulure", "diabete", "iatrogenie", "atteinte chronique surface",
                               "centrale"]
        self.categorie_nk_mapping = {"NA": -1}
        for code, etiology in enumerate(categorie_nk_to_map):
            self.categorie_nk_mapping[etiology] = code
        self.mapping_dict["categorie_nk"] = self.categorie_nk_mapping
        # each key is a tuple of reg_ex, and the value is a key to etiology_mapping
        # the key should be lower case
        self.categorie_nk_patterns = {("infect.*", "infetcieuse"): "infectieuse",
                                      "brulure": "brulure", "diabete": "diabete",
                                      "iatrogenie": "iatrogenie",
                                      "atteinte chronique surface": "atteinte chronique surface",
                                      "central.*": "centrale"}
        self.column_to_clean_with_reg_exp["categorie_NK"] = [self.categorie_nk_mapping, self.categorie_nk_patterns]

        cause_nk_to_map = ["hsv", "base", "kt", "2e kt", "3e kt", "ains", "vzv", "collyres atb", "gougerot"]
        self.cause_nk_mapping = {"NA": -1}
        for code, etiology in enumerate(cause_nk_to_map):
            self.cause_nk_mapping[etiology] = code
        self.mapping_dict["cause_NK"] = self.cause_nk_mapping
        # each key is a tuple of reg_ex, and the value is a key to etiology_mapping
        # the key should be lower case
        self.cause_nk_patterns = {"hsv": "hsv",
                                  "base": "base", "kt": "kt",
                                  "2.*kt": "2e kt",
                                  "3.*kt": "3e kt", "ains": "ains", "vzv": "vzv",
                                  "col.*atb": "collyres atb",
                                  "goug.*rot": "gougerot"}
        self.column_to_clean_with_reg_exp["cause_NK"] = [self.cause_nk_mapping, self.cause_nk_patterns]

        facteur_favorisant_to_map = ["brulure", "decompensation endotheliale"]
        self.facteur_favorisant_mapping = {"NA": -1}
        for code, etiology in enumerate(facteur_favorisant_to_map):
            self.facteur_favorisant_mapping[etiology] = code
        self.mapping_dict["facteur_favorisant"] = self.facteur_favorisant_mapping
        # each key is a tuple of reg_ex, and the value is a key to etiology_mapping
        # the key should be lower case
        self.facteur_favorisant_patterns = {"brulure.*": "brulure",
                                            "deompensation endotheliale":"decompensation endotheliale"}
        self.column_to_clean_with_reg_exp["facteur_favorisant"] = [self.facteur_favorisant_mapping,
                                                                   self.facteur_favorisant_patterns]

        facteur_associe_to_map = []
        self.facteur_associe_mapping = {"NA": -1}
        for code, etiology in enumerate(facteur_associe_to_map):
            self.facteur_associe_mapping[etiology] = code
        self.mapping_dict["facteur_associe"] = self.facteur_associe_mapping
        # each key is a tuple of reg_ex, and the value is a key to etiology_mapping
        # the key should be lower case
        self.facteur_associe_patterns = {}
        self.column_to_clean_with_reg_exp["facteur_associe"] = [self.facteur_associe_mapping,
                                                                self.facteur_associe_patterns]

        etat_corneen_to_map = ["preperforatif", "perfore"]
        self.etat_corneen_mapping = {"NA": -1}
        for code, etiology in enumerate(etat_corneen_to_map):
            self.etat_corneen_mapping[etiology] = code
        self.etat_corneen_cases = {"préperfratif": "preperforatif",
                                   "preerforatif": "preperforatif",
                                   "preperfratif": "preperforatif",
                                   "preforatif": "preperforatif"}
        self.mapping_dict["etat_corneen"] = self.etat_corneen_mapping
        self.column_to_clean_dict["etat_corneen"] = [self.etat_corneen_mapping, self.etat_corneen_cases]

        mode_anesthesie_to_map = ["ag", "top diaz", "topique", "alr"]
        self.mode_anesthesie_mapping = {"NA": -1}
        for code, etiology in enumerate(mode_anesthesie_to_map):
            self.mode_anesthesie_mapping[etiology] = code
        self.mode_anesthesie_cases = {"to pdiaz": "top diaz",
                                      "top diz": "top diaz",
                                      "top daz": "top diaz",
                                      "top": "topique"}
        self.mapping_dict["mode_anesthesie"] = self.mode_anesthesie_mapping
        self.column_to_clean_dict["mode_anesthesie"] = [self.mode_anesthesie_mapping, self.mode_anesthesie_cases]

        self.side_mapping = {"NA": -1, "OG": 0, "OD": 1, "OD_OG": 2}
        # to replace by reg exp
        self.side_special_cases = {"OG PUIS OD EN 07": "OD_OG", "0D": "OD", "OG PUIS OD": "OD_OG",
                                   "OD + OG": "OD_OG", "OG + OD": "OD_OG"}
        self.mapping_dict["cote"] = self.side_mapping
        self.column_to_clean_dict["cote"] = [self.side_mapping, self.side_special_cases]

        nb_couches_to_map = ["monocouche", "double couche", "multicouches"]
        self.nb_couches_mapping = {"NA": -1}
        for code, etiology in enumerate(nb_couches_to_map):
            self.nb_couches_mapping[etiology] = code
        self.mapping_dict["nb_couches"] = self.nb_couches_mapping
        # each key is a tuple of reg_ex, and the value is a key to etiology_mapping
        # the key should be lower case
        self.nb_couches_patterns = {"mon.*": "monocouche",
                                    "double.*": "double couche",
                                    "multi.*": "multicouches"}
        self.column_to_clean_with_reg_exp["nb_couches"] = [self.nb_couches_mapping,
                                                           self.nb_couches_patterns]
        self.clean()

    def clean_column_with_reg_exp(self, column_name, map_dict, pattern_dict):
        if self.keep_original:
            index_column = self.df_clean.columns.get_loc(column_name)
            self.df_clean.insert(loc=index_column + 1, column=column_name + "_originale",
                                 value=self.df_clean.loc[:, column_name],
                                 allow_duplicates=False)
        for index, cell_text in enumerate(self.df_clean.loc[:, column_name]):
            if pd.isna(cell_text):
                self.df_clean.at[index, column_name] = map_dict["NA"]
                continue
            if isinstance(cell_text, int):
                self.df_clean.at[index, column_name] = map_dict["NA"]
                continue
            cell_text = unidecode.unidecode(cell_text)
            cell_text = cell_text.lower().strip()
            pattern_found = False
            for patterns, key_value in pattern_dict.items():
                if isinstance(patterns, str):
                    patterns = [patterns]
                for pattern in patterns:
                    # print(f"pattern {pattern}")
                    match_object = re.search(pattern, cell_text, flags=0)
                    if match_object is not None:
                        pattern_found = True
                        self.df_clean.at[index, column_name] = map_dict[key_value]
                if pattern_found:
                    break
            if not pattern_found:
                # we could put 'NA" or create a new category from the one found
                use_na = False
                # "?" is missing with regexp, we could also remove the ? from the string
                if ("?" in cell_text) or use_na:
                    self.df_clean.at[index, column_name] = map_dict["NA"]
                else:
                    add_key_to_map_dict(map_dict=map_dict, key_to_add=cell_text)
                    pattern_dict[cell_text] = cell_text
                    self.df_clean.at[index, column_name] = map_dict[cell_text]

    def clean_column(self, column_name, map_dict, special_cases=None, use_upper=True):
        if self.keep_original:
            index_column = self.df_clean.columns.get_loc(column_name)

            self.df_clean.insert(loc=index_column + 1, column=column_name + "_originale",
                                 value=self.df_clean.loc[:, column_name],
                                 allow_duplicates=False)
        for index, value in enumerate(self.df_clean.loc[:, column_name]):
            if pd.isna(value):
                self.df_clean.at[index, column_name] = -1
            else:
                # removing accentuation or space at the beginning and end of the string
                # print(f"{column_name}: value {value}")
                value = unidecode.unidecode(value.strip())
                if value.upper() in map_dict:
                    self.df_clean.at[index, column_name] = map_dict[value.upper()]
                elif value.lower() in map_dict:
                    self.df_clean.at[index, column_name] = map_dict[value.lower()]
                elif value in map_dict:
                    self.df_clean.at[index, column_name] = map_dict[value]
                else:
                    if special_cases is not None:
                        if value.lower() in special_cases:
                            self.df_clean.at[index, column_name] = map_dict[special_cases[value.lower()]]
                        elif value.upper() in special_cases:
                            self.df_clean.at[index, column_name] = map_dict[special_cases[value.upper()]]
                        elif value in special_cases:
                            self.df_clean.at[index, column_name] = map_dict[special_cases[value]]
                        else:
                            print(f"not in special_cases value: {value}")

    def clean(self):
        super().clean()
        for column_name, values in self.column_to_clean_dict.items():
            if len(values) > 1:
                self.clean_column(column_name=column_name, map_dict=values[0],
                                  special_cases=values[1])
            else:
                self.clean_column(column_name=column_name, map_dict=values[0])
        # self.clean_column(column_name="sexe", map_dict=self.sexe_mapping)
        # self.clean_column(column_name="etat_corneen", map_dict=self.etat_corneen_mapping,
        #                   special_cases=self.etat_corneen_cases)
        # self.clean_column(column_name="cote", map_dict=self.side_mapping, special_cases=self.side_special_cases)
        for column_name, dict_list in self.column_to_clean_with_reg_exp.items():
            self.clean_column_with_reg_exp(column_name=column_name, map_dict=dict_list[0],
                                           pattern_dict=dict_list[1])

    def produce_stats(self, file_name):
        # df.loc[df['column_name'] == some_value]
        with open(file_name, "w", encoding='UTF-8') as file:
            file.write("Number of patients by etiology:")
            file.write("\n")
            n_patients_list = []
            etiology_dict = {}
            index = 0
            for value, code in self.etiology_mapping.items():
                # print(f"code {code}, value {value}")
                # mask = self.df_clean['etiologie'] == str(code)
                # print(f"mask {mask}")
                # n_patients = len(self.df_clean[mask].index)
                n_patients = len(self.df_clean.loc[self.df_clean['etiologie'] == code].index)
                n_patients_list.append(n_patients)
                etiology_dict[index] = value
                index += 1
            indices_sorted = np.argsort(n_patients_list)
            for index in indices_sorted[::-1]:
                file.write(f"- {etiology_dict[index]}: {n_patients_list[index]}")
                file.write("\n")

            file.write("\n")
            file.write("\n")

            # details of neurotrophique causes
            # dict take key as str reprensenting the categorie of NK, and the vlaue is a dict with cause_NK as key
            # and value an int representing the number of patients, "NA" is unknown
            nk_category_dict = {}
            n_patients_by_nk_category = {}
            self.reverse_categorie_nk_mapping = {}
            for item, value in self.categorie_nk_mapping.items():
                self.reverse_categorie_nk_mapping[value] = item
            self.reverse_cause_nk_mapping = {}
            for item, value in self.cause_nk_mapping.items():
                self.reverse_cause_nk_mapping[value] = item
            neurotrophique_indices = self.df_clean.loc[self.df_clean['etiologie'] ==
                                                       self.etiology_mapping["neurotrophique"]].index
            for index, value in enumerate(self.df_clean.loc[neurotrophique_indices, "categorie_NK"]):
                if pd.isna(value):
                    n_patients_by_nk_category["NA"] = n_patients_by_nk_category.get("NA", 0) + 1
                else:
                    str_value = self.reverse_categorie_nk_mapping[value]
                    n_patients_by_nk_category[str_value] = n_patients_by_nk_category.get(str_value, 0) + 1
                    if str_value not in nk_category_dict:
                        nk_category_dict[str_value] = dict()
                    value_cause_nk = self.df_clean.loc[neurotrophique_indices[index], "cause_NK"]
                    value_cause_nk = self.reverse_cause_nk_mapping[value_cause_nk]
                    if pd.isna(value_cause_nk):
                        value_cause_nk = "NA"
                    nk_category_dict[str_value][value_cause_nk] = nk_category_dict[str_value].get(value_cause_nk, 0) + 1

            file.write("Number of patients by Neurotrophique categories with causes:")
            file.write("\n")

            list_nk_categories = list(n_patients_by_nk_category.keys())
            list_nk_categories_nb = []
            for cat in list_nk_categories:
                list_nk_categories_nb.append(n_patients_by_nk_category[cat])

            n_patients_by_nk_category_sorted_indices = np.argsort(list_nk_categories_nb)
            for index in n_patients_by_nk_category_sorted_indices[::-1]:
                file.write(f"# {list_nk_categories[index]}: {list_nk_categories_nb[index]}\n")
                n_patients_by_nk_cause_dict = nk_category_dict[list_nk_categories[index]]
                list_nk_causes = list(n_patients_by_nk_cause_dict.keys())
                list_nk_causes_nb = []
                for cat in list_nk_causes:
                    list_nk_causes_nb.append(n_patients_by_nk_cause_dict[cat])

                n_patients_by_nk_causes_sorted_indices = np.argsort(list_nk_causes_nb)
                for index in n_patients_by_nk_causes_sorted_indices[::-1]:
                    file.write(f"- {list_nk_causes[index]}: {list_nk_causes_nb[index]}")
                    file.write("\n")

                file.write("\n")
            file.write("\n")
            file.write("\n")

            file.write(f"Number of patients by promoting and associated factors for each etiology\n")
            file.write("\n")
            file.write("\n")
            # for each etiology, except neurotrophique, we want the number of patients
            # for each "facteur_favorisant" and "facteur_associe"
            self.reverse_facteur_favorisant_mapping = {}
            for item, value in self.facteur_favorisant_mapping.items():
                self.reverse_facteur_favorisant_mapping[value] = item
            self.reverse_facteur_associe_mapping = {}
            for item, value in self.facteur_associe_mapping.items():
                self.reverse_facteur_associe_mapping[value] = item

            for etiology_str, etiology_code in self.etiology_mapping.items():
                if etiology_str == "neurotrophique":
                    continue

                file.write(f"/// Etiology: {etiology_str}\n")
                n_patients_by_facteur_favorisant = {}
                n_patients_by_facteur_associe = {}

                etiology_indices = self.df_clean.loc[self.df_clean['etiologie'] ==
                                                     etiology_code].index

                for index, value in enumerate(self.df_clean.loc[etiology_indices, "facteur_favorisant"]):
                    if pd.isna(value):
                        n_patients_by_facteur_favorisant["NA"] = n_patients_by_facteur_favorisant.get("NA", 0) + 1
                    else:
                        str_value = self.reverse_facteur_favorisant_mapping[value]
                        n_patients_by_facteur_favorisant[str_value] = \
                            n_patients_by_facteur_favorisant.get(str_value, 0) + 1

                for index, value in enumerate(self.df_clean.loc[etiology_indices, "facteur_associe"]):
                    if pd.isna(value):
                        n_patients_by_facteur_associe["NA"] = n_patients_by_facteur_associe.get("NA", 0) + 1
                    else:
                        str_value = self.reverse_facteur_associe_mapping[value]
                        n_patients_by_facteur_associe[str_value] = \
                            n_patients_by_facteur_associe.get(str_value, 0) + 1

                # then printing the values
                # first ordering them
                list_facteur_favorisant = list(n_patients_by_facteur_favorisant.keys())
                list_facteur_favorisant_nb = []
                for cat in list_facteur_favorisant:
                    list_facteur_favorisant_nb.append(n_patients_by_facteur_favorisant[cat])

                list_facteur_associe = list(n_patients_by_facteur_associe.keys())
                list_facteur_associe_nb = []
                for cat in list_facteur_associe:
                    list_facteur_associe_nb.append(n_patients_by_facteur_associe[cat])

                file.write(f"/ promoting factors:\n")
                n_patients_by_facteur_favorisant_sorted_indices = np.argsort(list_facteur_favorisant_nb)
                for index in n_patients_by_facteur_favorisant_sorted_indices[::-1]:
                    file.write(f"# {list_facteur_favorisant[index]}: {list_facteur_favorisant_nb[index]}\n")

                file.write(f"/ associated factors:\n")
                n_patients_by_facteur_associe_sorted_indices = np.argsort(list_facteur_associe_nb)
                for index in n_patients_by_facteur_associe_sorted_indices[::-1]:
                    file.write(f"# {list_facteur_associe[index]}: {list_facteur_associe_nb[index]}\n")
                file.write("\n")
            # nb of unique patients, and nb of patients depending on how greffe
            # we could use df.duplicated to count unique patients
            indices_found = []
            # key represent the number of grafts, value the number of patients with this number of grafts
            patients_count = SortedDict()
            for index, value in enumerate(self.df_clean.loc[:, "nom"]):
                if index in indices_found:
                    # patients already counted
                    continue
                last_name = value
                first_name = self.df_clean.loc[index, "prenom"]
                birth_date = self.df_clean.loc[index, "date_naissance"]
                # getting the list of indices
                patients_indices = self.df_clean.loc[(self.df_clean['nom'] == last_name) &
                                                     (self.df_clean['prenom'] == first_name) &
                                                     (self.df_clean['date_naissance'] == birth_date)].index
                n_grafts = len(patients_indices)
                patients_count[n_grafts] = patients_count.get(n_grafts, 0) + 1
                indices_found.extend(patients_indices)

            total_unique_patients = 0
            for value in patients_count.values():
                total_unique_patients += value
            file.write("Number of patients:\n")
            file.write(f"- Unique patients: {total_unique_patients}\n")
            for n_grafts, value in patients_count.items():
                file.write(f"- Patients with {n_grafts} grafts: {value}\n")


"""
class CleanerMulti(Cleaner):
    def __init__(self, df_data):
        super().__init__(df_data=df_data)

        self.sexe_new_columns = ["femme", "homme"]
        self.sexe_mapping = {"F": self.sexe_new_columns[0], "M": self.sexe_new_columns[1]}

        self.side_new_columns = ["cote_gauche", "cote_droit"]
        self.side_mapping = {"OG": self.side_new_columns[0], "OD": self.side_new_columns[1],
                             "OD_OG": [self.side_new_columns[0], self.side_new_columns[1]]}
        self.side_special_cases = {"OG PUIS OD EN 07": "OD_OG", "0D": "OD", "OG PUIS OD":  "OD_OG",
                                   "OD + OG": "OD_OG"}

        self.etiologies = ["abces_cornee", "ulcere_trophique", "abces_amibien", "ulcere_neurotrophique", "diabete",
                           "ectropion"]
        # each key is a tuple of reg_ex, and the value is a key to etiology_mapping
        self.etiology_patterns = {"abces.*cornee": "abces_cornee", "ulcere.* trophique": "ulcere_trophique",
                                  "abces.*amibien": "abces_amibien", "ulcere neurotrophique": "ulcere_neurotrophique",
                                  "diabete": "diabete", "ectropion": "ectropion"}

        self.clean()

    def clean_column_with_reg_exp(self, column_name, new_columns, pattern_dict):
        index_column = self.df_clean.columns.get_loc(column_name)
        for new_col in new_columns:
            self.df_clean.insert(loc=index_column, column=column_name + "_" + new_col,
                                 value=np.zeros(self.n_lines, dtype="int8"),
                                 allow_duplicates=False)

        for index, cell_text in enumerate(self.df_clean.loc[:, column_name]):
            no_match = True
            if not pd.isna(cell_text):
                cell_text = unidecode.unidecode(cell_text)
                cell_text = cell_text.lower()
                for patterns, key_value in pattern_dict.items():
                    # key_value is the name of one of the new columbs
                    if isinstance(patterns, str):
                        patterns = [patterns]
                    for pattern in patterns:
                        match_object = re.search(pattern, cell_text, flags=0)
                        if match_object is not None:
                            no_match = False
                            self.df_clean.at[index, column_name + "_" + key_value] = 1
            if no_match:
                for new_col in new_columns:
                    self.df_clean.at[index, column_name + "_" + new_col] = - 1

    def clean_column(self, column_name, new_columns, map_dict, special_cases=None, use_upper=True):
        # for simple cases
        index_column = self.df_clean.columns.get_loc(column_name)
        for new_col in new_columns:
            self.df_clean.insert(loc=index_column, column=new_col, value=np.zeros(self.n_lines, dtype="int8"),
                                 allow_duplicates=False)

        for index, value in enumerate(self.df_clean.loc[:, column_name]):
            if pd.isna(value):
                for new_col in new_columns:
                    self.df_clean.at[index, new_col] = -1
            else:
                value = value.strip().upper()

                col_value = None
                if value in map_dict:
                    col_value = map_dict[value]
                else:
                    if special_cases is not None:
                        if value in special_cases:
                            col_value = map_dict[special_cases[value]]
                if col_value is not None:
                    if isinstance(col_value, str):
                        self.df_clean.at[index, col_value] = 1
                    else:
                        for val in col_value:
                            self.df_clean.at[index, val] = 1
                else:
                    print(f"value {value}")

        self.df_clean = self.df_clean.drop(columns=column_name)

    def clean(self):
        super().clean()
        self.clean_column(column_name="sexe", new_columns=self.sexe_new_columns, map_dict=self.sexe_mapping)
        self.clean_column(column_name="cote", new_columns=self.side_new_columns, map_dict=self.side_mapping,
                          special_cases=self.side_special_cases)
        self.clean_column_with_reg_exp(column_name="etiologie", new_columns=self.etiologies,
                                       pattern_dict=self.etiology_patterns)
"""


def main():
    root_path = "/Users/pappyhammer/Documents/academique/these_inmed/"
    path_data = root_path + "these_lucie/"
    path_results = root_path + "these_lucie/clean_data/"

    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")

    use_mutli_sheets_excel = False
    single_sheet_file_name = "gma recueil Lucie 24 Juilletouf.xlsx"

    if use_mutli_sheets_excel:
        original_file_name = "GMA Toulouse.xlsx"
        df_summary = pd.read_excel(path_data + original_file_name, sheet_name=f"Feuil1")
        list_dfs = []
        names_col = None
        n_columns_full = 43
        n_columns_empty = 2
        # going through all the sheetss
        for n in np.arange(2, 7):
            df = pd.read_excel(path_data + original_file_name, sheet_name=f"Feuil{n}")
            df = df.iloc[:, :n_columns_full + n_columns_empty]
            if n == 4:
                # switiching in Feuil4 column date of birth and gender column to respect the order of others
                columnsTitles = list(df.columns)
                tmp = columnsTitles[3]
                columnsTitles[3] = columnsTitles[4]
                columnsTitles[4] = tmp
                df = df.reindex(columns=columnsTitles)
            columns_name = list(df.columns)
            print(f"{n}: columns_name: {columns_name}")
            # changing names of the columns so they are the same in all the sheets
            columns_name[24] = "catégorie NK"
            columns_name[26] = "facteur favorisant"
            columns_name[27] = "facteur associé"
            columns_name[29] = "CF (nb de j)"
            columns_name[30] = "antifongiques"
            columns_name[32] = "date ttt chir"
            columns_name[35] = "taille ulcère"
            df.columns = columns_name
            if names_col is None:
                names_col = list(df.columns)
            else:
                for index, col_name in enumerate(list(df.columns)):
                    if col_name not in names_col:
                        print(f"{n}: {col_name} not in columns, index: {index}")
            # print(f"{n}: shape: {df.shape}")
            # print(f"{n}: columns: {df.columns}")
            list_dfs.append(df)
        # raise Exception("TOTO")

        df_data = pd.concat(list_dfs)
        dates_columns = ["date greffe", "date naissance"]
        # dealing with data format
        for date_name in dates_columns:
            df_data[date_name] = pd.to_datetime(df_data[date_name])
            df_data[date_name] = df_data[date_name].dt.strftime('%d/%m/%Y')

        writer = pd.ExcelWriter(f'{path_results}/GMA_Toulouse_single_sheet.xlsx')
        # df_summary.to_excel(writer, 'summary', index=False)
        df_data.to_excel(writer, 'data', index=False)
        writer.save()
        # df_data = pd.read_excel(os.path.join(path_results, single_sheet_file_name), sheet_name="data")
    else:
        original_file_name = single_sheet_file_name
        df_data = pd.read_excel(os.path.join(path_data, original_file_name), sheet_name="Feuil1")

    # print(f"shape: {df_data.shape}")
    # print(f"columns: {df_data.columns}")
    use_multi_cleaner = False

    cleaner_coder_with_original = CleanerCoder(df_data=df_data.copy(), keep_original=True)
    cleaner_coder_without_original = CleanerCoder(df_data=df_data.copy(), keep_original=False)

    # if use_multi_cleaner:
    #     cleaner_multi = CleanerMulti(df_data=df_data.copy())

    # print(f"shape: {cleaner_coder.df_clean.shape}")
    # print(f"columns coder: {cleaner_coder.df_clean.columns}")
    # if use_multi_cleaner:
    #     print(f"shape: {cleaner_multi.df_clean.shape}")
    #     print(f"columns multi: {cleaner_multi.df_clean.columns}")

    cleaner_coder_without_original.produce_stats(file_name=f'{path_results}/stats_Lucie_{time_str}.txt')

    writer = pd.ExcelWriter(f'{path_results}/these_lucie_data_clean_code_with_original_{time_str}.xlsx')
    # df_summary.to_excel(writer, 'summary', index=False)
    cleaner_coder_with_original.df_clean.to_excel(writer, 'data', index=False)
    writer.save()

    cleaner_coder_with_original.do_anonymization()
    writer = pd.ExcelWriter(f'{path_results}/these_lucie_data_clean_code_with_original_anonymized_{time_str}.xlsx')
    # df_summary.to_excel(writer, 'summary', index=False)
    cleaner_coder_with_original.df_clean.to_excel(writer, 'data', index=False)
    writer.save()

    writer = pd.ExcelWriter(f'{path_results}/these_lucie_data_clean_code_{time_str}.xlsx')
    # df_summary.to_excel(writer, 'summary', index=False)
    cleaner_coder_without_original.df_clean.to_excel(writer, 'data', index=False)
    writer.save()

    cleaner_coder_without_original.do_anonymization()
    writer = pd.ExcelWriter(f'{path_results}/these_lucie_data_clean_code_anonymized_{time_str}.xlsx')
    # df_summary.to_excel(writer, 'summary', index=False)
    cleaner_coder_without_original.df_clean.to_excel(writer, 'data', index=False)
    writer.save()

    with open(f'{path_results}/code_mapping_{time_str}.txt', "w", encoding='UTF-8') as file:
        for dict_name, values in cleaner_coder_without_original.mapping_dict.items():
            file.write(f"{dict_name}: ")
            first = True
            for key, value in values.items():
                if not first:
                    file.write(f" / ")
                file.write(f"{value} -> {key}")
                first = False

            file.write("\n")
            file.write("\n")

    if use_multi_cleaner:
        writer = pd.ExcelWriter(f'{path_results}/these_lucie_data_clean_multi_{time_str}.xlsx')
        # df_summary.to_excel(writer, 'summary', index=False)
        cleaner_multi.df_clean.to_excel(writer, 'data', index=False)
        writer.save()


main()
