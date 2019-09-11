import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        # for index, value in enumerate(self.df_clean.loc[:, "date_greffe"]):
        #     date_time = datetime.strptime(value, '%d/%m/%Y')
        #     self.df_clean.at[index, "date_greffe"] = date_time.year

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
    def __init__(self, df_data, keep_original, path_results):
        """

        :param df_data:
        :param keep_original: if True, keep the original column adding "_original" to the name, still adding the encoded
        column
        """
        super().__init__(df_data=df_data)

        self.path_results = path_results
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

        statut_cristallin_to_map = ["phaque", "pke odg", "pke"]
        self.statut_cristallin_mapping = {"NA": -1}
        for code, statut_cristallin in enumerate(statut_cristallin_to_map):
            self.statut_cristallin_mapping[statut_cristallin] = code
        self.mapping_dict["statut_cristallin"] = self.statut_cristallin_mapping
        self.statut_cristallin_patterns = {("pke og", "pkeog", "eic od", r"\bpke\b (?!odg$)"): "pke",
                                           "pke odg": "pke odg"}
        """PKE =  pke og / 3 -> pke / 4 -> eic od"""
        self.column_to_clean_with_reg_exp["statut_cristallin"] = [self.statut_cristallin_mapping,
                                                                  self.statut_cristallin_patterns]

        loc_ulcere_to_map = ["paracentral", "central", "superieur", "inferieur", "conjonctive", "peripherique",
                             "colerette", "sclere"]
        self.loc_ulcere_mapping = {"NA": -1}
        for code, loc_ulcere in enumerate(loc_ulcere_to_map):
            self.loc_ulcere_mapping[loc_ulcere] = code
        self.mapping_dict["localisation_ulcere"] = self.loc_ulcere_mapping

        self.loc_ulcere_patterns = {(r"\bcenral\b", r"\bcentral\b", r"\bcentrale\b"): "central",
                                    "conjonctival": "conjonctive",
                                    "inf": "inferieur",
                                    "sup": "superieur",
                                    ("temporal", "nasal", "limbique"): "peripherique"}
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

        ttt_compl_to_map = ["ciclo 2%","cure chir statique palpebrale", "verre scleral",
                            "greffe cornee", "greffe bouchon", "glac",
                             "tarsorraphie", "greffe csl",
                            "recouvrement conjonctival", "debridement des berges", "poncage edta",
                            "recul conjonctival", "mitomycine"]
        self.ttt_compl_mapping = {"NA": -1}
        for code, ttt_compl in enumerate(ttt_compl_to_map):
            self.ttt_compl_mapping[ttt_compl] = code
        self.mapping_dict["ttt_complementaire"] = self.ttt_compl_mapping
        """
        tarsorraphie  definitive / 8 -> cure chir statique palpebrale / 9 -> tarsorraphie 

        """
        self.ttt_compl_patterns = {".*ciclo.*": "ciclo 2%",

                                   ("recouvrement conjontival"): "recouvrement conjonctival",

                                   ("debridement berges", "debridement"): "debridement des berges",

                                   ("grattage mecanique"): "poncage edta",

                                   ("mito 0,2%", "mitomycine 0,4%"): "mitomycine",

                                   "recul conjunctival": "recul conjonctival",

                                   "verres scleraux": "verre scleral",

                                   "inscription liste greffe": "glac",

                                   ("dsaek", "altk", "klap", r"\bkt\b.*(?!bouchon$)", "dsaek"): "greffe cornee",

                                   ("epikeratoplastie", "greffe limbique", "KT bouchon"): "greffe bouchon",

                                   ("tarsorraphie externe", "tarsorraphie provisoire", ".*tarsorraphie.*definitive.*"):
                                       "tarsorraphie",

                                   ("autogreffe csl", "autogreffe de limbe", "allogreffe limbe",
                                    "allogreffe csl"): "greffe csl",

                                   (".*entropion.*", ".*ectropion.*", "greffe muqueuse buccale"):
                                       "cure chir statique palpebrale"}
        self.column_to_clean_with_reg_exp["ttt_complementaire"] = [self.ttt_compl_mapping,
                                                                   self.ttt_compl_patterns]

        ttt_compl_2_to_map = ttt_compl_to_map
        self.ttt_compl_2_mapping = {"NA": -1}
        for code, ttt_compl_2 in enumerate(ttt_compl_2_to_map):
            self.ttt_compl_2_mapping[ttt_compl_2] = code
        self.mapping_dict["ttt_complementaire_2"] = self.ttt_compl_2_mapping
        self.ttt_compl_2_patterns = self.ttt_compl_patterns.copy()
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
        self.complication_patterns = {("abes cornee", "abces"): "abces cornee"}
        self.column_to_clean_with_reg_exp["complication"] = [self.complication_mapping,
                                                             self.complication_patterns]

        prelevement_to_map = ["staph aureus", "moraxella"]
        self.prelevement_mapping = {"NA": -1}
        for code, prelevement in enumerate(prelevement_to_map):
            self.prelevement_mapping[prelevement] = code
        self.mapping_dict["prelevement"] = self.prelevement_mapping
        self.prelevement_patterns = {("staphylocoque", "staph aureus"): "staph aureus",
                                     ("moraxela", "moraxella"): "moraxella"}
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

        etiologies_to_map = ["inflammatoire",
                             "mecanique", "keratite infectieuse",
                             "insuffisance limbique",
                             "anomalie statique palpebrale", "decompensation bulleuse epitheliale",
                             "destruction aigue surface", "neurotrophique", "keratopathie en bandelette",
                             "reconstruction", "refection BF"]
        self.etiology_mapping = {"NA": -1}
        for code, etiology in enumerate(etiologies_to_map):
            self.etiology_mapping[etiology] = code
        self.mapping_dict["etiologie"] = self.etiology_mapping
        # each key is a tuple of reg_ex, and the value is a key to etiology_mapping
        # the key should be lower case
        """
        destruction aigue surface / 14 -> "destuction aigue de la surface" / 15 -> atteinte aigue surface


        """
        self.etiology_patterns = {("ulcere.* inflammatoire", "inflammaoire", "inflammatoire"): "inflammatoire",
                                  "keratite.*infectieuse": "keratite infectieuse",
                                  "insuffisance limbique": "insuffisance limbique",
                                  "anomalie statique.*rale": "anomalie statique palpebrale",
                                  "decompensation bulleuse.*": "decompensation bulleuse epitheliale",
                                  ("destruction aigue surface", "destuction aigue de la surface",
                                   "atteinte aigue surface"): "destruction aigue surface",
                                  ("neurotrophique", "neurotrophiqe"): "neurotrophique",
                                  ("keratopathie.*bandelette", "keratite.*bandelette",
                                   "keratite en bandelette"): "keratopathie en bandelette",
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
                                      ("iatrogene", "iatrogenie"): "iatrogenie",
                                      "atteinte chronique surface": "atteinte chronique surface",
                                      "central.*": "centrale"}
        self.column_to_clean_with_reg_exp["categorie_NK"] = [self.categorie_nk_mapping, self.categorie_nk_patterns]

        cause_nk_to_map = ["hsv", "greffe cornee", "tumeurs cerebrales",
                           "sup ou egal a 2 greffes cornee", "ains", "vzv", "collyres", "pr + gougerot",
                           "thermocoagulation ganglion gasser", "rosacee et blepharite", "brulure"]
        self.cause_nk_mapping = {"NA": -1}
        for code, etiology in enumerate(cause_nk_to_map):
            self.cause_nk_mapping[etiology] = code
        self.mapping_dict["cause_NK"] = self.cause_nk_mapping
        # each key is a tuple of reg_ex, and the value is a key to etiology_mapping
        # the key should be lower case
        self.cause_nk_patterns = {"hsv": "hsv",
                                  ("brulure", "base"): "brulure",
                                  ("rosacee", "blepharite"): "rosacee et blepharite",
                                  ("metastases cerebrales", "tumeur cerebrale",
                                   "tumeur cervelet", "neurinome", "tumeur rocher",
                                   "exerese meningiome", "ch.*rurgie meningiome", "meningiome"):"tumeurs cerebrales",
                                  ("kt", "altk", "greffe lamellaire", "tatouage corneen"): "greffe cornee",
                                  ("2.*kt", "3.*kt"): "sup ou egal a 2 greffes cornee",
                                  "ains": "ains", "vzv": "vzv",
                                  ("coagulation ganglion gasser", "thermocoagulation ganglion gasser"):
                                      "thermocoagulation ganglion gasser",
                                  ("col.*atb", "liquide conservation lentilles",
                                   "collyres", "collyres antiglaucomateux", "chimiotherapie", "tumeur rocher"):
                                      "collyres",
                                  ("pr", "goug.*rot"): "pr + gougerot"}
        self.column_to_clean_with_reg_exp["cause_NK"] = [self.cause_nk_mapping, self.cause_nk_patterns]

        facteur_favorisant_to_map = ["brulure", "decompensation endotheliale", "patho corneenne congenitale",
                                     "greffe", "exerese pterygion",
                                     "tumeur conjonctivale", "dystrophie de cornee",
                                     "pr + gougerot",
                                     "patho corneenne congenitale", "keratoconjonctivite allergique",
                                     "facteurs locaux", "defect conjonctival/ scleral",
                                     "chir refractive", "entropion", "lagophtalmie", "vernal",
                                     "mooren"
                                     ]
        self.facteur_favorisant_mapping = {"NA": -1}
        for code, etiology in enumerate(facteur_favorisant_to_map):
            self.facteur_favorisant_mapping[etiology] = code
        self.mapping_dict["facteur_favorisant"] = self.facteur_favorisant_mapping
        # each key is a tuple of reg_ex, and the value is a key to etiology_mapping
        # the key should be lower case

        """
        18  mooren, 57 , pseudomooren  + 52 -> puk
        """

        self.facteur_favorisant_patterns = {("brulure.*", "base"): "brulure",
                                            ("pseudomooren", "puk"): "mooren",

                                            ("lasik", "post pkr", "pkr"): "chir refractive",

                                            ("kt", "2e kt", "rejet greffe", "greffe lamellaire"): "greffe",

                                            ("pr", "gougerot", "spa"): "pr + gougerot",

                                            ("orbitopathie dysthyroidienne"): "lagophtalmie",

                                            ('cil trichiasique', "cils trichiasiques"): "entropion",

                                            ("pterygoide", "cicatrice pter.*", "pterygion"): "exerese pterygion",

                                            ("exerese tumeur conjonctivale", "melanome", "bowen",
                                             "maladie de bowen", "kyste inclusion",
                                             "kyste conj.*", "carcinome",
                                             "carcinome epidermoide", "exerese melanose acquise",
                                             "exerese lesion", "exerese tumeur",
                                             "exerese kyste d'inclusion", "exerese lymphome conjonctival",
                                             "exerese lesion conjonctivale", "lymphome malt"): "tumeur conjonctivale",

                                             ("dyskeratose", "cogan", "salzman", "hydrops",
                                              "keratite.*bandelette", "poncage keratite en bandelette",
                                              "keratoglobe", "keratalgie recidivante",
                                              "keratopathie bandelette", "ICE syndrome"):
                                            "dystrophie de cornee",

                                            "keratoconjonctivite atopique": "keratoconjonctivite allergique",

                                            ("glaucome congenital", "scerocorneee", "aniridie congenitale",
                                             "sclerocornee"): "patho corneenne congenitale",

                                            ("deompensation endotheliale", "keratopathie bulleuse"):
                                                "decompensation endotheliale",

                                            ("ctc locale", "ctc po", "ctc topique", "blepharite", "corps etranger",
                                             "ce", "blepharite", "post pke", "traumatique", "cellulite infectieuse",
                                             "sur taie", "chimiotherapie",
                                             "incision", "trauma vegetal", "sur fil"): "facteurs locaux",

                                            ("ablation symblepharon", "dehiscence conjonctivale",
                                             "refection bf", "extrusion bille evisceration / scleromalacie",
                                             "dehiscence keratoprothese", "tumeur palpebrale",
                                             "reconstruction palpebrale",
                                             "POC biopsie conjonctivale", "cure symblepharon",
                                             "scleromalacie", "dellen post pterygion",
                                             "plaie cornee", "reouverture plaie", "conjonctivoplastie",
                                             "refection bulle filtration "):
                                                "defect conjonctival/ scleral",

                                            ("fonte stromale", "alteration chronique surface",
                                             "gnv", "trijumeau"): "NA",

                                            ("kcv"): "vernal"}
        """

scleromalacie/ dellen post pterygion / plaie cornee / 37 -> reouverture plaie/ conjonctivoplastie

                """
        self.column_to_clean_with_reg_exp["facteur_favorisant"] = [self.facteur_favorisant_mapping,
                                                                   self.facteur_favorisant_patterns]

        facteur_associe_to_map = ["kt"]
        self.facteur_associe_mapping = {"NA": -1}
        for code, etiology in enumerate(facteur_associe_to_map):
            self.facteur_associe_mapping[etiology] = code
        self.mapping_dict["facteur_associe"] = self.facteur_associe_mapping
        # each key is a tuple of reg_ex, and the value is a key to etiology_mapping
        # the key should be lower case
        self.facteur_associe_patterns = {"2e klap": "kt"}
        self.column_to_clean_with_reg_exp["facteur_associe"] = [self.facteur_associe_mapping,
                                                                self.facteur_associe_patterns]

        etat_corneen_to_map = ["preperforatif", "perfore"]
        self.etat_corneen_mapping = {"NA": -1}
        for code, etiology in enumerate(etat_corneen_to_map):
            self.etat_corneen_mapping[etiology] = code
        self.etat_corneen_cases = {"preperfratif": "preperforatif",
                                   "preerforatif": "preperforatif",
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
            if pd.isna(cell_text) or (cell_text == ""):
                self.df_clean.at[index, column_name] = map_dict["NA"]
                continue
            if isinstance(cell_text, int):
                self.df_clean.at[index, column_name] = map_dict["NA"]
                continue
            # if column_name == "facteur_favorisant":
            #     print(f"cell_text {cell_text}")
            cell_text = unidecode.unidecode(cell_text)
            cell_text = cell_text.lower().strip()
            if cell_text == "":
                self.df_clean.at[index, column_name] = map_dict["NA"]
                continue
            # if column_name == "facteur_favorisant":
            #     print(f"unicode cell_text {cell_text}")
            pattern_found = False
            for patterns, key_value in pattern_dict.items():
                if isinstance(patterns, str):
                    patterns = [patterns]
                for pattern in patterns:
                    # print(f"pattern {pattern}")
                    match_object = re.search(pattern, cell_text, flags=0)
                    if match_object is not None:
                        pattern_found = True
                        # print(f"{patterns}: key_value {key_value}")
                        self.df_clean.at[index, column_name] = map_dict[key_value]
                if pattern_found:
                    break
            if not pattern_found:
                # we could put 'NA" or create a new category from the one found
                use_na = False
                # "?" is missing with regexp, we could also remove the ? from the string
                if ("?" in cell_text) or (cell_text == "") or use_na:
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

    def clean_numerical_columns(self, column_name):
        """
        Removing '?'
        Returns:

        """
        if self.keep_original:
            index_column = self.df_clean.columns.get_loc(column_name)

            self.df_clean.insert(loc=index_column + 1, column=column_name + "_originale",
                                 value=self.df_clean.loc[:, column_name],
                                 allow_duplicates=False)
        for index, value in enumerate(self.df_clean.loc[:, column_name]):
            if pd.isna(value) or isinstance(value, str): # and ("?" in value)):
                self.df_clean.at[index, column_name] = -1


    def clean(self):
        super().clean()
        for column_name in ["taille_ulcere", "nb_de_greffe"]:
            self.clean_numerical_columns(column_name=column_name)
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

    def get_duration_bw_grafts(self, df, first_graft, second_graft):
        """
        Get duration in days between 2 graft.
        Args:
            df:
            first_graft: Index representing the first graft to compare (like 2 for the third graft)
            second_graft:Index representing the second graft to compare with the first (like 4 for the fifth graft)

        Returns:

        """
        graft_dates_list = []
        for index, value in enumerate(df.loc[:, "date_greffe"]):
            date_time = datetime.strptime(value, '%d/%m/%Y')
            graft_dates_list.append(date_time)
            # self.df_clean.at[index, "date_greffe"] = date_time.year
        graft_dates_list.sort()
        # number of days of difference between the 2 first grafts
        delta = graft_dates_list[second_graft] - graft_dates_list[first_graft]
        return delta.days

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
            durations_bw_1_st_2nd_graft_in_days = []
            durations_bw_2_nd_3rd_graft_in_days= []
            for index, value in enumerate(self.df_clean.loc[:, "nom"]):
                if index in indices_found:
                    # patients already counted
                    continue
                last_name = value
                first_name = self.df_clean.loc[index, "prenom"]
                birth_date = self.df_clean.loc[index, "date_naissance"]
                # getting the list of indices
                df_filter = self.df_clean.loc[(self.df_clean['nom'] == last_name) &
                                                     (self.df_clean['prenom'] == first_name) &
                                                     (self.df_clean['date_naissance'] == birth_date)]
                patients_indices = df_filter.index
                if len(patients_indices) > 1:
                    durations_bw_1_st_2nd_graft_in_days.append(self.get_duration_bw_grafts(df_filter, first_graft=0,
                                                                                           second_graft=1))
                if len(patients_indices) > 2:
                    durations_bw_2_nd_3rd_graft_in_days.append(self.get_duration_bw_grafts(df_filter, first_graft=1,
                                                                                           second_graft=2))

                n_grafts = len(patients_indices)
                patients_count[n_grafts] = patients_count.get(n_grafts, 0) + 1
                indices_found.extend(patients_indices)

            file.write(f"Mean & std duration between the 2 first grafts: {np.mean(durations_bw_1_st_2nd_graft_in_days)} "
                       f"days, {np.std(durations_bw_1_st_2nd_graft_in_days)} days"
                       f" ({len(durations_bw_1_st_2nd_graft_in_days)} patients)\n")

            file.write(
                f"Mean & std duration between the second and third grafts: {np.mean(durations_bw_2_nd_3rd_graft_in_days)} "
                f"days, {np.std(durations_bw_2_nd_3rd_graft_in_days)} days"
                f" ({len(durations_bw_2_nd_3rd_graft_in_days)} patients)\n")

            # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
            # + 11 diverting
            colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                      '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
                      '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
                      '#74add1', '#4575b4', '#313695']
            filename = "delai_greffes"
            plot_box_plot_data(data_dict={"1-2": durations_bw_1_st_2nd_graft_in_days,
                                          "2-3": durations_bw_2_nd_3rd_graft_in_days},
                               filename=filename,
                               path_results=self.path_results,
                               y_label="Delai (jours)", colors=colors, with_scatters=True)

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

def plot_box_plot_data(data_dict, filename,
                         y_label, path_results, colors=None,
                         y_lim=None,
                         x_label=None, with_scatters=True,
                         y_log=False,
                         title=None,
                         scatters_with_same_colors=None,
                         scatter_size=20,
                         scatter_alpha=0.5,
                         n_sessions_dict=None,
                         background_color="black",
                         link_medians=True,
                         color_link_medians="red",
                         labels_color="white",
                         with_y_jitter=None,
                         x_labels_rotation=None,
                         fliers_symbol=None,
                         save_formats="pdf"):
    """

    :param data_dict:
    :param n_sessions_dict: should be the same keys as data_dict, value is an int reprenseing the number of sessions
    that gave those data (N), a n will be display representing the number of poins in the boxplots if n != N
    :param title:
    :param filename:
    :param y_label:
    :param y_lim: tuple of int,
    :param scatters_with_same_colors: scatter that have the same index in the data_dict,, will be colors
    with the same colors, using the list of colors given by scatters_with_same_colors
    :param param: Contains a field name colors used to color the boxplot
    :param save_formats:
    :return:
    """
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(12, 12))
    colorfull = (colors is not None)

    median_color = background_color if colorfull else labels_color

    ax1.set_facecolor(background_color)

    fig.patch.set_facecolor(background_color)

    labels = []
    data_list = []
    medians_values = []
    for age, data in data_dict.items():
        data_list.append(data)
        # print(f"data {data}")
        medians_values.append(np.median(data))
        label = age
        if n_sessions_dict is None:
            # label += f"\n(n={len(data)})"
            pass
        else:
            n_sessions = n_sessions_dict[age]
            if n_sessions != len(data):
                label += f"\n(N={n_sessions}, n={len(data)})"
            else:
                label += f"\n(N={n_sessions})"
        labels.append(label)
    sym = ""
    if fliers_symbol is not None:
        sym = fliers_symbol
    bplot = plt.boxplot(data_list, patch_artist=colorfull,
                        labels=labels, sym=sym, zorder=30)  # whis=[5, 95], sym='+'
    # color=["b", "cornflowerblue"],
    # fill with colors

    # edge_color="silver"

    for element in ['boxes', 'whiskers', 'fliers', 'caps']:
        plt.setp(bplot[element], color="white")

    for element in ['means', 'medians']:
        plt.setp(bplot[element], color=median_color)

    if colorfull:
        if colors is None:
            colors = param.colors[:len(data_dict)]
        else:
            while len(colors) < len(data_dict):
                colors.extend(colors)
            colors = colors[:len(data_dict)]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            r, g, b, a = patch.get_facecolor()
            # for transparency purpose
            patch.set_facecolor((r, g, b, 0.8))

    if with_scatters:
        for data_index, data in enumerate(data_list):
            # Adding jitter
            x_pos = [1 + data_index + ((np.random.random_sample() - 0.5) * 0.5) for x in np.arange(len(data))]

            if with_y_jitter is not None:
                y_pos = [value + (((np.random.random_sample() - 0.5) * 2) * with_y_jitter) for value in data]
            else:
                y_pos = data
            font_size = 3
            colors_scatters = []
            if scatters_with_same_colors is not None:
                while len(colors_scatters) < len(y_pos):
                    colors_scatters.extend(scatters_with_same_colors)
            else:
                colors_scatters = [colors[data_index]]
            ax1.scatter(x_pos, y_pos,
                        color=colors_scatters[:len(y_pos)],
                        alpha=scatter_alpha,
                        marker="o",
                        edgecolors=background_color,
                        s=scatter_size, zorder=1)
    if link_medians:
        ax1.plot(np.arange(1, len(medians_values) + 1), medians_values, zorder=36, color=color_link_medians,
                 linewidth=2)

    # plt.xlim(0, 100)
    if title:
        plt.title(title)

    ax1.set_ylabel(f"{y_label}", fontsize=30, labelpad=20)
    if y_lim is not None:
        ax1.set_ylim(y_lim[0], y_lim[1])
    if x_label is not None:
        ax1.set_xlabel(x_label, fontsize=30, labelpad=20)
    ax1.xaxis.label.set_color(labels_color)
    ax1.yaxis.label.set_color(labels_color)
    if y_log:
        ax1.set_yscale("log")

    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.tick_params(axis='y', colors=labels_color)
    ax1.tick_params(axis='x', colors=labels_color)
    xticks = np.arange(1, len(data_dict) + 1)
    ax1.set_xticks(xticks)
    # removing the ticks but not the labels
    ax1.xaxis.set_ticks_position('none')
    # sce clusters labels
    ax1.set_xticklabels(labels)
    if x_labels_rotation is not None:
        for tick in ax1.get_xticklabels():
            tick.set_rotation(x_labels_rotation)

    # padding between ticks label and  label axis
    # ax1.tick_params(axis='both', which='major', pad=15)
    fig.tight_layout()
    # adjust the space between axis and the edge of the figure
    # https://matplotlib.org/faq/howto_faq.html#move-the-edge-of-an-axes-to-make-room-for-tick-labels
    # fig.subplots_adjust(left=0.2)

    if isinstance(save_formats, str):
        save_formats = [save_formats]

    for save_format in save_formats:
        fig.savefig(f'{path_results}/{filename}'
                    f'.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())

    plt.close()


def main():
    root_path = "/Users/pappyhammer/Documents/academique/these_inmed/"
    path_data = root_path + "these_lucie/"
    path_results = root_path + "these_lucie/clean_data/"

    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    path_results = path_results + f"{time_str}"
    os.mkdir(path_results)

    use_mutli_sheets_excel = False
    single_sheet_file_name = "gma recueil Lucie 4AOUT.xlsx"

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

    cleaner_coder_with_original = CleanerCoder(df_data=df_data.copy(), keep_original=True, path_results=path_results)
    cleaner_coder_without_original = CleanerCoder(df_data=df_data.copy(), keep_original=False, path_results=path_results)

    # if use_multi_cleaner:
    #     cleaner_multi = CleanerMulti(df_data=df_data.copy())

    # print(f"shape: {cleaner_coder.df_clean.shape}")
    # print(f"columns coder: {cleaner_coder.df_clean.columns}")
    # if use_multi_cleaner:
    #     print(f"shape: {cleaner_multi.df_clean.shape}")
    #     print(f"columns multi: {cleaner_multi.df_clean.columns}")


    do_regression = False

    if do_regression:
        RegressionAnalysis(cleaner_coder_without_original.df_clean, path_results, time_str)
        return

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


class RegressionAnalysis:

    def __init__(self, df, path_results, time_str):
        # df as pandas data frame
        self.df = df
        """
        aed.efficacy (-2 : no seizure or no AED, -1 : no info, 0 : seizure control, 
        1 : no control achieved, 2 partial control)

        specific aed (1 : seizure free, 2 : seizure reduction, 3 : no effect, 4 : worsening, -1 no info)
        """


        # print(f"index : { self.df['cohort']}")
        self.path_results = path_results
        self.time_str = time_str
        self.clean_df()

        # self.discretize_age_onset()
        #
        # self.create_patient_id()
        # self.create_mutation_regions()
        # self.create_scb_efficacy()

        # Check for Null values
        print(f"N patients {len(self.df)}")
        print("Null values: ")
        # print(self.df.isnull().sum())
        for key, value in self.df.isnull().sum().items():
            print(f"{key} {value}")

        # print(self.df.describe())
        # columns = ["age.onset", "EEG.onset", "gender", "seizure.type.onset", "dev.before", "dev.after",
        #            "region", "mri.onset", "mri.followup", "ofc.birth", "ofc.evolution", "AED.efficacy",
        #            'scb.efficacy'] + self.scb_names
        columns = list(self.df.columns)
        columns_to_remove = ["patient_id", "date_greffe", "nom", "prenom", "date_naissance", "ATCD_generaux",
                             "ATCD_ophtalmo", "AVL_oeil_controlat", "date_ttt_chir", "date_1er_ttt"]
        # TODO: need to remove "PL+" from AVL œil controlat
        for column_name in columns:
            # For now removing AVP columns
            # TODO: remove the P and keep only the number
            if "AV" in column_name or "AVP" in column_name or "AV_oeil_controlat" in column_name:

                columns_to_remove.append(column_name)
        columns = list(set(columns) - set(columns_to_remove))
        print(f"columns removed {columns_to_remove}")
        print(f"columns analyzed {columns}")
        # self.plot_distribution(columns=columns)
        self.plot_corr_heatmap(columns=columns)
        # columns = ["age.onset", "region", "EEG.onset", "gender",]
        # self.create_pairplots(columns=columns)

    def create_pairplots(self, columns, save_formats="png"):
        # fig, ax1 = plt.subplots(nrows=1, ncols=1,
        #                         gridspec_kw={'height_ratios': [1]},
        #                         figsize=(12, 12))
        # print(f"self.df[columns] {self.df[columns]}")
        # TODO: addd hue='Survived', replace Survived by the field stating AED or SCB efficacy
        grid = sns.pairplot(self.df[columns].astype(float), dropna=True,
                            diag_kind='kde')  # , height=1.2, # diag_kind='kde', palette='seismic'
        # diag_kws=dict(shade=True), plot_kws=dict(s=10))
        grid.set(xticklabels=[])
        # grid = grid.map_upper(col_nan_scatter)
        # plt.show()
        grid.savefig("output.png")

        if isinstance(save_formats, str):
            save_formats = [save_formats]
        filename = f"pairplots"
        for save_format in save_formats:
            grid.savefig(f'{self.path_results}/{filename}'
                         f'_{self.time_str}.{save_format}')
        #
        # for save_format in save_formats:
        #     fig.savefig(f'{self.path_results}/{filename}'
        #                 f'_{self.time_str}.{save_format}',
        #                 format=f"{save_format}",
        #                 facecolor=fig.get_facecolor())
        #
        # plt.close()

    def plot_corr_heatmap(self, columns, save_formats="pdf"):
        corr_df = self.df[columns]
        colormap = plt.cm.RdBu
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                gridspec_kw={'height_ratios': [1]},
                                figsize=(12, 12))
        # ax1.title('Pearson Correlation of Features', y=1.05, size=15)
        sns.heatmap(corr_df.astype(float).corr(), linewidths=0.1, vmax=1.0,
                    square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 3})

        if isinstance(save_formats, str):
            save_formats = [save_formats]
        filename = f"corr_heatmap"

        for save_format in save_formats:
            fig.savefig(f'{self.path_results}/{filename}'
                        f'_{self.time_str}.{save_format}',
                        format=f"{save_format}",
                        facecolor=fig.get_facecolor())

        plt.close()

    def plot_distribution(self, columns):

        background_color = "black"
        labels_color = "white"
        save_formats = ["pdf"]

        for column in columns:
            fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                    gridspec_kw={'height_ratios': [1]},
                                    figsize=(12, 12))
            ax1.set_facecolor(background_color)

            fig.patch.set_facecolor(background_color)
            self.df.hist(column=column, ax=ax1, bins=100)

            fig.tight_layout()
            ax1.tick_params(axis='y', colors=labels_color)
            ax1.tick_params(axis='x', colors=labels_color)

            if isinstance(save_formats, str):
                save_formats = [save_formats]
            filename = f"{column}_hist"

            for save_format in save_formats:
                fig.savefig(f'{self.path_results}/{filename}'
                            f'_{self.time_str}.{save_format}',
                            format=f"{save_format}",
                            facecolor=fig.get_facecolor())

            plt.close()

    def clean_df(self):
        """
        CLeaning the data_frame, for exemple instead of a code saying NA, we will put NaN
        :return:
        """
        # Fill empty and NaNs values with NaN
        self.df = self.df.fillna(np.nan)

        columns_name = list(self.df.columns)
        column_to_clean = columns_name

        # if the code is not -1, then we precise which is it, there could be more than one code
        # code_na = {"age.onset": (-2, -1), "EEG.onset": (-2, -1), "seizure.type.onset": (-3, -1),
        #            "AED.efficacy": (-2, -1)}
        code_na = {}
        for column_name in column_to_clean:
            print(f"## column name: {column_name}")
            codes_to_replace = [-1]
            if column_name in code_na:
                codes_to_replace = code_na[column_name]
                if isinstance(codes_to_replace, int):
                    codes_to_replace = [codes_to_replace]
            for code in codes_to_replace:
                self.df.loc[self.df[column_name] == code, column_name] = np.nan

        # keeping only patients that have seizures
        # self.df = self.df[self.df['age.onset'].notnull()]

        # keeping only those with missense mutations
        # self.df = self.df[self.df['mut.function'] == 0]
        # changing the code -2, meaning others, by 8
        # self.df.loc[self.df["seizure.type.onset"] == -2, "seizure.type.onset"] = 8
        # self.df["aa.change.position"] = self.df["aa.change.position"].astype(int)



main()
