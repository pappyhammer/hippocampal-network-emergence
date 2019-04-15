import numpy as np
import pandas as pd
from datetime import datetime
import unidecode
# reg exp
import re


def add_key_to_map_dict(map_dict, key_to_add):
    """
    Take a dict that match a string to a code and key_to_add to the dict using the next int among the max values one
    :param map_dict:
    :param  key_to_add:
    :return:
    """
    values = list(map_dict.values())
    new_value = np.max(values) + 1
    map_dict[key_to_add] = new_value

class Cleaner:
    def __init__(self, df_data):
        self.df_data = df_data
        self.df_clean = df_data
        self.n_lines = self.df_clean.shape[0]
        self.dates_columns = ["date_greffe", "date_naissance"]

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

        # dealing with data format
        for date_name in self.dates_columns:
            self.df_clean[date_name] = pd.to_datetime(self.df_clean[date_name])
            self.df_clean[date_name] = self.df_clean[date_name].dt.strftime('%d/%m/%Y')

        # setting indices
        self.df_clean = self.df_clean.set_index(np.arange(self.n_lines))


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

        # code for each column values
        self.sexe_mapping = {"na": -1, "F": 0, "M": 1}
        self.mapping_dict["sexe"] = self.sexe_mapping

        etiologies_to_map = ["ulcere inflammatoire", "ulcere mecanique", "keratite infectieuse",
                             "insuffisance limbique",
                             "anomalie statique palpebrale", "decompensation bulleuse epitheliale",
                             "destruction aigue surface", "neurotrophique", "BF perforee", "Keratopathie en bandelette",
                             "reconstruction", "refection BF"]
        self.etiology_mapping = {"NA": -1}
        for code, etiology in enumerate(etiologies_to_map):
            self.etiology_mapping[etiology] = code
        self.mapping_dict["etiologie"] = self.etiology_mapping
        # each key is a tuple of reg_ex, and the value is a key to etiology_mapping
        # the key should be lower case
        self.etiology_patterns = {"ulcere.* inflammatoire": "ulcere inflammatoire",
                                  "keratite.*infectieuse": "keratite infectieuse",
                                  "insuffisance limbique": "insuffisance limbique",
                                  "anomalie statique.*brale": "anomalie statique palpebrale",
                                  "decompensation bulleuse.*": "decompensation bulleuse epitheliale",
                                  "destruction aigue surface": "destruction aigue surface",
                                  "neurotrophique": "neurotrophique", "BF perforee": "BF perforee",
                                  "keratopathie.*bandelette": "Keratopathie en bandelette",
                                  "recon.*ction": "reconstruction", "refection.*bf": "refection BF"}

        categorie_nk_to_map = ["infectieuse", "brulure", "diabete", "iatrogenie", "atteinte chronique surface",
                               "centrale"]
        self.categorie_nk_mapping = {"NA": -1}
        for code, etiology in enumerate(categorie_nk_to_map):
            self.categorie_nk_mapping[etiology] = code
        self.mapping_dict["categorie_nk"] = self.categorie_nk_mapping
        # each key is a tuple of reg_ex, and the value is a key to etiology_mapping
        # the key should be lower case
        self.categorie_nk_patterns = {"infect.*":"infectieuse",
                                      "brulure":"brulure", "diabete":"diabete",
                                      "iatrogenie":"iatrogenie",
                                      "atteinte chronique surface":"atteinte chronique surface",
                               "central.*":"centrale"}

        cause_nk_to_map = ["hsv", "base", "kt", "2e kt", "3e kt", "ains", "vzv", "collyres atb", "gougerot"]
        self.cause_nk_mapping = {"NA": -1}
        for code, etiology in enumerate(cause_nk_to_map):
            self.cause_nk_mapping[etiology] = code
        self.mapping_dict["cause_nk"] = self.cause_nk_mapping
        # each key is a tuple of reg_ex, and the value is a key to etiology_mapping
        # the key should be lower case
        self.cause_nk_patterns = {"hsv": "hsv",
                                      "base": "base", "kt": "kt",
                                      "2.*kt": "2e kt",
                                      "3.*kt": "3e kt", "ains": "ains", "vzv": "vzv",
                                      "col.*atb": "collyres atb",
                                  "goug.*rot":"gougerot"}

        etat_corneen_to_map = ["preperforatif", "perfore"]
        self.etat_corneen_mapping = {"NA": -1}
        for code, etiology in enumerate(etat_corneen_to_map):
            self.etat_corneen_mapping[etiology] = code
        self.etat_corneen_cases = {"préperfratif": "preperforatif",
                                   "preperfratif": "preperforatif",
                                   "preforatif": "preperforatif"}

        self.mapping_dict["état cornéen"] = self.etat_corneen_mapping

        self.side_mapping = {"NA": -1, "OG": 0, "OD": 1, "OD_OG": 2}
        # to replace by reg exp
        self.side_special_cases = {"OG PUIS OD EN 07": "OD_OG", "0D": "OD", "OG PUIS OD":  "OD_OG",
                                   "OD + OG": "OD_OG"}
        self.mapping_dict["côté"] = self.side_mapping

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
        self.clean()

    def clean_column_with_reg_exp(self, column_name, map_dict, pattern_dict):
        if self.keep_original:
            index_column = self.df_clean.columns.get_loc(column_name)
            self.df_clean.insert(loc=index_column+1, column=column_name+"_originale",
                                 value=self.df_clean.loc[:, column_name],
                                 allow_duplicates=False)
        for index, cell_text in enumerate(self.df_clean.loc[:, column_name]):
            if pd.isna(cell_text):
                self.df_clean.at[index, column_name] = map_dict["NA"]
                continue
            cell_text = unidecode.unidecode(cell_text)
            cell_text = cell_text.lower()
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
            self.df_clean.insert(loc=index_column+1, column=column_name+"_originale",
                                 value=self.df_clean.loc[:, column_name],
                                 allow_duplicates=False)
        for index, value in enumerate(self.df_clean.loc[:, column_name]):
            if pd.isna(value):
                self.df_clean.at[index, column_name] = -1
            else:
                # removing accentuation or space at the beginning and end of the string
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
                file.write(f"### {list_nk_categories[index]}: {list_nk_categories_nb[index]}\n")
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
            file.write("\n")
            # file.write("\n")
            # file.write("\n")

            # file.write("Number of patients by causes of Neurotrophique categories:")
            # file.write("\n")
            #
            # for nk_category, n_patients_by_nk_cause_dict in nk_category_dict.items():
            #     file.write(f"## {nk_category}:\n")
            #     list_nk_causes = list(n_patients_by_nk_cause_dict.keys())
            #     list_nk_causes_nb = []
            #     for cat in list_nk_causes:
            #         list_nk_causes_nb.append(n_patients_by_nk_cause_dict[cat])
            #
            #     n_patients_by_nk_causes_sorted_indices = np.argsort(list_nk_causes_nb)
            #     for index in n_patients_by_nk_causes_sorted_indices[::-1]:
            #         file.write(f"- {list_nk_causes[index]}: {list_nk_causes_nb[index]}")
            #         file.write("\n")
            #     file.write("\n")
            #     file.write("\n")

    def clean(self):
        super().clean()
        self.clean_column(column_name="sexe", map_dict=self.sexe_mapping)
        self.clean_column(column_name="etat_corneen", map_dict=self.etat_corneen_mapping,
                          special_cases=self.etat_corneen_cases)
        self.clean_column(column_name="cote", map_dict=self.side_mapping, special_cases=self.side_special_cases)
        self.clean_column_with_reg_exp(column_name="etiologie", map_dict=self.etiology_mapping,
                                       pattern_dict=self.etiology_patterns)
        self.clean_column_with_reg_exp(column_name="nb_couches", map_dict=self.nb_couches_mapping,
                                       pattern_dict=self.nb_couches_patterns)
        self.clean_column_with_reg_exp(column_name="categorie_NK", map_dict=self.categorie_nk_mapping,
                                       pattern_dict=self.categorie_nk_patterns)
        self.clean_column_with_reg_exp(column_name="cause_NK", map_dict=self.cause_nk_mapping,
                                       pattern_dict=self.cause_nk_patterns)

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

def main():
    root_path = "/Users/pappyhammer/Documents/academique/these_inmed/"
    path_data = root_path + "these_lucie/"
    path_results = root_path + "these_lucie/clean_data/"

    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")

    original_file_name = "GMA Toulouse.xlsx"
    df_summary = pd.read_excel(path_data + original_file_name, sheet_name=f"Feuil1")
    list_dfs = []
    names_col = None
    n_columns_full = 43
    n_columns_empty = 2
    # going through all the sheets
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

    # print(f"shape: {df_data.shape}")
    # print(f"columns: {df_data.columns}")
    use_multi_cleaner = False

    cleaner_coder_with_original = CleanerCoder(df_data=df_data.copy(), keep_original=True)
    cleaner_coder_without_original = CleanerCoder(df_data=df_data.copy(), keep_original=False)

    if use_multi_cleaner:
        cleaner_multi = CleanerMulti(df_data=df_data.copy())

    # print(f"shape: {cleaner_coder.df_clean.shape}")
    # print(f"columns coder: {cleaner_coder.df_clean.columns}")
    if use_multi_cleaner:
        print(f"shape: {cleaner_multi.df_clean.shape}")
        print(f"columns multi: {cleaner_multi.df_clean.columns}")

    writer = pd.ExcelWriter(f'{path_results}/these_lucie_data_clean_code_with_original_{time_str}.xlsx')
    # df_summary.to_excel(writer, 'summary', index=False)
    cleaner_coder_with_original.df_clean.to_excel(writer, 'data', index=False)
    writer.save()

    writer = pd.ExcelWriter(f'{path_results}/these_lucie_data_clean_code_{time_str}.xlsx')
    # df_summary.to_excel(writer, 'summary', index=False)
    cleaner_coder_without_original.df_clean.to_excel(writer, 'data', index=False)
    writer.save()

    cleaner_coder_without_original.produce_stats(file_name=f'{path_results}/stats_Lucie_{time_str}.txt')

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