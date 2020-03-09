import numpy as np
import pandas as pd
from utils.data_loader import load_cdt_dataset, load_split_data
from us_data_ncc_run_script import get_network, make_separate_plots
import os
import json


class TableResults:
    def __init__(self, file_name, folder_split_data='SplitData', folder_figs_for_projects='figs_for_projects',
                 folder_data='Data'):
        self.file_name = file_name
        with open(f'{file_name}.json') as lst_file_json:
            json_models = json.load(lst_file_json)
            self.json_models = json_models
        self.table = []
        self.header = ['kind', 'test', 'model', ' accuracy', 'symmetry']
        self.split_data_path = self.get_path_file(folder_split_data)
        self.figs_for_projects_path = self.get_path_file(folder_figs_for_projects)
        self.data_path = self.get_path_file(folder_data)
        self.dict_models = self.create_model_dict()


    def create_model_dict(self):
        dict_models = dict()
        for json_model in self.json_models:
            model_name = json_model
            model_params = json.load(f'{json_model}.json')
            kind = self.create_model_kind(model_params)
            model_lst_vals = self.create_model_lst_vals(model_params, kind)
            dict_models[model_name] = model_lst_vals
        return dict_models

    @staticmethod
    def get_path_file(folder_name):
        folder_path = os.path.join(os.getcwd(), folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    @staticmethod
    def create_model_kind(m_params):
        nun_effects = m_params["nun_effects"]
        loaded_model_name = m_params["loaded_model_name"]
        freeze_encoder = m_params["freeze_encoder"]
        if nun_effects == 1:
            return 'causal'
        elif nun_effects == 2 and loaded_model_name:
            return 'confounded (load)'
        elif nun_effects == 2 and freeze_encoder:
            return 'confounded (freeze)'
        else:
            return 'confounded (new)'

    def create_model_lst_vals(self, m_params, kind):
        nun_effects = m_params["nun_effects"]
        data_file_1 = m_params["data_file_1"]
        data_file_2 = m_params["data_file_2"]
        test_name = '_'.join([data_file_1, data_file_2])
        if nun_effects == 1:
            return [[nun_effects, kind, 'Tubingen', self.data_path, 'Tubingen'],
                    [nun_effects, kind, test_name.strip('_'), self.split_data_path, '_'.join([test_name, 'test'])]]
        else:
            return [[nun_effects, kind, test_name.strip('_'), self.split_data_path, '_'.join([test_name, 'test'])],
                    [nun_effects, kind, 'Z -> X', self.data_path, '_'.join([data_file_2, 'causal_z_x'])],
                    [nun_effects, kind, 'Z -> Y', self.data_path, '_'.join([data_file_2, 'causal_z_y'])]]

    def create_results(self, device='cpu'):
        dict_models = self.dict_models
        for model_full_name in dict_models.keys():
            model_name = model_full_name.split('model')[0].strip('_')
            for (nun_effects, kind, test_name, folder_path, file_name) in dict_models[model_full_name]:
                network = get_network(model_full_name, num_effect=nun_effects)
                X_test, labels_test = load_split_data(folder_path, file_name)
                err_total, _, _, symmetry_check = network.compute_values(X_test, labels_test, device=device)
                self.table.append([kind, test_name, model_name, 1 - err_total, symmetry_check])

    def save_table(self):
        df = pd.DataFrame(self.table, columns=self.header)
        file_name = os.path.join(self.figs_for_projects_path, f'{self.file_name}.csv')
        df.to_csv(file_name)


if __name__ == '__main__':
    best_files = ['best_average_accuracy', 'best_symmetry', 'best_validation_accuracy']
    for best_file in best_files:
        table_result = TableResults(best_files)
        table_result.create_results(device='cpu')
        table_result.save_table()