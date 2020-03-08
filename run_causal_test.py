import numpy as np
import pandas as pd
from utils.data_loader import load_cdt_dataset, load_split_data
from us_data_ncc_run_script import get_network, make_separate_plots
import os


def create_results(dict_model_names, device='cpu'):
    # test : ['tubingen', 'synthesis']
    # kind : ['causal', 'confounded']
    d = {'model_name': [('test', 'kind')]}
    results = {
        'kind': ['causal', 'confounded', '...'],
        'test': ['data'],
        'model name': [],
        'accuracy': [],
        'symmetry': []
    }
    split_data_path = os.path.join(os.getcwd(), "SplitData")
    os.makedirs(split_data_path, exist_ok=True)
    figs_for_projects_path = os.path.join(os.getcwd(), "figs_for_projects")
    os.makedirs(figs_for_projects_path, exist_ok=True)

    test_set_funcs = {'tubingen': load_cdt_dataset, 'causal': load_split_data,
                      'confounded new': load_split_data, 'confounded load': load_split_data,
                      'confounde freeze': load_split_data}

    for model_name in dict_model_names.keys():
        network = get_network(model_name)
        for test, kind in dict_model_names[model_name]:
            load_test_set = test_set_funcs[test]
            X_test, labels_test = load_test_set(folder_path=split_data_path, file_name=model_name + '_test')
            err_total, _, _, symmetry_check = network.compute_values(X_test, labels_test, device=device)
            results['model name'] = model_name
            results['accuracy'].append(1 - err_total)
            results['symmetry'].append(symmetry_check)
            results['kind'].append(kind)
            results['test'] = test

    df_results = pd.DataFrame(results)
    full_path = os.path.join(figs_for_projects_path, 'table_results.csv')
    df_results.to_csv(full_path, index=False)
    return results


if __name__ == '__main__':
    split_data_name = 'medium_1_causal__test'
    model_name = 'medium_experiment_1_model_200308_145006'
    dict_model_names = {'medium_experiment_1_model_200308_145006': [('medium_1_causal__test', 'causal'),
                                                                    ('medium_1_causal__test', 'tubingen')]}

# loaded_model_name = 'small_experiment_4_model_200308_115124'
# X_test, labels_test = load_cdt_dataset()
# a = 0
#
# network = get_network(loaded_model_name, )
#
# # train network
# # make_plots(logged_values, plot_path=plots_path, model_name=FLAGS.save_model_name)
# make_separate_plots(logged_values, plot_path=plots_path, model_name=FLAGS.save_model_name)
