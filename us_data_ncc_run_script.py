import argparse
import json
import os

import jsbeautifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

from CausalDiscuveryToolboxClone.Models.NCC import NCC
from utils.causal_pair_generation_ncc import plot_joint_distribution
from utils.data_loader import load_data, create_labels  # load_correct,
from utils.data_loader import load_splitted_data
from utils.visualization import make_plots, make_separate_plots
from itertools import product


def save_json(output_path, model_name, **kwargs):
    output_path = os.path.join(output_path, model_name + '.json')
    with open(output_path, "w", encoding="utf8") as f:
        opts = jsbeautifier.default_options()
        opts.indent_size = 4
        f.write(jsbeautifier.beautify(json.dumps(kwargs), opts))


# def create_data(folder_path, file_name, label_val, **kwargs):
#     # data = load_data(path.join(getcwd(), 'Data'), 'temp_causal')[0]
#     data = load_data(folder_path, file_name)[0]
#     labels = create_labels(len(data), label_val)
#     return data, labels


def get_network(filename='', freeze_encoder=False, num_effect=1, **kwargs):
    assert num_effect in [1, 2]
    obj = NCC()
    if filename:  # transfer learning
        obj.load_model(os.path.join(os.getcwd(), 'Models'), file_path=filename + '.pth')
    else:
        obj.get_model(**kwargs)
    if freeze_encoder:
        obj.freeze_weights('encoder')
    if num_effect == 2:
        obj.anti = False
    return obj


def train(obj, train_data, train_labels, validation_data, validation_labels, epochs=10, learning_rate=1e-4,
          optimizer='rms', **kwargs):
    obj.create_loss(learning_rate, optimizer=optimizer)
    logs = obj.train(train_data, train_labels, validation_data, validation_labels, epochs=epochs, batch_size=16, )
    return logs


def split_data(dat, lab, train_size=0.8):
    return train_test_split(dat, lab, train_size=train_size)


def create_df(pair_1, pair_2, labels=()):
    # pair_1_x, pair_1_y = np.hstack(pair_1[:, 0]), np.hstack(pair_1[:, 1])
    # pair_2_x, pair_2_y = np.hstack(pair_2[:, 0]), np.hstack(pair_2[:, 1])
    pair_1_x, pair_1_y = np.hstack(pair_1)
    pair_2_x, pair_2_y = np.hstack(pair_2)
    aggregated_pair_1 = np.hstack([pair_1_x.reshape(-1, 1), pair_1_y.reshape(-1, 1)])
    aggregated_pair_2 = np.hstack([pair_2_x.reshape(-1, 1), pair_2_y.reshape(-1, 1)])
    aggregated = np.vstack([aggregated_pair_1, aggregated_pair_2])
    df = pd.DataFrame(aggregated, columns=['x', 'y'])
    df['kind'] = [labels[0]] * aggregated_pair_1.shape[0] + [labels[1]] * aggregated_pair_2.shape[0]
    return df


def save_model(obj, folder_path: str, name: str):
    obj.save_model(folder_path, file_path=f'{name}.pth')


def run_model(FLAGS, **kwargs):
    # create save path
    plots_path = os.path.join(os.getcwd(), "Results", FLAGS.save_model_name)
    model_path = os.path.join(os.getcwd(), "Models")
    jsons_path = os.path.join(os.getcwd(), "Jsons")
    splitted_data_path = os.path.join(os.getcwd(), "SplitData")
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(jsons_path, exist_ok=True)
    os.makedirs(splitted_data_path, exist_ok=True)

    name = '_'.join([FLAGS.data_file_1, FLAGS.data_file_2])
    X_tr, y_tr = load_splitted_data(splitted_data_path, name + '_train')
    X_val, y_val = load_splitted_data(splitted_data_path, name + '_val')

    # TODO: moved to split script - replace with load of train and validation sets
    # load data

    # data_folder_path = os.path.join(os.getcwd(), 'Data')
    # data, labels = create_data(data_folder_path, FLAGS.data_file_1, 0)
    # if FLAGS.num_effects == 2:
    #     if not FLAGS.data_file_2:
    #         raise ValueError('no second file inserted')
    #     data2, labels2 = create_data(data_folder_path, FLAGS.data_file_2, 1)
    #
    #     # plot distributions
    #     if kwargs.get('plot_distributions', False):
    #         df = create_df(data, data2, (FLAGS.data_file_1, FLAGS.data_file_2))
    #         plot_joint_distribution(df=df, folder_path=plots_path, name=FLAGS.save_model_name)
    #
    #     data = data + data2
    #     labels = np.hstack((labels, labels2))
    #
    # # split data
    # X_tr, X_test, y_tr, y_test = split_data(data, labels)
    # X_tr, X_val, y_tr, y_val = split_data(X_tr, y_tr)

    # load/initialize network
    network = get_network(FLAGS.loaded_model_name, FLAGS.freeze_encoder, FLAGS.num_effects,
                          n_hiddens=FLAGS.n_hiddens,
                          kernel_size=3,
                          dropout_rate=FLAGS.dropout_rate,
                          additional_num_hidden_layers=FLAGS.additional_num_hidden_layers)

    # train network
    network.create_loss(learning_rate=FLAGS.learning_rate, optimizer=FLAGS.optimizer)
    logged_values = network.train(X_tr, y_tr, X_val, y_val, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, )

    # make_plots(logged_values, plot_path=plots_path, model_name=FLAGS.save_model_name)
    make_separate_plots(logged_values, plot_path=plots_path, model_name=FLAGS.save_model_name)
    save_model(network, folder_path=model_path, name=FLAGS.save_model_name)
    save_json(jsons_path, FLAGS.save_model_name, **vars(FLAGS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_effects', default=1, choices={1, 2})
    # parser.add_argument('-data_file_1', default='final_data_causal_1_causal')
    # parser.add_argument('-data_file_2', default='final_data_confounded_1_confounded')
    parser.add_argument('-data_file_1', default='small_causal_1_causal')
    parser.add_argument('-data_file_2', default='small_confounded_1_confounded')
    parser.add_argument('-save_model_name', default='small_experiment_1_model')
    parser.add_argument('-epochs', default=20, type=int)
    parser.add_argument('-loaded_model_name', default='')
    parser.add_argument('-freeze_encoder', default=0, choices={0, 1})

    parser.add_argument('-learning_rate', default=1e-4)
    parser.add_argument('-optimizer', default='rms', choices={'rms', 'adam'})
    parser.add_argument('-batch_size', default=16, choices={8, 16, 32, 64})

    parser.add_argument('-n_hiddens', default=100, choices={50, 100, 500})
    parser.add_argument('-dropout_rate', default=0., choices={0., 0.1, 0.25, 0.3})
    parser.add_argument('-additional_num_hidden_layers', default=0, choices={0, 1})

    arguments = parser.parse_args()
    run_grid = True
    save_model_name = arguments.save_model_name
    if run_grid:
        learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
        optimizers = ['rms', 'adam']
        additional_num_hidden_layers = [0, 1]
        dropout_rates = [0., 0.1, 0.25, 0.3]
        num_hiddens = [50, 100, 500]
    else:
        learning_rates = [1e-4]
        optimizers = ['rms']
        additional_num_hidden_layers = [0]
        dropout_rates = [0.3]
        num_hiddens = [100]
    grid = product(learning_rates, optimizers, additional_num_hidden_layers, dropout_rates, num_hiddens)
    for lr, opt, add_layers, p, n_hidd in grid:
        arguments.learning_rate = lr
        arguments.optimizer = opt
        arguments.additional_num_hidden_layers = add_layers
        arguments.dropout_rate = p
        arguments.n_hiddens = n_hidd
        arguments.save_model_name = save_model_name + datetime.now().strftime('_%y%m%d_%H%M%S')
        run_model(arguments)
