import argparse
import json
import os
from datetime import datetime
from itertools import product

import jsbeautifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from CausalDiscuveryToolboxClone.Models.NCC import NCC
from utils.data_loader import load_split_data
from utils.visualization import make_separate_plots


def save_json(output_path, model_name, **kwargs):
    output_path = os.path.join(output_path, model_name + '.json')
    with open(output_path, "w", encoding="utf8") as f:
        opts = jsbeautifier.default_options()
        opts.indent_size = 4
        f.write(jsbeautifier.beautify(json.dumps(kwargs), opts))


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
    split_data_path = os.path.join(os.getcwd(), "SplitData")
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(jsons_path, exist_ok=True)
    os.makedirs(split_data_path, exist_ok=True)

    name = '_'.join([FLAGS.data_file_1, FLAGS.data_file_2])
    X_tr, y_tr = load_split_data(split_data_path, name + '_train')
    X_val, y_val = load_split_data(split_data_path, name + '_val')

    # load/initialize network
    network = get_network(FLAGS.loaded_model_name, FLAGS.freeze_encoder, FLAGS.num_effects,
                          n_hiddens=FLAGS.n_hiddens,
                          kernel_size=3,
                          dropout_rate=FLAGS.dropout_rate,
                          additional_num_hidden_layers=FLAGS.additional_num_hidden_layers)

    # train network
    network.create_loss(learning_rate=FLAGS.learning_rate, optimizer=FLAGS.optimizer)
    try:
        network.train(X_tr, y_tr, X_val, y_val, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, )
    except KeyboardInterrupt:
        pass

    logged_values = network.get_log_dict()

    make_separate_plots(logged_values, plot_path=plots_path, model_name=FLAGS.save_model_name)
    save_model(network, folder_path=model_path, name=FLAGS.save_model_name)
    save_json(jsons_path, FLAGS.save_model_name, **vars(FLAGS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_effects', default=1, choices={1, 2})
    parser.add_argument('-data_file_1', default='medium_2_causal')
    parser.add_argument('-data_file_2', default='')
    # parser.add_argument('-data_file_1', default='medium_3_causal')
    # parser.add_argument('-data_file_2', default='medium_3_confounded')
    parser.add_argument('-save_model_name', default='large_experiment_1_model')
    parser.add_argument('-epochs', default=3, type=int)
    parser.add_argument('-loaded_model_name', default='')
    parser.add_argument('-freeze_encoder', default=0, choices={0, 1})

    parser.add_argument('-learning_rate', default=1e-4)
    parser.add_argument('-optimizer', default='rms', choices={'rms', 'adam', 'momentum'})
    parser.add_argument('-batch_size', default=16, choices={8, 16, 32, 64})

    parser.add_argument('-n_hiddens', default=100, choices={50, 100, 500})
    parser.add_argument('-dropout_rate', default=0., choices={0., 0.1, 0.25, 0.3})
    parser.add_argument('-additional_num_hidden_layers', default=0, choices={0, 1})

    arguments = parser.parse_args()

    if arguments.num_effects == 1:
        arguments.data_file_1 = 'large_3_causal'
        arguments.data_file_2 = ''
    elif arguments.num_effects == 2:
        arguments.data_file_1 = 'large_4_causal'
        arguments.data_file_2 = 'large_4_confounded'

    run_grid = False
    save_model_name = arguments.save_model_name
    if run_grid:
        learning_rates = [0.01]
        optimizers = ['rms', 'momentum']
        additional_num_hidden_layers = [0, 1]
        dropout_rates = [0.0, 0.1, 0.25, 0.3]
        num_hiddens = [50, 100, 500]
    else:
        learning_rates = [1e-2]
        optimizers = ['rms', 'momentum']
        additional_num_hidden_layers = [0]
        dropout_rates = [0.0]
        num_hiddens = [50, 100, 500]
    grid = product(num_hiddens, learning_rates, optimizers, additional_num_hidden_layers, dropout_rates)
    print('lr, opt, add_layers, p, n_hidd:')
    for n_hidd, lr, opt, add_layers, p in grid:
        print(lr, opt, add_layers, p, n_hidd)
        arguments.learning_rate = lr
        arguments.optimizer = opt
        arguments.additional_num_hidden_layers = add_layers
        arguments.dropout_rate = p
        arguments.n_hiddens = n_hidd
        arguments.save_model_name = save_model_name + datetime.now().strftime('_%y%m%d_%H%M%S')
        run_model(arguments)
