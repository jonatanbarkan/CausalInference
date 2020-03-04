# generator = CausalPairGenerator('linear')
# data, labels = generator.generate(100, npoints=500)
# # generator.to_csv('generated_pairs')
# d = data.values

# from cdt.causality.pairwise import NCC
from CausalDiscuveryToolboxClone.Models.NCC import NCC
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from cdt.data import load_dataset
from sklearn.model_selection import train_test_split
from CausalDiscuveryToolboxClone.DataGeneration import functions
import numpy as np
from utils.visualization import make_plots
from os import path, getcwd
from scipy.special import expit
from utils.data_loader import load_data, create_labels  # load_correct,
from utils.blah import compare_models
import argparse
from utils.causal_pair_generation_ncc import plot_joint_distribution


# # data, labels = load_dataset('tuebingen')
# # data, labels = functions.swap_cause_effect(data, labels)
# data = load_data(path.join(getcwd(), 'Data'), 'temp_causal')[0]
#
# X_tr, X_val_test, y_tr, y_val_test = train_test_split(data, labels, train_size=.8)
# X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=.5)
#
# # obj = NCC()
# # obj.fit(X_tr, y_tr, 50, learning_rate=1e-2, us=True)
# NCC_parts = ['encoder', 'classifier', None]
# freeze_part = 'encoder'
# obj = NCC()
# epochs = 10
# model_type = "causal_NCC"
# error_dict, symmetry_check_dict = obj.train_and_validate(X_tr, y_tr, X_val, y_val, epochs=epochs,
#                                                          batch_size=32, learning_rate=1e-3)
#
# make_plots(error_dict, symmetry_check_dict, epochs, model_type, step=1)
# obj.save_model(path.join(getcwd(), 'Models'), file_path=f'model_type={model_type}_freeze={freeze_part}.pth')
#
# loaded_obj = NCC()
# loaded_obj.load_model(path.join(getcwd(), 'Models'), file_path=f'model_type={model_type}_freeze={freeze_part}.pth')
# loaded_obj.freeze_weights(freeze_part)
# error_dict2, symmetry_check_dict2 = loaded_obj.train_and_validate(X_tr, y_tr, X_val, y_val, epochs=epochs,
#                                                                   batch_size=32, learning_rate=1e-2)
#
# compare_models(obj, loaded_obj)


def create_data(folder_path, file_name, label_val, **kwargs):
    # data = load_data(path.join(getcwd(), 'Data'), 'temp_causal')[0]
    data = load_data(folder_path, file_name)[0]
    labels = create_labels(len(data), label_val)
    return data, labels


def get_network(filename='', freeze_encoder=False, num_effect=1):
    assert num_effect in [1, 2]
    obj = NCC()
    if filename:  # transfer learning
        obj.load_model(path.join(getcwd(), 'Models'), file_path=filename + '.pth')
    else:
        obj.get_model()
    if freeze_encoder:
        obj.freeze_weights('encoder')
    if num_effect == 2:
        obj.anti = False
    return obj


def train(obj, train_data, train_labels, validation_data, validation_labels, epochs=10, learning_rate=1e-4,
          optimizer='rms'):
    obj.create_loss(learning_rate, optimizer=optimizer)
    logs = obj.train(train_data, train_labels, validation_data, validation_labels, epochs=epochs, batch_size=16, )
    return logs


# def train_test_val_split(data, labels, train_size=0.8, val_size=0.2):
#     X_tr, X_test, y_tr, y_test = train_test_split(data, labels, train_size=train_size)
#     X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, train_size=val_size)
#     return


def split_data(dat, lab, train_size=0.8):
    return train_test_split(dat, lab, train_size=train_size)


# This example uses the predict() method
# logits = obj.predict(X_test)
# output = expit(logits.values)

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


def save_model(obj, name: str):
    obj.save_model(path.join(getcwd(), 'Models'), file_path=f'Model_{name}.pth')


def main(FLAGS):
    # load data
    folder_path = path.join(getcwd(), 'Data')
    data, labels = create_data(folder_path, FLAGS.data_file_1, 0)
    if FLAGS.num_effects == 2:
        if not FLAGS.data_file_2:
            raise ValueError('no second file inserted')
        data2, labels2 = create_data(folder_path, FLAGS.data_file_2, 1)

        # plot distributions
        if False:
            df = create_df(data, data2, (FLAGS.data_file_1, FLAGS.data_file_2))
            plot_joint_distribution(df=df, path_and_name=path.join(getcwd(), 'Models', FLAGS.save_model_name))

        # data = np.vstack((data, data2))
        data = data + data2
        labels = np.hstack((labels, labels2))

    # split data
    X_tr, X_test, y_tr, y_test = split_data(data, labels)
    X_tr, X_val, y_tr, y_val = split_data(X_tr, y_tr)

    # load/initialize network
    network = get_network(FLAGS.loaded_model_name, FLAGS.freeze_encoder, FLAGS.num_effects)

    # train network
    logged_values = train(network, X_tr, y_tr, X_val, y_val, epochs=FLAGS.epochs, )

    make_plots(logged_values, model_name=FLAGS.save_model_name)
    save_model(network, FLAGS.save_model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-size', default=500)
    parser.add_argument('-num_effects', default=2)
    parser.add_argument('-save', default=True)
    parser.add_argument('-file_model', default='temp_causal')
    parser.add_argument('-data_file_1', default='temp_causal')
    parser.add_argument('-data_file_2', default='temp_confounded')
    parser.add_argument('-save_model_name', default='temp')
    parser.add_argument('-epochs', default=3)
    parser.add_argument('-loaded_model_name', default='')
    parser.add_argument('-freeze_encoder', default=0)
    arguments = parser.parse_args()
    main(arguments)
