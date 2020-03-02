# generator = CausalPairGenerator('linear')
# data, labels = generator.generate(100, npoints=500)
# # generator.to_csv('generated_pairs')
# d = data.values

# from cdt.causality.pairwise import NCC
from CausalDiscuveryToolboxClone.Models.NCC import NCC
import networkx as nx
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


def create_data(folder_path, file_name, label_val):
    # data = load_data(path.join(getcwd(), 'Data'), 'temp_causal')[0]
    data = load_data(folder_path, file_name)[0]
    labels = create_labels(data.shape[0], label_val)
    return data, labels


def get_network(filename='', freeze_encoder=False):
    obj = NCC()
    if filename:  # transfer learning
        obj.load_model(path.join(getcwd(), 'Models'), file_path=filename + '.pth')
    else:
        obj.get_model()
    if freeze_encoder:
        obj.freeze_weights('encoder')
    return obj


def train(obj, train_data, train_labels, validation_data, validation_labels, epochs=10, learning_rate=1e-4,
          optimizer='rms'):
    obj.create_loss(learning_rate, optimizer=optimizer)
    logs = obj.train(train_data, train_labels, validation_data, validation_labels, epochs=epochs, batch_size=32, )
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

def main(args):
    # load data
    data, labels = create_data(path.join(getcwd(), 'Data'), args.file_model, 0)

    # split data
    X_tr, X_test, y_tr, y_test = split_data(data, labels)
    X_tr, X_val, y_tr, y_val = split_data(X_tr, y_tr)

    # load/initialize network
    network = get_network()

    # train network
    logged_values = train(network, X_tr, y_tr, X_val, y_val, epochs=args.epochs, )

    make_plots(logged_values, model_name='temp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-size', default=500)
    parser.add_argument('-num_effects', default=2)
    parser.add_argument('-save', default=True)
    parser.add_argument('-file_model', default='temp_confounded')
    parser.add_argument('-epochs', default=3)
    arguments = parser.parse_args()
    main(arguments)
