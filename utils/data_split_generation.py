import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.causal_pair_generation_ncc import plot_joint_distribution
from utils.data_loader import load_data, create_labels


def create_data(folder_path, file_name, label_val, **kwargs):
    # data = load_data(path.join(getcwd(), 'Data'), 'temp_causal')[0]
    data = load_data(folder_path, file_name)[0]
    labels = create_labels(len(data), label_val)
    return data, labels


def create_df(pair_1, pair_2, labels=()):
    pair_1_x, pair_1_y = np.hstack(pair_1)
    pair_2_x, pair_2_y = np.hstack(pair_2)
    aggregated_pair_1 = np.hstack([pair_1_x.reshape(-1, 1), pair_1_y.reshape(-1, 1)])
    aggregated_pair_2 = np.hstack([pair_2_x.reshape(-1, 1), pair_2_y.reshape(-1, 1)])
    aggregated = np.vstack([aggregated_pair_1, aggregated_pair_2])
    df = pd.DataFrame(aggregated, columns=['x', 'y'])
    df['kind'] = [labels[0]] * aggregated_pair_1.shape[0] + [labels[1]] * aggregated_pair_2.shape[0]
    return df


def split_data_to_disc(args, **kwargs):
    name = '_'.join([args.data_file_1, args.data_file_2])
    data_folder_path = os.path.join(os.getcwd(), '..', 'Data')
    split_data_folder_path = os.path.join(os.getcwd(), '..', 'SplitData')
    os.makedirs(split_data_folder_path, exist_ok=True)
    data, labels = create_data(data_folder_path, args.data_file_1, 0)
    if args.data_file_2:
        data2, labels2 = create_data(data_folder_path, args.data_file_2, 1)

        # plot distributions
        if kwargs.get('plot_distributions', True):
            df = create_df(data, data2, (args.data_file_1, args.data_file_2))
            plot_joint_distribution(df=df, folder_path=data_folder_path, name=name)

        data = data + data2
        labels = np.hstack((labels, labels2))

    # split data
    X_tr, X_test, y_tr, y_test = train_test_split(data, labels, train_size=kwargs.get('train_size', 0.8))
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, train_size=kwargs.get('train_size', 0.8))

    np.savez_compressed(os.path.join(split_data_folder_path, name + '_val_data'), data=np.hstack(X_val[:6]))
    np.savez_compressed(os.path.join(split_data_folder_path, name + '_train_data'), data=X_tr)
    np.savez_compressed(os.path.join(split_data_folder_path, name + '_test_data'), data=X_test)

    np.savez_compressed(os.path.join(split_data_folder_path, name + '_train_labels'), labels=y_tr)
    np.savez_compressed(os.path.join(split_data_folder_path, name + '_val_labels'), labels=y_val)
    np.savez_compressed(os.path.join(split_data_folder_path, name + '_test_labels'), labels=y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_file_1', default='small_causal_1_causal')
    parser.add_argument('-data_file_2', default='small_confounded_1_confounded')
    # parser.add_argument('-data_file_2', default='')
    arguments = parser.parse_args()
    split_data_to_disc(arguments)
