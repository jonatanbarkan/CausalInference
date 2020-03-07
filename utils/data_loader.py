import numpy as np
import os
from cdt.data import load_dataset

# TODO: check labels (in load_cdt_dataset)


def load_cdt_dataset(name='tuebingen'):
    df_data, df_labels = load_dataset(name)
    data = [np.vstack((df_data.iloc[i, 0], df_data.iloc[i, 1])) for i in range(len(df_data))]
    labels = df_labels.values.squeeze()
    return data, labels

def load_data(path: str, name: str, array_names=('data',)):
    loaded = np.load(os.path.join(path, name + '.npz'), allow_pickle=True)
    # return [loaded[arr_name] for arr_name in array_names]
    return [convert_numpy_object_to_list_array(loaded[arr_name]) for arr_name in array_names]


def convert_numpy_object_to_list_array(arr):
    return [np.stack(a) for a in arr]


def load_correct(path: str, name: str, array_names=('data', )):
    # data, labels = load_data(path, name, array_names)
    data = load_data(path, name, array_names)
    data = convert_numpy_object_to_list_array(data)
    return data, np.zeros(data.shape[0])
    # return data, labels.reshape(-1, 1)

def create_labels(size, val):
    return np.zeros(size) + val


def load_splitted_data(folder_path, file_name, names=('data', 'labels', 'vec_unstack_idxes')):
    loaded = np.load(os.path.join(folder_path, file_name + '.npz'), allow_pickle=True)
    X_stacked, y, vec_unstack_idxes = [loaded[name] for name in names]
    X = [X_stacked[:, vec_unstack_idxes[i - 1]:vec_unstack_idxes[i]] for i in range(1, vec_unstack_idxes.size)]
    return X, y


if __name__ == '__main__':
    data_causal = load_data(os.path.join(os.path.dirname(os.getcwd()), 'Data'), 'temp_causal')[0]
    data_confounded = load_data(os.path.join(os.path.dirname(os.getcwd()), 'Data'), 'temp_confounded')[0]
    a = 0
