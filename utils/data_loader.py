import numpy as np
import os


def load_data(path: str, name: str, array_names=('data',)):
    loaded = np.load(os.path.join(path, name + '.npz'), allow_pickle=True)
    return [loaded[arr_name] for arr_name in array_names]


def convert_numpy_object_to_list_array(arr):
    return [np.stack(a) for a in arr]


def load_correct(path: str, name: str, array_names=('data', 'labels')):
    data, labels = load_data(path, name, array_names)
    data = convert_numpy_object_to_list_array(data)
    return data, labels.reshape(-1, 1)


def create_labels(size, val):
    return np.zeros(size) + val


if __name__ == '__main__':
    data_causal = load_data(os.path.join(os.path.dirname(os.getcwd()), 'Data'), 'temp_causal')[0]
    data_confounded = load_data(os.path.join(os.path.dirname(os.getcwd()), 'Data'), 'temp_confounded')[0]
    a = 0
