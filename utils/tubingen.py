import pandas as pd
import numpy as np
import os
from utils.data_split_generation import save_sub_dataset


def create_df_pair_label(folder_Tubingen, file_name='pairmeta'):
    file_path = os.path.join(folder_Tubingen, file_name + '.txt')
    df = pd.read_csv(file_path, sep="\t", header=None)
    df = df[0].str.split(expand=True)
    df.columns = ['pair', 'start A', 'end A', 'start B', 'end B', 'weight']
    df['pair'] = df['pair'].map(lambda el: f'pair{el}')
    df = df.astype({'pair': str, 'start A': int, 'end A': int, 'start B': int, 'end B': int, 'weight': float})
    rel_df = df.loc[(df['end A'] - df['start A']) + (df['end B'] - df['start B']) == 0]
    # rel_df.insert()['label'] = 1
    rel_df.insert(rel_df.shape[1], 'label', 1)
    rel_df.loc[(rel_df['end A'] < rel_df['start B']), 'label'] = 0
    return rel_df[['pair', 'label']]


def create_cause_effect_objects(df):
    res = []
    header = ['A', 'B', 'label']
    data = []
    labels = []
    for (pair_id, label) in df.itertuples(index=False):
        if pair_id == 'pair0081' or pair_id == 'pair0082' or pair_id == 'pair0083':  # has more than two columns
            continue
        X = np.loadtxt(os.path.join(folder_pairs, f'{pair_id}.txt'))
        X_normalized = (X - X.mean(0)) / X.std(0)
        res.append([X_normalized[:, 0], X_normalized[:, 1], label])
        data.append(X.T)
        labels.append(label)
    labels = np.array(labels)
    return res, header, data, labels

def save_df(res, header, folder_name, file_name):
    df_cause_effect = pd.DataFrame(res, columns=header)
    df_cause_effect.to_csv(os.path.join(folder_name, f'{file_name}.csv'), index=False)


if __name__ == '__main__':
    folder_Tubingen = os.path.join(os.path.dirname(os.getcwd()), 'Tubingen')
    folder_pairs = os.path.join(folder_Tubingen, 'pairs')
    folder_Data = os.path.join(os.path.dirname(os.getcwd()), 'Data')
    df_pair_label = create_df_pair_label(folder_Tubingen)
    res, header, data, labels = create_cause_effect_objects(df_pair_label)
    save_df(res, header, folder_Data, 'Tubingen_cause_effect_pairs')
    save_sub_dataset(data, labels, folder_Data, 'Tubingen')



