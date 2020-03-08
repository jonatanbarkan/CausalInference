import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import json

sns.set_style('whitegrid')
import os


def concatenate(dataframe):
    pd.concat([dataframe['epochs'], dataframe['causal_train']], axis=1)


path = os.path.join(os.getcwd(), 'Results')
models = os.listdir(path)
experiment = 'small_experiment'
model_hp = {}
for model in models:
    if model.startswith(experiment):
        try:
            with open(os.path.join(os.getcwd(), 'Jsons', model + '.json'), 'r') as f:
                hp = json.load(f)
                model_hp[model] = hp
        except FileNotFoundError:
            pass

dfs = []
for (dirpath, dirnames, filenames) in os.walk(path):
    for file in filenames:
        if file.endswith('.csv') and file.startswith(experiment):
            df = pd.read_csv(os.path.join(dirpath, file), skiprows=[1])
            if len(df) > 6:
                df = df.rename(
                    columns={k: k + '_train' if i % 2 == 1 else k.replace('.1', '') + '_validation' for (i, k) in
                             enumerate(df.columns)})
                df = df.rename(columns={'Unnamed: 0_validation': 'epochs'})
                name = file.replace('_accuracy.csv', '')
                minidx = df.total_validation.argmin()
                model_hp[name]['max_validation_accuracy'] = 1 - df.total_validation.iat[minidx]
                model_hp[name]['first_total_validation_error'] = 1 - df.total_validation.iat[0]
                model_hp[name]['epoch'] = df.epochs.iloc[minidx]
                model_hp[name]['train_error'] = df.total_train.iat[minidx]
                model_hp[name]['symmetry_success'] = df.symmetry_validation.iat[minidx]
                model_hp[name]['success_average'] = (df.symmetry_validation.iat[minidx] + 1 - df.total_validation.iat[minidx]) / 2
                # concatenate(df)
                # dfs.append(df)
            else:
                del model_hp[file.replace('_accuracy.csv', '')]

hyper_df = pd.DataFrame(model_hp).T
# hyper_df.reset_index(level=0, inplace=True)
hyper_df.sort_values(by=['symmetry_success'], inplace=True)
sns.scatterplot(x='max_validation_accuracy', y='symmetry_success', data=hyper_df, hue='learning_rate')
plt.savefig(experiment + '_accuracy_vs_symmetry_lr')
plt.close()
sns.scatterplot(x='max_validation_accuracy', y='symmetry_success', data=hyper_df, hue='dropout_rate')
plt.savefig(experiment + 'accuracy_vs_symmetry_dropout')
plt.close()
sns.scatterplot(x='max_validation_accuracy', y='symmetry_success', data=hyper_df, hue='n_hiddens')
plt.savefig(experiment + 'accuracy_vs_symmetry_hidden_neurons')
plt.close()

with open('best_symmetry.json', 'w') as f:
    json.dump(hyper_df.sort_values(by=['symmetry_success']).tail().save_model_name.tolist(), f)
with open('best_validation_accuracy.json', 'w') as f:
    json.dump(hyper_df.sort_values(by=['max_validation_accuracy']).tail().save_model_name.tolist(), f)
with open('best_average_accuracy.json', 'w') as f:
    json.dump(hyper_df.sort_values(by=['success_average']).tail().save_model_name.tolist(), f)
a = 0
