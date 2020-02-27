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
from utils.data_loader import load_correct
from utils.blah import compare_models

# data, labels = load_dataset('tuebingen')
# data, labels = functions.swap_cause_effect(data, labels)
data, labels = load_correct(path.join(getcwd(), 'Data'), 'full_xy')
X_tr, X_val_test, y_tr, y_val_test = train_test_split(data, labels, train_size=.8)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=.5)

# obj = NCC()
# obj.fit(X_tr, y_tr, 50, learning_rate=1e-2, us=True)
NCC_parts = ['encoder', 'classifier', None]
freeze_part = 'encoder'
obj = NCC()
epochs = 10
model_type = "causal_NCC"
error_dict, symmetry_check_dict = obj.train_and_validate(X_tr, y_tr, X_val, y_val, epochs=epochs,
                                                         batch_size=32, learning_rate=1e-3)

make_plots(error_dict, symmetry_check_dict, epochs, model_type, step=1)
obj.save_model(path.join(getcwd(), 'Models'), file_path=f'model_type={model_type}_freeze={freeze_part}.pth')

loaded_obj = NCC()
loaded_obj.load_model(path.join(getcwd(), 'Models'), file_path=f'model_type={model_type}_freeze={freeze_part}.pth')
loaded_obj.freeze_weights(freeze_part)
error_dict2, symmetry_check_dict2 = loaded_obj.train_and_validate(X_tr, y_tr, X_val, y_val, epochs=epochs,
                                                                  batch_size=32, learning_rate=1e-2)

compare_models(obj, loaded_obj)

# This example uses the predict() method
# logits = obj.predict(X_test)
# output = expit(logits.values)
