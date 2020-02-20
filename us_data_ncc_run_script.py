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
import os
from scipy.special import expit
from utils.data_loader import load_correct

# data, labels = load_dataset('tuebingen')
# data, labels = functions.swap_cause_effect(data, labels)
data, labels = load_correct(os.path.join(os.getcwd(), 'Data'), 'temp')
X_tr, X_te, y_tr, y_te = train_test_split(data, labels, train_size=.9)

obj = NCC()
obj.fit(X_tr, y_tr, 50, learning_rate=1e-2, us=True)
# This example uses the predict() method
logits = obj.predict(X_te)
output = expit(logits.values)
