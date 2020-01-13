from cdt.data import CausalPairGenerator
import numpy as np

# generator = CausalPairGenerator('linear')
# data, labels = generator.generate(100, npoints=500)
# # generator.to_csv('generated_pairs')
# d = data.values

# from cdt.causality.pairwise import NCC
from CausalDiscuveryToolboxClone.NCC import NCC
import networkx as nx
import matplotlib.pyplot as plt
from cdt.data import load_dataset
from sklearn.model_selection import train_test_split

from scipy.special import expit

data, labels = load_dataset('tuebingen')
d0 = data.values[0]

X_tr, X_te, y_tr, y_te = train_test_split(data, labels, train_size=.5)

obj = NCC()
obj.fit(X_tr, y_tr, 50, learning_rate=1e-2)
# This example uses the predict() method
logits = obj.predict(X_te)
output = expit(logits.values)
# This example uses the orient_graph() method. The dataset used
# can be loaded using the cdt.data module
data2, graph2 = load_dataset("sachs")
output2 = obj.orient_graph(data2, nx.Graph(graph2))
# To view the directed graph run the following command
nx.draw_networkx(output2, font_size=8)
plt.show()

a = 0
