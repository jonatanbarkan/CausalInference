# from cdt.causality.pairwise import NCC
from CausalDiscuveryToolboxClone.Models.NCC import NCC
import networkx as nx
import matplotlib.pyplot as plt
from cdt.data import load_dataset
from sklearn.model_selection import train_test_split
from CausalDiscuveryToolboxClone.DataGeneration import functions
import scipy
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline
import numpy as np
from scipy.special import expit


# data, labels = load_dataset('tuebingen')
# data, labels = functions.swap_cause_effect(data, labels)


def draw_mixture_weights(k_i):
    mw = np.abs(np.random.standard_normal(k_i))
    return mw / np.sum(mw)


def draw_mechanism():
    pass


def draw_cause(k_i, r_i, s_i, m):
    w = draw_mixture_weights(k_i)
    mu = np.random.normal(0., r_i)
    sd = np.abs(np.random.normal(0., s_i))
    return np.dot(w, np.random.normal(loc=mu, scale=sd, size=(k_i, m)))


n = 5
m = 7

r = 5 * np.random.random(n)
s = 5 * np.random.random(n)
k = np.random.randint(1, 6, n)
d = np.random.randint(4, 6, n)
v = 5 * np.random.random(n)
X = []

for i in range(n):
    x_i = draw_cause(k[i], r[i], s[i], m)
    X.append(x_i)
a = 0
