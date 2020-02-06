# from cdt.causalityrwise import NCC
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
    x = np.dot(w, np.random.normal(loc=mu, scale=sd, size=(k_i, m)))
    return (x - x.mean()) / x.std()


def reduce_support(f, support):
    def supported(*args):
        x = args[0]
        y = f(x)
        cond = (x > support[1]) | (x < support[0])
        y[cond] = 0
        return y

    return supported


def create_mechanism(a_knots, b_knots, support):
    f = PchipInterpolator(a_knots, b_knots)
    return reduce_support(f, support)


def generate_noiseless_effect(f, cause):
    effect = f(cause)
    effect = (effect - effect.mean()) / effect.std()
    return effect


def generator(n, m):
    # n = 20
    # m = 30

    r = 5 * np.random.random(n)
    s = 5 * np.random.random(n)
    k = np.random.randint(1, 6, n)
    d = np.random.randint(4, 6, n)
    v = 5 * np.random.random(n)
    S = []

    for i in range(n):
        x_i = draw_cause(k[i], r[i], s[i], m)
        sd_i = x_i.std()
        support_i = [x_i.min() - sd_i, x_i.max() + sd_i]
        x_i_knots = np.linspace(*support_i, d[i])
        y_i_knots = np.random.normal(0., 1., d[i])
        f_i = create_mechanism(x_i_knots, y_i_knots, support_i)
        y_i = generate_noiseless_effect(f_i, x_i)

        e_i = np.random.normal(0., v[i], m)
        # v_x_knots = np.linspace(*support_i, d[i])
        v_y_knots = np.random.uniform(0, 5, d[i])
        v_spline = create_mechanism(x_i_knots, v_y_knots, support_i)
        v_i = v_spline(x_i)
        noise_i = e_i * v_i
        y_noisy = y_i + noise_i
        y_noisy = (y_noisy - y_noisy.mean()) / y_noisy.std()
        S.append(np.vstack((x_i, y_noisy)).T)
    # S = np.array(S)
    return S


def attach_label(d, cause_first=True):
    if cause_first:
        return d, 0
    else:
        return d, 1


def attach_labels_to_data(data):
    return tuple(map(attach_label, data))


def flip_labels(d):
    (d1, d2), l = d
    return (d1, d2), 1 - l


def flip_data(data_labels):
    return list(map(flip_labels, data_labels))


if __name__ == '__main__':
    n1 = 20
    m1 = 7
    data1 = generator(n1, m1)
    labels = np.ones((n1, 1))
    labeled_data = np.array(attach_labels_to_data(data1))
    # print(list(zip(*labeled_data)))
    ready = list(zip(*labeled_data))
    filpped_data = flip_data(labeled_data)
    print(filpped_data)
