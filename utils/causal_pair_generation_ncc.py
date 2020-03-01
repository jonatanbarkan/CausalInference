# from cdt.causality.pairwise import NCC
from CausalDiscuveryToolboxClone.Models.NCC import NCC
import networkx as nx
import matplotlib.pyplot as plt
from cdt.data import load_dataset
from sklearn.model_selection import train_test_split
from CausalDiscuveryToolboxClone.DataGeneration import functions
import scipy
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline, UnivariateSpline
import numpy as np
from scipy.special import expit
import os
import argparse

np.random.seed(0)


def draw_mixture_weights(k_i):
    mw = np.abs(np.random.standard_normal(k_i))
    return mw / np.sum(mw)


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


# def create_noise_mechanism(a_knots, b_knots, support):
#     f = UnivariateSpline(a_knots, b_knots)
#     return reduce_support(f, support)


def generate_noiseless_effect(f, cause):
    effect = f(cause)
    effect = (effect - effect.mean()) / effect.std()
    return effect


class CauseEffectPairs:
    def __init__(self, dataset_size, num_effects=1, dist_size_vec=None, r=None, s=None, k=None):
        self.dataset_size = dataset_size
        self.num_effects = num_effects
        # self.dist_size_vec = np.random.randint(100, 1500, dataset_size) if dist_size_vec is None else dist_size_vec
        self.dist_size_vec = 200 * np.ones(dataset_size, dtype=np.int32)
        self.r = 5 * np.random.random(dataset_size) if r is None else r
        self.s = 5 * np.random.random(dataset_size) if s is None else s
        self.k = np.random.randint(1, 6, dataset_size) if k is None else k

    def get_dist_size_vec(self):
        return self.dist_size_vec

    def get_r(self):
        return self.r

    def get_s(self):
        return self.s

    def get_k(self):
        return self.k

    def create_cause(self):
        n = self.dataset_size
        m = self.dist_size_vec
        r = self.r
        s = self.s
        k = self.k
        cause = []
        for i in range(n):
            cause_val = draw_cause(k[i], r[i], s[i], m[i])
            # cause.append(cause_val.reshape((1,  -1)))
            cause.append(cause_val)
        return cause

    def create_effect(self, cause):
        n = self.dataset_size
        m = self.dist_size_vec
        supports = [[cause_val.min() - cause_val.std(), cause_val.max() + cause_val.std()] for cause_val in cause]
        d = np.random.randint(4, 6, n)
        v = 5 * np.random.random(n)
        effect = []
        for i in range(n):
            cause_i_knots = np.linspace(*supports[i], d[i])
            effect_i_knots = np.random.normal(0., 1., d[i])
            f_i = create_mechanism(cause_i_knots, effect_i_knots, supports[i])
            tmp_effect_i = generate_noiseless_effect(f_i, cause[i])

            # e_i = np.random.normal(0., v[i], m[i]).reshape((1,  -1))
            e_i = np.random.normal(0., v[i], m[i])
            e_i__knots = np.random.uniform(0, 5, d[i])
            g_i = create_mechanism(cause_i_knots, e_i__knots, supports[i])
            v_i = g_i(cause[i])
            noise_i = e_i * v_i
            effect_i = tmp_effect_i + noise_i
            effect_i = (effect_i - effect_i.mean()) / effect_i.std()
            effect.append(effect_i)
        return effect

    def create_cause_effect_pairs(self):
        cause_effect_pairs = []
        cause = self.create_cause()
        for _ in range(self.num_effects):
            effect = self.create_effect(cause)
            cause_effect_pairs.append((cause, effect))

        return cause_effect_pairs


def save_causal(ce_pairs, folder_path, file_name, *args):
    cause, effect = ce_pairs[0]
    data = np.array([[cause[i], effect[i]] for i in range(len(cause))])
    # labels = np.zeros(data.shape[0])
    np.savez_compressed(os.path.join(folder_path, file_name), data=data)


def save_confounded(ce_pairs, folder_path, file_name, *args):
    n, m, r, s, k = args
    # effect_x, effect_y = [ce_pair[1] for ce_pair in ce_pairs]
    # z_cause, effect_x, effect_y = [ce_pair[1] for ce_pair in ce_pairs]
    cause_z, effect_x, effect_y = ce_pairs[0][0], ce_pairs[0][1], ce_pairs[1][1]
    # data_confounded = np.array(list(map(lambda i: np.hstack((effect_x[i], effect_y[i])), range(len(effect_x)))))
    data_confounded = np.array([[cause_z[i], effect_x[i], effect_y[i]] for i in range(len(effect_x))])
    # labels_confounded = np.zeros(len(effect_x))
    # cause_effect_pair = CauseEffectPairs(n, num_effects=1, dist_size_vec=m, r=r, s=s, k=k)
    # effect = cause_effect_pair.create_effect(effect_x)
    # cause, effect = cause_effect_pair.create_cause_effect_pairs()[0]
    # data_causal = np.array([[cause[i], effect[i]] for i in range(len(cause))])
    # labels_causal = np.ones(data_causal.shape[0])
    np.savez_compressed(os.path.join(folder_path, file_name), data=data_confounded)


def create_pairwise_dataset(args):
    n = args.size
    num_effects = args.num_effects
    assert num_effects in [1, 2]
    save = args.save
    file_name = args.file_name + '_{}'.format('confounded' if num_effects == 2 else 'causal')
    CE_pairs = CauseEffectPairs(n, num_effects)
    ce_pairs = CE_pairs.create_cause_effect_pairs()
    r, s, k = CE_pairs.get_r(), CE_pairs.get_s(), CE_pairs.get_k()
    m = CE_pairs.get_dist_size_vec()
    save_dict = {1: save_causal, 2: save_confounded}
    folder_path = os.path.dirname(os.getcwd())
    data_folder_path = os.path.join(folder_path, 'Data')
    if save:
        os.makedirs(data_folder_path, exist_ok=True)
        if num_effects in save_dict:
            save_data = save_dict[num_effects]
            save_data(ce_pairs, data_folder_path, file_name, n, m, r, s, k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-size', default=500)
    parser.add_argument('-num_effects', default=2)
    parser.add_argument('-save', default=True)
    parser.add_argument('-file_name', default='temp')
    arguments = parser.parse_args()
    create_pairwise_dataset(arguments)
    # data, labels = load_dataset('tuebingen')
    # data, labels = functions.swap_cause_effect(data, labels)
