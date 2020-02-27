import numpy as np
from scipy.interpolate import UnivariateSpline as sp
from sklearn.preprocessing import scale
from sklearn.mixture import GaussianMixture

np.random.seed(0)


def rp(k, s, d):
    return np.hstack((np.vstack([si * np.random.randn(k, d) for si in s]),
                      2 * np.pi * np.random.rand(k * len(s), 1))).T


def f1(x, w):
    return np.cos(np.dot(np.hstack((x, np.ones((x.shape[0], 1)))), w))


def f2(x, y, z):
    return np.hstack((f1(x, wx).mean(0), f1(y, wy).mean(0), f1(z, wz).mean(0)))


def cause(n, k, p1, p2):
    g = GaussianMixture(k)
    g.means_ = p1 * np.random.randn(k, 1)
    g.covars_ = np.power(abs(p2 * np.random.randn(k, 1) + 1), 2)
    g.weights_ = abs(np.random.rand(k, 1))
    g.weights_ = g.weights_ / sum(g.weights_)
    return scale(g.sample(n))


def noise(n, v):
    return v * np.random.rand(1) * np.random.randn(n, 1)


def mechanism(x, d):
    g = np.linspace(min(x) - np.std(x), max(x) + np.std(x), d)
    return sp(g, np.random.randn(d))(x.flatten())[:, np.newaxis]


def pair(n=1000, k=3, p1=2, p2=2, v=2, d=5):
    x = cause(n, k, p1, p2)
    return (x, scale(scale(mechanism(x, d)) + noise(n, v)))


def pairset(N):
    z1 = np.zeros((N, 3 * wx.shape[1]))
    z2 = np.zeros((N, 3 * wx.shape[1]))
    for i in range(N):
        (x, y) = pair()
        z1[i, :] = f2(x, y, np.hstack((x, y)))
        z2[i, :] = f2(y, x, np.hstack((y, x)))
    return (np.vstack((z1, z2)), np.hstack((np.ones(N), -np.ones(N))).ravel())


# generate random features
wx = rp(333, [0.2, 2, 20], 1)
wy = rp(333, [0.2, 2, 20], 1)
wz = rp(333, [0.2, 2, 20], 2)

# generate training data
print('generating training data...')
(x, y) = pairset(10000)
# (x_te,y_te,m_te) = tuebingen()

# load test data
(pairs, num_features, num_pairs) = boston_housing()
