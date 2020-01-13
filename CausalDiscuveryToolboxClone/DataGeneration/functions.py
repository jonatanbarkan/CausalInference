import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def swap_cause_effect(data: pd.DataFrame, labels: pd.DataFrame, percent=0.5, values=(1., -1)):
    # swap 'A' and 'B' for %percent of the rows and the flip the labels respectively
    assert 0. <= percent <= 1.
    X_tr, X_te, y_tr, y_te = train_test_split(data, labels, train_size=percent)
    invcols = list(reversed(X_tr.columns))
    X_tr.reindex(columns=invcols)
    if values == (1., -1):
        y_tr.iloc[:, 0] = y_tr.iloc[:, 0].map(lambda val: val * -1.)
    else:
        y_tr.iloc[:, 0] = y_tr.iloc[:, 0].map(lambda val: 1. - val)
    data = pd.concat([X_tr, X_te])
    labels = pd.concat([y_tr, y_te])
    return data, labels
