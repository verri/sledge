"""
Python package `sledge` semantic evaluation of clustering results.
"""

import pandas as pd
import numpy as np


def sledge_descriptors(X, labels, minimum_support=0.0):
    """
    Parameters
    ----------
    X : array_like
        TODO...
    labels: list of ints
        TODO...
    minimum_support: float
        Minimum support (percentage)
    """

    nclusters = max(labels) + 1

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])

    # 1-itemsets, for greater k
    support = X.groupby(pd.Series(labels)).apply(np.mean)
    print(support.transpose())

    # TODO: particularization and support threshold

    return support


def sledge_score(X, labels, aggregation='harmonic', return_descriptors=False):

    nclusters = max(labels) + 1
    descriptors = sledge_descriptors(X, labels)

    return [0 for _ in range(nclusters)]
