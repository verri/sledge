"""
Python package `sledge` semantic evaluation of clustering results.
"""

import pandas as pd
import numpy as np


def semantic_descriptors(X, labels, minimum_support=0.0):
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

    features = X.columns

    # 1-itemsets, for greater k
    support = X.groupby(pd.Series(labels)).apply(np.mean)

    # Particularization
    for feature in features:
        column = np.array(support[feature])

        max_support = np.array([np.max(np.delete(column, i))
                               for i in range(nclusters)])
        mean_support = np.array([np.mean(np.delete(column, i)) for i in
                                 range(nclusters)])

        toremove = column ** 2 <= mean_support * max_support
        support.loc[toremove, feature] = 0

    support[support < minimum_support] = 0
    return support.transpose()


def sledge_score_clusters(
        X,
        labels,
        aggregation='harmonic',
        return_descriptors=False):

    nclusters = max(labels) + 1
    descriptors = semantic_descriptors(X, labels)

    return [0 for _ in range(nclusters)]


def sledge_score(X, labels):
    return np.mean(sledge_score_clusters(X, labels))
