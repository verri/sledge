"""
Python package `sledge`: semantic evaluation of clustering results.
"""

import pandas as pd
import numpy as np
import math
from statistics import harmonic_mean


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

    # 1-itemsets, for greater k we need a different algorithm
    support = X.groupby(pd.Series(labels)).apply(np.mean)

    # Particularization
    for feature in features:
        column = np.array(support[feature])

        max_support = np.array([np.max(np.delete(column, i))
                               for i in range(nclusters)])
        mean_support = np.array([np.mean(np.delete(column, i)) for i in
                                 range(nclusters)])

        toremove = column ** 2 < mean_support * max_support
        support.loc[toremove, feature] = 0.0

    support[support < minimum_support] = 0.0
    return support


def sledge_score_clusters(
        X,
        labels,
        minimum_support=0.0,
        aggregation='harmonic',
        return_descriptors=False):

    nclusters = max(labels) + 1
    descriptors = semantic_descriptors(
        X, labels, minimum_support=minimum_support).transpose()

    # S: Average support for descriptors (features with particularized support
    # greater than zero)
    def mean_gt_zero(x): return 0 if np.count_nonzero(
        x) == 0 else np.mean(x[x > 0])
    support_score = [mean_gt_zero(descriptors[cluster])
                     for cluster in range(nclusters)]

    # L: Description set size deviation
    descriptor_set_size = np.array([np.count_nonzero(descriptors[cluster]) for
                                   cluster in range(nclusters)])
    # XXX: I am ignoring the clusters that have zero descriptors in the average
    # and considering length score equals zero if the cluster has no
    # descriptors
    average_set_size = np.mean(descriptor_set_size[descriptor_set_size > 0])
    length_score = [0 if set_size == 0 else 1.0 /
                    (1.0 +
                     abs(set_size -
                         average_set_size)) for set_size in descriptor_set_size]

    # E: Exclusivity
    descriptor_sets = np.array([frozenset(
        descriptors.index[descriptors[cluster] > 0]) for cluster in range(nclusters)])
    exclusive_sets = [
        descriptor_sets[cluster].difference(
            frozenset.union(
                *
                np.delete(
                    descriptor_sets,
                    cluster))) for cluster in range(nclusters)]
    exclusive_score = [0 if len(descriptor_sets[cluster]) == 0 else len(
        exclusive_sets[cluster]) / len(descriptor_sets[cluster]) for cluster in range(nclusters)]

    # D: Maximum ordered support difference
    # XXX: I implemented a little bit different from the paper. I always
    # consider that there is a _dummy_ descriptor with support equals zero.
    ordered_support = [np.sort(descriptors[cluster])
                       for cluster in range(nclusters)]
    diff_score = [math.sqrt(np.max(np.diff(ordered_support[cluster])))
                  for cluster in range(nclusters)]

    score = pd.DataFrame.from_dict({'S': support_score, 'L': length_score,
                                    'E': exclusive_score, 'D': diff_score})

    if aggregation == 'harmonic':
        score = score.transpose().apply(harmonic_mean)
    else:
        assert aggregation is None

    return score


def sledge_score(X, labels, minimum_support=0.0, aggregation='harmonic'):
    assert aggregation is not None
    return np.mean(sledge_score_clusters(X, labels,
                                         minimum_support=minimum_support,
                                         aggregation=aggregation))
