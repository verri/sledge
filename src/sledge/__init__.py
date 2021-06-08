"""
Python package `sledge`: semantic evaluation of clustering results.
"""

import pandas as pd
import numpy as np
import math
import statistics


def semantic_descriptors(X, labels, minimum_support=0.0):
    """
    Semantic descriptors based on feature support.

    This function computes the support of the present feature (1-itemsets
    composed by the features with value 1) of the samples in each cluster.

    Features in a cluster that do not meet either the *particularization
    criterion* or the `minimum_support` have their support zeroed.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        Feature array of each sample.  All features must be binary.
    labels: array-like of shape (n_samples,)
        Cluster labels for each sample starting in 0.
    minimum_support: float
        Minimum support threshold.

    Returns
    -------
    descriptors: array-like of shape (n_clusters, n_features)
        Matrix with the computed particularized support of features in each
        cluster.
    """

    n_clusters = max(labels) + 1

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])

    features = X.columns

    # 1-itemsets, for greater k we need a different algorithm
    support = X.groupby(pd.Series(labels)).apply(np.mean)

    # Particularization
    for feature in features:
        column = np.array(support[feature])

        max_support = np.array([np.max(np.delete(column, i))
                               for i in range(n_clusters)])
        mean_support = np.array([np.mean(np.delete(column, i)) for i in
                                 range(n_clusters)])

        toremove = column ** 2 < mean_support * max_support
        support.loc[toremove, feature] = 0.0

    support[support < minimum_support] = 0.0
    return support


def sledge_score_clusters(
        X,
        labels,
        minimum_support=0.0,
        aggregation='harmonic'):
    """
    SLEDge score for each cluster.

    This function computes the SLEDge score of each cluster.

    If `aggregation` is `None`, returns a matrix with values *S*, *L*, *E*, and
    *D* for each cluster.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        Feature array of each sample.  All features must be binary.
    labels: array-like of shape (n_samples,)
        Cluster labels for each sample starting in 0.
    minimum_support: float
        Minimum support threshold.
    aggregation: {'harmonic', 'geometric', 'median', None}
        Strategy to aggregate values of *S*, *L*, *E*, and *D*.

    Returns
    -------
    scores: array-like of shape (n_clusters,)
        SLEDge score for each cluster.
    score_matrix: array-like of shape (n_clusters, 4) if `aggregation` is None
        S,L,E,D score for each cluster.
    """

    n_clusters = max(labels) + 1
    descriptors = semantic_descriptors(
        X, labels, minimum_support=minimum_support).transpose()

    # S: Average support for descriptors (features with particularized support
    # greater than zero)
    def mean_gt_zero(x): return 0 if np.count_nonzero(
        x) == 0 else np.mean(x[x > 0])
    support_score = [mean_gt_zero(descriptors[cluster])
                     for cluster in range(n_clusters)]

    # L: Description set size deviation
    descriptor_set_size = np.array([np.count_nonzero(descriptors[cluster]) for
                                   cluster in range(n_clusters)])
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
        descriptors.index[descriptors[cluster] > 0]) for cluster in range(n_clusters)])
    exclusive_sets = [
        descriptor_sets[cluster].difference(
            frozenset.union(
                *
                np.delete(
                    descriptor_sets,
                    cluster))) for cluster in range(n_clusters)]
    exclusive_score = [0 if len(descriptor_sets[cluster]) == 0 else len(
        exclusive_sets[cluster]) / len(descriptor_sets[cluster]) for cluster in range(n_clusters)]

    # D: Maximum ordered support difference
    # XXX: I implemented a little bit different from the paper. I always
    # consider that there is a _dummy_ descriptor with support equals zero.
    ordered_support = [np.sort(descriptors[cluster])
                       for cluster in range(n_clusters)]
    diff_score = [math.sqrt(np.max(np.diff(ordered_support[cluster])))
                  for cluster in range(n_clusters)]

    score = pd.DataFrame.from_dict({'S': support_score, 'L': length_score,
                                    'E': exclusive_score, 'D': diff_score})

    if aggregation == 'harmonic':
        score = score.transpose().apply(statistics.harmonic_mean)
    elif aggregation == 'geometric':
        score = score.transpose().apply(statistics.geometric_mean)
    elif aggregation == 'median':
        score = score.transpose().apply(statistics.median)
    else:
        assert aggregation is None

    return score


def sledge_score(X, labels, minimum_support=0.0, aggregation='harmonic'):
    """
    SLEDge score.

    This function computes the average SLEDge score of all clusters.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        Feature array of each sample.  All features must be binary.
    labels: array-like of shape (n_samples,)
        Cluster labels for each sample starting in 0.
    minimum_support: float
        Minimum support threshold.
    aggregation: {'harmonic', 'geometric', 'median'}
        Strategy to aggregate values of *S*, *L*, *E*, and *D* for each cluster.

    Returns
    -------
    score: float
        Average SLEDge score.
    """
    assert aggregation is not None
    return np.mean(sledge_score_clusters(X, labels,
                                         minimum_support=minimum_support,
                                         aggregation=aggregation))


def sledge_curve(X, labels, minimum_support=0.0, aggregation='harmonic'):
    """
    SLEDge curve.

    This function computes the SLEDge curve.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        Feature array of each sample.  All features must be binary.
    labels: array-like of shape (n_samples,)
        Cluster labels for each sample starting in 0.
    minimum_support: float
        Minimum support threshold.
    aggregation: {'harmonic', 'geometric', 'median', None}
        Strategy to aggregate values of *S*, *L*, *E*, and *D*.

    Returns
    -------
    fractions: array-like of shape (>2,)
        Decreasing rate that element `i` is the fraction of clusters with
        SLEDge score > `thresholds[i]`.  `fractions[0]` is always `0`.
    thresholds: array-like of shape (>2, )
        Increasing thresholds of the cluster SLEDge score used to compute
        `fractions`.  `thresholds[0]` is alway `0` and `thresholds[-1]` is
        always `1`.
    """
    scores = sledge_score_clusters(X, labels, minimum_support=minimum_support,
                                   aggregation=aggregation)
    n_clusters = len(scores)

    thresholds = np.unique(scores)
    if thresholds[0] != 1:
        thresholds = np.concatenate((thresholds, [1]))
    if thresholds[len(thresholds) - 1] != 0:
        thresholds = np.concatenate(([0], thresholds))

    fractions = np.array(
        [np.count_nonzero(scores >= thr) / n_clusters for thr in thresholds])

    return fractions, thresholds
