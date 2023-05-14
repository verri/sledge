# SLEDge: Support, Length, Exclusivity and Difference for Group Evaluation

The package performs an evaluation of clustering results through the semantic relationship between the significant frequent patterns identified among the cluster items. The method uses an internal validation technique to evaluate the cluster rather than using distance-related metrics. However, the algorithm requires that the data be organized in CATEGORICAL FORM.

## Functions
 
### 1. Particularization of Descriptors
```sh
particularize_descriptors(descriptors, particular_threshold=1.0)
```
Particularization of descriptors based on support. This function particularizes descriptors using a threshold applied on the carrier (support maximum - support minimum) of the feature in the clusters.

#### Parameters

`particular_threshold` _float_: Particularization threshold. Given the relative support, 0.0 means that the entire range of relative support will be used, while 0.5 will be used half, and 1.0 only maximum support is kept.

`descriptors` _array-like of shape (n_clusters, n_features)_: Matrix with the support of features in each cluster.

#### Returns

`descriptors` _array-like of shape (n_clusters, n_features)_: Matrix with the computed particularized support of features in each cluster.

### 2. Semantic Descriptors

```sh
semantic_descriptors(X, labels, particular_threshold=None)
```
Semantic descriptors based on feature support. This function computes the support of the present feature (1-itemsets composed by the features with value 1) of the samples in each cluster. Features in a cluster that do not meet the particularization criterion have their support zeroed.

#### Parameters

`X` _array-like of shape (n_samples, n_features)_: Feature array of each sample. All features must be binary.

`labels` _array-like of shape (n_samples,)_: Cluster labels for each sample starting in 0.

`particular_threshold` _{None, float}_: Particularization threshold. None means no particularization strategy.

#### Returns

`descriptors` _array-like of shape (n_clusters, n_features)_: Matrix with the computed particularized support of features in each cluster.

### 3. Cluster scores

```sh
sledge_score_clusters(X, labels, particular_threshold=None, aggregation='harmonic')
```
SLEDge score for each cluster. This function computes the SLEDge score of each cluster. If aggregation is None, returns a matrix with values S, L, E, and D for each cluster.

#### Parameters

`X` _array-like of shape (n_samples, n_features)_: Feature array of each sample. All features must be binary.

`labels` _array-like of shape (n_samples,)_: Cluster labels for each sample starting in 0.

`particular_threshold` _{None, float}_: Particularization threshold. None means no particularization strategy.

`aggregation` _{'harmonic', 'geometric', 'median', None}_: Strategy to aggregate values of S, L, E, and D.

#### Returns

`scores` _array-like of shape (n_clusters,)_: SLEDge score for each cluster.

`score_matrix` _array-like of shape (n_clusters, 4) if aggregation is None_: S,L,E,D score for each cluster.

### 4. SLEDge score

```sh
sledge_score(X, labels, particular_threshold=None, aggregation='harmonic')
```
The SLEDge score. This function computes the average SLEDge score of all clusters.

#### Parameters

`X` _array-like of shape (n_samples, n_features)_: Feature array of each sample. All features must be binary.

`labels` _array-like of shape (n_samples,)_: Cluster labels for each sample starting in 0.

`particular_threshold` _{None, float}_: Particularization threshold. None means no particularization strategy.

`aggregation` _{'harmonic', 'geometric', 'median'}_: Strategy to aggregate values of S, L, E, and D for each cluster.

#### Returns

`score` _float_: Average SLEDge score.
