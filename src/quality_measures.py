import numpy as np
from sklearn.metrics import pairwise_distances as sk_pairwise_distances
from sklearn.neighbors import NearestNeighbors
from itertools import combinations as comb
from functools import partial

# from .logging import logger


# Helper functions
def square_matrix_entries(matrix):
    '''
    Return array of squares of entries. ie. (m_{ij}^2)
    '''
    matrix = np.array(matrix)
    return matrix**2


def centering_matrix(N):
    '''
    Returns the N x N centering matrix.
    '''
    I_N = np.identity(N)
    one_N = np.matrix(np.ones(N)).transpose()
    J = I_N - one_N * one_N.transpose()/N
    return J


def pairwise_distance_differences(high_distances=None, low_distances=None,
                                  high_data=None, low_data=None,
                                  metric='euclidean'):
    '''
    Computes $d_{ij}-||x_{i}-x_{j}||$. Computes pairwise distances in the
    high space and low space if they weren't passed in.

    Returns: (high_distances, low_distances, distance_difference)
    -------
    high_distances: np array of pairwise distances between high_data points
    low_distances: np array of pairwise distances between low_data points
    distance_difference: np array high_distances - low_distances

    >>> a = np.array([[7, 4, 0], [4, 5, 2], [9, 4, 3]])
    >>> b = np.array([[0, 6], [7, 1], [4, 9]])
    >>> pairwise_distance_differences(high_data=a, low_data=b)[0]
    array([[0.        , 3.74165739, 3.60555128],
           [3.74165739, 0.        , 5.19615242],
           [3.60555128, 5.19615242, 0.        ]])
    >>> pairwise_distance_differences(high_data=a, low_data=b)[1]
    array([[0.        , 8.60232527, 5.        ],
           [8.60232527, 0.        , 8.54400375],
           [5.        , 8.54400375, 0.        ]])
    >>> pairwise_distance_differences(high_data=a, low_data=b)[2]
    array([[ 0.        , -4.86066788, -1.39444872],
           [-4.86066788,  0.        , -3.34785132],
           [-1.39444872, -3.34785132,  0.        ]])

    >>> a = np.array([[0, 4, 7], [4, 0, 2], [7, 2, 0]])
    >>> b = np.array([[0, 4, 8], [4, 0, 1], [8, 1, 0]])
    >>> pairwise_distance_differences(high_distances=a, low_distances=b)[0]
    array([[0, 4, 7],
           [4, 0, 2],
           [7, 2, 0]])
    >>> pairwise_distance_differences(high_distances=a, low_distances=b)[1]
    array([[0, 4, 8],
           [4, 0, 1],
           [8, 1, 0]])
    >>> pairwise_distance_differences(high_distances=a, low_distances=b)[2]
    array([[ 0,  0, -1],
           [ 0,  0,  1],
           [-1,  1,  0]])
    '''
    if (high_distances is None) and (high_data is None):
        raise ValueError("One of high_distances or high_data is required")
    if (low_distances is None) and (low_data is None):
        raise ValueError("One of low_distances or low_data is required")
    if low_distances is None:
        low_distances = sk_pairwise_distances(low_data, metric=metric)
    if high_distances is None:
        high_distances = sk_pairwise_distances(high_data, metric=metric)

    difference_distances = high_distances-low_distances

    return high_distances, low_distances, difference_distances


def jaccard(A, B):
    '''
    Jaccard similarity of two sets.

    Parameters
    ----------
    A, B: (sets)
    Returns
    -------
    similarity in [0, 1]
    '''
    A = set(A)
    B = set(B)

    if A and B:
        return len(A.intersection(B))/len(A.union(B))
    else:
        return 0


def adaptedKendallTau(X, Y):
    '''
    Parameters
    ----------
    X, Y:
        2 lists X and Y of arbitrary length, consisting of unique elements (ie.
        X and Y represent posets).
    Returns
    -------
    similarity in [0,1]
    '''
    X = list(X)
    Y = list(Y)
    if not X or not Y:
        return 0
    elif len(set(X)) != len(X) or len(set(Y)) != len(Y):
        raise ValueError("There are repeated elements in X or Y")
    else:
        # intersection - list to keep the order as in X
        SI = [x for x in X if x in Y]
        # in X only
        SX = set(X).difference(SI)
        # in Y only
        SY = set(Y).difference(SI)
        tau = 0
        # SI: pairs in intersection
        length = len(SI)
        y = [Y.index(z) for z in SI]
        s = np.sum([a < b for a, b in comb(y, 2)])
        # pos-neg pairs a la Kendall-tau.
        tau += 2*s - length*(length-1) / 2
        # SX, resp. SY and SI: one element in intersection
        x = [X.index(z) for z in SX]
        y = [Y.index(z) for z in SY]
        i = [X.index(z) for z in SI]
        s = 2*np.sum([a > b for a in x for b in i]) - len(x)*len(i)
        tau += s
        s = 2*np.sum([a > b for a in y for b in i]) - len(y)*len(i)
        tau += s
        # the rest
        tau -= len(SX)*len(SY)
        tau += length*(length+1)/2
        tau /= (len(X)*len(Y))
        return ((tau+1)/2)


def list_similarity(label, list1, list2, how='jaccard'):
    """
    Compute similarity score of two lists.
    Parameters
    ----------
    label:
        name of the point
    list1, list2:
        lists of k-nearest neighbors of label
    how: {'jaccard', 'adapted-ktau'}
        similarity score method

    Returns: (label, score)
    """
    if how == 'jaccard':
        score = jaccard(list1, list2)
    elif how == 'adapted-ktau':
        score = adaptedKendallTau(list1, list2)
    else:
        raise ValueError("Unknown method: {}".format(how))
    return label, score


# Stress
def stress(high_distances=None, low_distances=None,
           high_data=None, low_data=None, metric='euclidean'):
    '''
    Compute the stress as defined in Metric MDS given $d_{ij}-||x_{i}-x_{j}||$.


    >>> a = np.array([[7, 4, 0], [4, 5, 2], [9, 4, 3]])
    >>> b = np.array([[0, 6], [7, 1], [4, 9]])
    >>> stress(high_data=a, low_data=b)
    8.576559679258732

    >>> a = np.array([[0, 4, 7], [4, 0, 2], [7, 2, 0]])
    >>> b = np.array([[0, 4, 8], [4, 0, 1], [8, 1, 0]])
    >>> stress(high_distances=a, low_distances=b)
    2.0
    '''
    dd = pairwise_distance_differences(high_distances=high_distances,
                                       low_distances=low_distances,
                                       high_data=high_data,
                                       low_data=low_data,
                                       metric=metric)[2]
    s_difference_distances = square_matrix_entries(dd)
    stress = np.sqrt(np.sum(s_difference_distances))
    return stress


def point_stress(high_distances=None, low_distances=None,
                 high_data=None, low_data=None, metric='euclidean'):
    '''
    Attempt at defining a notion of the contribution to stress by point.

    Do this by taking the square root of the row sums of
    $(d_{ij}-||x_{i}-x_{j}||)^2$

    >>> a = np.array([[7, 4, 0], [4, 5, 2], [9, 4, 3]])
    >>> b = np.array([[0, 6], [7, 1], [4, 9]])
    >>> point_stress(high_data=a, low_data=b)
    array([5.05673605, 5.90205055, 3.62665076])

    >>> a = np.array([[0, 4, 7], [4, 0, 2], [7, 2, 0]])
    >>> b = np.array([[0, 4, 8], [4, 0, 1], [8, 1, 0]])
    >>> point_stress(high_distances=a, low_distances=b)
    array([1.        , 1.        , 1.41421356])
    '''
    dd = pairwise_distance_differences(high_distances=high_distances,
                                       low_distances=low_distances,
                                       high_data=high_data,
                                       low_data=low_data,
                                       metric=metric)[2]

    s_difference_distances = square_matrix_entries(dd)
    point_stress = np.sqrt(np.sum(s_difference_distances, axis=1))
    return point_stress


def doubly_center_matrix(matrix):
    '''
    Doubly center the matrix. That is, -J * matrix * J.

    Note that the matrix input must be square.
    '''
    m, n = matrix.shape
    if m != n:
        raise ValueError(f"Input matrix is {m} x {n}. Matrix must be square")
    J = centering_matrix(m)
    new_matrix = -J * matrix * J
    return new_matrix / 2


def strain(high_distances=None, low_distances=None,
           high_data=None, low_data=None, metric='euclidean'):
    '''
    Compute the strain as defined in Classical MDS.
    '''
    hd, ld, _ = pairwise_distance_differences(high_distances=high_distances,
                                              low_distances=low_distances,
                                              high_data=high_data,
                                              low_data=low_data,
                                              metric=metric)
    if (hd == 0).all():
        raise ValueError("high_distances can't be the zero matrix")
    B = doubly_center_matrix(square_matrix_entries(hd))
    top = square_matrix_entries(B - square_matrix_entries(ld))
    result = np.sqrt(np.sum(top)/np.sum(square_matrix_entries(B)))
    return result


def point_strain(high_distances=None, low_distances=None,
                 high_data=None, low_data=None, metric='euclidean'):
    '''
    Compute the contribution of each point towards strain (as defined
    in Classical MDS). This is done by taking row sums of the numerator
    over the normalization factor.
    '''

    hd, ld, _ = pairwise_distance_differences(high_distances=high_distances,
                                              low_distances=low_distances,
                                              high_data=high_data,
                                              low_data=low_data,
                                              metric=metric)
    if (hd == 0).all():
        raise ValueError("high_distances can't be the zero matrix")
    B = doubly_center_matrix(square_matrix_entries(hd))
    top = square_matrix_entries(B - square_matrix_entries(ld))
    result = np.sum(top, axis=1)/np.sum(square_matrix_entries(B))
    return result


def rank_matrix(distance_matrix):
    '''
    Return a rank matrix where the (i, j) entry is the number of
    distances in row i that that are less than the value of the
    entry (i, j) in the distance matrix. Ties in distance are broken
    by lexicographical order of the column index (as in numpy's argsort).
    >>> rank_matrix(np.array([[0, 1, 5, 3],\
                              [1, 0 , 3, 5],\
                              [5, 3, 0, 1],\
                              [3, 5, 1, 0]]))
    array([[0, 1, 3, 2],
           [1, 0, 2, 3],
           [3, 2, 0, 1],
           [2, 3, 1, 0]], dtype=int32)

    >>> rank_matrix(np.array([[0, 1, 2, 3],\
                              [1, 0 , 1, 2],\
                              [2, 1, 0, 1],\
                              [3, 2, 1, 0]]))
    array([[0, 1, 2, 3],
           [1, 0, 2, 3],
           [3, 1, 0, 2],
           [3, 2, 1, 0]], dtype=int32)
    '''
    mat = np.zeros((distance_matrix.shape[0], distance_matrix.shape[1]),
                   dtype='int32')
    for i, row in enumerate(distance_matrix):
        mat[i] = np.argsort(np.argsort(row))
    return mat


def slower_rank_matrix(distance_matrix):
    '''
    Return a rank matrix where the (i, j) entry is the number of
    distances in row i that that are less than the value of the
    entry (i, j) in the distance matrix. Ties in distance are broken
    by lexicographical order of the column index (as in numpy's argsort).
    >>> slower_rank_matrix(np.array([[0, 1, 5, 3],\
                                     [1, 0 , 3, 5],\
                                     [5, 3, 0, 1],\
                                     [3, 5, 1, 0]]))
    array([[0, 1, 3, 2],
           [1, 0, 2, 3],
           [3, 2, 0, 1],
           [2, 3, 1, 0]], dtype=int32)
    >>> slower_rank_matrix(np.array([[0, 1, 2, 3],\
                                     [1, 0 , 1, 2],\
                                     [2, 1, 0, 1],\
                                     [3, 2, 1, 0]]))
    array([[0, 1, 2, 3],
           [1, 0, 2, 3],
           [3, 1, 0, 2],
           [3, 2, 1, 0]], dtype=int32)
    '''
    mat = np.zeros((distance_matrix.shape[0], distance_matrix.shape[1]),
                   dtype='int32')
    for i, row in enumerate(distance_matrix):
        sorted_row = np.argsort(row)
        s = np.empty(sorted_row.size, dtype='int32')
        for j in np.arange(sorted_row.size):
            s[sorted_row[j]] = j
        mat[i] = s
    return mat


def rank_to_knn(rank_matrix, n_neighbors=None):
    '''
    Given a rank matrix, return a list where each item
    contains the column indices of the k-nearest neighbors
    (where list indices correspond to row indices).

    >>> rank_to_knn(np.array([[0, 1, 2, 3],\
                              [1, 0, 2, 3],\
                              [3, 1, 0, 2],\
                              [3, 2, 1, 0]], dtype='int32'),\
                    n_neighbors=1)
    array([[0, 1],
           [0, 1],
           [1, 2],
           [2, 3]])
    '''
    if n_neighbors is None:
        raise ValueError("n_neighbors is required")
    knn = []
    for row in rank_matrix:
        knn.append(list(np.where(row <= n_neighbors)[0]))
    return np.array(knn)


def _trustworthiness_normalizating_factor(n_neighbors, n_points):
    '''
    Given the number of neighbors used in trustworthiness calculation,
    and the total number of points, return the normalization factor
    G_K for the trustworthiness calculation.
    '''
    K = n_neighbors
    N = n_points
    if K < (N / 2):
        G_K = N*K*(2*N - 3*K - 1)
    else:
        G_K = N*(N - K)*(N - K - 1)
    return G_K


def _np_set_difference(array1, array2):
    '''
    Compute the row by row set difference of two 2d arrays.

    >>> _np_set_difference(np.array([[0, 1, 2, 3],\
                                    [1, 0, 3, 4]]),\
                          np.array([[0, 1, 4, 5],\
                                    [2, 2, 3, 4]]))
    array([[2, 3],
           [0, 1]])
    '''
    set_diff = []
    for i, row in enumerate(array1):
        set_diff.append(np.setdiff1d(row, array2[i]))
    return np.array(set_diff)


def _sum_indices_to_point_scores(sum_indices, *, n_neighbors,
                                 rank_matrix):
    '''
    Given the indices to sum rank differences over for each point
    together with the rank matrix, compute the value of
    "untrustworthiness"/"discontinuity" of a point.
    '''
    point_scores = []
    N = rank_matrix.shape[0]
    G_K = _trustworthiness_normalizating_factor(n_neighbors, N)
    for i, indices in enumerate(sum_indices):
        score = 0
        for j in indices:
            score += (rank_matrix[i, j] - n_neighbors) * 2 / G_K
        point_scores.append(score)
    return np.array(point_scores)


def point_untrustworthiness(high_distances=None, low_distances=None,
                            high_data=None, low_data=None,
                            metric='euclidean', n_neighbors=None):
    '''
    Given high/low distances or data, compute the value of
    "untrustworthiness" of a point (this is the factor that a point
    contributes negatively to trustworthiness).
    '''
    hd, ld, _ = pairwise_distance_differences(high_distances=high_distances,
                                              low_distances=low_distances,
                                              high_data=high_data,
                                              low_data=low_data,
                                              metric=metric)

    if n_neighbors is None:
        raise ValueError("n_neighbors is required")
    high_rank = rank_matrix(hd)
    low_rank = rank_matrix(ld)
    high_knn = rank_to_knn(high_rank, n_neighbors=n_neighbors)
    low_knn = rank_to_knn(low_rank, n_neighbors=n_neighbors)
    set_difference = _np_set_difference(low_knn, high_knn)
    point_scores = _sum_indices_to_point_scores(set_difference,
                                                n_neighbors=n_neighbors,
                                                rank_matrix=high_rank)
    return point_scores


def trustworthiness(high_distances=None, low_distances=None,
                    high_data=None, low_data=None,
                    point_scores=None,
                    metric='euclidean',
                    n_neighbors=None):
    '''
    Given high/low distances or data, compute the value of
    trustworthiness of an embedding. Alternately, pass in point_scores,
    which should be the output from point_untrustworthiness.
    '''
    if point_scores is None:
        pt = point_untrustworthiness(high_data=high_data,
                                     low_data=low_data,
                                     high_distances=high_distances,
                                     low_distances=low_distances,
                                     metric=metric,
                                     n_neighbors=n_neighbors)
    else:
        pt = point_scores
    return 1 - sum(pt)


def point_discontinuity(high_distances=None, low_distances=None,
                        high_data=None, low_data=None,
                        metric='euclidean', n_neighbors=None):
    '''
    Given high/low distances or data, compute the value of
    "discontinuity" of a point (this is the factor that a point
    contributes negatively to continuity).
    '''
    hd, ld, _ = pairwise_distance_differences(high_distances=high_distances,
                                              low_distances=low_distances,
                                              high_data=high_data,
                                              low_data=low_data,
                                              metric=metric)

    if n_neighbors is None:
        raise ValueError("n_neighbors is required")
    high_rank = rank_matrix(hd)
    low_rank = rank_matrix(ld)
    high_knn = rank_to_knn(high_rank, n_neighbors=n_neighbors)
    low_knn = rank_to_knn(low_rank, n_neighbors=n_neighbors)
    set_difference = _np_set_difference(high_knn, low_knn)
    point_scores = _sum_indices_to_point_scores(set_difference,
                                                n_neighbors=n_neighbors,
                                                rank_matrix=low_rank)
    return point_scores


def continuity(high_distances=None, low_distances=None,
               high_data=None, low_data=None,
               point_scores=None,
               metric='euclidean',
               n_neighbors=None):
    '''
    Given high/low distances or data, compute the value of
    continuity of an embedding. Alternately, pass in point_scores,
    which should be the output from point_discontinuity.
    '''
    if point_scores is None:
        pt = point_discontinuity(high_data=high_data,
                                 low_data=low_data,
                                 high_distances=high_distances,
                                 low_distances=low_distances,
                                 metric=metric,
                                 n_neighbors=n_neighbors)
    else:
        pt = point_scores
    return 1 - sum(pt)


def point_generalized_1nn_error(*, data, classes, metric='euclidean'):
    '''
    Given data and associated classes (for each row), return an
    array with entry 0 if the row's nearest neighbor has the same class
    and a 1 if the row's nearest neighbor is from a different class.

    Parameters
    ----------
    data: np.array
    classes: 1d np.array
    metric: an sklearn metric to use on the data to find nearest neighbors

    Returns
    -------
    point_generalized_1nn_error: 1d np.array
    '''
    nbrs = NearestNeighbors(n_neighbors=2, metric=metric).fit(data)
    _, indices = nbrs.kneighbors(data)
    error = []
    for a, b in indices:
        if classes[a] == classes[b]:
            error.append(0)
        else:
            error.append(1)
    return np.array(error)


def generalized_1nn_error(data=None, classes=None, point_error=None,
                          metric='euclidean'):
    '''
    Given either data and associated classes (for each row), or
    point_error, return the proportion of datapoints whose nearest neighbor
    does not have the same class as it does.

    Parameters
    ----------
    data: np.array
    classes: 1d np.array
    point_error: 1d np.array
        output from point_generalized_1nn_error
    metric: an sklearn metric to use on the data to find nearest neighbors

    Returns
    -------
    generalized_1nn_error: (float)
    '''
    if point_error is None:
        point_error = point_generalized_1nn_error(data=data,
                                                  classes=classes,
                                                  metric=metric)
    error = np.sum(point_error)/len(point_error)
    return error


def generalized_1nn_error_scorer(estimator, X, y=None, metric='euclidean'):
    '''
    Given data, X, an estimator and associated classes, y, (for each row),
    return the proportion of datapoints whose nearest neighbor
    does not have the same class as it does.

    Note: This returns the generalized 1-nn classifier error times -1, as
    scorers get used in such a way that "greater is better".

    Parameters
    ----------
    estimator: sklearn estimator with a transform/fit_transform method
    X: np.array
    y: 1d np.array

    metric: an sklearn metric to use on the data to find nearest neighbors

    Returns
    -------
    -1 * generalized_1nn_error: (float)
    '''
    if getattr(estimator, "transform", None) is not None:
        data = estimator.transform(X)
    else:
        data = estimator.fit_transform(X)
    point_error = point_generalized_1nn_error(data=data,
                                              classes=y,
                                              metric=metric)
    error = np.sum(point_error)/len(point_error)
    return -1 * error


def point_nn_comparison(high_data=None, low_data=None,
                        high_distances=None, low_distances=None,
                        metric='euclidean', n_neighbors=20,
                        *, comparison_function):
    if high_distances is not None:
        nn = NearestNeighbors(n_neighbors=n_neighbors,
                              metric='precomputed').fit(high_distances)
        _, high_indices = nn.kneighbors()
    else:
        nn = NearestNeighbors(n_neighbors=n_neighbors,
                              metric=metric).fit(high_data)
        _, high_indices = nn.kneighbors()
    if low_distances is not None:
        nn = NearestNeighbors(n_neighbors=n_neighbors,
                              metric='precomputed').fit(low_distances)
        _, low_indices = nn.kneighbors()
    else:
        nn = NearestNeighbors(n_neighbors=n_neighbors,
                              metric=metric).fit(low_data)
        _, low_indices = nn.kneighbors()

    point_score = []
    for i, nbrs in enumerate(high_indices):
        _, score = list_similarity(i, nbrs, low_indices[i],
                                   how=comparison_function)
        point_score.append(score)
    return point_score


def nn_comparison(high_data=None, low_data=None, metric='euclidean',
                  high_distances=None, low_distances=None,
                  n_neighbors=20, point_score=None, aggregate_function='mean',
                  *, comparison_function):
    if point_score is None:
        pt = point_nn_comparison(high_data=high_data,
                                 low_data=low_data,
                                 high_distances=high_distances,
                                 low_distances=low_distances,
                                 metric=metric,
                                 n_neighbors=n_neighbors,
                                 comparison_function=comparison_function)
    else:
        pt = point_score
    if aggregate_function == 'mean':
        score = np.mean(pt)
    elif aggregate_function == 'median':
        score = np.median(pt)
    return score


def make_hi_lo_scorer(func, greater_is_better=True, **kwargs):
    """Make a sklearn-style scoring function for measures taking high/low data
    representations.

    Assumes the wrapped function expects `high_data` and `low_data` as
    parameters.

    greater_is_better : boolean, default=True
        Whether `func` is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `func`.
    """
    sign = 1 if greater_is_better else -1

    def wrapped_func(estimator, X, y=None, **wrap_kw):
        if getattr(estimator, "transform", None) is not None:
            low_data = estimator.transform(X)
        else:
            low_data = estimator.fit_transform(X)
        new_kwargs = {**kwargs, **wrap_kw}
        score = func(high_data=X, low_data=low_data, **new_kwargs)
        return sign * score
    return wrapped_func


def available_quality_measures():
    """Valid quality measures for evaluating dimension reductions.

    This function simply returns the valid quality metrics.
    It exists to allow for a description of the mapping for
    each of the valid strings.

    The valid quality metrics, and the function they map to, are:

    ============     ====================================
    metric           Function
    ============     ====================================
    '1nn-error'
    'adj-kendall-tau'
    'continuity'
    'jaccard'
    'quality'
    'stress'
    'strain'
    'trustworthiness'
    ============     ====================================

    """
    return DR_MEASURES


DR_MEASURES = {
    "1nn-error": generalized_1nn_error,
    "nn-adapted-ktau": partial(nn_comparison,
                                  comparison_function='adapted-ktau'),
    "continuity": continuity,
    "nn-jaccard": partial(nn_comparison,
                          comparison_function='jaccard'),
#    "quality":None,
    "stress": stress,
    "strain": strain,
    "trustworthiness": trustworthiness,
}


def available_scorers():
    """Valid scorers based on quality measures for evaluating
    dimension reductions.

    This function simply returns the valid scorer. As per the
    sklearn API, a scorer is a function that can be called with
    parameters (estimator, X, y), where estimator is the model
    that should be evaluated, X is data to be operated on,
    and y is the ground truth target for X (in the supervised case)
    or None (in the unsupervised case).

    It exists to allow for a description of the mapping for
    each of the valid strings.

    The valid scorers, and the functions they map to, are:

    ============     ====================================
    metric           Function
    ============     ====================================
    '1nn-error'
    'adj-kendall-tau'
    'continuity'
    'jaccard'
    'quality'
    'stress'
    'strain'
    'trustworthiness'
    ============     ====================================

    """
    return DR_SCORERS


DR_SCORERS = {
    "1nn-error": generalized_1nn_error_scorer,
    "nn-adapted-ktau": make_hi_lo_scorer(partial(nn_comparison,
                                                 comparison_function='adapted-ktau'),
                                         greater_is_better=True),
    "continuity": make_hi_lo_scorer(continuity, greater_is_better=True),
    "nn-jaccard": make_hi_lo_scorer(partial(nn_comparison,
                                            comparison_function='jaccard'),
                                    greater_is_better=True),
#    "quality":None,
    "stress": make_hi_lo_scorer(stress, greater_is_better=False),
    "strain": make_hi_lo_scorer(strain, greater_is_better=False),
    "trustworthiness": make_hi_lo_scorer(trustworthiness,
                                         greater_is_better=True),
}
