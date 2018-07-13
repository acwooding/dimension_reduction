import numpy as np
from sklearn.metrics import pairwise_distances as sk_pairwise_distances

import logging

LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"

logging.basicConfig(format=LOG_FORMAT, level=logging.INFO,
                    datefmt=DATE_FORMAT)
logger = logging.getLogger()


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

# Stress


def stress(high_distances=None, low_distances=None,
           high_data=None, low_data=None, metric='euclidean'):
    '''
    Compute the stress as defined in Metric MDS given $d_{ij}-||x_{i}-x_{j}||$.
    '''
    difference_distances = pairwise_distance_differences(high_distances=high_distances,
                                                    low_distances=low_distances,
                                                    high_data=high_data,
                                                    low_data=low_data,
                                                    metric=metric)[2]
    s_difference_distances = square_matrix_entries(difference_distances)
    stress = np.sqrt(np.sum(s_difference_distances))
    return stress


def point_stress(high_distances=None, low_distances=None,
                 high_data=None, low_data=None, metric='euclidean'):
    '''
    Attempt at defining a notion of the contribution to stress by point.

    Do this by taking the square root of the row sums of  $(d_{ij}-||x_{i}-x_{j}||)^2$
    '''
    difference_distances = pairwise_distance_differences(high_distances=high_distances,
                                                    low_distances=low_distances,
                                                    high_data=high_data,
                                                    low_data=low_data,
                                                    metric=metric)[2]

    s_difference_distances = square_matrix_entries(difference_distances)
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
    high_distances, low_distances, _ = pairwise_distance_differences(high_distances=high_distances,
                                                          low_distances=low_distances,
                                                          high_data=high_data,
                                                          low_data=low_data,
                                                          metric=metric)
    if (high_distances == 0).all():
        raise ValueError("high_distances can't be the zero matrix")
    B = doubly_center_matrix(square_matrix_entries(high_distances))
    top = square_matrix_entries(B - square_matrix_entries(low_distances))
    result = np.sqrt(np.sum(top)/np.sum(square_matrix_entries(B)))
    return result


def point_strain(high_distances=None, low_distances=None,
                 high_data=None, low_data=None, metric='euclidean'):
    '''
    Compute the contribution of each point towards strain (as defined
    in Classical MDS). This is done by taking row sums of the numerator
    over the normalization factor.
    '''

    high_distances, low_distances, _ = pairwise_distance_differences(high_distances=high_distances,
                                                          low_distances=low_distances,
                                                          high_data=high_data,
                                                          low_data=low_data,
                                                          metric=metric)
    if (high_distances == 0).all():
        raise ValueError("high_distances can't be the zero matrix")
    B = doubly_center_matrix(square_matrix_entries(high_distances))
    top = square_matrix_entries(B - square_matrix_entries(low_distances))
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


def faster_rank_matrix(distance_matrix):
    '''
    Return a rank matrix where the (i, j) entry is the number of
    distances in row i that that are less than the value of the
    entry (i, j) in the distance matrix. Ties in distance are broken
    by lexicographical order of the column index (as in numpy's argsort).
    >>> faster_rank_matrix(np.array([[0, 1, 5, 3],\
                                     [1, 0 , 3, 5],\
                                     [5, 3, 0, 1],\
                                     [3, 5, 1, 0]]))
    array([[0, 1, 3, 2],
           [1, 0, 2, 3],
           [3, 2, 0, 1],
           [2, 3, 1, 0]], dtype=int32)
    >>> faster_rank_matrix(np.array([[0, 1, 2, 3],\
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


def _trustworthiness_normalizating_factor(K, N):
    if K < (N / 2):
        G_K = N*K*(2*N - 3*K - 1)
    else:
        G_K = N*(N - K)*(N - K - 1)
    return G_K


def _knn_to_point_trustworthiness(high_knn, low_knn, n_neighbors=None,
                                  high_rank=None):
    if n_neighbors is None or high_rank is None:
        raise ValueError("n_neighbors and high_rank are required")
    point_scores = []
    N = high_knn.shape[0]
    G_K = _trustworthiness_normalizating_factor(n_neighbors, N)
    for i, low in enumerate(low_knn):
        trust_indices = set(low).difference(set(high_knn[i]))
        score = 0
        for j in trust_indices:
            score += (high_rank[i, j] - n_neighbors) * 2 / G_K
        point_scores.append(score)
    return np.array(point_scores)


def point_trustworthiness(high_distances=None, low_distances=None,
                          high_data=None, low_data=None,
                          metric='euclidean', n_neighbors=None):
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
    point_scores = _knn_to_point_trustworthiness(high_knn, low_knn,
                                                 n_neighbors=n_neighbors,
                                                 high_rank=high_rank)
    return point_scores


def trustworthiness(high_distances=None, low_distances=None,
                    high_data=None, low_data=None,
                    point_scores=None,
                    metric='euclidean',
                    n_neighbors=None):
    if point_scores is None:
        pt = point_trustworthiness(high_data=high_data,
                                   low_data=low_data,
                                   high_distances=high_distances,
                                   low_distances=low_distances,
                                   metric=metric,
                                   n_neighbors=n_neighbors)
    else:
        pt = point_scores
    return 1 - sum(pt)
