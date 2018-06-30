import numpy as np
from sklearn.metrics import pairwise_distances as sk_pairwise_distances


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


def pairwise_distances(high_distances=None, low_distances=None,
                       high_data=None, low_data=None, metric='euclidean'):
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
    _, _, difference_distances = pairwise_distances(high_distances=high_distances,
                                                    low_distances=low_distances,
                                                    high_data=high_data,
                                                    low_data=low_data,
                                                    metric=metric)
    s_difference_distances = square_matrix_entries(difference_distances)
    stress = np.sqrt(np.sum(s_difference_distances))
    return stress


def point_stress(high_distances=None, low_distances=None,
                 high_data=None, low_data=None, metric='euclidean'):
    '''
    Attempt at defining a notion of the contribution to stress by point.

    Do this by taking the row sums of  $(d_{ij}-||x_{i}-x_{j}||)^2$
    '''
    _, _, difference_distances = pairwise_distances(high_distances=high_distances,
                                                    low_distances=low_distances,
                                                    high_data=high_data,
                                                    low_data=low_data,
                                                    metric=metric)
    s_difference_distances = square_matrix_entries(difference_distances)
    point_stress = np.sum(s_difference_distances, axis=1)
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
    high_distances, low_distances, _ = pairwise_distances(high_distances=high_distances,
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
    over the corresponding row sum of the denominator.
    '''

    high_distances, low_distances, _ = pairwise_distances(high_distances=high_distances,
                                                          low_distances=low_distances,
                                                          high_data=high_data,
                                                          low_data=low_data,
                                                          metric=metric)
    if (high_distances == 0).all():
        raise ValueError("high_distances can't be the zero matrix")
    B = doubly_center_matrix(square_matrix_entries(high_distances))
    top = square_matrix_entries(B - square_matrix_entries(low_distances))
    result = np.sum(top, axis=1)/np.sum(square_matrix_entries(B), axis=1)
    return result
