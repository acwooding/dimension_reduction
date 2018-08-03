import sys
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis import given
import unittest
import numpy as np
import logging

import src.quality_measures as qm

LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"

logging.basicConfig(format=LOG_FORMAT, level=logging.INFO,
                    datefmt=DATE_FORMAT)
logger = logging.getLogger()


# old functions to test against while refactoring

def old_centering_matrix(N):
    '''
    Returns the N x N centering matrix.
    '''
    I_N = np.identity(N)
    one_N = np.matrix(np.ones(N)).transpose()
    J = I_N - one_N * one_N.transpose()/N
    return J


def old_doubly_center_matrix(matrix):
    '''
    Doubly center the matrix. That is, -J * matrix * J.

    Note that this matrix must be square.
    '''
    m, n = matrix.shape
    assert m == n, "Matrix must be square"
    J = old_centering_matrix(m)
    new_matrix = -J * matrix * J
    return new_matrix / 2


def old_strain(high_distances, low_distances):
    B = qm.doubly_center_matrix(qm.square_matrix_entries(high_distances))
    top = qm.square_matrix_entries(B - qm.square_matrix_entries(low_distances))
    result = np.sqrt(np.sum(top)/np.sum(qm.square_matrix_entries(B)))
    return result


def old_point_strain(high_distances, low_distances):
    B = qm.doubly_center_matrix(qm.square_matrix_entries(high_distances))
    top = qm.square_matrix_entries(B - qm.square_matrix_entries(low_distances))
    result = np.sum(top, axis=1)/np.sum(qm.square_matrix_entries(B))
    return result


def knn_to_point_untrustworthiness(high_knn, low_knn, n_neighbors=None,
                                   high_rank=None):
    '''
    Given the n_neighbors nearest neighbors in high space and low space,
    together with the rank matrix, compute the value of
    "untrustworthiness" of a point (this is the factor that a point
    contributes negatively to trustworthiness).
    '''
    if n_neighbors is None or high_rank is None:
        raise ValueError("n_neighbors and high_rank are required")
    point_scores = []
    N = high_knn.shape[0]
    G_K = qm._trustworthiness_normalizating_factor(n_neighbors, N)
    for i, low in enumerate(low_knn):
        trust_indices = set(low).difference(set(high_knn[i]))
        score = 0
        for j in trust_indices:
            score += (high_rank[i, j] - n_neighbors) * 2 / G_K
        point_scores.append(score)
    return np.array(point_scores)


def old_point_untrustworthiness(high_distances=None, low_distances=None,
                                high_data=None, low_data=None,
                                metric='euclidean', n_neighbors=None):
    '''
    Given high/low distances or data, compute the value of
    "untrustworthiness" of a point (this is the factor that a point
    contributes negatively to trustworthiness).
    '''
    hd, ld, _ = qm.pairwise_distance_differences(high_distances=high_distances,
                                                 low_distances=low_distances,
                                                 high_data=high_data,
                                                 low_data=low_data,
                                                 metric=metric)

    if n_neighbors is None:
        raise ValueError("n_neighbors is required")
    high_rank = qm.rank_matrix(hd)
    low_rank = qm.rank_matrix(ld)
    high_knn = qm.rank_to_knn(high_rank, n_neighbors=n_neighbors)
    low_knn = qm.rank_to_knn(low_rank, n_neighbors=n_neighbors)
    point_scores = knn_to_point_untrustworthiness(high_knn, low_knn,
                                                  n_neighbors=n_neighbors,
                                                  high_rank=high_rank)
    return point_scores


# Start of tests


@given(arrays(np.float, (3, 3), elements=st.floats(min_value=-100,
                                                   max_value=100)))
def test_square_matrix_entries(array):
    matrix = np.matrix(array)
    s_array = array**2
    assert (qm.square_matrix_entries(matrix) == s_array).all()


@given(st.integers(min_value=1, max_value=100))
def test_old_new_centering_matrix(N):
    assert (qm.centering_matrix(N) == old_centering_matrix(N)).all()


@given(st.integers(min_value=1, max_value=100))
def test_centering_matrix_output(N):
    matrix = qm.centering_matrix(N)
    assert matrix.shape == (N, N)


@given(arrays(np.float, (3, 3), elements=st.floats(min_value=-100,
                                                   max_value=100)))
def test_old_new_doubly_center_matrix(matrix):
    assert (qm.doubly_center_matrix(matrix) ==
            old_doubly_center_matrix(matrix)).all()


@given(arrays(np.float, (3, 3), elements=st.floats(min_value=-100,
                                                   max_value=100)),
       arrays(np.float, (3, 3), elements=st.floats(min_value=-100,
                                                   max_value=100)))
def test_old_new_strain(high_distances, low_distances):
    # all zeros raises an error. tested later.
    if not (high_distances == 0).all():
        assert (qm.strain(high_distances, low_distances) ==
                old_strain(high_distances, low_distances)).all()


@given(arrays(np.float, (3, 3), elements=st.floats(min_value=-100,
                                                   max_value=100)),
       arrays(np.float, (3, 3), elements=st.floats(min_value=-100,
                                                   max_value=100)))
def test_old_new_point_strain(high_distances, low_distances):
    # all zeros raises an error. tested later.
    if not (high_distances == 0).all():
        assert (qm.point_strain(high_distances, low_distances) ==
                old_point_strain(high_distances, low_distances)).all()

# TODO: Test various input styles.
@given(arrays(np.float, (3, 3), elements=st.floats(min_value=-100,
                                                   max_value=100)),
       arrays(np.float, (3, 3), elements=st.floats(min_value=-100,
                                                   max_value=100)),
       st.integers(min_value=1, max_value=3))
def test_old_new_point_untrustworthiness(high_distances, low_distances,
                                         n_neighbors):
    old = old_point_untrustworthiness(high_distances=high_distances,
                                      low_distances=low_distances,
                                      n_neighbors=n_neighbors)
    new = qm.point_untrustworthiness(high_distances=high_distances,
                                     low_distances=low_distances,
                                     n_neighbors=n_neighbors)
    assert all(old == new)

class TestEncoding(unittest.TestCase):

    @given(arrays(np.float, (3, 2), elements=st.floats(min_value=-100,
                                                       max_value=100)))
    def test_doubly_center_matrix_input(self, matrix):
        with self.assertRaises(ValueError):
            qm.doubly_center_matrix(matrix)

    @given(st.integers(min_value=1, max_value=100))
    def test_zero_input_strain(self, N):
        matrix = np.zeros((N, N))
        with self.assertRaises(ValueError):
            qm.strain(high_distances=matrix, low_distances=matrix)

    @given(st.integers(min_value=1, max_value=100))
    def test_zero_input_point_strain(self, N):
        matrix = np.zeros((N, N))
        with self.assertRaises(ValueError):
            qm.point_strain(high_distances=matrix, low_distances=matrix)


if __name__ == '__main__':
    unittest.main()
