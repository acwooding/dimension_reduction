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
