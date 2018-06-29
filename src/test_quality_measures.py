import sys
sys.path.append(".")

import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis import given

import numpy as np

import logging
import quality_measures as qm

LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"

logging.basicConfig(format=LOG_FORMAT, level=logging.INFO,
                    datefmt=DATE_FORMAT)
logger = logging.getLogger()

@given(arrays(np.float, (3, 3), elements=st.floats(min_value=0,
                                                max_value=1)))
def test_square_matrix_entries(array):
    matrix = np.matrix(array)
    s_array = array**2
    assert (qm.square_matrix_entries(matrix) == s_array).all()
