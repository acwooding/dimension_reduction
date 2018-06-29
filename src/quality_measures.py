import numpy as np
from sklearn.metrics import pairwise_distances as sk_pairwise_distances


# Helper functions
def square_matrix_entries(matrix):
    '''
    Return array of squares of entries. ie. (m_{ij}^2)
    '''
    matrix = np.array(matrix)
    return matrix**2

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
    _, _, difference_distances = pairwise_distances(high_distances=high_distances, low_distances=low_distances,
                                                    high_data=high_data, low_data=low_data, metric=metric)
    s_difference_distances = square_matrix_entries(difference_distances)
    stress = np.sqrt(np.sum(s_difference_distances))
    return stress

def point_stress(high_distances=None, low_distances=None,
                 high_data=None, low_data=None, metric='euclidean'):
    '''
    Attempt at defining a notion of the contribution to stress by point.
    
    Do this by taking the row sums of  $(d_{ij}-||x_{i}-x_{j}||)^2$
    '''
    _, _, difference_distances = pairwise_distances(high_distances=high_distances, low_distances=low_distances,
                                                    high_data=high_data, low_data=low_data, metric=metric)    
    s_difference_distances = square_matrix_entries(difference_distances)
    point_stress = np.sum(s_difference_distances, axis=1)
    return point_stress

