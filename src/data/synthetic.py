from  itertools import product
import numpy as np
from sklearn.utils import check_random_state

__all__ = ['synthetic_data', 'sample_sphere_surface', 'sample_ball', 'helix']

def _combn(iterable, repeat):
    """Equivalent of matlab's combn"""
    return list(product(iterable, repeat=repeat))

def _parameterized_swiss_roll(manifold_coords):
    """Given parameters t, y return a swiss roll

    Produces a 3d swiss roll with parameterized coordinates:
      [t cos(t), y, t sin(t)]

    Parameters
    ----------
    manifold_coordinates: A tuple (t, y) indicating swiss roll parameters

    Returns
    -------
    A tuple (coords, manifold_coords) where
        coords: [t cos(t), y, t sin(t)]
        manifold_coords: [t, y]

    """
    t, y = manifold_coords[:, 0], manifold_coords[:, 1]
    x = t * np.cos(t)
    z = t * np.sin(t)
    return np.column_stack((x,y,z))

def checkerboard(t, shift_factors=None, scale_factors=None, n_classes=2, n_occur=1):
    """Divide an n-dimensional array into classes via checkerboarding

    Parameters
    ----------
    n_classes: int or None
        Number of distinct classes
        if None, input vector will be rescaled and shifted, but not quantized.
    n_occur: int
        Number of occurrences of each class (along each axis)
    scale_factors: array of float
        Scale factor (to convert each column to the range 0...1)
    shift_factors: array of float
        Shift factor (to be added to each column to start the range at 0)

    Returns
    -------
    np.array the same shape as t, representing integer class for each data point

    """
    if t.ndim == 1:
        t = t.reshape((-1, 1))

    if scale_factors is None:
        scale_factors = np.ones(t.shape[1])
    if shift_factors is None:
        shift_factors = np.zeros(t.shape[1])

    if n_classes is None:
        return ((t + shift_factors) / scale_factors).reshape(-1)

    return np.remainder(np.sum(np.round(0.5 + (t + shift_factors) / scale_factors * n_classes * n_occur ), axis=1), n_classes)

def synthetic_data(n_points=1000, noise=0.05,
                   random_state=None, kind="unit_cube",
                   n_classes=None, n_occur=1, legacy_labels=False, **kwargs):
    """Make a synthetic dataset

    A sample dataset generators in the style of sklearn's
    `sample_generators`. This adds other functions found in the Matlab
    toolkit for Dimensionality Reduction

    Parameters
    ----------
    kind: {'unit_cube', 'swiss_roll', 'broken_swiss_roll', 'twinpeaks', 'difficult'}
        The type of synthetic dataset
    legacy_labels: boolean
        If True, try and reproduce the labels from the Matlab Toolkit for
        Dimensionality Reduction. (overrides any value in n_classes)
        This usually only works if algorithm-specific coefficient choices
        (e.g. `height` for swiss_roll) are left at their default values
    n_points : int, optional (default=1000)
        The total number of points generated.
    n_classes: None or int
        If None, target vector is based on underlying manifold coordinate
        If int, the manifold coordinate is bucketized into this many classes.
    n_occur: int
        Number of occurrences of a given class (along a given axis)
        ignored if n_classes = None
    noise : double or None (default=0.05)
        Standard deviation of Gaussian noise added to the data.
        If None, no noise is added.
    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling and noise.
        Pass an int for reproducible output across multiple function calls.

    Additional Parameters
    ---------------------
    difficult:
        n_dims: int (default 5)
            Number of dimensions to embed

    swiss_roll:
    broken_swiss_roll:
        height: float (default 30.)
            scaling to apply to y dimension

    Returns
    -------
    X : array of shape [n_points, 2]
        The generated samples.
    y : array of shape [n_points]
        The labels for class membership of each point.

    """
    generator = check_random_state(random_state)
    metadata = {
        "synthetic_type": kind,
        "n_points": n_points,
        "noise": noise
    }

    if kind == 'unit_cube':
        x = 2 * (generator.rand(n_points) - 0.5)
        y = 2 * (generator.rand(n_points) - 0.5)
        z = 2 * (generator.rand(n_points) - 0.5)
        X = np.column_stack((x, y, z))
        shift = np.array([1.])
        scale = np.array([2.])
        labels = checkerboard(X, shift_factors=shift, scale_factors=scale, n_occur=n_occur, n_classes=n_classes)
        metadata['manifold_coords'] = np.concatenate((x,y,z), axis=0).T

    elif kind == 'twinpeaks':
        inc = 1.5 / np.sqrt(n_points)
        x = np.arange(-1, 1, inc)
        xy = 1 - 2 * generator.rand(2, n_points)
        z = np.sin(np.pi * xy[0, :]) * np.tanh(3 * xy[1, :])
        X = np.vstack([xy, z * 10.]).T #  + noise * generator.randn(n_points, 3)
        t = xy.T
        metadata['manifold_coords'] = t

        if legacy_labels is True:
            labels = np.remainder(np.sum(np.round((X + np.tile(np.min(X, axis=0), (X.shape[0], 1))) / 10.), axis=1), 2)
        elif n_classes is None:
            labels = 1-z
        else:
            shift = np.array([1.])
            scale = np.array([2.])
            labels = checkerboard(t, shift_factors=shift, scale_factors=scale,
                                  n_classes=n_classes, n_occur=n_occur)

    elif kind == 'swiss_roll':
        height = kwargs.pop('height', 30.)
        t = 1.5 * np.pi * (1.0 + 2.0 * generator.rand(n_points))
        y = height * generator.rand(*t.shape)
        manifold_coords = np.column_stack((t, y))
        X = _parameterized_swiss_roll(manifold_coords)
        metadata['manifold_coords'] = manifold_coords
        if legacy_labels is True:
            labels = np.remainder(np.round(t / 2.) + np.round(height / 12.), 2)
        else:
            scale = np.array([3*np.pi])
            shift = np.array([-1.5*np.pi])
            labels =  checkerboard(t, shift_factors=shift, scale_factors=scale,
                                   n_classes=n_classes, n_occur=n_occur)


    elif kind == 'broken_swiss_roll':
        height = kwargs.pop('height', 30.)
        np1 = int(np.ceil(n_points / 2.0))
        t1 = 1.5 * np.pi * (1.0 + 2.0 * (generator.rand(np1) * 0.4))
        t2 = 1.5 * np.pi * (1.0 + 2.0 * (generator.rand(n_points - np1) * 0.4 + 0.6))
        t = np.concatenate((t1, t2))
        y = height * generator.rand(*t.shape)
        manifold_coords = np.column_stack((t, y))
        X = _parameterized_swiss_roll(manifold_coords)

        metadata['manifold_coords'] = manifold_coords
        if legacy_labels is True:
            labels = np.remainder(np.round(t / 2.) + np.round(height / 12.), 2)
        else:
            scale = np.array([3*np.pi])
            shift = np.array([-1.5*np.pi])
            labels = checkerboard(t, shift_factors=shift, scale_factors=scale,
                                  n_classes=n_classes, n_occur=n_occur)

    elif kind == 'difficult':
        n_dims = kwargs.pop("n_dims", 5)
        points_per_dim = int(np.round(float(n_points ** (1.0 / n_dims))))
        l = np.linspace(0, 1, num=points_per_dim)
        t = np.array(list(_combn(l, n_dims)))
        X = np.vstack((np.cos(t[:,0]),
                       np.tanh(3 * t[:,1]),
                       t[:,0] + t[:,2],
                       t[:,3] * np.sin(t[:,1]),
                       np.sin(t[:,0] + t[:,4]),
                       t[:,4] * np.cos(t[:,1]),
                       t[:,4] + t[:,3],
                       t[:,1],
                       t[:,2] * t[:,3],
                       t[:,0])).T
        tt = 1 + np.round(t)
        # Generate labels for dataset (2x2x2x2x2 checkerboard pattern)
        labels = np.remainder(tt.sum(axis=1), 2)
        metadata['n_dims'] = n_dims
        metadata['manifold_coords'] = t

    else:
        raise Exception(f"Unknown synthetic dataset type: {kind}")

    if noise is not None:
        X += noise * generator.randn(*X.shape)

    return X, labels, metadata

def sample_sphere_surface(n_points, n_dim=3, random_state=0, noise=None):
    '''Sample on the surface of a sphere

    See Wolfram Sphere Point Picking
    (Muller 1959, Marsaglia 1972)

    Other ways to do this: http://www-alg.ist.hokudai.ac.jp/~jan/randsphere.pdf,
    Use a very simple trick to color the points in a reasonable way
    '''

    generator = check_random_state(random_state)
    X = generator.randn(n_dim, n_points)
    X /= np.linalg.norm(X, axis=0)
    X = X.transpose()
    rgb = (X + 1.0) / 2.0  # scale to a 0...1 RGB value
    color = (rgb[:, 0] + rgb[:, 1] + rgb[:, 2]) / 3.0
    metadata = {
        "n_points": n_points,
        "n_dim": n_dim,
    }
    if noise is not None:
        X += noise * generator.randn(*X.shape)

    return X, color, metadata

def sample_ball(n_points, n_dim=3, random_state=0):
    '''Sample from a unit ball

    Use rejection sampling on the unit cube
    '''
    np.random.seed(random_state)
    points = []
    labels = []
    while len(points) < n_points:
        pt = np.random.uniform(-1.0, 1.0, n_dim)
        if np.linalg.norm(pt) < 1.0:
            points.append(pt)
            labels.append(np.linalg.norm(pt))
    X = np.array(points)
    t = np.array(labels)
    metadata = {
        "n_points": n_points,
        "n_dim": n_dim,
    }
    return X, t, metadata

def helix(n_points=1000, random_state=None, noise=0.05,
          n_twists=8, major_radius=2.0, minor_radius=1.0, n_classes=None,
          n_occur=1, legacy_labels=False):
    '''Sample from a toroidal helix; i.e. use the parameterization:

    x = R + r cos(nt)) * cos(t)
    y = R + r cos(nt)) * sin(t)
    z = r sin(nt)

    X = (x, y, z) + noise

    where $n$ is `n_twists`, $R$ is the `major_radius`
    and $r$ is the `minor_radius`, and $t$ ranges from 0 .. 2*pi

    Label is currently just $t$

    Parameters
    ----------
    major_radius:
        Major (equatorial) radius of the torus
    minor_radius:
        Minor (cross-section) radius of the torus
    legacy_labels: boolean
        If True, try and reproduce the labels from the Matlab Toolkit for
        Dimensionality Reduction. (overrides any value in n_classes)
        This usually only works if algorithm-specific coefficient choices
        (e.g. `n_twists`, radii) are left at their default values
    n_twists:
        Number of twists in the toroidal helix
    n_points:
        Number of points to return
    n_classes: int or None
        If None, target vector is the manifold coordinate (t), before noise
        If int, the manifold parameter is bucketized into this many classes.
          The non-bucketized manifold coordinates are returned as `metadata['manifold_coords']`
    random_state: int or None
        For seeding the random number generator
    noise : double or None (default=0.05)
        Standard deviation of Gaussian noise added to the data.
        If None, no noise is added.

    Returns
    -------
    X, y, metadata
    X = (x,y,z) + noise
    y = t or class number (see `n_classes`)
    '''
    generator = check_random_state(random_state)
    t = generator.uniform(0, 2*np.pi, (1, n_points))
    cosnt = np.cos(n_twists*t)
    x = (major_radius + minor_radius * cosnt) * np.cos(t)
    y = (major_radius + minor_radius * cosnt) * np.sin(t)
    z = minor_radius * np.sin(n_twists * t)
    X = np.concatenate((x,y,z))
    X = X.T
    #labels = np.linalg.norm(X, axis=1)
    if noise:
        X += noise * generator.randn(n_points, 3)

    metadata = {
        "n_points": n_points,
        "noise": noise,
        "n_twists": n_twists,
        "major_radius": major_radius,
        "minor_radius": minor_radius,
        "n_classes": n_classes,
    }

    labels = t.reshape(-1)
    if legacy_labels is True:
        labels = np.remainder(np.round(t * 1.5), 2).reshape(-1)
    elif n_classes is not None:
        metadata['manifold_coords'] = labels
        labels = checkerboard(labels, scale_factors=2*np.pi, n_classes=n_classes, n_occur=n_occur)
    return X, labels, metadata
