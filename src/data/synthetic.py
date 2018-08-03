from  itertools import product
import joblib.func_inspect as jfi
import numpy as np
from functools import partial
from sklearn.utils import check_random_state

def _combn(iterable, repeat):
    """Equivalent of matlab's combn"""
    return list(product(iterable, repeat=repeat))

def _parameterized_swiss_roll(t, random_state=None, k=21.0):
    """
    Given a parameter t, return a swiss roll

    k: constant in height (y-coordinate) computation
    random_state : int, RandomState instance or None (default)
        Pass an int for reproducible output across multiple function calls.
    """
    generator = check_random_state(random_state)
    x = t * np.cos(t)
    y = k * generator.rand(*t.shape)
    z = t * np.sin(t)
    t = np.squeeze(t)
    return np.concatenate((x,y,z)).T, t

def synthetic_data(n_points=1000, noise=0.05,
                   random_state=None, kind="unit_cube", **kwargs):
    """Make a synthetic dataset

    A sample dataset generators in the style of sklearn's
    `sample_generators`. This adds other functions found in the Matlab
    toolkit for Dimensionality Reduction

    Parameters
    ----------
    kind: {'unit_cube', 'broken_swiss_roll', 'twinpeaks', 'difficult'}
        The type of synthetic dataset
    n_points : int, optional (default=1000)
        The total number of points generated.
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
        x = 2 * (generator.rand(1, n_points) - 0.5)
        y = 2 * (generator.rand(1, n_points) - 0.5)
        z = 2 * (generator.rand(1, n_points) - 0.5)
        X = np.concatenate((x, y, z))
        X = X.T
        t = np.linalg.norm(X, axis=1)
    elif kind == 'twinpeaks':
        inc = 1.5 / np.sqrt(n_points)
        x = np.arange(-1, 1, inc)
        xy = 1 - 2 * generator.rand(2, n_points)
        z = np.sin(np.pi * xy[0, :]) * np.tanh(3 * xy[1, :])
        X = np.vstack([xy, z]).T #  + noise * generator.randn(n_points, 3)
        #X[:, 2] = X[:, 2] * 1
        # t = xy.T
        t = 1-z

    elif kind == 'broken_swiss_roll':
        np1, np2 = int(np.ceil(n_points / 2.0)), int(np.floor(n_points / 2.0))
        t1 = 1.5 * np.pi * (1.0 + 2.0 * (generator.rand(1, np1) * 0.4))
        t2 = 1.5 * np.pi * (1.0 + 2.0 * (generator.rand(1, np2) * 0.4 + 0.6))

        X1,t1 = _parameterized_swiss_roll(t1, random_state=generator)
        X2,t2 = _parameterized_swiss_roll(t2, random_state=generator)

        X, t = np.concatenate((X1, X2)), np.concatenate((t1, t2))
    elif kind == 'difficult':
        n_dims = kwargs.pop("n_dims", 5)
        points_per_dim = int(np.round(float(n_points ** (1.0 / n_dims))))
        l = np.linspace(0, 1, num=points_per_dim)
        t = np.array(list(_combn(l, n_dims)))
        X = np.vstack((np.cos(t[:,0]),  np.tanh(3 * t[:,1]),  t[:,0] + t[:,2],  t[:,3] * np.sin(t[:,1]),  np.sin(t[:,0] + t[:,4]),
             t[:,4] * np.cos(t[:,1]), t[:,4] + t[:,3], t[:,1], t[:,2] * t[:,3],  t[:,0])).T
        tt = 1 + np.round(t)
        # Generate labels for dataset (2x2x2x2x2 checkerboard pattern)
        t = np.remainder(tt.sum(axis=1), 2)
        metadata['n_dims'] = n_dims

    else:
        raise Exception(f"Unknown synthetic dataset type: {kind}")

    if noise is not None:
        X += noise * generator.randn(*X.shape)

    return X, t, metadata

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
          n_twists=8, major_radius=2.0, minor_radius=1.0):
    '''Sample from a toroidal helix; i.e. use the parameterization: 

    x = R + r cos(nt)) * cos(t)
    y = R + r cos(nt)) * sin(t)
    z = r sin(nt)

    where $n$ is `n_twists`, $R$ is the `major_radius`
    and $r$ is the `minor_radius`, and $t$ ranges from 0 .. 2*pi

    Label is currently just $t$

    Parameters
    ----------
    major_radius:
        Major (equatorial) radius of the torus
    minor_radius:
        Minor (cross-section) radius of the torus
    n_twists:
        Number of twists in the toroidal helix
    n_points:
        Number of points to return
    random_state: int or None
        For seeding the random number generator
    noise : double or None (default=0.05)
        Standard deviation of Gaussian noise added to the data.
        If None, no noise is added.

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
    labels = t.reshape(-1)
    if noise:
        X += noise * generator.randn(n_points, 3)

    metadata = {
        "n_points": n_points,
        "noise": noise,
        "n_twists": n_twists,
        "major_radius": major_radius,
        "minor_radius": minor_radius
    }
    return X, labels, metadata
