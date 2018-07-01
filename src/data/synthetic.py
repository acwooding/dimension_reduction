import numpy as np


def sample_sphere_surface(n_points, n_dim=3, random_state=0):
    '''Sample on the surface of a sphere

    See Wolfram Sphere Point Picking
    (Muller 1959, Marsaglia 1972)

    Other ways to do this: http://www-alg.ist.hokudai.ac.jp/~jan/randsphere.pdf,
    Use a very simple trick to color the points in a reasonable way
    '''

    np.random.seed(random_state)
    vec = np.random.randn(n_dim, n_points)
    vec /= np.linalg.norm(vec, axis=0)
    vec = vec.transpose()
    rgb = (vec + 1.0) / 2.0  # scale to a 0...1 RGB value
    color = (rgb[:, 0] + rgb[:, 1] + rgb[:, 2]) / 3.0

    return vec, color

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
    return np.array(points), np.array(labels)

def helix(major_radius, minor_radius, n_twists=20, n_points=1000, random_state=None, random=False):
    '''
    Sample from a toroidal helix.

    major_radius:
        Major (equatorial) radius of the torus
    minor_radius:
        Minor (cross-section) radius of the torus
    n_twists:
        Number of twists in the toroidal helix
    n_points:
        Number of points to return
    random_state: int
        For seeding the random number generator
    random: boolean
        If True, randomly sample from a toroidal helical path
        If False, generate uniformly on the parameterized interval

    https://math.stackexchange.com/questions/324527/do-these-equations-create-a-helix-wrapped-into-a-torus

    '''
    if random_state is not None:
        np.random.seed(random_state)
        random = True

    if random is True:
        steps = np.random.uniform(0, 2*np.pi, n_points)
    else:
        steps = np.linspace(0, 2*np.pi, num=n_points)
    points = []
    labels = []
    for t in steps:
        x = (major_radius + minor_radius * np.cos(n_twists*t)) * np.cos(t)
        y = (major_radius + minor_radius * np.cos(n_twists*t)) * np.sin(t)
        z = minor_radius * np.sin(n_twists * t)
        points.append((x,y,z))
        labels.append(t / 2 / np.pi)
    return np.array(points), np.array(labels)
