import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

import math

import logging

logger = logging.getLogger()


def two_dim_label_viz(data, labels, cmap="Blues", s=10, **kwargs):
    """
    Plot data using labels as the color scheme.

    Parameters
    ----------
    data: 2d np array
    labels: 1d np array
    cmap: Default to "Blues"
    s: Default to 10

    Any other plt.scatter options as kwargs.
    """
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, s=s, **kwargs)
    plt.colorbar()


def two_dim_multiplot(data, labels_list, titles, ncols=2,
                      cmap="Blues", s=15, share_cbar=True, **kwargs):
    nrows = math.ceil(len(data)/ncols)

    if share_cbar:
        cbar_min = min(labels_list[0])
        cbar_max = max(labels_list[0])
        for label in labels_list:
            if label.dtype == 'O':
                logger.warning("Can't share colorbar "
                               "when labels are strings.")
                share_cbar = False
                break
            cbar_min = min(cbar_min, min(label))
            cbar_max = max(cbar_max, max(label))

    for i, d in enumerate(data):
        plt.subplot(nrows, ncols, i+1)
        if share_cbar:
            two_dim_label_viz(d, labels_list[i], cmap=cmap, s=s,
                              vmin=cbar_min, vmax=cbar_max, **kwargs)
        else:
            two_dim_label_viz(d, labels_list[i], cmap=cmap,
                              s=s, **kwargs)
        plt.title(titles[i])

def plot_3d_dataset(data, color_data, title='3d plot', figsize=(8,8), dim_list=None, cmap=None, **kwargs):
    '''Display a basic (colored) 3d plot
    dim_list:
        indices to use for each of the 3 dimensions of the plot
    title:
        title for the plot
    data:
        multidimensional data
    color_data:
        labels to use for coloring the points
    s: ??
    '''
    if dim_list is None:
        dim_list = [0, 1, 2]
    if cmap is None:
        cmap = plt.cm.Spectral

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, dim_list[0]], data[:, dim_list[1]], data[:, dim_list[2]],
             c=color_data, cmap=cmap)
    ax.view_init(10)
    plt.title(title)
    return ax

def sphere_plot(data, color_data, wireframe=False, title='sphere plot',
                s=50, zorder=10, dim_list=None, cmap=None,
                figsize=(8,8), **kwargs):
    '''
    '''
    if dim_list is None:
        dim_list = [0, 1, 2]
    if cmap is None:
        cmap = plt.cm.Spectral

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d', aspect='equal')
    if wireframe:
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2*np.pi, 40)
        x = np.outer(np.sin(theta), np.cos(phi))
        y = np.outer(np.sin(theta), np.sin(phi))
        z = np.outer(np.cos(theta), np.ones_like(phi))
        ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1, linewidth=1)
    ax.scatter(data[:, dim_list[0]], data[:, dim_list[1]], data[:, dim_list[2]],
             c=color_data, cmap=cmap, s=s, zorder=zorder, **kwargs)

    plt.title(title)
    return ax
