import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import ImageGrid
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
