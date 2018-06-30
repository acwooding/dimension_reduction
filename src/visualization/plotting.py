import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid


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
    plt.colorbar();


def two_dim_multiplot(data, labels, titles, cmap="Blues", s=15, **kwargs):
    ncols = 2
    nrows = int(len(data)/ncols)

    fig = plt.figure(figsize=(14*nrows, 5*ncols))

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(nrows, ncols),
                    axes_pad=0.05,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )

    cbar_min, cbar_max = min(labels[0]), max(labels[0])

    for label in labels:
        cbar_min = min(cbar_min, min(label))
        cbar_max = max(cbar_max, max(label))

    for i, ax in enumerate(grid):
        ax.set_axis_off()
        im = ax.scatter(data[i][:, 0], data[i][:, 1],
                        c=labels[i], cmap=cmap, s=s,
                        vmin=cbar_min, vmax=cbar_max)
        ax.set_title(titles[i])

    # when cbar_mode is 'single', for ax in grid, ax.cax = grid.cbar_axes[0]

    # cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im);
