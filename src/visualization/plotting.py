import matplotlib.pyplot as plt

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
    plt.scatter(data[:,0], data[:,1], c=labels, cmap=cmap, s=s, **kwargs)
    plt.colorbar();
