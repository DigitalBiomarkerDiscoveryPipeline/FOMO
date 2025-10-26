from tslearn.barycenters import softdtw_barycenter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_time_series_by_cluster(time_series, labels, ylim=(0,1), ncols=5):
    """Patterns as line plots"""
    unique_labels = np.unique(labels)
    nrows = int(np.ceil(len(unique_labels) / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    
    for i, label in enumerate(unique_labels):
        cluster_indices = np.where(labels == label)[0]
        cluster_series = []
        for idx in cluster_indices:
            axs.flatten()[i].plot(time_series[idx], "k-", alpha=0.2, label=f"Cluster {label} - Series {idx}")
            cluster_series.append(time_series[idx])
        
        barycenter = softdtw_barycenter(cluster_series, gamma=1., max_iter=50, tol=1e-3)
        axs.flatten()[i].plot(barycenter, "r-", linewidth=2, label=f"Barycenter {label}")
        axs.flatten()[i].set_ylim(ylim)
        axs.flatten()[i].set_title(f"Time Series in Cluster {label}")
    return fig, axs

def cluster_pattern_plots(pattern_matrix, labels, axes = None, noscale_labels = [], ncols = 1, show_colorbar=True, gamma=1.):
    '''Plot day-level (1D) or week-level (2D) pattern plot heatmaps as the barycenter of each cluster.
    
    Parameters
    ----------
    pattern_matrix: np.array
        2-D or 3-D matrix of patterns, which has shape ([number of time series in matrix], [length of each time series]) for 2D
        Or shape ([number of time series in matrix], [number of rows of time series], [length of each row of time series])
    labels: array-like
        Array of labels corresponding to each time series in `pattern_matrix`
    axes: matplotlib Axes array (default: None)
        Array of matplotlib axes for plotting the heatmaps. If None is provided, a fig and axes will be created.
    noscale_labels: list (default: [])
        List of labels found in `labels` which should not be passed through MinMaxScaler
    ncols: int (default: 1)
        Number of columns in figure. The number of rows will be calculated to fit the number of unique patterns
    show_colorbar: bool (default: True)
        If true, shows a colorbar on the right of the figure. Colorbar will always use the full range of the cmap.
    
    Returns
    -------
    fig: matplotlib Figure
    axes: matplotlib Axes
    '''
    unique_labels = np.unique(labels)
    
    nrows = int(np.ceil(len(unique_labels) / ncols))

    if axes is None:
        if len(pattern_matrix.shape) == 3:
            figsize = (4*ncols, 2*nrows) # Make week level plots taller
        else:
            figsize = (4*ncols, 1*nrows) # Make day level plots shorter
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    else:
        fig = plt.gcf()
        if  np.prod(axes.shape) < len(unique_labels):
            raise ValueError(f'Provided array of axes is too small. Expected at least {len(unique_labels)} axes, but provided axes array has only {np.prod(axes.shape)}')
    
    for label, ax in zip(unique_labels, axes.flatten()):
        clust_data = pattern_matrix[labels == label]
        clust_data = clust_data.reshape(clust_data.shape[0], -1)
        barycenter = softdtw_barycenter(clust_data, gamma=gamma, max_iter=50, tol=1e-3)
        if label in noscale_labels:
            scaled_barycenter = barycenter.flatten()
        else:
            scaled_barycenter = MinMaxScaler().fit_transform(barycenter).flatten()
        if len(pattern_matrix.shape) == 2: # day level plotting
            cax = ax.pcolormesh(np.expand_dims(scaled_barycenter,axis=0), cmap='Blues_r',
                            vmin = 0,
                            vmax = 1)#, edgecolors='k', linewidth=0.1)
            ax.set_yticks([0,1])
            ax.set_yticklabels([' ', ' '])
            ax.set_xticks(np.linspace(0,len(scaled_barycenter),5, dtype=int))

        elif len(pattern_matrix.shape) == 3: # week level plotting
            scaled_barycenter = scaled_barycenter.reshape(7, -1)
            cax = ax.pcolormesh(scaled_barycenter[::-1], # invert y axis to make the ordering Sunday-Saturday from top to bottom
                                cmap='Blues_r',
                           vmin = 0,
                           vmax = 1)#, edgecolors='k', linewidth=0.1)
            ax.set_yticks(np.arange(7) + 0.5)
            ax.set_yticklabels(['Sa','F','R','W','T','M','Su'])
            ax.set_xticks(np.linspace(0,scaled_barycenter.shape[1],5, dtype=int))
        ax.set_xticklabels([0,6,12,18,24])
        
        
        ax.set_title(f"n={len(clust_data)}")
        
    if show_colorbar:
        # Make a colorbar that always shows the full range of the colormap
        norm = mcolors.Normalize(vmin=0, vmax=1)
        mappable = cm.ScalarMappable(norm=norm, cmap='Blues_r')
        fig.colorbar(mappable, ax=axes.flatten())
    return fig, axes
