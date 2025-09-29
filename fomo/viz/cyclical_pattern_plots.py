def plot_cluster_pattern_plots(time_series, labels, noscale_labels = [], ncols = 3):
    unique_labels = np.unique(labels)
    
    nrows = int(np.ceil(len(unique_labels) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 1 * nrows))
    
    for label, ax in zip(unique_labels, axes.flatten()):
        clust_data = time_series[labels == label]
        barycenter = softdtw_barycenter(clust_data, gamma=1., max_iter=50, tol=1e-3)
        if label in noscale_labels:
            scaled_barycenter = barycenter.flatten()
        else:
            scaled_barycenter = MinMaxScaler().fit_transform(barycenter).flatten()
        cax = ax.pcolormesh(np.expand_dims(scaled_barycenter,axis=0), cmap='Blues_r',
                           vmin = 0,
                           vmax = 1)#, edgecolors='k', linewidth=0.1)
        ax.set_xticks(np.linspace(0,len(scaled_barycenter),5, dtype=int))
        ax.set_xticklabels([0,6,12,18,24])
        ax.set_yticks([0,1])
        ax.set_title(f"n={len(clust_data)}")
        ax.set_yticklabels([' ', ' '])

## usage:
pts_wclust=read_from_cloud('2year_minrange02_dtw_4h_wclust.csv')
aggloDTW_labels = pts_wclust['cluster'].replace({3: 0, 1 : 0, 2:0, 7 :1, 6:1})
plot_cluster_pattern_plots(scaled_variable, aggloDTW_labels, ncols=1)
plt.tight_layout()