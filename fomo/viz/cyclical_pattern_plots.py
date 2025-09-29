from tslearn.barycenters import softdtw_barycenter

def plot_time_series_by_cluster(time_series, labels, ylim=(0,1)):
    """Patterns as line plots"""
    unique_labels = np.unique(labels)
    ncols = 5
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


########## Add in inconsistent missingness
aggloDTW_labels
mod_aggloDTW_labels = aggloDTW_labels.copy()
label_map = {0: 2, 1:3, 2:0, 3:1, 4:0}
for original_value, mod_value in label_map.items():
    mod_aggloDTW_labels[aggloDTW_labels == original_value] = mod_value

flat_rows = data_matrix[row_ranges < min_range]
flat_missing_level = flat_rows.mean(axis=1)

flat_low_missing = flat_rows[flat_missing_level <= 0.5]
flat_high_missing = flat_rows[flat_missing_level > 0.5]
# scaled_flat = StandardScaler(with_std=False).fit_transform(flat_rows.T).T
# scaled_flat = flat_rows
all_scaled = np.concatenate([scaled_variable, flat_low_missing, flat_high_missing])
low_missing_labels = np.full(len(flat_low_missing), 999)
high_missing_labels = np.full(len(flat_high_missing), 998)
# unclustered_labels = np.full(len(scaled_flat), 999)
all_labels = np.concatenate([mod_aggloDTW_labels, low_missing_labels, high_missing_labels])

plot_cluster_pattern_plots(all_scaled, all_labels, noscale_labels = [998, 999], ncols=1)
plt.tight_layout()
plt.savefig('fig2b.pdf')