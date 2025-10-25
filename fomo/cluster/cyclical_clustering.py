#### Day level cyclical clustering
# get data
data_matrix = np.load('24h_data_2year.npy')

min_range = 0.2
row_ranges = np.ptp(data_matrix, axis=1)
variable_rows = data_matrix[row_ranges >= min_range]
drop_rows = data_matrix[row_ranges < min_range]
scaled_variable = StandardScaler(with_std=True).fit_transform(variable_rows.T).T

# get pairwise dtw distance
from dtaidistance import dtw
range_dist_matrix_full = dtw.distance_matrix_fast(scaled_variable, window=4, inner_dist='euclidean')
np.save('2year_minrange02_dtw_6h_radius.npy', range_dist_matrix_full)

# cluster
aggloDTW = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete', compute_full_tree=True, distance_threshold=32)
aggloDTW_labels = aggloDTW.fit_predict(range_dist_matrix_full)
np.array(np.unique(aggloDTW_labels, return_counts=True)).T

# vizualize
fig, axes = plot_time_series_by_cluster(1-scaled_variable, aggloDTW_labels, ylim=(-3,3))
plt.tight_layout()
axes[0].set_ylabel('Wear Level (Z score)')
for ax in axes:
    ax.set_title('')
    ax.set_xlabel('Hour of day')

### Week level cyclical clustering
data_matrix = np.load('2year_dayweek_avg_full.npy')
# dist_matrix_full = np.load('2year_variable030_dayweek_dtw_4h_radius.npy')

min_range = 0.3
row_ranges = np.ptp(full_data_matrix, axis=1)
variable_rows = full_data_matrix[row_ranges >= min_range]
scaled_variable_full = StandardScaler(with_std=True).fit_transform(variable_rows.T).T

aggloDTW = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete', compute_full_tree=True, distance_threshold=170)
aggloDTW_labels = aggloDTW.fit_predict(dist_matrix_full)
np.array(np.unique(aggloDTW_labels, return_counts=True)).T