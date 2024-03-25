import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from collections import defaultdict
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans


class Cluster_DTW_KMeans(TimeSeriesKMeans):

    """Wrapper class for TSLearn DTW KMeans, with relevant pre-processing and tools for clustering."""

    def __init__(self, n_clusters):
        super().__init__(n_clusters, metric='dtw')

    def fit(self, missing_data_matrix):
        # Some reshaping of missing_data_matrix
        reshaped_missing_data_matrix = to_time_series_dataset(missing_data_matrix)
        
        return super().fit(reshaped_missing_data_matrix)  

    @classmethod
    def find_n_clusters(cls, missing_data_matrix, clusters: list, method: str):
        if method == 'elbow':
            # calculate the sum of squared distances for each number of clusters
            sse = []
            for cluster in clusters:
                kmeans = cls(cluster)
                kmeans.fit(missing_data_matrix)
                sse.append(kmeans.inertia_)
            
            # calculate the elbow point
            elbow_point = np.diff(sse, 2)
            elbow_point = elbow_point.argmax() + 2
            
            # plot the elbow plot with dots at each number of clusters
            plt.plot(clusters, sse, 'o-')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Sum of Squares')
            plt.title('Optimal K Using Elbow Method')

            # draw a vertical line at the elbow point
            plt.axvline(x=elbow_point, color='black', linestyle='--')

            plt.show()
        
        elif method == 'silhouette':
            # calculate the silhouette score for each number of clusters
            silhouette_scores = []
            for cluster in clusters:
                if cluster == 1:
                    silhouette_scores.append(0)
                    continue
                kmeans = cls(cluster)
                kmeans.fit(missing_data_matrix)
                silhouette_scores.append(silhouette_score(missing_data_matrix, kmeans.labels_))
            
            # the optimal number of clusters is the one with the highest silhouette score
            optimal_k = clusters[silhouette_scores.index(max(silhouette_scores))]

            # plot the silhouette scores
            plt.plot(clusters, silhouette_scores, 'o-')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.title('Optimal K Using Silhouette Score')
            
            # draw a vertical line at the elbow point
            plt.axvline(x=optimal_k, color='black', linestyle='--')

            plt.show()
            
        elif method == 'gap':
            # set max reference distributions
            nrefs = 5
        
            # calculate the gap statistic for each number of clusters
            gaps = np.zeros((len(clusters),))
            results_dic = defaultdict(list)

            # iterate over the range of clusters
            for gap_index, k in enumerate(clusters):
                ref_disps = np.zeros(nrefs)

                # for n references, generate random sample and perform kmeans
                for i in range(nrefs):
                    # create a uniform random reference distribution
                    random_uniform_distribution = np.random.random_sample(size=missing_data_matrix.shape)

                    # fit to the distribution
                    kmeans = cls(k)
                    kmeans.fit(random_uniform_distribution)

                    # calculate the dispersion
                    ref_disp = kmeans.inertia_
                    ref_disps[i] = ref_disp
                
                # calculate dispersion on the original data
                kmeans = cls(k)
                kmeans.fit(missing_data_matrix)

                # get og dispersion
                orig_disp = kmeans.inertia_

                # calculate the gap statistic
                gap = np.log(np.mean(ref_disps)) - np.log(orig_disp)

                # store the results
                gaps[gap_index] = gap
                results_dic[k] = gap
            
            # best k
            optimal_k = gaps.argmax() + 1
                
            # plot the gap statistic
            plt.plot(results_dic.keys(), results_dic.values(), 'o-')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Gap Value')
            plt.title('Optimal K Using Gap Statistic')

            # draw a vertical line at the elbow point
            plt.axvline(x=optimal_k, color='black', linestyle='--')

            plt.show()

            # return optimal number of clusters
            return optimal_k
    
        else:
            raise ValueError("Invalid method. Please choose from 'elbow', 'silhouette', or 'gap'.")