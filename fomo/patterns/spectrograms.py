import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def generate_spectrograms(missing_data_matrix):
    freqs_arr = []
    times_arr = []
    Sxx_arr = []

    fs = 1 / (missing_data_matrix.columns[2] - missing_data_matrix.columns[1]).total_seconds()   # Sampling frequency

    for i in range(len(missing_data_matrix['person_id'])):
    
        # Generate spectrogram
        signal = missing_data_matrix.loc[i][1:]
        signal = signal.T.values

        frequencies, times, Sxx = spectrogram(signal, fs)

        # Keep freq, time, Sxx for plotting spectrogram later
        freqs_arr.append(frequencies)
        times_arr.append(times)
        Sxx_arr.append(Sxx)

    spectrograms = pd.DataFrame({
        'person_id': missing_data_matrix['person_id'],
        'frequencies': freqs_arr,
        'times': times_arr,
        'Sxx': Sxx_arr
    })

    return spectrograms

def cluster_spectrograms(spectrograms, ClusteringClass, n_clusters):

    # Convert person_id, Sxx to format suitable for clustering
    expanded_df = pd.DataFrame({
        'person_id': spectrograms['person_id'],
        'Sxx': [spec.flatten() for spec in spectrograms['Sxx']]
    })

    expanded_df = expanded_df.set_index('person_id')
    expanded_df = pd.DataFrame(expanded_df['Sxx'].tolist(), index=expanded_df.index)

    # Cluster based on specified class
    cluster = ClusteringClass(n_clusters = n_clusters)
    cluster.fit(expanded_df)
    clusters = cluster.labels_

    return clusters


def plot_spectrogram(times, frequencies, Sxx):

    plt.pcolormesh(times, frequencies*3600*24, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency (1 / day)')
    plt.xlabel('Time (min)')
    plt.title('Spectrogram Person 1')
    plt.colorbar(label='Intensity')
    plt.show()