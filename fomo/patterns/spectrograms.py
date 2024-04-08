import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


# generate_spectrogram(person_data, missingness_interval, window_length_minutes, noverlap=None, zero_pad=0) instead, 
# sampling frequency = 1/missingness_interval (from flag_missing_data())
# calculate nperseg from window_length_minutes / missingness_interval
# for anything above zero_pad=0 make nfft=nperseg+zero_pad

def generate_spectrogram(person_missing_data, missingness_interval, window_length_minutes, noverlap=None, zero_pad=0):

    """
    Generate spectrogram from the missingness data of one person. Returns frequency, time, Sxx from scipy.siginal.spectrogram

    person_missing_data - str, missingness data for one person
    missingness_interval - Integer, how big of a gap there is between data points in output table in minutes (aggregated
    raw data points into 'bins' of size interval_size)
    window_length_minutes - Integer, how many minutes there are of data in person_missing_data
    noverlap - for scipy.signal.spectrogram()
    zero_pad - Integer, how many zeros to pad for FFT
    """

    # Calculate sampling frequency
    fs = 1 / missingness_interval

    # Calculate nperseg
    nperseg = window_length_minutes / missingness_interval

    # Calculate nfft
    nfft = None if zero_pad == 0 else nperseg+zero_pad

    # Get spectrogram
    freq, time, Sxx = spectrogram(person_missing_data, fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    return freq, time, Sxx


def generate_spectrograms(missing_data_matrix, missingness_interval, window_length_minutes, noverlap=None, zero_pad=0):
    freqs_arr = []
    times_arr = []
    Sxx_arr = []

    for i in range(len(missing_data_matrix['person_id'])):
    
        # Generate spectrogram
        person_missing_data = missing_data_matrix.loc[i][1:]
        person_missing_data = person_missing_data.T.values

        frequencies, times, Sxx = generate_spectrogram(person_missing_data, missingness_interval, window_length_minutes, noverlap, zero_pad)

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