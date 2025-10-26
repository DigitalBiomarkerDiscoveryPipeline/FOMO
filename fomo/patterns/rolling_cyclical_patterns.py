
import pandas as pd
import numpy as np
def rolling_week_pattern(flagged_df, time_column=None, basis_rate=15, missingness_interval=60, n_weeks=1, n_overlap=0, missing_flags_column = 'Missing_Flag', return_index=False):
    if n_overlap >= n_weeks: raise ValueError("n_overlap should be less than n_weeks")
    if missingness_interval < basis_rate: raise ValueError("missingness_interval should be >= basis_rate")
    if missingness_interval % basis_rate: raise ValueError('missingness_interval should be a multiple of basis_rate')
    if 24*60 % missingness_interval: raise ValueError('missingness_interval should be chosen such that it divides one day into an integer number of intervals')
    
    # Convert Missing_Flag to integer to calculate percentages later
    flagged_df['Missing_Flag'] = flagged_df['Missing_Flag'].astype(int)
    
    if time_column is not None:
        matrix = flagged_df.set_index(time_column)
    else:
        matrix = flagged_df.sort_index()
        if flagged_df.index.name is None:
            time_column = 'index'
        else:
            time_column = flagged_df.index.name
    
    intervals_per_group = missingness_interval // basis_rate
    
    # Use rolling mean to calculate the percentage of missing values within each interval
    # Note: The window size is set to intervals_per_group, and min_periods is set to 1 to ensure that we get a value even if there's only one non-missing value in the window.
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=intervals_per_group) # makes sure that the timestamp for missing_data_matrix represents the beginning of the interval
    resampled_matrix = matrix.rolling(window=indexer, min_periods=1).mean()
    
    # let the index be just numbers and move the datatime to be a new column
    resampled_matrix = resampled_matrix.reset_index()

    # only keep datetime and missingness columns
    resampled_matrix = resampled_matrix[[time_column, missing_flags_column]]

    # Since rolling mean includes the current and previous (window-1) columns, we need to select every intervals_per_group-th row to get non-overlapping intervals
    resampled_matrix = resampled_matrix.iloc[[0] + [i for i in range(intervals_per_group, len(resampled_matrix), intervals_per_group)], :]

    # create intermediate pandas df that contains all values needed to calculate any set of input axes
    missingness_matrix_week = resampled_matrix.copy()

    # Number each week, starting at every Sunday
    missingness_matrix_week['Day'] = resampled_matrix[time_column].dt.day_name()
    start_date = missingness_matrix_week[time_column].iloc[0]
    start_of_week = start_date - pd.Timedelta(days = start_date.dayofweek + 1)
    missingness_matrix_week['Week'] = ((missingness_matrix_week[time_column] - start_of_week).dt.days //7)
    missingness_matrix_week['Day-Week'] = missingness_matrix_week['Day'] + missingness_matrix_week['Week'].astype(str)


    # Drop weeks without 7 days
    week_daynums = missingness_matrix_week.groupby('Week')['Day'].nunique()
    invalid_weeks = week_daynums[week_daynums<7].index
    missingness_matrix_week = missingness_matrix_week[~missingness_matrix_week['Week'].isin(invalid_weeks)]
    
    missingness_matrix_week = missingness_matrix_week.reset_index(drop=True)
    
    # Make array where each 2-dimensional week is a time point
    day_lengths = missingness_matrix_week.groupby('Day-Week').size().unique()
    if len(day_lengths) > 1: raise ValueError('All days must have the same number of data points. Is this time series regularly sampled?')
    day_length = day_lengths[0]
    week_array = missingness_matrix_week[missing_flags_column].to_numpy().reshape(-1, 7, day_length)
    
    # Calculate rolling mean of week_array, with the appropriate stride given n_overlap 
    cumsum = np.cumsum(week_array, axis=0)
    window = n_weeks
    stride = window - n_overlap
    invSize = 1. / window
    rolling_means = (cumsum[window-1:] - np.concatenate([np.zeros((1,week_array.shape[1],week_array.shape[2])), cumsum[:-window]],axis=0)) * invSize
    means_withstride = rolling_means[::stride]
    if return_index:
        # Return start date for each index
        index_stride = int((24*60) / missingness_interval * 7 * stride)
        return means_withstride, missingness_matrix_week['index'][::index_stride].reset_index(drop=True)
    return means_withstride


def rolling_day_pattern(flagged_df, time_column=None, basis_rate=15, missingness_interval=60, n_days=1, n_overlap=0, missing_flags_column = 'Missing_Flag', full_weeks_only=False, return_index=False):
    """Turn flagged time series into array of day patterns
    flagged_df: pd dataframe with time series index or column 
    time_column: which column is the datetime labels. If None, the index will be used. (use pd.to_datetime() to make the dtype datetime)
    basis_rate: minute sampling period from flagged_df
    missingness_interval: target period of missingness (when larger than basis rate, will get mean)
    n_days: number of days to use for pattern
    n_overlap: number of days to overlap, when n_days is >1 
    missing_flags_column: which column in flagged_df is the binary indicator of missingness
    full_weeks_only: when True, will remove incomplete sunday-saturday weeks from the end and start of the time series
        * e.g. n_days=7, n_overlap=0, full_weeks_only=True will make average patterns from Sunday-Saturday periods
    return_index: when True, will return time series labels (for start of each pattern)
    """
    if n_overlap >= n_days: raise ValueError("n_overlap should be less than n_days")
    if missingness_interval < basis_rate: raise ValueError("missingness_interval should be >= basis_rate")
    if missingness_interval % basis_rate: raise ValueError('missingness_interval should be a multiple of basis_rate')
    if 24*60 % missingness_interval: raise ValueError('missingness_interval should be chosen such that it divides one day into an integer number of intervals')
    
    # Convert Missing_Flag to integer to calculate percentages later
    flagged_df['Missing_Flag'] = flagged_df['Missing_Flag'].astype(int)
    
    if time_column is not None:
        matrix = flagged_df.set_index(time_column)
    else:
        matrix = flagged_df.sort_index()
        if flagged_df.index.name is None:
            time_column = 'index'
        else:
            time_column = flagged_df.index.name
    
    intervals_per_group = missingness_interval // basis_rate
    
    # Use rolling mean to calculate the percentage of missing values within each interval
    # Note: The window size is set to intervals_per_group, and min_periods is set to 1 to ensure that we get a value even if there's only one non-missing value in the window.
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=intervals_per_group) # makes sure that the timestamp for missing_data_matrix represents the beginning of the interval
    resampled_matrix = matrix.rolling(window=indexer, min_periods=1).mean()
    
    # let the index be just numbers and move the datatime to be a new column
    resampled_matrix = resampled_matrix.reset_index()

    # only keep datetime and missingness columns
    resampled_matrix = resampled_matrix[[time_column, missing_flags_column]]

    # Since rolling mean includes the current and previous (window-1) columns, we need to select every intervals_per_group-th row to get non-overlapping intervals
    resampled_matrix = resampled_matrix.iloc[[0] + [i for i in range(intervals_per_group, len(resampled_matrix), intervals_per_group)], :]

    # create intermediate pandas df that contains all values needed to calculate any set of input axes
    missingness_matrix_week = resampled_matrix.copy()

    # Number each week, starting at every Sunday
    missingness_matrix_week['Day'] = resampled_matrix[time_column].dt.day_name()
    start_date = missingness_matrix_week[time_column].iloc[0]
    start_of_week = start_date - pd.Timedelta(days = start_date.dayofweek + 1)
    missingness_matrix_week['Week'] = ((missingness_matrix_week[time_column] - start_of_week).dt.days //7)
    missingness_matrix_week['Day-Week'] = missingness_matrix_week['Day'] + missingness_matrix_week['Week'].astype(str)


    # Drop weeks without 7 days
    week_daynums = missingness_matrix_week.groupby('Week')['Day'].nunique()
    if full_weeks_only:
        invalid_weeks = week_daynums[week_daynums<7].index
        missingness_matrix_week = missingness_matrix_week[~missingness_matrix_week['Week'].isin(invalid_weeks)]
    
    missingness_matrix_week = missingness_matrix_week.reset_index(drop=True)
    
    # Make array where each 2-dimensional day is a time point
    day_lengths = missingness_matrix_week.groupby('Day-Week').size().unique()
    if len(day_lengths) > 1: raise ValueError('All days must have the same number of data points. Is this time series regularly sampled?')
    day_length = day_lengths[0]
    day_array = missingness_matrix_week[missing_flags_column].to_numpy().reshape(-1,day_length)
    
    # Calculate rolling mean of week_array, with the appropriate stride given n_overlap 
    cumsum = np.cumsum(day_array, axis=0)
    window = n_days
    stride = window - n_overlap
    invSize = 1. / window
    rolling_means = (cumsum[window-1:] - np.concatenate([np.zeros((1,day_array.shape[1])), cumsum[:-window]],axis=0)) * invSize
    means_withstride = rolling_means[::stride]
    if return_index:
        # Return start date for each index
        index_stride = int((24*60) / missingness_interval * stride)
        return means_withstride, missingness_matrix_week['index'][::index_stride].reset_index(drop=True)
    return means_withstride

def rolling_day_pattern_fast(flagged_df, time_column=None, basis_rate=15, missingness_interval=60, n_days=1, missing_flags_column='Missing_Flag', return_index=False):
    '''Fast version of rolling_day_pattern. Only works with n_overlap=0'''
    if missingness_interval < basis_rate: raise ValueError("missingness_interval should be >= basis_rate")
    if missingness_interval % basis_rate: raise ValueError('missingness_interval should be a multiple of basis_rate')
    if 24*60 % missingness_interval: raise ValueError('missingness_interval should be chosen such that it divides one day into an integer number of intervals')
    
    # Convert Missing_Flag to integer to calculate percentages later
    flagged_df['Missing_Flag'] = flagged_df['Missing_Flag'].astype(int)
    
    if time_column is not None:
        matrix = flagged_df.set_index(time_column)
    else:
        matrix = flagged_df.sort_index()
        if flagged_df.index.name is None:
            time_column = 'index'
        else:
            time_column = flagged_df.index.name
    
    intervals_per_group = missingness_interval // basis_rate
    
    missing_array = flagged_df['Missing_Flag'].to_numpy()
    
    # Aggregate
    day_pat = np.mean(
        missing_array.reshape(-1, intervals_per_group), # aggregate every interval
        axis=1
    ).reshape(-1,int(24//(missingness_interval/60))) # shape into matrix with shape (n days, length of a day in intervals)
    
    if return_index:
        # Reshape to each interval
        day_ind = flagged_df.index.to_numpy().reshape(-1, int(24//(missingness_interval/60)), intervals_per_group)
        day_ind = day_ind[:,0,0]
        return day_pat, day_ind
        
    
    return day_pat
