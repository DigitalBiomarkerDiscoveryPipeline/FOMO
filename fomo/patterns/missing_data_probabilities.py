import numpy as np
import pandas as pd

def missing_data_probabilities(flagged_df, time_column, axes, basis_rate=15, missingness_interval=15, missing_flags_column='Missing_Flag'):
    """
    Calculate missingness probability and variance matrices (or tensors) along a time axis
    An axis can be "day", "week", "year"
    
    Parameters
    ----------
    flagged_df: pandas.DataFrame
	    Dataframe with flagged missingness for one person. Flags can be obtained with flag_missing_data
    time_column: str
	    Name of column in dataframe which contains the time information. Column should have dtype datetime
    axes: str or list of str's ["day", "week", "year"]
	    Axis of the final output vector/matrix/tensor which denotes a unit/scale of time, along which to check for missingness
    basis_rate: Integer, how big of a gap there is between data points in output table in minutes
    missingness_interval: For fractional missingness, how big of interval to consider in final matrix.
	  missing_flags_column: str default='Missing_Flag'
		  Name of column in dataframe which contains boolean missingness flags. 
		  If dataframe was generated with flag_missing_data, the default value "Missing_Flag" should be correct
		  
    """
    
    # Step 1: Implement similar logic to missing_data_matrix, but for one person
    # Instead of making each time point be one column, each time point should be a row. Therefore, make a "Time" column in the intermediate dataframe, or make time the index of a pandas Series
    
     # Check if flagged_df has Missing Flag column
    if 'Missing_Flag' not in flagged_df.columns:
        raise Exception("No Missing Flags in provided dataframe")

    # Check if missingness_interval > basis_rate
    if missingness_interval < basis_rate:
        raise Exception("Resampling Error: missingness_interval must be greater than basis_rate.")
    
     # Convert Missing_Flag to integer to calculate percentages later
    flagged_df['Missing_Flag'] = flagged_df['Missing_Flag'].astype(int)

    # Create matrix where we take all columns but the datatime column
    matrix = flagged_df

    # perform rolling mean to get the percentage of missing values within each interval
    matrix = matrix.set_index(time_column)

    # Calculate the number of basis_rate intervals within each missingness_interval
    intervals_per_group = missingness_interval // basis_rate

    # Use rolling mean to calculate the percentage of missing values within each interval
    # Note: The window size is set to intervals_per_group, and min_periods is set to 1 to ensure that we get a value even if there's only one non-missing value in the window.
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=intervals_per_group) # makes sure that the timestamp for missing_data_matrix represents the beginning of the interval
    resampled_matrix = matrix.rolling(window=indexer, axis=0, min_periods=1).mean()

    # let the index be just numbers and move the datatime to be a new column
    resampled_matrix = resampled_matrix.reset_index()

    # only keep datetime and missingness columns
    resampled_matrix = resampled_matrix[[time_column, missing_flags_column]]

    # Since rolling mean includes the current and previous (window-1) columns, we need to select every intervals_per_group-th row to get non-overlapping intervals
    resampled_matrix = resampled_matrix.iloc[[0] + [i for i in range(intervals_per_group, len(resampled_matrix), intervals_per_group)], :]

    # get and store the unique time values, day values, and number of weeks
    time_values = resampled_matrix[time_column].dt.strftime('%H:%M').unique()
    day_values = resampled_matrix[time_column].dt.day_name().unique()
    month_values = resampled_matrix[time_column].dt.month_name().unique()
    num_weeks = np.ceil(len(resampled_matrix) / (len(day_values) * len(time_values))).astype(int)

    # create intermediate pandas df that contains all values needed to calculate any set of input axes
    missingness_matrix_week = resampled_matrix.copy()
    
    # every set of len(day_values) * len(time_values) rows will have the same week value up to the last value in the matrix
    missingness_matrix_week['Month'] = resampled_matrix[time_column].dt.month_name()
    missingness_matrix_week['Cal-Day'] = resampled_matrix[time_column].dt.day
    missingness_matrix_week['Week'] = np.repeat(np.arange(num_weeks), len(day_values) * len(time_values))[:len(resampled_matrix)]
    missingness_matrix_week['Day'] = resampled_matrix[time_column].dt.day_name()
    missingness_matrix_week['Time'] = resampled_matrix[time_column].dt.strftime('%H:%M')
    missingness_matrix_week['Full-Time'] = resampled_matrix[time_column].dt.day_name() + missingness_matrix_week['Week'].astype(str) + ':' + resampled_matrix[time_column].dt.strftime('%H:%M')
    missingness_matrix_week['Day-Week'] = resampled_matrix[time_column].dt.day_name() + missingness_matrix_week['Week'].astype(str)
    missingness_matrix_week['Day-Time'] = resampled_matrix[time_column].dt.day_name() + resampled_matrix[time_column].dt.strftime('%H:%M')
    day_week_values = missingness_matrix_week['Day-Week'].unique()

    # keep only the time and missingness columns, rename the missingness column to Missing_Fraction, and reset the index
    missingness_matrix_week = missingness_matrix_week.rename(columns={missing_flags_column: 'Missing_Fraction'})
    missingness_matrix_week = missingness_matrix_week.reset_index(drop=True)

    # Step 2: Unflatten time points. The first axis is in minutes and should be at the sampling rate of missingness_interval
    unflattened_matrix = None

    # make intermediate matrix for each of the unique axes
    if isinstance(axes, str) or len(axes) == 1:
        if axes == 'day' or axes[0] == 'day':
            # axes = day
            unflattened_day_matrix = missingness_matrix_week.pivot(index='Day-Week', columns='Time', values='Missing_Fraction')
            unflattened_day_matrix = unflattened_day_matrix.reindex(day_week_values)
            unflattened_matrix = unflattened_day_matrix.to_numpy() # note: day is a time_of_day x day (happens to correspond to day of week)
        elif axes == 'week' or axes[0] == 'week':
            # axes = week
            original_order = missingness_matrix_week['Day-Time'].unique()
            unflattened_week_matrix = missingness_matrix_week.pivot(index='Week', columns='Day-Time', values='Missing_Fraction').reindex(columns=original_order)
            unflattened_matrix = unflattened_week_matrix.to_numpy() # note: week is a unique_days x week
        else:
            raise Exception("Invalid axes input")
    elif isinstance(axes, list):
        if axes == ['day', 'week'] or axes == ['week', 'day']:
            """in this case, we need to have complete sets of weeks. thus, we will fill in any incomplete weeks with nans before creating the tensor"""
            # original unflattened day matrix with potentially missing days of the week
            unflattened_day_matrix = missingness_matrix_week.pivot(index='Day-Week', columns='Time', values='Missing_Fraction')
            unflattened_day_matrix = unflattened_day_matrix.reindex(day_week_values)

            # create a new list for every day_of_week + week combination
            full_day_week = []
            for w in range(num_weeks):
                for d in day_values:
                    full_day_week.append(d + str(w))

            # extend unflattened matrix to have all day_of_week + week combinations
            full_unflattened_matrix = unflattened_day_matrix.reindex(full_day_week, fill_value=np.nan)
            
            # transform pands df into 2d numpy array where each row is a day of the week and each column is a time of day
            full_unflattened_day_matrix = full_unflattened_matrix.to_numpy() # note: time_of_day x day (happens to correspond to day of week) that is full

            # create the 3d tensor by taking every len(day_values) rows and stacking them on top of each other
            unflattened_matrix = full_unflattened_day_matrix.reshape(num_weeks, len(day_values), len(time_values)) # note: day-week tensor is a weeks x days (Sun-Sat) x time_of_day (0-24) (depth x row x column)
        elif axes == ['week', 'year'] or axes == ['year', 'week']:
            """in this case, we aggregate the missingness on a per-day level (Monday0_Missingness, Monday1_Missingness...) and aggregate over weeks"""

            """ in this part, we calculate the average missingness per day_of_week for each month. it has been verified that this averaging takes into account the number
                of each day_of_week for that month in a particular year (Ex. There are 5 Fridays in June, 2021). Currently, the data missing for each weekday in a month
                is not appended as nan (as depending on the case, this might make it impossible to calculate week-year stats.) Thus, the probabilities here is only for
                given data.
            """
            # group by 'Week' and 'Day' and calculate the average of 'Missing_Fraction' for each day-week item
            unflattened_day_matrix = missingness_matrix_week.groupby(['Month', 'Cal-Day', 'Day'])['Missing_Fraction'].mean().reset_index()

            # group by 'Month' and 'Day'
            unflattened_day_matrix = unflattened_day_matrix.groupby(['Month', 'Day'])['Missing_Fraction'].mean().reset_index()
            
            # create a pivoted matrix where the index is the unique day, the columns are each week, and the value is the missingness
            unflattened_day_matrix = unflattened_day_matrix.pivot(index='Month', columns='Day', values='Missing_Fraction')
            
            # map month names to calendar order
            month_order = {
                'January': 1, 'February': 2, 'March': 3,
                'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9,
                'October': 10, 'November': 11, 'December': 12
            }
            day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

            # sort by day_order
            unflattened_day_matrix = unflattened_day_matrix[day_order]
            
            # add a new column for the month order
            unflattened_day_matrix['Month_Order'] = [month_order[month] for month in unflattened_day_matrix.index]

            # sort the DataFrame by this new column
            unflattened_day_matrix = unflattened_day_matrix.sort_values('Month_Order')

            unflattened_day_matrix = unflattened_day_matrix.drop('Month_Order', axis=1)

            # transform pands df into 2d numpy array where each row is a day of the month and each column is a day of week
            full_unflattened_day_matrix = unflattened_day_matrix.to_numpy() # note: month_of_year x day_of_week that is full

            # create the 3d tensor by taking every len(day_values) rows and stacking them on top of each other
            unflattened_matrix = full_unflattened_day_matrix.reshape(-1, len(month_values), len(day_values)) 

        else:
            raise Exception("Invalid axes input")
        
    # print(unflattened_matrix.shape)
    # print(unflattened_matrix)

    # Step 3: Aggregate
    # Get rid of the extra axis (which is something like "each example of these combination of time points")
    missingness_avg = None
    missingness_var = None

    # get the mean and variance of the missingness along the correct axis
    missingness_avg = np.nanmean(unflattened_matrix, axis=0)
    missingness_var = np.nanvar(unflattened_matrix, axis=0)
	  
    return missingness_avg, missingness_var