import numpy as np
import pandas as pd

def missing_data_probabilities(flagged_df, time_column, axes, basis_rate=15, missingness_interval=15, missing_flags_column='Missing_Flag'):
    """
    Calculate missingness probability and variance matrices (or tensors) along a time axis
    An axis can be "time_of_day", "day_of_week", "month_of_year"
    
    Parameters
    ----------
    flagged_df: pandas.DataFrame
	    Dataframe with flagged missingness for one person. Flags can be obtained with flag_missing_data
    time_column: str
	    Name of column in dataframe which contains the time information. Column should have dtype datetime
    axes: str or list of str's ["time_of_day", "day_of_week", "month_of_year"]
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
    resampled_matrix = matrix.rolling(window=intervals_per_group, axis=0, min_periods=1).mean()

    # let the index be just numbers and move the datatime to be a new column
    resampled_matrix = resampled_matrix.reset_index()

    # only keep datetime and missingness columns
    resampled_matrix = resampled_matrix[[time_column, missing_flags_column]]

    # Since rolling mean includes the current and previous (window-1) columns, we need to select every intervals_per_group-th row to get non-overlapping intervals
    resampled_matrix = resampled_matrix.iloc[[0] + [i for i in range(intervals_per_group, len(resampled_matrix), intervals_per_group)], :]

    # get and store the unique time values, day values, and number of weeks
    time_values = resampled_matrix[time_column].dt.strftime('%H:%M').unique()
    day_values = resampled_matrix[time_column].dt.day_name().unique()
    num_weeks = np.ceil(len(resampled_matrix) / (len(day_values) * len(time_values))).astype(int)

    # relate each of the datetimes to a day of the week (Monday-Sunday) and a time of day (0-23:59) and change to the form day_of_week:time_of_day
    # note: make sure this works for all years and doesn't just start on Monday
    missingness_matrix = resampled_matrix.copy()
    missingness_matrix['Time'] = resampled_matrix[time_column].dt.day_name() + ':' + resampled_matrix[time_column].dt.strftime('%H:%M')

    # create matrix with unique week values as well
    missingness_matrix_week = resampled_matrix.copy()
    # every set of len(day_values) * len(time_values) rows will have the same week value up to the last value in the matrix
    missingness_matrix_week['Week'] = np.repeat(np.arange(num_weeks), len(day_values) * len(time_values))[:len(resampled_matrix)]
    missingness_matrix_week['Full-Time'] = resampled_matrix[time_column].dt.day_name() + missingness_matrix_week['Week'].astype(str) + ':' + resampled_matrix[time_column].dt.strftime('%H:%M')
    missingness_matrix_week['Day-Week'] = resampled_matrix[time_column].dt.day_name() + missingness_matrix_week['Week'].astype(str)
    missingness_matrix_week['Time'] = resampled_matrix[time_column].dt.strftime('%H:%M')
    day_week_values = missingness_matrix_week['Day-Week'].unique()

    # keep only the time and missingness columns, rename the missingness column to Missing_Fraction, and reset the index
    missingness_matrix = missingness_matrix[['Time', missing_flags_column]]
    missingness_matrix = missingness_matrix.rename(columns={missing_flags_column: 'Missing_Fraction'})
    missingness_matrix = missingness_matrix.reset_index(drop=True)
    missingness_matrix_week = missingness_matrix_week.rename(columns={missing_flags_column: 'Missing_Fraction'})
    missingness_matrix_week = missingness_matrix_week.reset_index(drop=True)

    # Step 2: Unflatten time points. The first axis is in minutes and should be at the sampling rate of missingness_interval

    # make intermediate matrix for each of the unique time values where the row is the Day-Week and the column is the Time and the value is the Missing_Fraction (if a given Day-Week and Time combination is missing fil in with np.nan)
    unflattened_matrix = missingness_matrix_week.pivot(index='Day-Week', columns='Time', values='Missing_Fraction')
    unflattened_matrix = unflattened_matrix.reindex(day_week_values)
                      
    """
    For example, if you have missing data dataframe from step 1 that looks like:
    (In this case, missingness_interval was set to 12 hours, which is 720 minutes)
    index     Time           Missing_Fraction
    1         Monday:00:00        1.0
    2         Monday:12:00        0.9
    3         Tuesday:00:00       0.8
    4         Tuesday:12:00       0.7
    5         Wednesday:00:00     0.6
    6         Wednesday:12:00     0.5
    7         Monday:00:00        0.4         <- The next monday
    
    
    If axes = ['time_of_day'], then unflattening the matrix would look like:
            00:00 12:00
    [ mon. [1.0, 0.9],                       
      tue [0.8, 0.7],                       [ [1.0, 0.8, 0.6, 0.4],
      wed [0.6, 0.5],      or alternatively   [0.9, 0.7, 0.5, np.nan] ]
      mon2 [0.4, np.nan] ]                      
      
    In this case, you should see that all values that were taken at the same "time_of_day" are aligned vertically (in the left example),
	    or aligned horizontally (in the right example)
    
    Also of note, we had to fill in missing values that would ruin the 2x4 shape of the matrix
    
	  Step 3 will be aggregating along an axis to get a len(axes)-dimension output. In this example, it would be a 1-D array (1 value per time of day)
	  
	  
	  If axes = ['day_of_week', 'time_of_day'], our final result will be a 2-D matrix (for each combination of time of day and day of week),
		  which means that our unflattened matrix should be a 3-D tensor (we will aggregate along one axis to get the final 2-D matrix)
			You can think of this as a day_of_week by time_of_day matrix, which extends into the 3rd dimension for each example. 
			
		Another way to think about this might be by fragmenting and stacking at the 
		
		If our data looks like this, where M1 is Monday at time point 1 (for simplicity, I'm using a 4 day week)
		M1 M2 M3 T1 T2 T3 W1 W2 W3 R1 R2 R3 M1 M2 M3 T1 T2 T3 W1 W2 W3 R1 R2 R3
		----------------------------------------------------------------------> time
		
		You might first break this up by the smaller axis, "Time of day"
		
		M1 M2 M3 (next point has time point 1, so go to the next row)
		T1 T2 T3
		W1 W2 W3
		R1 R2 R3 
		M1 M2 M3 
		T1 T2 T3 
		W1 W2 W3 
		R1 R2 R3
		
		Then, break this up by the larger axis  "day of week":
		
	  [ M1 M2 M3 
		  T1 T2 T3
		  W1 W2 W3
		  R1 R2 R3 ]
		
		[ M1 M2 M3 
		  T1 T2 T3 
		  W1 W2 W3 
		  R1 R2 R3 ] 
		  
		If you visualize "stacking" these 2 matrices on top of each other, you can see how we end up at a 3-D array
		The shape of this array would be something like (number of times of day, number of days of week, number of weeks for this participant)
			In this simple example, that would be (3, 4, 2)
		
    """

    # Step 2.5: Create intermediate matrices for each of the axes splits
    """time_of_day_matrix: day_of_week x time_of_day
       day_of_week_matrix: week x day_of_week x time_of_day

       note: we fill missing days of week in the time_day_matrix with np.nan in order to create the day_of_week_matrix
    """

    # extend the unflattened matrix to finish the week and fill in missing values with np.nan
    # create a new list for every day_of_week + week combination
    full_day_week = []
    for w in range(num_weeks):
        for d in day_values:
            full_day_week.append(d + str(w))

    # extend unflattened matrix to have all day_of_week + week combinations
    full_unflattened_matrix = unflattened_matrix.reindex(full_day_week, fill_value=np.nan)

    # matrix for each of the axis splits
    # transform pands df into 2d numpy array where each row is a day of the week and each column is a time of day
    time_of_day_matrix = full_unflattened_matrix.to_numpy() # note: time of day is a matrix day_of_week x time_of_day
    # create the day_of_week matrix by taking every len(day_values) rows and stacking them on top of each other
    day_of_week_matrix = time_of_day_matrix.reshape(num_weeks, len(day_values), len(time_values)) # note: day of week is a week x matrix day_of_week x time_of_day (depth x row x column)

    # Step 3: Aggregate
    # Get rid of the extra axis (which is something like "each example of these combination of time points")
    missingness_avg = None
    missingness_var = None

    # for 1 input to axes
    if len(axes) == 1:
        # get the mean and variance of the missingness along the correct axis
        if axes[0] == 'time_of_day': # aggregate along day of week (row-wise)
            # get the mean and variance of the missingness along the correct axis
            missingness_avg = np.nanmean(time_of_day_matrix, axis=0)
            missingness_var = np.nanvar(time_of_day_matrix, axis=0)
        elif axes[0] == 'day_of_week': # aggregate along time of day (column-wise)
            # get the mean and variance of the missingness along the correct axis
            missingness_avg = np.nanmean(time_of_day_matrix, axis=1)
            missingness_var = np.nanvar(time_of_day_matrix, axis=1)
        else:
            pass
    # for 2 inputs to axes
    elif len(axes) == 2: # no if as we will always aggregate along the depth axis assuming each 2d matrix is formed correctly
        # get the mean and variance of the missingness along the correct axis
        missingness_avg = np.nanmean(day_of_week_matrix, axis=0) # aggregate along week (depth-wise)
        missingness_var = np.nanvar(day_of_week_matrix, axis=0) # aggregate along week (depth-wise)
    else:
        pass
	  
    return missingness_avg, missingness_var