import numpy as np
import pandas as pd

def flag_missing_data(df : pd.DataFrame, start_date = None, end_date = None, time_col: str = 'timestamp', freq='15min'):
    """Turn a pandas dataframe time series with only existing samples (no missing) into an array of missingness flags.
    
    Parameters
    ----------
    df: pd.DataFrame
        Time series pandas dataframe. Includes a column for the time stamp of each sample.

    start_date: str or datetime-like
        Day to start array of missing data flags. start_date will be adjusted to 00:00:00 on this day. If no value is provided, the first day of data will be used.

    end_date: str or datetime-like
        Day to end the array of missing data flags. end_date will be adjusted to 1 freq period before the end of this day. If no value is provided, the last day of data will be used.

    time_col: str
        Name of column in df where time stamps are stored.

    freq: str, pd.Timedelta, datetime.timedelta or pd.DateOffset, default '15min' (15 minutes)
        Frequency of missingness flags to generate.

    Returns
    -------
    flagged_df: pd.DataFrame
        Pandas dataframe with time series array of missingness flags (1=missing, 0 = not mising). Will include index 'datetime' for timestamps, and 'Missing_Flag' as a column.
    """

    # convert time_col to pandas datetime    
    df[time_col] = pd.to_datetime(df[time_col])

    # TODO: resample time_col to given freq .dt.floor(freq) should do it
    
    # check if start_date and end_date are defined
    if start_date is None:
        start_date = df[time_col].min()
    
    if end_date is None:
        end_date = df[time_col].max()

    # shift start_date and end_date to the start/end of their days
    start_date = start_date.replace(hour=0, minute=0)
    end_date = end_date.replace(hour=23, minute = 45) # TODO: make this flexible for any freq

    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

    flagged_df = pd.DataFrame(index=date_range)
    flagged_df['Missing_Flag'] = 1
    flagged_df.loc[df[time_col], 'Missing_Flag'] = 0

    return flagged_df

    
