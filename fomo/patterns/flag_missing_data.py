# IMPORTANT: RESAMPLE DOESNT GO BACK TO FILL IN VALUES BEFORE
# EX: PERSON 9 DOESNT HAVE DATA FOR THE FIRST PART SO NEED TO GO BACK AND FILL IN TRUE
# FOR ALL CELLS BEFORE THE FIRST TIME POINT WITH DATA FOR THAT PERSON

import pandas as pd

def flag_missing_data(df, column_to_query, study_period=(), interval_size=15):
    """
    Flags missing data for a specific column within a given study period.
    Parameters:
    - df: DataFrame with 'person_id', 'date', and other columns.** date or datetime should be flexible
    - column_to_query: either string or list, the name of the column(s) to check for missing data.
    - study_period: Tuple or list, containing the start and end dates of the study period ('YYYY-MM-DD', 'YYYY-MM-DD').
    - interval_size: Integer, how big of a gap there is between data points in output table in minutes (aggregated
    raw data points into 'bins' of size interval_size)

    Returns:
    - DataFrame with an additional column indicating missing data for the queried column (called 'Missing_Flag')
    """

    # Check to make sure column_to_query is a column in df
    if column_to_query not in df.columns:
        raise Exception("Column to query is not a column in the provided DataFrame")

    # Preprocessing of dataset
    df['datetime'] = pd.to_datetime(df['datetime']) # Convert to datetime objects

    # Determine study_period
    if len(study_period) == 2:
        start_date = pd.to_datetime(study_period[0])
        end_date = pd.to_datetime(study_period[1])

    # If no study period provided, use entire dataset
    else:
        start_date = df['datetime'].min()
        end_date = df['datetime'].max()

    # Filter to only have study period
    df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

    # Group by person_id, resample based on interval_size provided
    df.set_index('datetime', inplace=True)
    
    # Resample data based on interval_size, take mean heart_rate of all entries in that time interval
    resampled = df.groupby('person_id', as_index=False).resample(f'{interval_size}min').mean()

    # Fill in person_id values for new rows from resampling
    resampled.loc[:,'person_id'] = resampled.loc[:,'person_id'].ffill()

    # Create Missing_Flag column
    resampled['Missing_Flag'] = resampled[column_to_query].isna()
    
    # Prepare data to be returned
    resampled = resampled.reset_index().drop('level_0', axis=1)
    resampled = resampled[['person_id', 'datetime', column_to_query, 'Missing_Flag']]
    resampled = resampled.astype({'person_id': int})

    return resampled

def run_fmd():

    heart_rate_data = pd.read_csv('sample_hr.csv')
    heart_rate_data = heart_rate_data.drop('Unnamed: 0', axis=1)
    print(heart_rate_data)

    flagged_data = flag_missing_data(heart_rate_data, 'heart_rate')
    flagged_data.to_csv('output.txt', sep='\t', index=False)

if __name__ == "__main__":
    run_fmd()