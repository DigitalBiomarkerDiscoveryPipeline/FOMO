import pandas as pd
import numpy as np

def explore_adherence(df, method='hr_adherence', step_threshold=None):
    """
    Determines adherence based on heart rate or step data, with error handling for incorrect DataFrame inputs.
    Parameters:
    - df: DataFrame with daily activity summary or minute-level heart rate data.
    - method: Specifies the method to calculate adherence ('hr_adherence' for heart rate adherence and 'steps' for step count adherence).
    - step_threshold: The threshold for steps to consider for step adherence calculation.

    Returns: DataFrame with columns: person_id, date, adherence

    Raises:
    - ValueError: If an incorrect DataFrame is passed for the specified method.
    """
    # Error handling for incorrect DataFrame inputs
    if method not in ['hr_adherence', 'steps_adherence']:
        raise ValueError("The method can only be either hr_adhenrence or steps_adherence.")
        
    if method == 'hr_adherence' and 'heart_rate' not in df.columns:
        raise ValueError("The heart_rate_df must contain a 'datetime' column for hr_adherence method.")
    elif method == 'steps_adherence':
        if 'steps' not in df.columns:
            raise ValueError("The activity_df must contain a 'steps' column for steps_adherence method.")
        if step_threshold is None:
            raise ValueError("You have to set your step_threshold for step_adherence method.")

    # Initialize an empty DataFrame for the final result
    adherence_result = pd.DataFrame()
    
    if method == 'hr_adherence':
        hr = df.copy()
        # Process heart rate data for adherence calculation
        hr['datetime'] = pd.to_datetime(hr['datetime'])
        hr.sort_values(by=['person_id', 'datetime'], inplace=True)
        hr['date'] = hr['datetime'].dt.date
        hr['minute'] = hr['datetime'].dt.floor('T')
        df_unique_minutes = hr.drop_duplicates(subset=['person_id', 'date', 'minute'])
        hr_adherence = df_unique_minutes.groupby(['person_id', 'date']).size().reset_index(name='unique_minutes')
        hr_adherence['adherence'] = hr_adherence['unique_minutes'] / 1440
        adherence_result = hr_adherence[['person_id', 'date', 'adherence']]
    
    elif method == 'steps_adherence':
        activity = df.copy()
        # Calculate step adherence
        activity['adherence'] = np.where(activity['steps'] > step_threshold, 1, 0)
        adherence_result = activity[['person_id', 'date', 'adherence']]
    
    return adherence_result
