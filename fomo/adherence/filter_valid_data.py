import pandas as pd
import numpy as np


def filter_valid_data(adherency, adherent_threshold=0.5, num_consec_days=0):
    """
    Filters a DataFrame for days meeting specified adherence levels and consecutive day requirements.
    Adds a 'strikes' column to count consecutive days of adherence.
    
    Parameters:
    - adherency: DataFrame with adherence column.
    - adherent_threshold: Float, adherence threshold to filter the data on.
    - num_consec_days: Integer, number of consecutive day requirement.
    
    Returns:
    - DataFrame with days meeting adherence criteria and a 'strikes' column for consecutive adherence.
    """ 
    # Validate input
    if "adherence" not in adherency.columns:
        raise ValueError("DataFrame adherency must include an 'adherence' column.")
    if adherent_threshold < 0 or num_consec_days < 0:
        raise ValueError("Thresholds or num_consec_days must be non-negative.")
    
    # Ensure 'date' is a datetime object
    adherency['date'] = pd.to_datetime(adherency['date'])
    adherency.sort_values(by=['person_id', 'date'], inplace=True)

    # Calculate adherence flag
    adherency['adherence_flag'] = adherency['adherence'] >= adherent_threshold

    # Calculate 'strikes' for consecutive adherence by person_id, considering continuous dates
    adherency['next_date'] = adherency.groupby('person_id')['date'].shift(-1)
    adherency['is_continuous'] = (adherency['next_date'] - adherency['date']).dt.days == 1
    adherency['strikes'] = 0  # Initialize 'strikes' column

    for person_id, group in adherency.groupby('person_id'):
        strike_count = 0
        for i, row in group.iterrows():
            if row['adherence_flag']:
                strike_count += 1
            else:
                strike_count = 0
            adherency.at[i, 'strikes'] = strike_count
            # Reset strike count if the next day is not continuous
            if not row['is_continuous']:
                strike_count = 0

    # Filter based on threshold and consecutive days
    filtered = adherency[adherency['adherence_flag']]
    if num_consec_days > 0:
        filtered['included'] = filtered['strikes'] >= num_consec_days
        filtered_list = filtered.index[filtered['strikes'] == num_consec_days].tolist()
        for num in filtered_list:
            filtered.loc[num - num_consec_days + 1:num, 'included'] = True
        filtered = filtered[filtered['included'] == True]
    
    # Return the filtered DataFrame without temporary columns
    return filtered[['person_id', 'date', 'adherence', 'strikes']]
