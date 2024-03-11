import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


def generate_per_person_graph(df, interval, compare=True):
    # Calculate the optimal threshold
    optimal_threshold = find_optimal_threshold(df,interval*10)
    min_val = np.min(df['adherency'])
    max_val = np.max(df['adherency'])
    thres_range = np.linspace(min_val, max_val, interval)
    
    # Calculate mean and median values for each threshold
    mean_vals = [calc_num_day_per_person(df, thres) for thres in thres_range]
    median_vals = [calc_num_day_per_person(df, thres, 'median') for thres in thres_range]

    ci_upper = [val * 1.05 for val in mean_vals]  
    ci_lower = [val * 0.95 for val in mean_vals] 
    iqr_upper = [calc_num_day_per_person(df, thres, '75percentile') for thres in thres_range] 
    iqr_lower = [calc_num_day_per_person(df, thres, '25percentile') for thres in thres_range]  

    # Create the figure
    fig = go.Figure()

    # Add the mean line
    fig.add_trace(go.Scatter(x=thres_range, y=mean_vals, mode='lines', name='mean', line=dict(color='blue')))
    
    # Add the median line if compare is True
    if compare:
        fig.add_trace(go.Scatter(x=thres_range, y=median_vals, mode='lines', name='median', line=dict(color='orange')))

    # Add shaded area for the 95% CI of the mean
    fig.add_trace(go.Scatter(x=thres_range, y=ci_upper, fill=None, mode='lines', line=dict(color='lightblue'), showlegend=False))
    fig.add_trace(go.Scatter(x=thres_range, y=ci_lower, fill='tonexty', mode='lines', line=dict(color='lightblue'), fillcolor='rgba(173,216,230,0.4)', name='95% CI of Mean'))

    # # Add shaded area for the IQR
    fig.add_trace(go.Scatter(x=thres_range, y=iqr_upper, fill=None, mode='lines', line=dict(color='lightsalmon'), showlegend=False))
    fig.add_trace(go.Scatter(x=thres_range, y=iqr_lower, fill='tonexty', mode='lines', line=dict(color='lightsalmon'), fillcolor='rgba(250,128,114,0.4)', name='IQR'))

    # Add a point for the optimal threshold on the mean line
    fig.add_trace(go.Scatter(x=[optimal_threshold], y=[calc_num_day_per_person(df, optimal_threshold)], mode='markers', name='Optimal Threshold', marker=dict(color='red', size=12), text=f'Optimal: {optimal_threshold:.2f}', hoverinfo='text'))

    # Update the layout
    fig.update_layout(
        title='Total Number of Days per Participant by Adherence Threshold',
        xaxis_title='Adherence Threshold',
        yaxis_title='Total number of days with adherence level per participants',
        legend_title='Category',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Add annotation for the optimal threshold
    fig.add_annotation(x=optimal_threshold, y=calc_num_day_per_person(df, optimal_threshold),
                       text=f"Optimal threshold: {optimal_threshold:.2f}",
                       showarrow=True, arrowhead=1, yshift=10)

    return fig


# Use the second derivatives to find the optimal value
# If there are multiple values, uers can use cutoff points to get the closest one, default set to 0.5
def find_optimal_threshold(df,thres):
    min_val = np.min(df['adherency'])
    max_val = np.max(df['adherency'])
    cutoff = (max_val-min_val)/2
    adherence_thresholds = np.linspace(min_val, max_val, thres)
    total_numbers = [calc_tot_num_by_adherence(df, thres) for thres in adherence_thresholds]
    # find the second derivative
    first_grad = np.gradient(total_numbers, adherence_thresholds)
    second_grad = np.gradient(first_grad, adherence_thresholds)
    # find the smallest second derivative
    min_grad = np.min(np.abs(second_grad))
    # find values equal to the smallest second derivative
    idxes = np.where(np.abs(second_grad) == min_grad)[0]
    # find the ones closest to the cutoff points
    closest_idx = np.argmin(np.abs(adherence_thresholds[idxes] - cutoff))
    optimal_threshold = adherence_thresholds[idxes[closest_idx]]
    return optimal_threshold


# number of days per participant
def calc_num_day_per_person(df,thres,mode='mean',round=False):
    df['flag'] = df['adherency'].apply(lambda x: True if x > thres else False)
    if mode == 'mean':
        val = np.mean(df.groupby('person_id')['flag'].sum())
    elif mode == 'median':
        val = np.median(df.groupby('person_id')['flag'].sum())
    elif mode == '75percentile':
        val = np.percentile(df.groupby('person_id')['flag'].sum(), 75)
    elif mode == '25percentile':
        val = np.percentile(df.groupby('person_id')['flag'].sum(), 25)
    if round:
        val = np.round(val)
    return val

# total number of days
def calc_tot_num_by_adherence(df,thres):
    df['flag'] = df['adherency'].apply(lambda x: True if x > thres else False)
    num_day = sum(df['flag']==True)
    return num_day


def valid_3_a_day(df, person_id, date):
    # Filter records for the given person_id
    person_data = df[df['person_id'] == person_id]

    # Define time periods
    periods = [
        ('03:00:00', '11:00:00'), # 3 am to 11 am
        ('11:00:00', '15:00:00'), # 11 am to 3 pm
        ('15:00:00', '03:00:00')  # 3 pm to 3 am (next day)
    ]
    
    # Check for at least one entry in each period
    for start, end in periods:
        if start < end:
            # For periods within the same day
            if not person_data.between_time(start, end).loc[date].empty:
                continue
            else:
                return False
        else:
            # For the period that spans over to the next day
            next_day = pd.to_datetime(date) + pd.Timedelta(days=1)
            if not person_data.between_time(start, '23:59:59').loc[date].empty or not person_data.between_time('00:00:00', end).loc[str(next_day.date())].empty:
                continue
            else:
                return False         
    return True


def generate_kde_graph(df):
    # Create figure and axis without showing it
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot histogram on ax1
    sns.histplot(df, ax=ax1, color='green')
    ax1.set_xlabel('Number of Days Available (Adherence = 1) Per Person')
    ax1.set_ylabel('Count')
    
    # Create a second y-axis for density
    ax2 = ax1.twinx()
    sns.kdeplot(df, ax=ax2, color="darkgreen")
    ax2.set_ylabel('Density')
    
    # Set title
    plt.title('Adhrence Result Overview')
    plt.close(fig)

    return fig


def explore_adherence(df, method='hr_adherence', threshold=None, visualize=False):
    """
    Determines adherence based on heart rate or step data, with error handling for incorrect DataFrame inputs.
    Parameters:
    - df: DataFrame with either daily activity summary or minute-level heart rate data.
    - method (str): The method to calculate adherence. Options include:
        - 'hr_adherence': Calculates adherence based on heart rate data availability across the entire day.
        - 'steps_adherence': Calculates adherence based on whether daily steps exceed a specified `threshold`.
        - '>=nhour': Considers a day adherent if heart rate data is present in at least `threshold` number of 
                    distinct hours.
        - '3-a-day': A day is considered valid if the tracker registered data in 3 predefined periods.
        - '3-of-4 windows': Assesses adherence based on presence of heart rate data in at least three of four specified daily windows.
        - 'step_threshold': The threshold for steps to consider for step adherence calculation.

    Returns: DataFrame with columns: person_id, date, adherence

    Raises:
    - ValueError: If an incorrect DataFrame is passed for the specified method.
    """
    # Error handling for incorrect DataFrame inputs
    if method not in ['hr_adherence', 'steps_adherence', ">=nhour", "3-a-day", "3-of-4 windows"]:
        raise ValueError("The method must be one of the following: 'hr_adherence', 'steps_adherence', '>=10hour', '3-a-day', or '3-of-4 windows'.")
        
    if method in ['hr_adherence', ">=nhour", "3-a-day", "3-of-4 windows"] and 'heart_rate' not in df.columns:
        raise ValueError("The heart_rate_df must contain a 'heart_rate' column for heart rate related method.")
    elif method == 'steps_adherence':
        if 'steps' not in df.columns:
            raise ValueError("The activity_df must contain a 'steps' column for steps_adherence method.")
        if threshold is None:
            raise ValueError("You have to set your threshold for step_adherence method.")
    
    if method == '>=nhour' and threshold is None:
        raise ValueError("You have to set your threshold for '>=nhour' method.")
    

    # Initialize an empty DataFrame for the final result
    adherence_result = pd.DataFrame()
    
    if method in ['hr_adherence', ">=nhour", "3-a-day", "3-of-4 windows"]:
        hr = df.copy()
        # Process heart rate data for adherence calculation
        hr['datetime'] = pd.to_datetime(hr['datetime'])
        hr.sort_values(by=['person_id', 'datetime'], inplace=True)
        hr['date'] = hr['datetime'].dt.date
        hr['hour'] = hr['datetime'].dt.hour
        hr['minute'] = hr['datetime'].dt.floor('T')
        
        if method == 'hr_adherence':
            df_unique_minutes = hr.drop_duplicates(subset=['person_id', 'date', 'minute'])
            hr_adherence = df_unique_minutes.groupby(['person_id', 'date']).size().reset_index(name='unique_minutes')
            hr_adherence['adherency'] = hr_adherence['unique_minutes'] / 1440
            if threshold is None:
                adherence_result = hr_adherence[['person_id', 'date', 'adherency']]
            else:
                if 0 <= threshold <= 1:
                    hr_adherence['adherence'] = (hr_adherence['adherency'] >= threshold).astype(int)
                    adherence_result = hr_adherence[['person_id', 'date', 'adherence']]
                else:
                    raise ValueError("The threshold for heart rate adherence mu be between 0 an 1.")
            
        
        if method == ">=nhour":
            # A day is considered valid if the tracker registered data in at least ten different 1-hour windows.
            df_hours = hr.groupby(['person_id', 'date'])['hour'].nunique().reset_index(name='hours_count')
            df_hours['adherence'] = (df_hours['hours_count'] >= threshold).astype(int)
            adherence_result = df_hours[['person_id', 'date', 'adherence']]
        
        if method == "3-a-day":
            # A day is considered valid if the tracker registered data in 3 predefined periods: 
            # 3 am to 11 am, 11 am to 3 pm, and 3 pm to 3 am
            # A participant must register at least one step in 3 windows anchored to the morning, afternoon, and evening
        ## intraday step data & overnight issue
            results = []
            for (person_id, date), group in hr.groupby(['person_id', 'date']):
                group = group.set_index('datetime')
                # print(group)
                adherence = 1 if valid_3_a_day(group, person_id, str(date)) else 0
                results.append({'person_id': person_id, 'date': date, 'adherence': adherence})
            adherence_result = pd.DataFrame(results)
            
        if method == "3-of-4 windows":
            # A day is considered valid if the tracker registered data in at least three of four periods: 
            # 12 am to 6 am, 6 am to 12 pm, 12 pm to 6 pm, and 6 pm to 12 am.
            results = []
            windows = [(0, 6), (6, 12), (12, 18), (18, 24)]
            for (person_id, date), group in hr.groupby(['person_id', 'date']):
                # print(person_id, date)
                group = group.set_index('datetime')
                window_counts = 0
                for start, end in windows:
                    if group.between_time(f'{start}:00', f'{end-1}:59').any().any() == True:
                        window_counts+=1
                adherence = 1 if window_counts >= 3 else 0
                results.append({'person_id': person_id, 'date': date, 'adherence': adherence})
            adherence_result = pd.DataFrame(results)
    
    elif method == 'steps_adherence':
        activity = df.copy()
        # Calculate step adherence
        activity['adherence'] = np.where(activity['steps'] > threshold, 1, 0)
        adherence_result = activity[['person_id', 'date', 'adherence']]
        
    # Plot
    if visualize == True:
        if method == 'hr_adherence' and threshold == None:
            fig = generate_per_person_graph(adherence_result, 50)
        else:
            ### KDE
            df_filtered = adherence_result[adherence_result['adherence'] == 1]
            # Calculate number of days available (adherence=1) per participant
            df_grouped_filtered = df_filtered.groupby('person_id').size()
            
            fig = generate_kde_graph(df_grouped_filtered)
    
    return adherence_result, fig
