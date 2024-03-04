import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
import re



df = pd.read_csv('/Users/wuyuyou/Desktop/sample_adherence.csv')




def calc_tot_num_by_adherence(df,thres):
    df['flag'] = df['adherence'].apply(lambda x: True if x > thres else False)
    num_day = sum(df['flag']==True)
    return num_day

# number of days per participant
def calc_num_day_per_person(df,thres,mode='mean',round=False):
    df['flag'] = df['adherence'].apply(lambda x: True if x > thres else False)
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

# Use the second derivatives to find the optimal value
# If there are multiple values, uers can use cutoff points to get the closest one, default set to 0.5
def find_optimal_threshold(df,thres):
    min_val = np.min(df['adherence'])
    max_val = np.max(df['adherence'])
    cutoff = (max_val-min_val)/2
    adherence_thresholds = np.linspace(min_val, max_val, thres)
    total_numbers = [calc_num_day_per_person(df, thres) for thres in adherence_thresholds]
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

def generate_per_person_graph(df, interval, compare=True):
    # Calculate the optimal threshold
    optimal_threshold = find_optimal_threshold(df,interval*10)
    min_val = np.min(df['adherence'])
    max_val = np.max(df['adherence'])
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



def intra_day_viz(df, start_day, time_span, gran='15min',diff=True):
    ## start_day check valid
    start_day = datetime.strptime(start_day, '%Y-%m-%d').date()
    unique_days = df['datetime'].dt.date.unique()
    if start_day not in unique_days:
        print("Invalid start day.")
        return
    #print('Start day check pass.')
    ## time_span check valid
    if (df['datetime'].max() - df['datetime'].min()) < pd.Timedelta(days=time_span) or (start_day + pd.Timedelta(days=time_span)) > hr['datetime'].dt.date.max():
        print("Invalid time span.")
        return
    #print('Time span check pass.')
    viz_df = df[df['datetime'].dt.date>=start_day]
    viz_df = viz_df[viz_df['datetime'] <= pd.Timestamp(start_day + pd.Timedelta(days=time_span))]
    ## adjust granularity
    viz_df['rounded_datetime'] = viz_df['datetime'].dt.floor(gran)
    group_df = viz_df.groupby(['person_id','rounded_datetime']).agg({'heart_rate': lambda x: x.any(), 'datetime': 'first'}).reset_index()
    group_df['rounded_datetime'] = group_df['rounded_datetime'].dt.strftime('%H:%M')
    group_df['is_weekend'] = group_df['datetime'].dt.weekday.isin([5, 6])
    if diff:
        days_count_df = group_df.groupby(['person_id','rounded_datetime','is_weekend'])['rounded_datetime'].count().reset_index(name='total_rounded_datetime')
        average_days_per_person = days_count_df.groupby(['rounded_datetime','is_weekend'])['total_rounded_datetime'].mean().reset_index(name='average_days_per_person')
        average_days_per_person['average_days_per_person'] = average_days_per_person['average_days_per_person'].round(3)
        weekend_data = average_days_per_person[average_days_per_person['is_weekend']]
        weekday_data = average_days_per_person[~average_days_per_person['is_weekend']]
        weekday_trace = go.Scatter(x=weekday_data['rounded_datetime'], 
                            y=weekday_data['average_days_per_person'], 
                            mode='lines', 
                            name='Weekday')
        weekend_trace = go.Scatter(x=weekend_data['rounded_datetime'], 
                            y=weekend_data['average_days_per_person'], 
                            mode='lines', 
                            name='Weekend', 
                            line=dict(color='green'))
        layout = go.Layout(title='Average Days per Person over One-Day Period',
                        xaxis=dict(title='Date time'),
                        yaxis=dict(title='Average Days per Person'))
        fig = go.Figure(data=[weekday_trace, weekend_trace], layout=layout)
    else:
        days_count_df = group_df.groupby(['person_id','rounded_datetime'])['rounded_datetime'].count().reset_index(name='total_rounded_datetime')
        average_days_per_person = days_count_df.groupby('rounded_datetime')['total_rounded_datetime'].mean().reset_index(name='average_days_per_person')
        # ## visualize

        fig = px.line(average_days_per_person, x='rounded_datetime', y='average_days_per_person', 
                title='Average Days per Person over One-Day Period',
                labels={'rounded_datetime': 'Date time', 'average_days_per_person': 'Average Days per Person'})
    return fig