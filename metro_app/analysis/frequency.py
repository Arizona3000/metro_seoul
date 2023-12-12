from ml_logic.data import get_data_frequency
from ml_logic.preprocess import general_preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go



def get_frequency_graph(df, date_for_average: str, date_for_pred: str, day: str, station: str, line: int):
    """
    Function that returns a graph showing the average crowding in a specific station compared to the crowding
    on a specific day for the same station.\n
    df --> dataframe crowding
    (df_pred) --> df prediction, add this feature later
    date_for_average --> what is the average you want, how many days of data, input a date here
    date_for_pred --> the date of prediction
    day --> the day of the week "Monday", "Friday"...
    station --> the station of interest
    line --> the line of interest (will maybe be removed)
    """

    df_for_avegarge = df.copy()
    df_for_pred = df.copy()

    frequency = get_data_frequency(df_for_avegarge, date_for_average)
    frequency_ss = frequency[(frequency['day']==day) & (frequency['station_name'] == station) & (frequency['line']==line)]
    frequency_ss = frequency_ss.reset_index()

    prepro_df =general_preprocessing(df_for_pred)
    seouls_s = prepro_df[(prepro_df['date'] == date_for_pred) & (prepro_df['station_name']==station) & (prepro_df['line']==line)]
    seouls_station = seouls_s.groupby(['date','station_name'])[seouls_s.columns[5:]].sum().reset_index()


    # Create a grouped bar chart using plotly.graph_objects
    fig = go.Figure()

    # Add bars for the average on the right
    fig.add_trace(go.Bar(
        x=frequency_ss.columns[5:],
        y=list(frequency_ss.iloc[0].values[5:]),
        name="Average",
        marker_color='lightgray',
        offset=-0.2  # Offset to position on the right
    ))

    # Add bars for the prediction on the left
    fig.add_trace(go.Bar(
        x=seouls_station.columns[5:],
        y=list(seouls_station.iloc[0].values[5:]),
        name="Prediction",
        marker_color='darkblue',
        offset=0.2  # Offset to position on the left
    ))

    # Update layout
    fig.update_layout(
        xaxis_title='Time of the day (h)',
        yaxis_title='Number of people',
        title=f'Average Crowding and Prediction on {day} at {station}',
        barmode='group',  # Grouped bar chart
        plot_bgcolor='rgba(255, 255, 255, 0.9)',
        paper_bgcolor='rgba(255, 255, 255, 0.9)',
        font=dict(color='black'),
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey')
    )

    return fig.show()
