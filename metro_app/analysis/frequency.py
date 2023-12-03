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

    prepro_df =general_preprocessing(df_for_pred)
    seouls_s = prepro_df[(prepro_df['date'] == date_for_pred) & (prepro_df['station_name']==station) & prepro_df['line']==line]
    seouls_station = seouls_s.groupby(['date','station_name'])[seouls_s.columns[5:]].sum().reset_index()

    fig = px.line(x=seouls_station.columns[2:],
              y=list(seouls_station.iloc[0].values[2:]),
              color=px.Constant("Day"),
              line_shape='spline',
              color_discrete_sequence=['darkblue'])

    fig.add_bar(x=frequency_ss.columns[4:],
                y=list(frequency_ss.iloc[0].values[4:]),
                name="Average",
                marker_color='lightgray')

    fig.update_layout(
        xaxis_title='Time of the day (h)',
        yaxis_title='Number of people',
        title=f'Fréquentation moyenne le vendredi a {frequency_ss.station_name.values[0]}',
        plot_bgcolor='rgba(255, 255, 255, 0.9)',
        paper_bgcolor='rgba(255, 255, 255, 0.9)',
        font=dict(color='black'),
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey')
    )

    return fig.show()
