# heatmap def
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from metro_app.ml_logic.preprocess import general_preprocessing
from ml_logic.data import plot_data, lines_data



def global_network():

    """This function returns a map where the global network of Seoul metro is displayed"""

    df = plot_data()

    fig = px.density_mapbox(df, lat="lat", lon="lng", z='n_passengers', radius=13,
                        center=dict(lat=37.5519, lon=126.9918), zoom=10,animation_frame="hour",
                        color_continuous_scale = px.colors.sequential.Plasma_r,
                        hover_name = 'station_name',
                        range_color = [50, 1500],
                        mapbox_style="carto-positron")

    fig.show()




def commuter_network(station, line):

    """This function returns a graph where data from the specific lines is plot"""

    df = plot_data()
    lines_dictionary = lines_data()

    df = df[(df['line'] == line)]


    fig = px.density_mapbox(df, lat="lat", lon="lng", z='n_passengers', radius=13,
                        center=dict(lat=37.5519, lon=126.9918), zoom=10,animation_frame="hour",
                        color_continuous_scale = px.colors.sequential.Plasma_r,
                        hover_name = 'station_name',
                        range_color = [50, 1500],
                        mapbox_style="carto-positron")

    # I draw the line of the metro

    for _,row in lines_dictionary[f'line_{line}'].iterrows():
        fig.add_trace(go.Scattermapbox(mode='lines',
                                lon=[row['lng'], row['next_lng']],
                                lat=[row['lat'], row['next_lat']],
                                line_color='lightgrey',
                                line=dict(width=2),
                                name=None,
                                showlegend=False
                                ))

    lat = df.loc[df['station_name'] == station, 'lat'][0]
    lon = df.loc[df['station_name'] == station, 'lng'][0]

    # I draw the point where I am right now


    fig.add_trace(go.Scattermapbox(
        lat=[lat],
                lon=[lon],
                mode='markers',
                marker=dict(size=8, color='salmon'),
                showlegend=False))


    # I draw the points of the metro

    fig.add_trace(go.Scattermapbox(
            lat=df['lat'],
            lon=df['lng'],
            mode='markers',
            marker=dict(size=4, color='lightsalmon'),
            showlegend=False))




    fig.show()
