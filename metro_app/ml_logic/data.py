# Put the data imports/exports here

import pandas as pd
from preprocess import general_preprocessing


def get_data_frequency(data, date_limit):
    df = general_preprocessing(data)
    df = df.groupby(['date', 'line', 'station_number', 'station_name'])[['01',
       '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13',
       '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']].sum().reset_index()

    df['day'] = df['date'].dt.day_name()
    df = df[df['date']>date_limit]

    df = df.groupby(['day', 'line', 'station_number', 'station_name'])[['01',
       '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13',
       '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']].mean().reset_index()

    return df
