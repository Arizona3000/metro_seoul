from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    MSTL
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.metrics import mean_absolute_percentage_error as mape

from preprocess import general_preprocessing, model_data_preprocessing
from gcp.setup import upload

def choose_model(df, station_list, key_to_json):

    df = general_preprocessing(df)
    df = model_data_preprocessing(df)

    df = df.groupby(['datetime', 'station_name', 'station_number', 'line'])['value'].sum().reset_index()

    for station, station_line in station_list:

        df_station = df[(df['station_name'] == station.strip()) & (df['line'] == int(station_line))]

        ## MSTL

        df_model_mstl = df_station[['station_name', 'datetime', 'value']]
        df_model_mstl.rename(columns={'station_name':'unique_id', 'datetime':'ds', 'value':'y'}, inplace=True)

        train_size = int(len(df_model_mstl) * 0.8)
        train, test = df_model_mstl.iloc[:train_size, :], df_model_mstl.iloc[train_size:, :]

        models = [MSTL(
            season_length=[24, 24 * 7, 24 * 7 * 2, 24 * 7 * 52], # seasonalities of the time series  ##Ajouter 24*7*365,25
            trend_forecaster=AutoARIMA()
        )]

        sf = StatsForecast(
            models=models,
            df=train,
            freq='H',
            n_jobs=-1,
            )

        forecast = sf.forecast(df = train, h = 24)

        mape_mstl = mape(forecast['MSTL'].values, test.head(24)['y'].values)


        ## Prophet




        mape_prophet = 1


        ## Station number

        station_number = df_station.station_number.unique()[0]


        if mape_prophet > mape_mstl:
            print('yes')
            serialized_model = pickle.dumps(sf)
            print(upload(serialized_model,file_name=f'models/MSTL/MSTL_station_{station_number}.pkl', path_to_json_key=key_to_json))
            print('Check')

        print('finish')

        break
