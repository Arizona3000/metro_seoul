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
from model import prophet_train_predict
from gcp.setup import upload, upload_from_file

def choose_model(df, station_list, number_hours, key_to_json):

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

        sf.fit(train)

        forecast = sf.predict(h = number_hours)

        station_number = df_station.station_number.unique()[0]

        forecast2 = forecast.copy()
        forecast2['MSTL'] = forecast2.apply(lambda row: 0 if 0 <= row['ds'].hour <= 4 else row['MSTL'], axis=1)

        forecast2['station_number'] = station_number

        mape_mstl = mape(forecast2['MSTL'].values, test.head(number_hours)['y'].values)


        ## Prophet

        model, mape_prophet, prediction = prophet_train_predict(df_station, days=3,
                                                                changepoint_prior_scale=0.2, seasonality_prior_scale=10.0,
                                                                holidays_prior_scale=10.0, seasonality_mode='multiplicative')

        prediction_prophet = prediction[['ds', 'yhat']]

        station_name = df_station.station_name.unique()[0]


        prediction_prophet['unique_id'] = station_name
        prediction_prophet['station_number'] = station_number


        prediction_prophet = prediction[['unique_id', 'ds', 'yhat']]

        prediction_prophet.rename(columns={'yhat':'MSTL'}, inplace=True)



        ## Station number

        #station_number = df_station.station_number.unique()[0]


        if mape_prophet > mape_mstl:
            #print('yes')
            serialized_model = pickle.dumps(sf)
            print(upload(serialized_model,file_name=f'models/model_station_{station_number}.pkl', path_to_json_key=key_to_json))
            print(upload_from_file(forecast2, file_name=f'data/pred/data_pred_station_{station_number}.pkl', path_to_json_key=key_to_json ))
            print('File has been uploaded to gcp')

        else:
            serialized_model = pickle.dumps(model)
            print(upload(serialized_model,file_name=f'models/model_station_{station_number}.pkl', path_to_json_key=key_to_json))
            print(upload_from_file(forecast2, file_name=f'data/pred/data_pred_station_{station_number}.pkl', path_to_json_key=key_to_json ))
            print('File has been uploaded to gcp')
