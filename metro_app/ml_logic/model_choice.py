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

from metro_app.ml_logic.preprocess import general_preprocessing, model_data_preprocessing
from metro_app.ml_logic.model import prophet_train_predict
from gcp.setup import upload, upload_from_file

def choose_model(df):

    df = general_preprocessing(df)
    df = model_data_preprocessing(df)

    df = df[df['entry/exit'] == 'exit']

    station_list = [list(row) for row in df[['station_name', 'line']].values]
    unique_list = [list(item) for item in set(tuple(row) for row in station_list)]
    df = df.groupby(['datetime', 'station_name', 'station_number', 'line'])['value'].sum().reset_index()

    list_mape = []

    for station, station_line in unique_list:

        try :

            df_station = df[(df['station_name'] == station.strip()) & (df['line'] == int(station_line))]

            station_number = df_station.station_number.unique()[0]

            model, mape_prophet, prediction = prophet_train_predict(df_station, days=3,
                                                                    changepoint_prior_scale=0.2, seasonality_prior_scale=10.0,
                                                                    holidays_prior_scale=10.0, seasonality_mode='multiplicative')

            if mape_prophet == 0 :
                pass

            else:

                list_mape.append(mape_prophet)

                prediction_prophet = prediction[['ds', 'yhat']]

                station_name = df_station.station_name.unique()[0]


                prediction_prophet['unique_id'] = str(station).strip()
                prediction_prophet['station_number'] = station_number
                prediction_prophet['line'] = station_line



                prediction_prophet = prediction_prophet[['unique_id','station_number','line', 'ds', 'yhat']]

                prediction_prophet.rename(columns={'yhat':'MSTL'}, inplace=True)

                print(f'the mape for {station_name} is {mape_prophet}')

                serialized_model = pickle.dumps(model)
                upload(serialized_model,file_name=f'models/exit/model_station_{station_number}.pkl')
                upload(prediction_prophet.to_csv(), file_name=f'data/pred/exit/data_pred_station_{station_number}.csv')
                print('File has been uploaded to gcp (prophet)')

        except:
            pass


    average = sum(list_mape) / len(list_mape)
    return average
