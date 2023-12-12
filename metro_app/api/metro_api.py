from fastapi import FastAPI
import requests
import pandas as pd
import pickle
import sys
from io import StringIO

from metro_app.ml_logic.preprocess import general_preprocessing, model_data_preprocessing
from gcp.setup import view_file
from metro_app.ml_logic.model import prophet_predict
from metro_app.ml_logic.data import get_data_frequency

app = FastAPI()

app.state.crowd_str = view_file('data/crowd2020-2023.csv')

@app.get("/")
async def root():
    return {"message":
        "bienvenue sur l'api qui prédit l'affluence du métro de Séoul, créée par un groupe d'amis passionnés de bien-être dans les transports, convaincus qu'il existe des solutions durables à des conditions saines de transport en commun dans les grandes métropoles"}


@app.get("/ping")
def pong():
    return {"ping": "pong!"}


@app.get('/home')
def testvic(group:str):

    if group == 'entry':
        prediction_csv_bytes = view_file(f'data/all_predictions_entry.csv')

    elif group =='exit':
        prediction_csv_bytes = view_file(f'data/all_predictions_exit.csv')

    else:
        prediction_csv_bytes = view_file(f'data/all_predictions.csv')

    prediction_csv = pd.read_csv(StringIO(str(prediction_csv_bytes,'utf-8')))
    prediction_csv2 = prediction_csv[['station_name', 'station_number', 'line', 'ds', 'MSTL', 'lat', 'lng']]
    prediction_csv2 = prediction_csv2.drop_duplicates()
    return prediction_csv2.to_dict('list')


@app.get('/home/station')
def model_station(station: int, days:int, pred:str, group:str):

    if group == 'entry': # Entry
        model_pkl = view_file(f'models/entry/model_station_{station}.pkl')
        model = pickle.loads(model_pkl)

    elif group == 'exit': # Exit
        model_pkl = view_file(f'models/exit/model_station_{station}.pkl')
        model = pickle.loads(model_pkl)

    else: # All
        model_pkl = view_file(f'models/model_station_{station}.pkl')
        model = pickle.loads(model_pkl)

    prediction = prophet_predict(model=model, days=days)
    prediction['day'] = prediction['ds'].dt.day_name()

    if pred == 'tmrw': # tmrw
        prediction = prediction[(prediction['ds']>'2023-06-29') & (prediction['ds']<'2023-06-30')]

    if pred == 'aftmrw': # After tmrw
        prediction = prediction[(prediction['ds']>'2023-06-30')]

    else: # Today
        prediction = prediction.head(24)


    dict_pred = prediction.to_dict('list')
    print('pred_dict done')

    #Frequency per day of the week

    #crowd_str = view_file('data/crowd2020-2023.csv')
    crowd_str = app.state.crowd_str
    assert crowd_str is not None
    crowd = pd.read_csv(StringIO(str(crowd_str,'utf-8')))
    print('crowd imported')

    average_per_day = get_data_frequency(crowd, '2023-04-01')
    average_per_day = average_per_day[average_per_day['station_number']==station]
    print('average_per_day done')

    average_day_dict = average_per_day.to_dict('list')
    #print(dict_pred | average_day_dict)

    #Itineraire

    iti_bytes = view_file(f'data/iti_demo')
    iti = pd.read_csv(StringIO(str(iti_bytes,'utf-8')))
    iti = iti.to_dict('list')


    dict_final = {'prediction_station' : dict_pred,
                  'average_per_day': average_day_dict,
                  'itineraire' : iti}

    return dict_final
