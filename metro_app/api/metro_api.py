from fastapi import FastAPI
import requests
import pandas as pd
import pickle
import sys
from io import StringIO
# sys.path.append('/Users/thomas_metral/code/thomas-metral/metro_seoul/metro_app')
# sys.path.append('/Users/victor/code/Arizona3000/metro_seoul/metro_app')
# sys.path.append('/Users/victor/code/Arizona3000/metro_seoul')
# sys.path.append('/Users/victor/code/Arizona3000/metro_seoul/metro_app/ml_logic')

from metro_app.ml_logic.preprocess import general_preprocessing, model_data_preprocessing
from gcp.setup import view_file
from metro_app.ml_logic.model import prophet_predict
from metro_app.ml_logic.data import get_data_frequency
# import sys
# sys.path.append('/Users/thomas_metral/code/thomas-metral/metro_seoul')

app = FastAPI()

app.state.crowd_str = view_file('data/crowd2020-2023.csv')

@app.get("/")
async def root():
    return {"message":
        "bienvenue sur l'api qui prédit l'affluence du métro de Séoul, créée par un groupe d'amis passionnés de bien-être dans les transports, convaincus qu'il existe des solutions durables à des conditions saines de transport en commun dans les grandes métropoles"}


@app.get("/ping")
def pong():
    return {"ping": "pong!"}

# http://localhost:8000/predict_seongsu?days=2

# @app.get('/predict_seongsu')
# def predict_seongsu(days:int):

#     # load pickle model
#     print("loading model...")
#     pickled_model = pickle.load(open('model_test.pkl', 'rb'))
#     print("loading done.")

#     #result = model.predict(data reshaped)
#     prediction = prophet_predict(pickled_model, days=days)

#     # return the result data as dict
#     return prediction.to_dict('list')


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




# @app.get('/testvicmodel')
# def testvicmodel(station : int, days : int):

#     model_pkl = view_file(f'models/model_station_{station}.pkl')
#     model = pickle.loads(model_pkl)

#     ######### Remplacer par fonction predict de thomas ##############

#     prediction = prophet_predict(model=model, days=days)

#     # future = model.make_future_dataframe(periods=3*24, include_history=False, freq='h')  # Forecasting for the next week
#     # future['is_morning_peak'] = ((future['ds'].dt.hour >= 8) & (future['ds'].dt.hour <= 9) & (future['ds'].dt.dayofweek <= 5)).astype(int)
#     # future['is_afternoon_peak'] = ((future['ds'].dt.hour >= 17) & (future['ds'].dt.hour <= 19) & (future['ds'].dt.dayofweek <= 5)).astype(int)
#     # future['is_closed'] = ((future['ds'].dt.hour >= 0) & (future['ds'].dt.hour <= 4)).astype(int)
#     # future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
#     # #print(future.columns)

#     # #print(future.head(20))

#     # # Generate predictions
#     # forecast = model.predict(future)
#     # forecast.loc[forecast['is_closed'] != 0, 'yhat'] = 0
#     # forecast.loc[forecast['morning_peak'] > 2, 'yhat'] = forecast.loc[forecast['morning_peak'] > 2, 'yhat'] * 1.2
#     # forecast.loc[forecast['afternoon_peak'] > 2, 'yhat'] = forecast.loc[forecast['afternoon_peak'] > 2, 'yhat'] * 1.18
#     # prediction = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

#     # ######### Remplacer par fonction predict de thomas ##############


#     return prediction.to_dict('list')


# @app.get('/derniertest')
# def derniertest():

#     df = pd.read_csv('/Users/victor/code/Arizona3000/metro_seoul/raw_data/crowd2020-2023.csv') # Ajouter le csv sur gcp

#     df1 = general_preprocessing(df)
#     df2 = model_data_preprocessing(df1)
#     station_number_list = df2['station_number'].drop_duplicates().to_list()

#     df_concat = pd.DataFrame(columns = ['Unnamed: 0', 'unique_id', 'station_number', 'line', 'ds', 'MSTL'])

#     for station in station_number_list:
#         try:
#             file = view_file(f'data/pred/data_pred_station_{station}.csv', '/Users/victor/gcp/metro-seoul-86af79318438.json')
#             #print(f'station test .{station}.')
#             df_to_merge = pd.read_csv(StringIO(str(file,'utf-8')))
#             df_to_merge['ds'] = df_to_merge['ds'].astype(str)
#             #print(df_to_merge)
#             df_concat = pd.concat([df_concat,df_to_merge])

#         except:
#             print(f'no file for station {station}')

#     return df_concat.to_dict('list')
