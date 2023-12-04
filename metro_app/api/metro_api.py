from fastapi import FastAPI
import requests
import pandas as pd
import pickle
import sys
sys.path.append('/Users/thomas_metral/code/thomas-metral/metro_seoul/metro_app')
from ml_logic.preprocess import general_preprocessing, model_data_preprocessing
from ml_logic.model import prophet_predict
# import sys
# sys.path.append('/Users/thomas_metral/code/thomas-metral/metro_seoul')

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# http://localhost:8000/predict_prophet?days=2

@app.get('/predict_prophet')
def predict_prophet(days:int):

    # get the data and put it in the right format
    df = pd.read_csv('/Users/thomas_metral/code/thomas-metral/metro_seoul/raw_data/crowd2020-2023.csv')

    # call preproc and all
    df = pd.read_csv('/Users/thomas_metral/code/thomas-metral/metro_seoul/raw_data/crowd2020-2023.csv')
    df1 = general_preprocessing(df)
    df2 = model_data_preprocessing(df1)
    df2.reset_index(inplace=True)
    df3 = df2.groupby(['datetime', 'station_name', 'station_number', 'line'])['value'].sum().reset_index()
    data = df3[(df3['station_name'] == 'Seongsu') &
               (df3['line'] == 2)]

    # load pickle model
    print("loading model...")
    pickled_model = pickle.load(open('model_test.pkl', 'rb'))
    print("loading done.")

    #result = model.predict(data reshaped)
    mape, prediction = prophet_predict(pickled_model, data, days=days)

    # return the result data as dict
    return dict(prediction)
