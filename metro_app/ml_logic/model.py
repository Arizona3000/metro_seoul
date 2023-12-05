
################################################################################
                            # PROPHET MODEL #
################################################################################

import pandas as pd
import numpy as np
import seaborn as sns
from prophet import Prophet
import itertools
from sklearn.metrics import mean_absolute_percentage_error as mape
import plotly.express as px



def prophet_train_predict(df, days=3,
                  changepoint_prior_scale=0.2, seasonality_prior_scale=10.0,
                  holidays_prior_scale=10.0, seasonality_mode='multiplicative'):
    """
    Train a Prophet model with best tested parameters.
    Predicts values for one station during a given horizon (days).
    df needs to be preprocessed with general_preprocessing(df) and
    model_data_preprocessing(df)
    Holidays from South Korea are taken into account.
    -> returns the model, the mape and the prediction
    """

    data = df.rename(columns={'datetime':'ds', 'value' : 'y'})

    data_train = data.iloc[:-24*days]
    data_test = data.iloc[-24*days:]

    #shorten the train data
    data_train = data_train.iloc[-24*500:]

    data_train['is_morning_peak'] = ((data_train['ds'].dt.hour == 8) & (data_train['ds'].dt.dayofweek <= 5)).astype(int)
    data_train['is_afternoon_peak'] = ((data_train['ds'].dt.hour == 18) & (data_train['ds'].dt.dayofweek <= 5)).astype(int)
    data_train['is_closed'] = ((data_train['ds'].dt.hour >= 0) & (data_train['ds'].dt.hour <= 4)).astype(int)

    #Instantiate the model with defined params
    model = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    holidays_prior_scale=holidays_prior_scale,
                    seasonality_mode=seasonality_mode)

    #Adding regressors and seasonality for better predictions
    model.add_regressor(name='is_closed', mode='multiplicative')
    model.add_seasonality(name='morning_peak', period=1, fourier_order=8, condition_name='is_morning_peak')
    model.add_seasonality(name='afternoon_peak', period=1, fourier_order=8, condition_name='is_afternoon_peak')
    model.add_country_holidays(country_name='SK')
    model.fit(data_train)

    # Create a future dataframe for predictions
    future = model.make_future_dataframe(periods=days*24, include_history=False, freq='h')  # Forecasting for the next week
    future['is_morning_peak'] = ((future['ds'].dt.hour >= 8) & (future['ds'].dt.hour <= 9) & (future['ds'].dt.dayofweek <= 5)).astype(int)
    future['is_afternoon_peak'] = ((future['ds'].dt.hour >= 17) & (future['ds'].dt.hour <= 19) & (future['ds'].dt.dayofweek <= 5)).astype(int)
    future['is_closed'] = ((future['ds'].dt.hour >= 0) & (future['ds'].dt.hour <= 4)).astype(int)
    future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
    print(future.columns)

    print(future.head(20))

    # Generate predictions
    forecast = model.predict(future)
    forecast.loc[forecast['is_closed'] != 0, 'yhat'] = 0
    forecast.loc[forecast['morning_peak'] > 2, 'yhat'] = forecast.loc[forecast['morning_peak'] > 2, 'yhat'] * 1.2
    forecast.loc[forecast['afternoon_peak'] > 2, 'yhat'] = forecast.loc[forecast['afternoon_peak'] > 2, 'yhat'] * 1.18
    prediction = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    mape_prophet = mape(prediction['yhat'].values, data_test['y'].values)


    return model, mape_prophet, prediction

def prophet_predict(model, days): #removed df

    # data = df.rename(columns={'datetime':'ds', 'value' : 'y'})
    # data_test = data.iloc[-24*days:]

    # Create a future dataframe for predictions
    future = model.make_future_dataframe(periods=days*24, include_history=False, freq='h')  # Forecasting for the next week
    future['is_morning_peak'] = ((future['ds'].dt.hour >= 8) & (future['ds'].dt.hour <= 9) & (future['ds'].dt.dayofweek <= 5)).astype(int)
    future['is_afternoon_peak'] = ((future['ds'].dt.hour >= 17) & (future['ds'].dt.hour <= 19) & (future['ds'].dt.dayofweek <= 5)).astype(int)
    future['is_closed'] = ((future['ds'].dt.hour >= 0) & (future['ds'].dt.hour <= 4)).astype(int)
    future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
    print(future.columns)

    print(future.head(20))

    # Generate predictions
    forecast = model.predict(future)
    forecast.loc[forecast['is_closed'] != 0, 'yhat'] = 0
    forecast.loc[forecast['morning_peak'] > 2, 'yhat'] = forecast.loc[forecast['morning_peak'] > 2, 'yhat'] * 1.2
    forecast.loc[forecast['afternoon_peak'] > 2, 'yhat'] = forecast.loc[forecast['afternoon_peak'] > 2, 'yhat'] * 1.18
    prediction = forecast[['ds', 'yhat']]

    # mape_prophet = mape(prediction['yhat'].values, data_test['y'].values)

    return prediction


def plot_evaluate(df, prediction, days=3):

    data = df.rename(columns={'datetime':'ds', 'value' : 'y'})
    data_test = data.iloc[-24*days:]
    comparaison = prediction.merge(data_test, on='ds')
    fig = px.line(comparaison, x="ds", y=["y", 'yhat'])

    fig.show()



################################################################################
                            # AUTOARIMA MODEL  #
################################################################################

#voir model_choice.py

################################################################################
                            # LSTM MODEL
################################################################################


from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np


def initialize_lstm(df):
    """
    Define train and test split
    Initialize the Neural Network with random weights
    """
    train, test = df.iloc[:,:int(df.shape[1]*0.8)], df.iloc[:,int(df.shape[1]*0.8):]

    X_train = np.array(train.iloc[:,:-24]).reshape((df.shape[0],int(df.shape[1]*0.8)-24,1))
    y_train = np.array(train.iloc[:,-24:]).reshape(df.shape[0],24,1)

    X_test = np.array(test.iloc[:,:-24]).reshape((df.shape[0],int(1 - (df.shape[1]*0.8))-24,1))
    y_test = np.array(test.iloc[:,-24:]).reshape(df.shape[0],24,1)                    #O.2 test split


    model = Sequential()

    model.add(InputLayer((X_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences= True))
    model.add(LSTM(64))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(24, activation = 'linear'))

    model.summary()

    print("✅ Model initialized")

    return model, X_train, y_train, X_test, y_test



def compile_lstm(model, learning_rate = 0.01):
    """
    Compile the Neural Network
    """
    model.compile(loss = MeanSquaredError(), optimizer= Adam(learning_rate = learning_rate),
                  metrics = [RootMeanSquaredError()])

    print("✅ Model compiled")

    return model



def train_lstm(model,
               X_train,
               y_train,
               batch_size=32,
               patience=4,
               epochs=1,
               validation_split=0.3):
    """
    Train the LSTM model and select a specific patience and number of epochs
    """
    cp = ModelCheckpoint('model/', save_best_only=True) #only save the best model (the one with the smallest validation loss)
    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1)


    history = model.fit(X_train,
                        y_train,
                        batch_size = batch_size,
                        validation_split=validation_split,
                        epochs = epochs,
                        callbacks = [es, cp],
                        verbose =1)
    model = load_model('model/')

    return history, model



def evaluate_lstm(model, X_test, y_test, batch_size = 32):
    """
    Predict the model on the train set and print a DataFrame comparing the
    predictions and the actual values
    """

    metrics = model.evaluate(X_test, y_test, batch_size=batch_size,
        verbose=1)

    loss = metrics[0]
    rmse = metrics[1]

    return metrics


def predict_lstm(model, X_train):
    predictions = model.predict(X_train)

    return predictions
