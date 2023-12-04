
################################################################################
                            # PROPHET MODEL #
################################################################################

import pandas as pd
import numpy as np
import seaborn as sns
from prophet import Prophet
import itertools


def train_prophet(df, changepoint_prior_scale=0.002, seasonality_prior_scale=3.0,
                  holidays_prior_scale=0.02, seasonality_mode='multiplicative'):
    """
    Train a Prophet model with default parameters based on a cross validation.
    returns a model fitted on a dataframe with one station.
    df needs to be preprocessed with prophet_preprocessing_one_station function.
    Holidays from South Korea are taken into account.
    """
    m = Prophet(changepoint_prior_scale, seasonality_prior_scale,
                  holidays_prior_scale, seasonality_mode)

    m.add_country_holidays(country_name='KR')

    m.fit(df)
    return m

def predict_prophet(model, days):
    """
    Predicts the number of people in a given metro stop.
    according to model (m) for a given horizon (days)
    """

    future = model.make_future_dataframe(periods=days*24, freq='h', include_history=False)
    forecast = model.predict(future)
    prediction = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    return prediction



################################################################################
                            # AUTOARIMA MODEL  #
################################################################################


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


def initialize_lstm(df):
    """
    Define train and test split
    Initialize the Neural Network with random weights
    """
    train, test = df.iloc[:,:int(df.shape[1]*0.8)], df.iloc[:,int(df.shape[1]*0.8):]

    X_train = np.array(train.iloc[:,:-24]).reshape((df.shape[0],int(df.shape[1]*0.8)-24,1))
    y_train = np.array(train.iloc[:,-24:]).reshape(df.shape[0],24,1)

    X_test = np.array(test.iloc[:,:-24]).reshape((df.shape[0],int(df.shape[1]*0.2)-24,1))
    y_test = np.array(test.iloc[:,-24:]).reshape(df.shape[0],24,1)                   #O.2 test split


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

    loss = metrics["loss"]
    rmse = metrics["root_mean_squared_error"]

    return metrics


def predict_lstm(model, X_train):
    predictions = model.predict(X_train)

    return predictions
