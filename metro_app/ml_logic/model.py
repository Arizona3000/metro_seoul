
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
                            # LSTM MODEL  #
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

def df_to_X_y(df, window_size=8760):
    """
    Define X and y for lstm model with a specific number of observations
    (window_size where 8760 = 24*7*52 -> one year of data )
    """
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)



def initialize_lstm(input_shape: tuple, X, y,df):
    """
    Define train, test and validation split
    Initialize the Neural Network with random weights
    """
    X_train, y_train = X[:0.7*len(df)], y[:0.7*len(df)]                     #0.7 train split
    X_val, y_val = X[0.7*len(df):0.8*len(df)], y[0.7*len(df):0.8*len(df)]   #0.1 val split
    X_test, y_test =  X[0.8*len(df):], y[0.8*len(df):]                      #O.2 test split


    model = Sequential()

    model.add(InputLayer(input_shape))
    model.add(LSTM(64))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))

    model.summary()

    print("✅ Model initialized")

    return model, X_train, y_train, X_val, y_val, X_test, y_test



def compile_lstm(model, learning_rate = 0.01):
    """
    Compile the Neural Network
    """
    model.compile(loss = MeanSquaredError(), optimizer= Adam(learning_rate = learning_rate),
                  metrics = [RootMeanSquaredError()])

    print("✅ Model compiled")

    return model



def train_lstm(model, X_train, y_train, X_val, y_val, patience, epochs):
    """
    Train the LSTM model and select a specific patience and number of epochs
    """
    cp = ModelCheckpoint('model/', save_best_only=True) #only save the best model (the one with the smallest validation loss)
    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1)


    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs = epochs, callbacks = [es, cp])
    model = load_model('model/')

    return model



def predict_lstm(model, X_train, y_train):
    """
    Predict the model on the train set and print a DataFrame comparing the
    predictions and the actual values
    """
    train_predictions = model.predict(X_train).flatten()

    train_results = pd.DataFrame(data ={'Train Predictions': train_predictions, 'Actuals': y_train})

    return train_results



def visualisation_lstm_train(train_results, interval):
    """
    Visualisation of the two trends on a specified interval (interval)
    """
    plt.figure(figsize=(16,9))
    plt.plot(train_results['Train Predictions'][:interval])
    plt.plot(train_results['Actuals'][:interval])
    plt.show()
