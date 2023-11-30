# Put the code for the model here
import pandas as pd
from prophet import Prophet
import itertools
import numpy as np
import seaborn as sns

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
