# Put preprocessing here


import pandas as pd
import numpy as np


def general_preprocessing(df):
    """
    Preprocess data, rename columns, add all hours, reorder columns, convert to datetime\n
    df --> expecting dataframe crowding metro
    """

    #rename
    df.rename(columns = {'06': '05'}, inplace = True )
    df.rename(columns = {'06-07': '06'}, inplace = True)
    df.rename(columns = {'07-08': '07'}, inplace = True)
    df.rename(columns = {'08-09': '08'}, inplace = True)
    df.rename(columns = {'09-10': '09'}, inplace = True)
    df.rename(columns = {'10-11': '10'}, inplace = True)
    df.rename(columns = {'11-12': '11'}, inplace = True)
    df.rename(columns = {'12-13': '12'}, inplace = True)
    df.rename(columns = {'13-14': '13'}, inplace = True)
    df.rename(columns = {'14-15': '14'}, inplace = True)
    df.rename(columns = {'15-16': '15'}, inplace = True)
    df.rename(columns = {'16-17': '16'}, inplace = True)
    df.rename(columns = {'17-18': '17'}, inplace = True)
    df.rename(columns = {'18-19': '18'}, inplace = True)
    df.rename(columns = {'19-20': '19'}, inplace = True)
    df.rename(columns = {'20-21': '20'}, inplace = True)
    df.rename(columns = {'21-22': '21'}, inplace = True)
    df.rename(columns = {'22-23': '22'}, inplace = True)
    df.rename(columns = {'23-24': '23'}, inplace = True)

    #add columns (missing hours)
    df['24']=0
    df['01'] = 0
    df['02'] = 0
    df['03'] = 0
    df['04'] = 0

    #reordering columns
    df = df[['date', 'line', 'station_number', 'station_name', 'entry/exit',
             '01', '02', '03', '04', '05', '06','07', '08', '09', '10', '11',
             '12', '13', '14', '15', '16', '17', '18','19', '20', '21', '22',
             '23', '24']]

    #preprocessing date & line
    df['date'] = pd.to_datetime(df['date'])
    df['line'] = df['line'].astype('int')

    return df


def model_data_preprocessing(df):
    """
    Preprocess dataframe for Auto-Arima model\n
    df --> preprocessed crowding dataframe expected
    """

    #melt - reshape
    df_final = pd.melt(df, id_vars=['date','line', 'station_number',
            'station_name', 'entry/exit'], value_vars=['01','02','03',
            '04', '05', '06', '07', '08', '09', '10', '11', '12', '13',
            '14','15', '16', '17', '18', '19', '20', '21', '22', '23',
            '24'])
    df_final['variable'] = df_final['variable'].apply(int)


    #datetime
    df_final['datetime'] = pd.to_datetime(df_final['date']) + pd.to_timedelta(df_final.variable, unit = 'h')

    #details
    df_final.drop(columns = ['date', 'variable'], inplace=True)
    df_final.set_index('datetime', inplace = True)

    return df_final


def prophet_preprocessing_one_station(df, station_number, entry_exit='entry'):
    """
    Preprocess dataframe for Prophet model using general_preprocessing and
    model_data_preprocessing functions. User can choose one station number
    to isolate and weather passengers are entering or exiting
    -> returns a dataframe with the following mandatory columns:
    'ds' as datetime
    'y' as target (number of people entering or exiting)
    """
    #using the two functions above to start preprocessing
    df = general_preprocessing(df)
    df = model_data_preprocessing(df)

    #Isolating the entries of one station
    df = df[df['station_number'] == station_number]
    df = df[df['entry/exit'] == entry_exit]

    #renaming columns adapted to prophet
    df.rename(columns={'value':'y'}, inplace=True)
    df.index.names = ['ds']

    #sorting dates as ascending
    df.sort_index(ascending=True, inplace=True)

    #resetting index to have datetimes as a column
    df.reset_index(inplace=True)

    return df
