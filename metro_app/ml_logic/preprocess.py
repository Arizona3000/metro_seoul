import pandas as pd
import numpy as np


################################################################################
                            # GENERAL PREPROCESSING #
################################################################################


def general_preprocessing(df):
    """
    Preprocess data, rename columns, add all hours, reorder columns, convert to datetime\n
    df --> expecting dataframe crowding metro
    """

    #translate
    dic = {
        '신촌': 'Sinchon',
        '교대(법원.검찰청)': 'Seoul National Univ. of Education(Court & Prosecutors Office)',
        '화랑대(서울여대입구)': 'Hwarangdae',
        '남한산성입구(성남법원.검찰청)': 'Namhansanseong',
        '동대문역사문화공원(DDP)(DDP)': 'Dongdaemun History Culture Park',
        '낙성대': 'Nakseongdae',
        '용마산': 'Yongmasan'
        }

    df.replace({'station_name': dic}, inplace = True)

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



################################################################################
                         # MODEL DATA PREPROCESSING #
################################################################################


def model_data_preprocessing(df):  #intégrer genal_preprocessing
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




################################################################################
                         # PROPHET PREPROCESSING #
################################################################################


def prophet_preprocessing_one_station(df, station_name, metro_line, entry_exit='exit'):
    """
    Preprocess dataframe for Prophet model using preprocess lstm function.
    User can isolate one station and choose between entry or exit.
    -> returns a dataframe with the following mandatory columns:
    'ds' as datetime
    'y' as target (number of people entering or exiting)
    """
    df = preprocess_lstm(df, entry_exit)

    #choosing which station to look at for a given line
    station = f"{station_name} {metro_line}"
    df = pd.DataFrame(df[station])

    #setting the right columns for prophet
    df.reset_index(inplace=True)
    df.rename(columns={'datetime': 'ds', station : 'y'}, inplace=True)

    return df




################################################################################
                          # LSTM PREPROCESSING # #A REFAIRE!!!!!!!!!!
################################################################################


def preprocess_lstm(df, exit_entry:str):
    """
    Preprocessing for lstm model\n
    df --> Crowd dataframe expected\n
    exit_entry --> Choose between entry or exit depending on what you need\n
    """

    df = general_preprocessing(df)
    df = model_data_preprocessing(df)

    df.reset_index(inplace=True)

    df_exit = df[df['entry/exit'] == exit_entry]
    df_exit['station_name_line'] = df_exit['station_name'] + ' ' + df_exit['line'].apply(str)
    df_exit.reset_index(inplace = True)

    df_lstm_exit = pd.pivot(df_exit.drop(columns = ['station_name','line', 'station_number',
                                            'entry/exit']), index=['station_name_line'],
                                            columns='datetime', values='value')

    df_lstm_exit_T = df_lstm_exit.T

    for column in df_lstm_exit_T.columns:
        if df_lstm_exit_T[column].isna().sum() > 0:
            df_lstm_exit_T.drop(columns=column, inplace=True)
        else:
            pass

    return df_lstm_exit_T
