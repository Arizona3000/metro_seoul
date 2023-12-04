# Put the data imports/exports here
import sys
sys.path.append("/Users/yannickdeza/code/wagon_project/metro_seoul")
import pandas as pd
import io
from metro_app.ml_logic.preprocess import general_preprocessing
from gcp.setup import view_file
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def get_data_frequency(data, date_limit):
    """
    Function that returns a dataframe of average crowding per station per day\n
    data --> the crowding dataframe
    date_limit --> the number of days you want for the average
    """
    df_prepro = general_preprocessing(data)
    df_prepro = df_prepro.groupby(['date', 'line', 'station_number', 'station_name'])[['01',
       '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13',
       '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']].sum().reset_index()

    df_prepro['day'] = df_prepro['date'].dt.day_name()
    df_prepro = df_prepro[df_prepro['date']>date_limit]

    df_prepro = df_prepro.groupby(['day', 'line', 'station_number', 'station_name'])[['01',
       '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13',
       '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']].mean().reset_index()

    return df_prepro



def get_data_for_timetable(file_name_gcp, day, hours, number_of_trains, key_to_json):
    """
    Function to get access to the time of the next metro at a specific hour.
    The dataframe returned contains the station, the line, the day of interest and
    the direction of the metro. \n
    file_name_gcp --> the name of the file in the bucket\n
    day --> between 'weekdays', 'saturday', 'sunday'\n
    hours --> the hours requested by the user\n
    number_of_trains --> the number of rows you want in the dataframe\n
    key_to_json --> the key to your json file\n
    """

    timetable_string = view_file(f'data/timetable/{file_name_gcp}', key_to_json)
    timetable = pd.read_csv(io.StringIO(timetable_string.decode('utf-8')))

    timetable[day] = timetable[day].apply(lambda x: '00' + x[2:] if x.startswith('24') else x)
    timetable[day] = pd.to_datetime(timetable[day], format='%H:%M')
    timetable_copy = timetable.copy()

    user_time = pd.to_datetime(hours, format='%H:%M')

    timetable_copy['time_diff'] = abs(timetable_copy['weekdays'] - user_time)
    closest_time_row = timetable_copy.sort_values(by='time_diff').head(number_of_trains)

    return pd.DataFrame(closest_time_row)




def prediction_data():

    """This function pre-process the data on passenger flows to make predictions on a single day"""

    data = pd.read_csv('raw_data/passenger_flow.csv')
    new_data = general_preprocessing(data)

    # 2) I filter for a specific date
    new_data = data[data['date'] == '2023-06-30']

    # 3) I drop columns I don't care about
    new_data.drop(columns=['Unnamed: 0', 'station_number', 'date', '24', '01', '02', '03', '04'], inplace=True)

    # 4) I rmeove duplicates
    new_data.drop_duplicates(inplace=True)

    return new_data


def translation_data():

    """ This function pre-process the translation data,
    where each metro station is translated from korean to english"""

    map = pd.read_excel('raw_data/translation_data.xlsx')
    #1) I rename columns
    map.drop(map.columns[0:2], axis=1, inplace=True)
    map.rename(columns={map.columns[1]: 'station_name', map.columns[0]: 'korean_name'}, inplace=True)
    # 2) I remove duplicates
    map.drop_duplicates(inplace=True)

    return map


def location_data():

    """This function pre-process the location data,
    where korean station names are associated with GPS coordinates (lat & lon)"""

    loc = pd.read_csv('raw_data/location_data.csv')
    loc.drop(columns=['Unnamed: 0'], inplace=True)
    loc.rename(columns={'name': 'korean_name'}, inplace=True)
    # I clean the "line" column and I only keep lines from 1 to 8
    loc['line'] = loc.line.str.extract('(\d+)')
    loc['line'].replace('2', np.nan, inplace=True)
    loc.dropna(inplace=True)
    loc['line'] = loc.line.str.lstrip('0')
    loc['line'].replace('9', np.nan, inplace=True)
    loc.dropna(inplace=True)

    return loc


def merge_location_translation():

    """ This function merges the processed location dataframe
    with the processed translation dataframe"""

    location = location_data()
    translation = translation_data()
    merged = location.merge(translation, on='korean_name')
    merged.drop_duplicates(inplace=True)

    return merged



def plot_data():

    """This function returns the dataframe for plotting on a map the various metro_stations"""

    predictions = prediction_data()
    location_and_translation = merge_location_translation()

    location_and_translation.drop(columns=['line', 'no'], inplace=True)

    complete = predictions.merge(location_and_translation, on='station_name')
    complete.drop(columns=['korean_name'], inplace=True)
    columns_to_unpivot = complete.columns[3:22].tolist()
    final_df = pd.melt(complete, id_vars=['station_name', 'line', 'entry/exit', 'lat', 'lng'],
            value_vars=columns_to_unpivot)
    final_df.rename(columns={'value': 'n_passengers', 'variable': 'hour'}, inplace=True)

    final_df.drop_duplicates(inplace=True)


    return final_df




def lines_data():
    """ This function returns a dictionary where each key represents a metro line
    (e.g. line_1, line_2, etc...), and each value corresponds to a dataframe
    containing the ordered list of metro stations in that line, with info on
    lat and lon of the following station
    """
    location_and_translation = merge_location_translation()
    location_and_translation = location_and_translation.drop(columns=['korean_name'])
    lines = {}

    for i in range(1, 9, 1):
        line = location_and_translation[location_and_translation['line'] == f'{i}']
        line['next_lat'] = line['lat'].shift(-1, axis=0)
        line['next_lng'] = line['lng'].shift(-1, axis=0)

        lines[f'line_{i}'] = line

    return lines
