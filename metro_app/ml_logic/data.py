# Put the data imports/exports here

import pandas as pd
import io
from preprocess import general_preprocessing
from gcp.setup import view_file


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

def get_data_for_map(df):
    pass


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
