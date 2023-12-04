from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import os

def cred(path_to_json_key : str):
    """
    Get access to your account by giving identification key\n
    path_to_json_key --> path to your json key
    """
    credentials = ServiceAccountCredentials.from_json_keyfile_name(path_to_json_key,
                                                               scopes=['https://www.googleapis.com/auth/cloud-platform'])
    return storage.Client(credentials=credentials, project='metro-seoul')

def upload(pickle, file_name : str, path_to_json_key : str):
    """
    Upload a file on the bucket 'seoul_bucket', from string \n
    file_path --> your path to the file you want to upload \n
    file_name --> the name you want to give to your file\n
    path_to_json_key --> path to your json key\n
    """
    try:
        client = cred(path_to_json_key)
        bucket = client.get_bucket('seoul_bucket')
        blob = bucket.blob(file_name)
        #blob.upload_from_filename(pickle)
        print(pickle)
        blob.upload_from_string(pickle)
        return 'uploaded file!'
    except Exception as e:
        return f'An error occurred'

def upload_file(file, file_name : str, path_to_json_key : str):
    """
    Upload a file on the bucket 'seoul_bucket', from file name\n
    file_path --> your path to the file you want to upload \n
    file_name --> the name you want to give to your file\n
    path_to_json_key --> path to your json key\n
    """
    try:
        client = cred(path_to_json_key)
        bucket = client.get_bucket('seoul_bucket')
        blob = bucket.blob(file_name)
        #blob.upload_from_filename(pickle)
        print(file)
        blob.upload_from_filename(file)
        return 'uploaded file!'
    except Exception as e:
        return f'An error occurred'

def upload_from_file(file, file_name : str, path_to_json_key : str):
    """
    Upload a file on the bucket 'seoul_bucket', from file\n
    file_path --> your path to the file you want to upload \n
    file_name --> the name you want to give to your file\n
    path_to_json_key --> path to your json key\n
    """
    try:
        client = cred(path_to_json_key)
        bucket = client.get_bucket('seoul_bucket')
        blob = bucket.blob(file_name)
        #blob.upload_from_filename(pickle)
        print(file)
        blob.upload_from_file(file)
        return 'uploaded file!'
    except Exception as e:
        return f'An error occurred'

def download(destination_file_name : str, file_name : str, path_to_json_key : str):
    """
    Download a file on the bucket 'seoul_bucket'\n
    destination_file_name --> name and destination of the file you download\n
    file_name --> the name of the file\n
    path_to_json_key --> path to your json key\n
    """
    try:
        client = cred(path_to_json_key)
        bucket = client.bucket('seoul_bucket')
        blob = bucket.blob(file_name)
        blob.download_to_filename(destination_file_name)
        return 'uploaded file!'
    except Exception as e:
        return e

def view_file(file_name : str, path_to_json_key : str):
    """
    View a file on the bucket 'seoul_bucket'\n
    destination_file_name --> name and destination of the file you download\n
    file_name --> the name of the file\n
    path_to_json_key --> path to your json key\n
    """
    try:
        client = cred(path_to_json_key)
        bucket = client.bucket('seoul_bucket')
        blob = bucket.blob(file_name)
        #blob = bucket.blob(storage_name)
        x = blob.download_as_string()
        #print(x.decode('utf-8'))
        #blob.download_to_filename(destination_file_name)
        return x
    except Exception as e:
        return e
