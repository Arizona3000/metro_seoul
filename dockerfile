# create an image from an environment
FROM python:3.10.6-buster

# COPY -> select the folder you need
COPY requirements_api.txt /requirements.txt
RUN pip install -r requirements.txt

COPY metro_app/api /api
COPY setup.py setup.py
RUN pip install --upgrade pip

CMD uvicorn metro_app.api.fast:app --host 0.0.0.0 --port $PORT
