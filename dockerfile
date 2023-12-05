# create an image from an environment
FROM python:3.10.6-buster

# COPY -> select the folder you need
COPY metro_app.api /api
COPY requirements_api.txt /requirements.txt

# RUN run terminal command
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# install your package
COPY model model
COPY setup.py setup.py
RUN pip install .

CMD uvicorn metro_app.api.fast:app --host 0.0.0.0 --port $PORT
