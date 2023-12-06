FROM python:3.10.6-buster
# COPY -> select the folder you need
COPY metro_app/api metro_app/api
COPY metro_app/ml_logic metro_app/ml_logic
COPY requirements_api.txt requirements.txt
COPY gcp gcp
# RUN run terminal command
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# install your package
COPY setup.py setup.py
RUN pip install .
CMD uvicorn metro_app.api.metro_api:app --host 0.0.0.0 --port 8000
