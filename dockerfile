# create an image from an environment
FROM python:3.10.6-buster

# COPY -> select the folder you need
COPY metro_app /metro_app
COPY requirements_api.txt /requirements.txt

# RUN run terminal command
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
