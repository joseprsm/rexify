FROM python:3.8-slim-buster AS mlflow

RUN pip install mlflow

EXPOSE 5000


FROM python:3.8-slim-buster AS training_pipeline

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt


FROM python:3.8-slim-buster AS backend

RUN pip install fastapi pydantic sqlalchemy pandas


FROM python:3.8-slim-buster AS frontend

RUN pip install streamlit pandas matplotlib
