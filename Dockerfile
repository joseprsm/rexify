FROM python:3.8-slim-buster AS mlflow

RUN pip install mlflow

EXPOSE 5000


FROM python:3.8-slim-buster AS training_pipeline

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt


FROM python:3.8-slim-buster AS backend

RUN pip install fastapi pydantic sqlalchemy pandas uvicorn

RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && pip install psycopg2

COPY rexify/app rexify/app

COPY setup.py setup.py
COPY setup.cfg setup.cfg

RUN pip install -e .

CMD python rexify/app/main.py


FROM python:3.8-slim-buster AS frontend

RUN pip install streamlit pandas matplotlib
