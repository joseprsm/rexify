FROM python:3.8-slim-buster AS mlflow

RUN pip install mlflow

EXPOSE 5000


FROM puckel/docker-airflow AS airflow

WORKDIR /usr/local/airflow

COPY requirements.txt requirements.txt

RUN pip install --user -r requirements.txt

COPY setup.py .
COPY setup.cfg .
COPY rexify .

RUN pip install --user -e .


FROM python:3.8-slim-buster AS backend

RUN pip install fastapi pydantic sqlalchemy pandas uvicorn apache-airflow

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
