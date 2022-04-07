FROM python:3.8-slim-buster

COPY requirements.txt requirements.txt

RUN pip install --user -r requirements.txt

COPY config.ini .
COPY setup.py .
COPY setup.cfg .
COPY rexify rexify

RUN pip install -e .

CMD streamlit run rexify/app.py

