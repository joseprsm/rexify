FROM python:3.8-slim-buster AS base

WORKDIR src

FROM base AS frontend

RUN pip install streamlit pandas
COPY rexify/app.py ./app.py
CMD python -m streamlit run app.py

FROM base AS preprocess

RUN pip install scikit-learn==1.0.2
COPY rexify/components/preprocess/task.py ./preprocess.py
ENTRYPOINT ["python", "-m", "preprocess.py"]

FROM base AS rexify

COPY config.ini .
COPY setup.py .
COPY setup.cfg .
RUN pip install -e .

FROM rexify AS train

COPY rexify/components/train/task.py ./train.py
ENTRYPOINT ["python", "-m", "train.py"]

FROM base AS index

RUN pip install scann
COPY rexify/components/index/task.py ./index.py
ENTRYPOINT ["python", "-m", "index.py"]