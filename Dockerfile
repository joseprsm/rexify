FROM python:3.8-slim-buster AS base

WORKDIR src

RUN pip install rexify numpy pandas

FROM base AS preprocess

RUN pip install scikit-learn

COPY rexify/components/load/task.py ./load.py

ENTRYPOINT ["python", "-m", "load.py"]

FROM base AS tf

RUN pip install tensorflow tensorflow_recommenders

FROM tf AS train

COPY rexify/components/train/task.py ./train.py

ENTRYPOINT ["python", "-m", "train.py"]

FROM tf AS index

RUN pip install scann

COPY rexify/components/index/task.py ./index.py

ENTRYPOINT ["python", "-m", "index.py"]

FROM tf AS rank

COPY rexify/components/rank/task.py ./rank.py

ENTRYPOINT ["python", "-m", "rank.py"]