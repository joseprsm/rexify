FROM python:3.8-slim-buster AS preprocess

RUN pip install scikit-learn==1.0.2

WORKDIR src

COPY rexify/components/preprocess/task.py ./preprocess.py

ENTRYPOINT ["python", "-m", "preprocess.py"]

FROM python:3.8-slim-buster AS base

COPY config.ini .
COPY setup.py .
COPY setup.cfg .
COPY rexify rexify

RUN pip install -e .

FROM base AS train

WORKDIR src

COPY rexify/components/train/task.py ./train.py

ENTRYPOINT ["python", "-m", "train.py"]

FROM base AS index

RUN pip install scann

WORKDIR src

COPY rexify/components/index/task.py ./index.py

ENTRYPOINT ["python", "-m", "index.py"]