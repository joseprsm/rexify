FROM python:3.8.6 AS base

WORKDIR /usr/local/rexify

RUN pip install numpy pandas

FROM base AS rexify

COPY rexify/features rexify/features
COPY rexify/models rexify/models
COPY rexify/utils.py rexify/utils.py
COPY rexify/__init__.py rexify/__init__.py
COPY setup.cfg .
COPY setup.py .

RUN pip install .

FROM base AS load

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

ENTRYPOINT ['python', '-m', 'index.py']