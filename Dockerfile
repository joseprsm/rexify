FROM python:3.8.6 AS base

WORKDIR /usr/local/rexify

RUN pip install numpy~=1.19.5 pandas~=1.3.3 click~=8.1.2 scikit-learn~=1.0.2

FROM base AS rexify

COPY rexify/features rexify/features
COPY rexify/models rexify/models
COPY rexify/utils.py rexify/utils.py
COPY rexify/__init__.py rexify/__init__.py
COPY setup.cfg .
COPY setup.py .

RUN pip install .

FROM rexify AS load

COPY rexify/components/load/task.py ./load.py

ENTRYPOINT ["python", "-m", "load.py"]

FROM rexify AS tf

RUN pip install tensorflow==2.6.0 tensorflow_recommenders==0.6.0

FROM tf AS train

COPY rexify/components/train/task.py ./train.py

ENTRYPOINT ["python", "-m", "train.py"]


FROM tf AS index

RUN pip install scann==1.2.3

COPY rexify/components/index/task.py ./index.py

ENTRYPOINT ['python', '-m', 'index.py']