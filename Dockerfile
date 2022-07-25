FROM python:3.8.6 AS base

WORKDIR /usr/local/rexify

COPY requirements.txt .

RUN pip install -r requirements.txt

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

FROM rexify AS train

COPY rexify/components/train/task.py ./train.py

ENTRYPOINT ["python", "-m", "train.py"]

FROM rexify AS index

COPY rexify/components/index/task.py ./index.py

ENTRYPOINT ['python', '-m', 'index.py']

FROM rexify AS retrieval

COPY rexify/components/retrieval/task.py ./retrieval.py

ENTRYPOINT ["python", '-m', 'retrieval.py']