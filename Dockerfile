FROM python:3.8.6 AS base

WORKDIR /usr/local/rexify

RUN pip install rexify

FROM base AS load

COPY rexify/components/load/task.py ./load.py

ENTRYPOINT ["python", "-m", "load.py"]

FROM base AS train

COPY rexify/components/train/task.py ./train.py

ENTRYPOINT ["python", "-m", "train.py"]

FROM base AS index

COPY rexify/components/index/task.py ./index.py

ENTRYPOINT ['python', '-m', 'index.py']

FROM base AS retrieval

COPY rexify/components/retrieval/task.py ./retrieval.py

ENTRYPOINT ["python", '-m', 'retrieval.py']


FROM base AS demo

RUN pip install streamlit==1.11.1

COPY demo demo

ENTRYPOINT ["streamlit", "run", "demo/app.py"]
