FROM python:3.8-slim-buster AS preprocess

RUN pip install scikit-learn==1.0.2

WORKDIR src

COPY rexify/components/preprocess/task.py ./preprocess.py

ENTRYPOINT ["python", "-m", "preprocess.py"]

FROM python:3.8-slim-buster AS train

COPY config.ini .
COPY setup.py .
COPY setup.cfg .
COPY rexify rexify

RUN pip install -e .

WORKDIR src

COPY rexify/components/train/task.py ./train.py

ENTRYPOINT ["python", "-m", "train.py"]
