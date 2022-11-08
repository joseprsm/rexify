ARG version=0.0.9

FROM python:3.10

RUN pip install rexify==$version
