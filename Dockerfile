ARG python="3.10"
ARG filesystem="gcs"

FROM python:${python} AS base

RUN if [[ $(uname -m) != *arm* ]]; then pip install scann; fi

RUN pip install pandas numpy scikit-learn fsspec rexify

FROM base AS fs-s3

RUN pip install s3fs

FROM base AS fs-gcs

RUN pip install gcsfs

FROM fs-${filesystem} AS final
