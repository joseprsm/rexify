#!/bin/env/bash

cd docs
rm -rf _build
python -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html