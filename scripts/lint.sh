#!/usr/bin/env bash

set -e
set -x

flake8 rexify tests
black rexify tests --check
isort rexify tests scripts --check-only