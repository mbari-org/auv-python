# From ur_l/python_docker/Dockerfile
FROM python:3.8.1-slim as python-base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# builder-base is used to build dependencies
FROM python-base as builder-base
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        build-essential

# Freeze poetry version to avoid breaking changes
ENV POETRY_VERSION=1.1.6
RUN curl -sSL https://install.python-poetry.org | python3 -

# We copy our Python requirements here to cache them
# and install only runtime deps using poetry

WORKDIR $PYSETUP_PATH
COPY ./poetry.lock ./pyproject.toml ./
RUN poetry self update $POETRY_VERSION && poetry install --no-dev

# 'production' stage installs all dev deps and can be used to develop code.
# For example using docker-compose to mount local volume under /app
FROM python-base as production
ENV TARGET_ENV=production

# Copying poetry and venv into image
COPY --from=builder-base $POETRY_HOME $POETRY_HOME
COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH

# venv already has runtime deps installed we get a quicker install
WORKDIR $PYSETUP_PATH
#RUN poetry install
RUN poetry update

RUN mkdir /dev_app
WORKDIR /dev_app

# 'lint' stage was for running black, isort, safety
FROM production AS lint
COPY src ./src

# 'production' stage uses the clean 'python-base' stage and copies
# in only our runtime deps that were installed in the 'builder-base'
FROM python-base as production
ENV TARGET_ENV=production

COPY --from=builder-base $VENV_PATH $VENV_PATH

# Need rsync in final production image for use by src/data/process_i2map.py
RUN apt-get update && apt-get install -y rsync

COPY ./src /src
WORKDIR /src