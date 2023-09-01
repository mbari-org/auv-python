# Adapted from from https://github.com/MBARIMike/ur_l/python_docker/Dockerfile
FROM python:3.10.9 as python-base
# Set DOCKER_USER_ID in shell, e.g. export DOCKER_USER_ID=$(id -u)
ARG DOCKER_USER_ID
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
ENV POETRY_VERSION=1.2.2
RUN curl -sSL https://install.python-poetry.org | python3 -

# We copy our Python requirements here to cache them
# and install only runtime deps using poetry

WORKDIR $PYSETUP_PATH
COPY ./poetry.lock ./pyproject.toml ./
RUN poetry self update $POETRY_VERSION && poetry install --only main

# 'production' stage uses the clean 'python-base' stage and copies
# in only our runtime deps that were installed in the 'builder-base'
FROM python-base as production
ARG DOCKER_USER_ID
ENV TARGET_ENV=production

RUN adduser docker_user --uid $DOCKER_USER_ID --gid 100 --disabled-password --gecos ""

COPY --from=builder-base $VENV_PATH $VENV_PATH

# Need git & rsync in final production image for use by src/data/process_i2map.py
RUN apt-get update && apt-get install -y git rsync

# Make sure the uid used in production (docker_user) can write to /data
RUN mkdir /data && chown docker_user:100 /data
USER docker_user

COPY ./src /src
COPY ./.git /.git

# So that repo.head.object.hexsha can be used to record the commit hash
RUN git config --global add.safe.directory /

# Make WORKDIR / so that relative paths (e.g. src/data/process_i2map.py) work
WORKDIR /
