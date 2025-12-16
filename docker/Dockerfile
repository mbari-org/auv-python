# Adapted from from https://github.com/MBARIMike/ur_l/python_docker/Dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS python-base

# Set DOCKER_USER_ID in shell, e.g. export DOCKER_USER_ID=$(id -u)
ARG DOCKER_USER_ID
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

# Install system dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        build-essential \
        git \
        rsync \
        # GMT6 dependencies
        gmt \
        libgmt-dev \
        gmt-dcw \
        gmt-gshhg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR $PYSETUP_PATH

# Copy dependency files first for better cache
COPY ./pyproject.toml ./uv.lock ./

# Install Python dependencies using uv (from pyproject.toml)
RUN uv pip install --system .

# Create user and set permissions
RUN adduser --disabled-password --gecos "" --uid $DOCKER_USER_ID docker_user \
    && mkdir /data && chown docker_user:100 /data

USER docker_user

# Copy source code and .git for commit hash tracking
COPY ./src /src
COPY ./.git /.git

# So that repo.head.object.hexsha can be used to record the commit hash
RUN git config --global --add safe.directory /

# Set working directory
WORKDIR /
