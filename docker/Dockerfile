# From SeafloorMapping
# Base the build off of recent stable GDAL image
FROM osgeo/gdal:ubuntu-small-3.3.0 as python

# Python build stage
FROM python as python-build-stage

# Install apt packages
RUN apt-get update && apt-get install --no-install-recommends -y \
  # dependencies for building Python packages
  build-essential \
  python3-dev \
  python3-pip


# Python 'run' stage
FROM python as python-run-stage

ARG APP_HOME=/app

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

WORKDIR ${APP_HOME}

# Install required system dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
  # Translations dependencies
  python3-pip \
  # Install here rather than from .devcontainer/devcontainer.json
  git less wget zsh ssh-client \
  # cleaning up unused files
  && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
  && rm -rf /var/lib/apt/lists/*

# All absolute dir copies ignore workdir instruction. All relative dir copies are wrt to the workdir instruction
COPY --from=python-build-stage /usr/lib/libgdal* /usr/lib/
RUN ldconfig

# Copy poetry requirements and install  
COPY ./poetry.lock ./pyproject.toml .

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
RUN poetry install --no-dev


# copy application code to WORKDIR
COPY ./src/data /src/data
COPY . ${APP_HOME}

ENTRYPOINT [ "poetry", "shell" ]


