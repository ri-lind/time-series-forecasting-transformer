# Use a slim Debian Bookworm image as base
FROM debian:bookworm-slim

# Install required packages for building Python and for pyenv/poetry
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Set Poetry configuration environment variables.
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_HOME=/usr/local \
    POETRY_CACHE_DIR=/tmp/pypoetry \
    POETRY_VERSION=2.0.1

# Install pyenv using the official installer script.
RUN curl -fsSL https://pyenv.run | bash

# Install Poetry via its official installer.
RUN curl -fsSL https://install.python-poetry.org | python3 -

# Set environment variables for pyenv and Poetry.
ENV PYENV_ROOT=/root/.pyenv
ENV PATH=/root/.pyenv/shims:/root/.pyenv/bin:/root/.local/bin:$PATH

# Create a profile script so that interactive shells automatically load pyenv.
RUN mkdir -p /etc/profile.d && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /etc/profile.d/pyenv.sh && \
    echo 'export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"' >> /etc/profile.d/pyenv.sh && \
    echo 'eval "$(pyenv init - bash)"' >> /etc/profile.d/pyenv.sh && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> /etc/profile.d/pyenv.sh && \
    chmod +x /etc/profile.d/pyenv.sh

# Install additional Python versions via pyenv.
# Here we install Python 3.10.16 and Python 3.13.0.
RUN pyenv install 3.10.16 && \
    pyenv install 3.13.0 && \
    pyenv rehash

# Set up the project working directory.
WORKDIR /code

# Copy Poetry configuration files first to leverage Docker caching.
# Ensure these files are in your build context.
COPY poetry.lock pyproject.toml /code/

# Set the local Python version and install project dependencies with Poetry.
# The extra group "torch" is installed here; adjust as needed.
RUN pyenv local 3.13.0 && \
    poetry lock && \
    poetry install -E torch --no-interaction --no-ansi --no-root


# Default command: launch an interactive bash shell.
EXPOSE 8888
CMD ["/bin/bash", "-c", "source /etc/profile.d/pyenv.sh && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''"]
