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

# Install pyenv using the official installer script
RUN curl -fsSL https://pyenv.run | bash

# Install Poetry via its official installer
RUN curl -fsSL https://install.python-poetry.org | python3 -

# Set environment variables for pyenv and Poetry.
# (In this container the default user is root.)
ENV PYENV_ROOT=/root/.pyenv
ENV PATH=/root/.pyenv/bin:/root/.local/bin:$PATH

# Create a profile script so that interactive shells automatically load pyenv
RUN mkdir -p /etc/profile.d && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /etc/profile.d/pyenv.sh && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> /etc/profile.d/pyenv.sh && \
    echo 'eval "$(pyenv init - bash)"' >> /etc/profile.d/pyenv.sh && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> /etc/profile.d/pyenv.sh && \
    chmod +x /etc/profile.d/pyenv.sh

# Install additional Python versions via pyenv.
# Here we install Python 3.10.16 and Python 3.14.0.
RUN pyenv install 3.10.16 && \
    pyenv install 3.14.0 && \
    pyenv rehash

# Optionally, set a global or local version:
# For example, to set 3.14.0 as the global Python:
RUN pyenv global 3.14.0

# Default command: launch an interactive bash shell.
CMD ["/bin/bash"]
