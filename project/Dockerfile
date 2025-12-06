ARG PYTHON_VERSION=3.11
FROM --platform=$BUILDPLATFORM python:${PYTHON_VERSION}-slim

ARG PUID=1000
ARG PGID=1000
ENV DEBIAN_FRONTEND=noninteractive

# basic deps for building python packages & a non-root user
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        openssh-client \
        && rm -rf /var/lib/apt/lists/*

# create group/user to match host uid/gid (optional)
RUN groupadd -g ${PGID} devgroup || true
RUN useradd -m -u ${PUID} -g ${PGID} -s /bin/bash devuser || true

# create workspace
ENV WORKSPACE_DIR=/workspace/project
RUN mkdir -p ${WORKSPACE_DIR} && chown devuser:devgroup ${WORKSPACE_DIR}

# switch to non-root user by default
USER devuser
WORKDIR ${WORKSPACE_DIR}

# create a virtualenv path and ensure venv dir is writable
ENV VIRTUAL_ENV=/workspace/.venv
RUN python -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# copy only requirements for cache efficiency (if you have requirements.txt)
# If you do not have requirements, omit these COPY/INSTALL lines
COPY --chown=devuser:devgroup requirements.txt ${WORKSPACE_DIR}/requirements.txt
RUN if [ -f requirements.txt ]; then pip install --upgrade pip && pip install -r requirements.txt; fi

# default entrypoint is to keep container running (compose overrides)
CMD ["bash"]
