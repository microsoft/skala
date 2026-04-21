# syntax=docker/dockerfile:1
FROM ubuntu:noble

RUN apt-get update --quiet \
    && apt-get install --yes --quiet --no-install-recommends \
    wget \
    ca-certificates \
    && apt-get clean --yes \
    && rm -rf /var/lib/apt/lists/*

SHELL [ "/bin/bash", "-c" ]

# mamba installation (mamba>2.0)
ENV MAMBA_DIR=/opt/miniforge3
ENV MAMBA_ROOT_PREFIX=${MAMBA_DIR}
ENV PATH=${PATH}:${MAMBA_DIR}/bin

RUN wget --no-hsts --quiet --output-document=miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash miniforge.sh -b -p ${MAMBA_DIR} \
    && rm miniforge.sh \
    # Remove python3.1 symlink if it exists, as it causes issues with conda
    # https://github.com/conda/conda/issues/11423
    && (test -L ${MAMBA_DIR}/bin/python3.1 && unlink ${MAMBA_DIR}/bin/python3.1 || true) \
    && ${MAMBA_DIR}/bin/mamba update -n base --all -y \
    && mamba clean --all --yes \
    # Mamba initialization script
    && echo "eval $(mamba shell hook --shell bash)" >> /etc/profile.d/source_mamba.sh \
    # for interactive shells:
    && echo "source /etc/profile.d/source_mamba.sh" >> /etc/bash.bashrc

# for non-interactive, not login shells:
# https://www.solipsys.co.uk/images/BashStartupFiles1.png
ENV BASH_ENV="/etc/profile.d/source_mamba.sh"

# create environment (ENV_VARIANT: "cpu" or "gpu")
ARG ENV_VARIANT=cpu
COPY ./environment-${ENV_VARIANT}.yml ./environment.yml
# CONDA_OVERRIDE_CUDA lets mamba solve CUDA deps without a GPU present during build
# Cache the mamba package downloads so rebuilds after environment.yml changes are faster.
# The cache mount is not part of the image layer, so mamba clean is unnecessary.
RUN --mount=type=cache,target=${MAMBA_DIR}/pkgs \
    CONDA_OVERRIDE_CUDA="12.0" mamba env create --file environment.yml \
    && rm environment.yml