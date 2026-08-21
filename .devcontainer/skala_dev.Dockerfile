# syntax=docker/dockerfile:1
FROM ubuntu:noble

RUN apt-get update --quiet \
    && apt-get install --yes --quiet --no-install-recommends \
    ca-certificates \
    wget \
    && apt-get clean --yes \
    && rm -rf /var/lib/apt/lists/*

SHELL [ "/bin/bash", "-c" ]

ARG PIXI_VERSION=v0.75.0
RUN wget --no-hsts --quiet \
    --output-document=/tmp/pixi.tar.gz \
    "https://github.com/prefix-dev/pixi/releases/download/${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz" \
    && tar --extract --gzip --file=/tmp/pixi.tar.gz --directory=/usr/local/bin \
    && rm /tmp/pixi.tar.gz \
    && pixi --version