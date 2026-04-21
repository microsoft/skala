#!/usr/bin/env bash
# run this file from repo root

# exit if any command fails
set -e 
set -o pipefail

FILE=".devcontainer/skala_dev.Dockerfile"
ENV_VARIANT="${1:-cpu}"
BASE_NAME="skala-dev-${ENV_VARIANT}"
TAG=${BASE_NAME}":"$(date +"%Y%m%dT%H%M%S")

echo "Building Docker image with tag \"${TAG}\" (ENV_VARIANT=${ENV_VARIANT})" 

# To ignore the cache, use --no-cache
docker build \
    --progress=plain \
    --build-arg ENV_VARIANT="${ENV_VARIANT}" \
    --tag=${TAG} \
    --file=${FILE} \
    . \
    2>&1 | tee -a "build_${TAG}.log"

docker tag ${TAG} ${BASE_NAME}":latest"