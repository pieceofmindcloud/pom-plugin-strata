# Builds the Strata runtime this plugin runs on a POM node.
#
# The base is the image the POM itself is compiled in (`build-base` of the POM
# Dockerfile): the same CUDA toolkit and the same Ubuntu, so the engine links
# the same static CUDA libraries the POM does, and the Python packages are
# built for the Python of the POM node image (Ubuntu 24.04, Python 3.12).
ARG CUDA_VERSION=13.4.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      git \
      ninja-build \
      python3 \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY docker/build-runtime-in-container.sh /usr/local/bin/build-runtime
ENTRYPOINT ["/usr/local/bin/build-runtime"]
