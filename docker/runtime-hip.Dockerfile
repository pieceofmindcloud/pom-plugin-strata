# Builds the Strata HIP runtime (AMD cards) this plugin runs on a POM node.
#
# ROCm's own development image: the HIP compiler, hipBLAS and hipBLASLt the
# engine links. A POM node has no ROCm, so the build copies the libraries the
# engine needs into the runtime (docker/build-runtime-hip-in-container.sh).
# Ubuntu 24.04, the POM node image's: the Python packages target its 3.12.
ARG ROCM_VERSION=7.2.4
FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      git \
      ninja-build \
      patchelf \
      python3 \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY docker/build-runtime-hip-in-container.sh /usr/local/bin/build-runtime
ENTRYPOINT ["/usr/local/bin/build-runtime"]
