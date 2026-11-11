# Builds the Strata runtime this plugin runs on a POM node.
#
# The base is the image the POM itself is compiled in (`build-base` of the POM
# Dockerfile): the same CUDA toolkit and the same Ubuntu, so the engine links
# the same static CUDA libraries the POM does, and the Python packages are
# built for the Python of the POM node image (Ubuntu 24.04, Python 3.12).
ARG CUDA_VERSION=13.4.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
# A CI runner's mirror sometimes drops a download (exit 100 with no other change): apt retries each file, and a
# failed update is tried again once before the build gives up.
# UBUNTU_MIRROR (build-runtime.sh passes it when the CI runner cannot resolve archive.ubuntu.com): the Ubuntu
# archive and security sources point there instead.
ARG UBUNTU_MIRROR=
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries \
    && if [ -n "$UBUNTU_MIRROR" ]; then \
         sed -i -e "s|http://archive.ubuntu.com/ubuntu|$UBUNTU_MIRROR|g" -e "s|http://security.ubuntu.com/ubuntu|$UBUNTU_MIRROR|g" \
           /etc/apt/sources.list.d/*.sources /etc/apt/sources.list 2>/dev/null || true; \
       fi
RUN (apt-get update || (sleep 30 && apt-get update)) && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      ccache \
      cmake \
      git \
      ninja-build \
      python3 \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY docker/build-runtime-in-container.sh /usr/local/bin/build-runtime
ENTRYPOINT ["/usr/local/bin/build-runtime"]
