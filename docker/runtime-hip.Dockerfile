# Builds the Strata HIP runtime (AMD cards) this plugin runs on a POM node.
#
# ROCm's own development image: the HIP compiler, hipBLAS and hipBLASLt the
# engine links. A POM node has no ROCm, so the build copies the libraries the
# engine needs into the runtime (docker/build-runtime-hip-in-container.sh).
# Ubuntu 24.04, the POM node image's: the Python packages target its 3.12.
ARG ROCM_VERSION=7.2.4
FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete

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
      patchelf \
      python3 \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY docker/build-runtime-hip-in-container.sh /usr/local/bin/build-runtime
ENTRYPOINT ["/usr/local/bin/build-runtime"]
