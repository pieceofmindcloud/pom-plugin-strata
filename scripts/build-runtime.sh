#!/usr/bin/env bash
# Builds the Strata runtime (engine, server, pack tools and their Python
# packages) for Linux x86_64 in the POM's CUDA build image, into
# build/runtime/. With --archive it also writes the release asset
# dist-release/strata-runtime-linux-x86_64.tar.gz.
#
# A development node reads build/runtime/ straight from the mounted source
# (see docs/README.md), so nothing has to be published to try a change.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck disable=SC1091
source "$root/strata.lock"
archs="${CUDA_ARCHITECTURES:-86;120}"
cuda_version="${CUDA_VERSION:-13.4.1}"
strata_source="${STRATA_SOURCE_DIR:-}"
archive=0
version="${RUNTIME_VERSION:-}"

usage() {
  printf '%s\n' \
    'Usage: scripts/build-runtime.sh [--archive] [--version <semver>]' \
    'Environment: CUDA_ARCHITECTURES (default "86;120", the POM default),' \
    '  CUDA_VERSION (default 13.4.1, the POM default),' \
    '  STRATA_SOURCE_DIR (a local Strata git checkout to take the locked commit from).'
}

while (($# > 0)); do
  case "$1" in
    --archive) archive=1; shift ;;
    --version) version="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'build-runtime: unknown option %s\n' "$1" >&2; exit 2 ;;
  esac
done

work="$root/build/runtime-src"
rm -rf "$work"
mkdir -p "$work"

# The Strata source at the locked commit, never a working tree: untracked
# files of a local checkout do not reach the build.
if [[ -n "$strata_source" ]]; then
  git -C "$strata_source" cat-file -e "${STRATA_COMMIT}^{commit}"
  git -C "$strata_source" archive "$STRATA_COMMIT" | tar -x -C "$work"
else
  git clone --quiet --filter=blob:none "$STRATA_REPOSITORY" "$work/.clone"
  git -C "$work/.clone" archive "$STRATA_COMMIT" | tar -x -C "$work"
  rm -rf "$work/.clone"
fi

# llama.cpp at the commit Strata pins: a local copy of exactly that commit, or
# GitHub's archive of it.
llama="$work/third_party/llama.cpp"
local_llama="${strata_source:+$strata_source/third_party/llama.cpp}"
if [[ -n "$local_llama" && -f "$local_llama/ggml/CMakeLists.txt" ]] \
    && grep -q "$LLAMA_CPP_COMMIT" <(git -C "$strata_source" show "$STRATA_COMMIT:setup.py"); then
  mkdir -p "$llama"
  cp -a "$local_llama/." "$llama/"
else
  curl -fsSL "https://github.com/ggml-org/llama.cpp/archive/${LLAMA_CPP_COMMIT}.tar.gz" \
    | tar -xz -C "$work/third_party"
  mv "$work/third_party/llama.cpp-${LLAMA_CPP_COMMIT}" "$llama"
fi

image="pom-plugin-strata-runtime:cuda${cuda_version}"
docker build --quiet --build-arg CUDA_VERSION="$cuda_version" \
  -f "$root/docker/runtime.Dockerfile" -t "$image" "$root" >/dev/null

mkdir -p "$root/build/cache"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e CUDA_ARCHITECTURES="$archs" \
  -e STRATA_COMMIT="$STRATA_COMMIT" \
  -e LLAMA_CPP_COMMIT="$LLAMA_CPP_COMMIT" \
  -e RUNTIME_VERSION="$version" \
  -v "$work:/src:ro" \
  -v "$root:/plugin:ro" \
  -v "$root/build/cache:/work" \
  -v "$root/build:/out" \
  "$image"

if ((archive)); then
  mkdir -p "$root/dist-release"
  asset="$root/dist-release/strata-runtime-linux-x86_64.tar.gz"
  tar -C "$root/build" -czf "$asset" runtime
  sha256="$(sha256sum "$asset" | awk '{print $1}')"
  size="$(wc -c < "$asset" | tr -d '[:space:]')"
  printf 'runtime_asset=%s\nruntime_sha256=%s\nruntime_size=%s\n' "$asset" "$sha256" "$size"
fi
