#!/usr/bin/env bash
# Builds the Strata runtime (engine, server, pack tools and their Python
# packages) for Linux x86_64 in the POM's CUDA build image, into
# build/runtime/ - or, with --hip, for AMD cards in ROCm's image, into
# build/runtime-hip/ with the ROCm libraries the engine needs. With --archive
# it also writes the release asset dist-release/strata-runtime[-hip]-linux-x86_64.tar.gz.
#
# The architectures default to the development machines' cards (CUDA 86;120,
# HIP gfx1101); a release passes CUDA_ARCHITECTURES and HIP_ARCHITECTURES with
# the families it supports (.github/workflows/release.yml).
#
# A development node reads build/runtime/ straight from the mounted source
# (see docs/README.md), so nothing has to be published to try a change.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck disable=SC1091
source "$root/strata.lock"
archs="${CUDA_ARCHITECTURES:-86;120}"
hip_archs="${HIP_ARCHITECTURES:-gfx1101}"
cuda_version="${CUDA_VERSION:-13.4.1}"
rocm_version="${ROCM_VERSION:-7.2.4}"
hip=0
native=0
strata_source="${STRATA_SOURCE_DIR:-}"
archive=0
worktree=0
version="${RUNTIME_VERSION:-}"

usage() {
  printf '%s\n' \
    'Usage: scripts/build-runtime.sh [--archive] [--worktree] [--version <semver>]' \
    '  --worktree builds STRATA_SOURCE_DIR as it is on disk (development; incremental, not for a release)' \
    '  --hip      the AMD runtime (ROCm image, build/runtime-hip) instead of the CUDA one' \
    '  --native   with --hip: build on this host with ROCM_PATH (e.g. AMD'"'"'s TheRock wheels), no image' \
    'Environment: CUDA_ARCHITECTURES (default "86;120", the POM default),' \
    '  HIP_ARCHITECTURES (default "gfx1101"), ROCM_VERSION (default 7.2.4),' \
    '  CUDA_VERSION (default 13.4.1, the POM default),' \
    '  STRATA_SOURCE_DIR (a local Strata git checkout to take the locked commit from).'
}

while (($# > 0)); do
  case "$1" in
    --archive) archive=1; shift ;;
    --worktree) worktree=1; shift ;;
    --hip) hip=1; shift ;;
    --native) native=1; shift ;;
    --version) version="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'build-runtime: unknown option %s\n' "$1" >&2; exit 2 ;;
  esac
done

work="$root/build/runtime-src"
((hip == 0)) || work="$root/build/runtime-hip-src"
if ((worktree)); then
  [[ -n "$strata_source" && -f "$strata_source/third_party/llama.cpp/ggml/CMakeLists.txt" ]] \
    || { echo 'build-runtime: --worktree needs STRATA_SOURCE_DIR with third_party/llama.cpp' >&2; exit 2; }
  ((archive == 0)) || { echo 'build-runtime: --worktree builds are not releases (no --archive)' >&2; exit 2; }
  work="$strata_source"
  STRATA_COMMIT="$(git -C "$strata_source" rev-parse HEAD)-dirty"
else
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

fi

mkdir -p "$root/build/cache"
if ((hip && native)); then
  [[ -n "${ROCM_PATH:-}" && -x "$ROCM_PATH/llvm/bin/clang++" ]] \
    || { echo 'build-runtime: --native needs ROCM_PATH with llvm/bin/clang++ (a ROCm with the HIP development files)' >&2; exit 2; }
  SRC_DIR="$work" PLUGIN_DIR="$root" WORK_DIR="$root/build/cache" OUT_DIR="$root/build" \
    HIP_ARCHITECTURES="$hip_archs" STRATA_COMMIT="$STRATA_COMMIT" LLAMA_CPP_COMMIT="$LLAMA_CPP_COMMIT" \
    RUNTIME_VERSION="$version" "$root/docker/build-runtime-hip-in-container.sh"
  runtime_dir=runtime-hip
  asset_name=strata-runtime-hip-linux-x86_64.tar.gz
elif ((hip)); then
  image="pom-plugin-strata-runtime:rocm${rocm_version}"
  docker build --quiet --build-arg ROCM_VERSION="$rocm_version" \
    -f "$root/docker/runtime-hip.Dockerfile" -t "$image" "$root" >/dev/null
  runtime_dir=runtime-hip
  asset_name=strata-runtime-hip-linux-x86_64.tar.gz
  arch_env=(-e HIP_ARCHITECTURES="$hip_archs")
else
  image="pom-plugin-strata-runtime:cuda${cuda_version}"
  docker build --quiet --build-arg CUDA_VERSION="$cuda_version" \
    -f "$root/docker/runtime.Dockerfile" -t "$image" "$root" >/dev/null
  runtime_dir=runtime
  asset_name=strata-runtime-linux-x86_64.tar.gz
  arch_env=(-e CUDA_ARCHITECTURES="$archs")
fi

((hip && native)) || docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  "${arch_env[@]}" \
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
  asset="$root/dist-release/$asset_name"
  # the archive unpacks as runtime/ whatever the backend (runtime::install)
  tar -C "$root/build" -czf "$asset" --transform "s,^${runtime_dir},runtime," "$runtime_dir"
  sha256="$(sha256sum "$asset" | awk '{print $1}')"
  size="$(wc -c < "$asset" | tr -d '[:space:]')"
  printf 'runtime_asset=%s\nruntime_sha256=%s\nruntime_size=%s\n' "$asset" "$sha256" "$size"
fi
