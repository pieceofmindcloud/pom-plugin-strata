#!/usr/bin/env bash
# Builds the Strata runtime (engine, server, pack tools and their Python
# packages) for the platform it runs on, into build/runtime/:
#   Linux x86_64   in the POM's CUDA build image - or, with --hip, for AMD
#                  cards, into build/runtime-hip/ with the ROCm libraries the
#                  engine needs: on a development machine with the host ROCm
#                  of scripts/setup-host-rocm.sh, natively with it; in CI (or
#                  with --image) in ROCm's image, tens of GB;
#   macOS aarch64  natively, the Metal engine (scripts/build-runtime-macos.sh);
#   Windows x86_64 natively under Git Bash with MSVC and the CUDA toolkit
#                  (scripts/build-runtime-windows.sh).
# With --archive it also writes the release asset
# dist-release/strata-runtime[-hip]-<platform>.tar.gz.
#
# The engine is the slow part: with ENGINE_KEY set (scripts/runtime-key.sh) a
# compiled engine kept in build/cache/engine/<backend> under the same key is
# reused instead of compiled, so a runtime whose engine inputs did not change
# (only the bridge, the app, the Python packages) builds in minutes.
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
image_only=0
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
    '             (the default outside CI when scripts/setup-host-rocm.sh installed one)' \
    '  --image    with --hip: build in ROCm'"'"'s image even when the host has a ROCm (what CI does)' \
    'Environment: CUDA_ARCHITECTURES (default "86;120", the POM default),' \
    '  HIP_ARCHITECTURES (default "gfx1101"), ROCM_VERSION (default 7.2.4),' \
    '  CUDA_VERSION (default 13.4.1, the POM default),' \
    '  STRATA_SOURCE_DIR (a local Strata git checkout to take the branch head from).'
}

while (($# > 0)); do
  case "$1" in
    --archive) archive=1; shift ;;
    --worktree) worktree=1; shift ;;
    --hip) hip=1; shift ;;
    --native) native=1; shift ;;
    --image) image_only=1; shift ;;
    --version) version="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'build-runtime: unknown option %s\n' "$1" >&2; exit 2 ;;
  esac
done

# A development machine with the host ROCm of scripts/setup-host-rocm.sh
# builds the HIP runtime with it; CI keeps ROCm's image (ROCM_VERSION).
rocm_home="${STRATA_ROCM_HOME:-$HOME/.local/share/pom-strata-rocm}"
if ((hip && !native && !image_only)) && [[ "${GITHUB_ACTIONS:-}" != true && -x "$rocm_home/bin/rocm-sdk" ]]; then
  native=1
  echo "build-runtime: --hip with the host ROCm in $rocm_home (--image for ROCm's image)"
fi
if ((hip && native)) && [[ -x "$rocm_home/bin/rocm-sdk" ]]; then
  export ROCM_PATH="${ROCM_PATH:-$("$rocm_home/bin/rocm-sdk" path --root)}"
  # its ninja, patchelf and the Python (with pip) the runtime's packages are installed with
  export PATH="$rocm_home/bin:$PATH" PYTHON="${PYTHON:-$rocm_home/bin/python}"
fi

work="$root/build/runtime-src"
((hip == 0)) || work="$root/build/runtime-hip-src"
if ((worktree)); then
  [[ -n "$strata_source" && -f "$strata_source/third_party/llama.cpp/ggml/CMakeLists.txt" ]] \
    || { echo 'build-runtime: --worktree needs STRATA_SOURCE_DIR with third_party/llama.cpp' >&2; exit 2; }
  ((archive == 0)) || { echo 'build-runtime: --worktree builds are not releases (no --archive)' >&2; exit 2; }
  work="$strata_source"
  STRATA_COMMIT="$(git -C "$strata_source" rev-parse HEAD)-dirty"
else
# The head of the Strata branch now (or STRATA_COMMIT, which CI resolves once for the whole run): it is recorded in
# RUNTIME.json.
STRATA_COMMIT="${STRATA_COMMIT:-$(git ls-remote "$STRATA_REPOSITORY" "refs/heads/$STRATA_BRANCH" | cut -f1)}"
[[ -n "$STRATA_COMMIT" ]] || { echo "build-runtime: branch $STRATA_BRANCH not found in $STRATA_REPOSITORY" >&2; exit 2; }
echo "build-runtime: Strata $STRATA_BRANCH at $STRATA_COMMIT"
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
case "$(uname -s)" in
  Darwin) host_os=macos ;;
  MINGW*|MSYS*|CYGWIN*) host_os=windows ;;
  *) host_os=linux ;;
esac
if [[ "$host_os" != linux ]]; then
  ((hip == 0)) || { echo 'build-runtime: --hip builds on Linux only' >&2; exit 2; }
  runtime_dir=runtime
  if [[ "$host_os" == macos ]]; then
    asset_name=strata-runtime-macos-aarch64.tar.gz
  else
    asset_name=strata-runtime-windows-x86_64.tar.gz
  fi
  SRC_DIR="$work" OUT_DIR="$root/build/$runtime_dir" CACHE_DIR="$root/build/cache" \
    STRATA_COMMIT="$STRATA_COMMIT" LLAMA_CPP_COMMIT="$LLAMA_CPP_COMMIT" RUNTIME_VERSION="$version" \
    ENGINE_KEY="${ENGINE_KEY:-}" "$root/scripts/build-runtime-${host_os}.sh"
elif ((hip && native)); then
  [[ -n "${ROCM_PATH:-}" && -x "$ROCM_PATH/llvm/bin/clang++" ]] \
    || { echo 'build-runtime: --native needs ROCM_PATH with llvm/bin/clang++ (a ROCm with the HIP development files)' >&2; exit 2; }
  # its own work dir: the image's CMake cache records the container's paths (/src, /work)
  mkdir -p "$root/build/cache/native"
  SRC_DIR="$work" PLUGIN_DIR="$root" WORK_DIR="$root/build/cache/native" OUT_DIR="$root/build" \
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

[[ "$host_os" != linux ]] || ((hip && native)) || docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  "${arch_env[@]}" \
  -e STRATA_COMMIT="$STRATA_COMMIT" \
  -e LLAMA_CPP_COMMIT="$LLAMA_CPP_COMMIT" \
  -e RUNTIME_VERSION="$version" \
  -e ENGINE_KEY="${ENGINE_KEY:-}" \
  -e CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-4G}" \
  -v "$work:/src:ro" \
  -v "$root:/plugin:ro" \
  -v "$root/build/cache:/work" \
  -v "$root/build:/out" \
  "$image"

# Strata's app (web shell, host server and the Node.js it runs on), next to the engine and server
"$root/scripts/build-app.sh" "$work" "$root/build/$runtime_dir"

if ((archive)); then
  mkdir -p "$root/dist-release"
  asset="$root/dist-release/$asset_name"
  # the archive unpacks as runtime/ whatever the backend (runtime::install)
  # (macOS's tar has no --transform; its runtime is already runtime/)
  transform=()
  [[ "$runtime_dir" == runtime ]] || transform=(--transform "s,^${runtime_dir},runtime,")
  tar -C "$root/build" -czf "$asset" "${transform[@]}" "$runtime_dir"
  sha256="$("$root/scripts/sha256.sh" "$asset")"
  size="$(wc -c < "$asset" | tr -d '[:space:]')"
  printf 'runtime_asset=%s\nruntime_sha256=%s\nruntime_size=%s\n' "$asset" "$sha256" "$size"
fi
