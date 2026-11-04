#!/usr/bin/env bash
# The keys that decide whether a runtime, or its engine, has to be built again.
#
#   scripts/runtime-key.sh <cuda|hip|metal|cuda-windows> [strata commit]
#
# Prints:
#   strata_commit=  the Strata commit the build takes (the head of the locked branch, or the one given)
#   engine_key=     what the compiled engine depends on: the Strata and llama.cpp commits, the backend, its
#                   architectures and the files that compile it. Same key, same engine: the runtime workflow
#                   restores it from the Actions cache instead of compiling it (scripts/build-runtime.sh).
#   runtime_key=    the engine key plus what is assembled around it (the bridge, the app's Node.js and harness
#                   pins, the Python requirements come with the Strata commit). Same key as the runtime already in
#                   the strata-runtime release: nothing to build at all.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck disable=SC1091
source "$root/strata.lock"
backend="${1:?usage: scripts/runtime-key.sh <cuda|hip|metal|cuda-windows> [strata commit]}"
commit="${2:-${STRATA_COMMIT:-}}"
[[ -n "$commit" ]] || commit="$(git ls-remote "$STRATA_REPOSITORY" "refs/heads/$STRATA_BRANCH" | cut -f1)"
[[ -n "$commit" ]] || { echo "runtime-key: branch $STRATA_BRANCH not found in $STRATA_REPOSITORY" >&2; exit 2; }

case "$backend" in
  cuda) files=(docker/runtime.Dockerfile docker/build-runtime-in-container.sh cmake/pom-static-cuda.cmake)
        archs="${CUDA_ARCHITECTURES:-86;120} ${CUDA_VERSION:-13.4.1}" ;;
  hip) files=(docker/runtime-hip.Dockerfile docker/build-runtime-hip-in-container.sh)
       archs="${HIP_ARCHITECTURES:-gfx1101} ${ROCM_VERSION:-7.2.4}" ;;
  metal) files=(scripts/build-runtime-macos.sh); archs="apple" ;;
  cuda-windows) files=(scripts/build-runtime-windows.sh); archs="${CUDA_ARCHITECTURES:-86;120}" ;;
  *) echo "runtime-key: unknown backend $backend" >&2; exit 2 ;;
esac

hash() { "$root/scripts/sha256.sh" /dev/stdin; }
engine_key="$(
  {
    printf '%s\n' "$backend" "$commit" "$LLAMA_CPP_COMMIT" "$archs"
    for file in "${files[@]}"; do printf '%s\n' "$file"; cat "$root/$file"; done
  } | hash
)"
runtime_key="$(
  {
    printf '%s\n' "$engine_key"
    for file in runtime/pom_strata_server.py runtime/pom_sessions.py scripts/build-app.sh scripts/build-runtime.sh strata.lock; do
      printf '%s\n' "$file"; cat "$root/$file"
    done
  } | hash
)"
printf 'strata_commit=%s\nengine_key=%s\nruntime_key=%s\n' "$commit" "${engine_key:0:24}" "${runtime_key:0:24}"
