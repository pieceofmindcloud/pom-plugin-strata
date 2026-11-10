#!/usr/bin/env bash
# Installs a ROCm with the HIP development files on this Linux host, from
# AMD's TheRock wheels, so `scripts/build-runtime.sh --hip` builds the AMD
# runtime natively instead of in ROCm's development image (tens of GB).
#
# It goes into a Python venv (default ~/.local/share/pom-strata-rocm, or
# STRATA_ROCM_HOME) together with the build tools the host may lack (ninja,
# patchelf). build-runtime.sh finds it there; CI keeps building in the image.
#
# The wheels carry one GPU family's libraries: ROCM_GPU_FAMILY (default
# gfx110X-all, RDNA3, the development machines' Radeon). HIP_ARCHITECTURES of
# a native build must stay within that family.
set -euo pipefail

home="${STRATA_ROCM_HOME:-$HOME/.local/share/pom-strata-rocm}"
family="${ROCM_GPU_FAMILY:-gfx110X-all}"
version="${ROCM_SDK_VERSION:-7.13.0}"
index="https://repo.amd.com/rocm/whl/$family/"

[[ "$(uname -s)" == Linux ]] || { echo 'setup-host-rocm: Linux only' >&2; exit 2; }
command -v uv >/dev/null || { echo 'setup-host-rocm: needs uv (https://docs.astral.sh/uv/)' >&2; exit 2; }

echo "setup-host-rocm: ROCm $version ($family) into $home"
[[ -x "$home/bin/python" ]] || uv venv --seed --python 3.12 "$home"
# ROCm from AMD's index alone (it answers 403 for what it does not have, which
# stops uv), the build tools from PyPI.
uv pip install --python "$home/bin/python" --index-url "$index" "rocm[devel,libraries]==$version"
uv pip install --python "$home/bin/python" ninja patchelf
# The devel wheel unpacks the SDK on first use.
"$home/bin/rocm-sdk" init >/dev/null
root="$("$home/bin/rocm-sdk" path --root)"
[[ -x "$root/llvm/bin/clang++" ]] || { echo "setup-host-rocm: no HIP compiler under $root" >&2; exit 1; }
echo "setup-host-rocm: ROCM_PATH=$root ($(du -sh "$home" | cut -f1))"
