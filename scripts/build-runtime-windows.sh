#!/usr/bin/env bash
# The Strata runtime for Windows x86_64 (CUDA), built natively under Git Bash.
# Called by scripts/build-runtime.sh, which prepares the source and adds the app.
#
# Inputs (environment): SRC_DIR, OUT_DIR, CACHE_DIR, STRATA_COMMIT,
# LLAMA_CPP_COMMIT, RUNTIME_VERSION, ENGINE_KEY (optional) as on macOS, and
# CUDA_ARCHITECTURES (default "86;120"). Needs the MSVC environment (vcvars64,
# or ilammy/msvc-dev-cmd in CI), the CUDA toolkit in CUDA_PATH, cmake and ninja.
#
# Layout, the Linux one with Windows names:
#   engine/strata.exe, engine/strata-dsv41.exe, engine/strata-vision.exe and
#   the DLLs they load, beside them: Strata's own and cuBLAS (Windows has no
#   static cuBLAS; cudart is static), so a node needs only the NVIDIA driver
#   python/                     a standalone CPython 3.12 (python/python.exe)
#   pylib/, strata/, pom_*.py   as on Linux
set -euo pipefail

src="${SRC_DIR:?}"
out="${OUT_DIR:?}"
cache="${CACHE_DIR:?}"
plugin="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
archs="${CUDA_ARCHITECTURES:-86;120}"
jobs="${BUILD_JOBS:-$NUMBER_OF_PROCESSORS}"
[[ -n "${CUDA_PATH:-}" && -f "$CUDA_PATH/bin/nvcc.exe" ]] || { echo 'build-runtime-windows: CUDA_PATH has no bin/nvcc.exe' >&2; exit 2; }
command -v cl.exe >/dev/null 2>&1 || { echo 'build-runtime-windows: no cl.exe: run it in the MSVC environment' >&2; exit 2; }

# Strata's pin (packaging/package_terminal.py)
python_url="https://github.com/astral-sh/python-build-standalone/releases/download/20261009/cpython-3.12.15%2B20261009-x86_64-pc-windows-msvc-install_only.tar.gz"
python_sha256="194c8f0d9e0de6f1c4576e5dfa67819a48c764be87608b74b8ad43387a57b93b"

fetch() { # url sha256 file
  [[ -f "$3" && "$(sha256sum "$3" | awk '{print $1}')" == "$2" ]] && return 0
  curl -fsSL -o "$3.part" "$1"
  [[ "$(sha256sum "$3.part" | awk '{print $1}')" == "$2" ]] || { echo "build-runtime-windows: $1 does not match its pin" >&2; exit 1; }
  mv "$3.part" "$3"
}
# CMake wants C:/... paths, not Git Bash's /c/...
win() { cygpath -m "$1"; }

launcher=()
if command -v ccache >/dev/null 2>&1; then
  # nvcc stays outside ccache here: only the host C/C++ goes through it
  export CCACHE_DIR="${CCACHE_DIR:-$(win "$cache/ccache")}" CCACHE_BASEDIR="$(win "$src")" CCACHE_NOHASHDIR=1 \
    CCACHE_COMPILERCHECK=content CCACHE_SLOPPINESS=include_file_mtime,include_file_ctime,time_macros,pch_defines
  launcher=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
fi

mkdir -p "$cache/downloads"
rm -rf "$out.tmp"
mkdir -p "$out.tmp/engine" "$out.tmp/strata"

engine_cache="$cache/engine/cuda-windows"
if [[ -n "${ENGINE_KEY:-}" && -f "$engine_cache/ENGINE.key" && "$(cat "$engine_cache/ENGINE.key")" == "$ENGINE_KEY" ]]; then
  echo "== engine: reused ($ENGINE_KEY)"
  cp -a "$engine_cache/engine/." "$out.tmp/engine/"
else
  echo "== engine: CUDA $("$CUDA_PATH/bin/nvcc.exe" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'), archs $archs"
  build="$cache/build"
  cmake -G Ninja "${launcher[@]}" -S "$(win "$src")" -B "$(win "$build")" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_CUDA=ON \
    -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_PORTABLE=ON \
    -DSTRATA_Q6K_EXPERTS=ON \
    -DSTRATA_MMQ_KQUANTS=ON \
    -DSTRATA_BUILD_DSV41=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$archs" \
    -DCMAKE_CUDA_RUNTIME_LIBRARY=Static \
    -DCMAKE_CUDA_COMPILER="$(win "$CUDA_PATH/bin/nvcc.exe")" \
    -DCUDAToolkit_ROOT="$(win "$CUDA_PATH")" \
    -DSTRATA_GGML_DIR="$(win "$src/third_party/llama.cpp")"
  cmake --build "$(win "$build")" --target strata strata-dsv41 -j "$jobs"
  cp "$build/strata.exe" "$build/strata-dsv41.exe" "$out.tmp/engine/"
  # Strata's own DLLs (strata_kq.dll and the like), then cuBLAS from the toolkit (bin/x64 from CUDA 13 on, bin before)
  find "$build" -maxdepth 1 -name '*.dll' -exec cp {} "$out.tmp/engine/" \;
  for dll in cublas64_ cublasLt64_; do
    found="$(find "$CUDA_PATH/bin" -maxdepth 2 -name "${dll}*.dll" | head -1)"
    [[ -n "$found" ]] || { echo "build-runtime-windows: no ${dll}*.dll in $CUDA_PATH/bin" >&2; exit 1; }
    cp "$found" "$out.tmp/engine/"
  done

  echo "== vision encoder (CPU)"
  cmake -G Ninja "${launcher[@]}" -S "$(win "$src/tools/vision")" -B "$(win "$cache/build-vision")" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_DIR="$(win "$src/third_party/llama.cpp")" \
    -DSTRATA_PORTABLE=ON \
    -DSTRATA_VISION_CUDA=OFF
  cmake --build "$(win "$cache/build-vision")" --target strata-vision -j "$jobs"
  cp "$cache/build-vision/bin/strata-vision.exe" "$out.tmp/engine/"
  find "$cache/build-vision/bin" -maxdepth 1 -name '*.dll' -exec cp {} "$out.tmp/engine/" \;

  # the engine must start: a missing DLL fails here, not on a user's node (no GPU needed for --version)
  "$out.tmp/engine/strata-vision.exe" --version >/dev/null
  if [[ -n "${ENGINE_KEY:-}" ]]; then
    rm -rf "$engine_cache" && mkdir -p "$engine_cache"
    cp -a "$out.tmp/engine" "$engine_cache/engine"
    printf '%s' "$ENGINE_KEY" > "$engine_cache/ENGINE.key"
  fi
fi

echo "== server and tools"
cp -a "$src/serve" "$src/tools" "$src/data" "$src/LICENSE" "$out.tmp/strata/"
for pkg in strata_control strata_cli; do [[ ! -d "$src/$pkg" ]] || cp -a "$src/$pkg" "$out.tmp/strata/"; done
mkdir -p "$out.tmp/strata/third_party/llama.cpp"
cp -a "$src/third_party/llama.cpp/gguf-py" "$out.tmp/strata/third_party/llama.cpp/"
find "$out.tmp/strata" -name '__pycache__' -prune -exec rm -rf {} +
find "$out.tmp/strata/serve" "$out.tmp/strata/strata_control" "$out.tmp/strata/strata_cli" -name 'test_*.py' -delete 2>/dev/null || true
cp "$plugin/runtime/pom_strata_server.py" "$plugin/runtime/pom_sessions.py" "$out.tmp/"

echo "== python: standalone CPython 3.12 and the packages"
fetch "$python_url" "$python_sha256" "$cache/downloads/cpython-windows.tar.gz"
tar -xzf "$cache/downloads/cpython-windows.tar.gz" -C "$out.tmp"
python="$out.tmp/python/python.exe"
grep -vE '^\s*(#|$)|^(cmake|ninja)==' "$src/requirements.txt" > "$cache/requirements.txt"
"$python" -m pip install --quiet --no-cache-dir --disable-pip-version-check \
  --only-binary=:all: --target "$(win "$out.tmp/pylib")" -r "$(win "$cache/requirements.txt")"

"$python" - "$(win "$out.tmp")" <<'PY'
import json, os, sys
meta = {
    "strata_commit": os.environ.get("STRATA_COMMIT", ""),
    "llama_cpp_commit": os.environ.get("LLAMA_CPP_COMMIT", ""),
    "backend": "cuda",
    "platform": "windows-x86_64",
    "cuda_architectures": os.environ.get("CUDA_ARCHITECTURES", "86;120").split(";"),
    "python": "%d.%d" % sys.version_info[:2],
    "runtime_version": os.environ.get("RUNTIME_VERSION", ""),
    "engine_key": os.environ.get("ENGINE_KEY", ""),
}
with open(os.path.join(sys.argv[1], "RUNTIME.json"), "w") as f:
    json.dump(meta, f, indent=1)
print(json.dumps(meta))
PY

rm -rf "$out"
mv "$out.tmp" "$out"
echo "== runtime ready: $out"
