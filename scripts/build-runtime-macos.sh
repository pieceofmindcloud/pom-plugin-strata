#!/usr/bin/env bash
# The Strata runtime for an Apple Silicon Mac (Metal), built natively. Called
# by scripts/build-runtime.sh, which prepares the source and adds the app.
#
# Inputs (environment): SRC_DIR (the Strata source at the locked commit, with
# third_party/llama.cpp), OUT_DIR (the runtime directory to write), CACHE_DIR
# (downloads, the compiler cache and the engine cache), STRATA_COMMIT,
# LLAMA_CPP_COMMIT, RUNTIME_VERSION, ENGINE_KEY (optional).
#
# Layout, the Linux one plus what a Mac lacks:
#   engine/strata, engine/strata-dsv41, engine/strata-vision
#   engine/metal/                the kernel sources the Metal engines compile when
#                                they start, found beside the executable
#                                (Strata's metal/src/resource_path.h)
#   python/                      a standalone CPython 3.12 (macOS ships 3.9)
#   pylib/, strata/, pom_*.py    as on Linux
# Needs the Command Line Tools, cmake and ninja.
set -euo pipefail

src="${SRC_DIR:?}"
out="${OUT_DIR:?}"
cache="${CACHE_DIR:?}"
plugin="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
jobs="${BUILD_JOBS:-$(sysctl -n hw.ncpu)}"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'build-runtime-macos: builds on an Apple Silicon Mac' >&2; exit 2; }

# Strata's pins (.github/workflows/metal-build.yml, packaging/package_terminal.py)
metal_cpp_url="https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip"
metal_cpp_sha256="4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4"
python_url="https://github.com/astral-sh/python-build-standalone/releases/download/20261009/cpython-3.12.15%2B20261009-aarch64-apple-darwin-install_only.tar.gz"
python_sha256="83dcb04777c798bd95f3687acf9ae84b09860272ee6bdfb05be8bcf9447f75b8"

fetch() { # url sha256 file
  [[ -f "$3" && "$(shasum -a 256 "$3" | awk '{print $1}')" == "$2" ]] && return 0
  curl -fsSL -o "$3.part" "$1"
  [[ "$(shasum -a 256 "$3.part" | awk '{print $1}')" == "$2" ]] || { echo "build-runtime-macos: $1 does not match its pin" >&2; exit 1; }
  mv "$3.part" "$3"
}

launcher=()
if command -v ccache >/dev/null 2>&1; then
  export CCACHE_DIR="${CCACHE_DIR:-$cache/ccache}" CCACHE_BASEDIR="$src" CCACHE_NOHASHDIR=1 CCACHE_COMPILERCHECK=content \
    CCACHE_SLOPPINESS=include_file_mtime,include_file_ctime,time_macros,pch_defines
  launcher=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_OBJCXX_COMPILER_LAUNCHER=ccache)
fi

mkdir -p "$cache/downloads"
rm -rf "$out.tmp"
mkdir -p "$out.tmp/engine" "$out.tmp/strata"

engine_cache="$cache/engine/metal"
if [[ -n "${ENGINE_KEY:-}" && -f "$engine_cache/ENGINE.key" && "$(cat "$engine_cache/ENGINE.key")" == "$ENGINE_KEY" ]]; then
  echo "== engine: reused ($ENGINE_KEY)"
  cp -a "$engine_cache/engine/." "$out.tmp/engine/"
else
  echo "== engine: Metal"
  fetch "$metal_cpp_url" "$metal_cpp_sha256" "$cache/downloads/metal-cpp.zip"
  rm -rf "$cache/metal-cpp" && mkdir -p "$cache/metal-cpp"
  unzip -q "$cache/downloads/metal-cpp.zip" -d "$cache/metal-cpp"
  build="$cache/build-metal"
  cmake -G Ninja "${launcher[@]}" -S "$src/metal" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMETAL_CPP_DIR="$cache/metal-cpp/metal-cpp" \
    -DSTRATA_GGML_DIR="$src/third_party/llama.cpp" \
    -DSTRATA_BUILD_ID="${STRATA_COMMIT:0:12}"
  cmake --build "$build" --target strata strata_ds4 -j "$jobs"
  install -m 0755 "$build/strata" "$out.tmp/engine/strata"
  install -m 0755 "$build/strata_ds4" "$out.tmp/engine/strata-dsv41"

  # The kernels both engines compile when they start, beside them (resource_path.h): the Qwen engine reads
  # metal/engine-msl, the DeepSeek one metal/src/engine/msl.
  metal="$out.tmp/engine/metal"
  mkdir -p "$metal/kernels" "$metal/engine-msl" "$metal/src/engine/msl" "$metal/data"
  cp "$src"/metal/kernels/*.metal "$metal/kernels/"
  cp "$src"/metal/src/engine/msl/*.metal "$metal/engine-msl/"
  cp "$src"/metal/src/engine/msl/*.metal "$metal/src/engine/msl/"
  cp "$src/third_party/ggml/ggml-common.h" "$metal/"
  cp "$src/data/expert-profile.bin" "$src/data/draft_vocab.bin" "$metal/data/"

  echo "== vision encoder (Metal built in, run on the CPU unless asked)"
  cmake -G Ninja "${launcher[@]}" -S "$src/tools/vision" -B "$cache/build-vision" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_DIR="$src/third_party/llama.cpp" \
    -DSTRATA_PORTABLE=ON \
    -DSTRATA_VISION_METAL=ON \
    -DSTRATA_VISION_CUDA=OFF
  cmake --build "$cache/build-vision" --target strata-vision -j "$jobs"
  install -m 0755 "$cache/build-vision/bin/strata-vision" "$out.tmp/engine/strata-vision"

  # only macOS's own libraries: a Mac running POM has nothing else to give
  for program in strata strata-dsv41 strata-vision; do
    if otool -L "$out.tmp/engine/$program" | tail -n +2 | awk '{print $1}' | grep -vE '^(/usr/lib/|/System/Library/)'; then
      echo "build-runtime-macos: $program links a library that is not part of macOS" >&2
      exit 1
    fi
  done
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
install -m 0644 "$plugin/runtime/pom_strata_server.py" "$out.tmp/pom_strata_server.py"
install -m 0644 "$plugin/runtime/pom_sessions.py" "$out.tmp/pom_sessions.py"

echo "== python: standalone CPython 3.12 and the packages"
fetch "$python_url" "$python_sha256" "$cache/downloads/cpython-macos.tar.gz"
tar -xzf "$cache/downloads/cpython-macos.tar.gz" -C "$out.tmp"
python="$out.tmp/python/bin/python3"
grep -vE '^\s*(#|$)|^(cmake|ninja)==' "$src/requirements.txt" > "$cache/requirements.txt"
"$python" -m pip install --quiet --no-cache-dir --disable-pip-version-check \
  --only-binary=:all: --target "$out.tmp/pylib" -r "$cache/requirements.txt"

"$python" - "$out.tmp" <<'PY'
import json, os, sys
meta = {
    "strata_commit": os.environ.get("STRATA_COMMIT", ""),
    "llama_cpp_commit": os.environ.get("LLAMA_CPP_COMMIT", ""),
    "backend": "metal",
    "platform": "macos-aarch64",
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
