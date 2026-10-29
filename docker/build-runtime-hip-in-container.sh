#!/usr/bin/env bash
# Runs inside docker/runtime-hip.Dockerfile. Inputs: /src (the Strata source at
# the locked commit, with third_party/llama.cpp), /plugin (this repository, read
# only). Output: /out/runtime-hip, the directory the plugin runs Strata from on
# AMD cards: the engine, its ROCm libraries in lib/ (the node has none), the
# server, the tools and their Python packages.
#
# `scripts/build-runtime.sh --hip --native` runs it on the host instead, with
# SRC_DIR, PLUGIN_DIR, WORK_DIR, OUT_DIR and ROCM_PATH (a ROCm with the HIP
# development files, as AMD's TheRock wheels install).
set -euo pipefail
src="${SRC_DIR:-/src}"
plugin="${PLUGIN_DIR:-/plugin}"
work="${WORK_DIR:-/work}"

archs="${HIP_ARCHITECTURES:?HIP_ARCHITECTURES is required}"
jobs="${BUILD_JOBS:-$(nproc)}"
rocm="${ROCM_PATH:-/opt/rocm}"
build="$work/build-hip"
out="${OUT_DIR:-/out}/runtime-hip"

# ccache, kept in the mounted work cache (CI restores it between runs): a rebuild compiles only what changed.
# The source is unpacked fresh every build, so files are compared by content, not by time.
launcher=()
if command -v ccache >/dev/null 2>&1; then
  export CCACHE_DIR="${CCACHE_DIR:-$work/ccache}" CCACHE_BASEDIR="$src" CCACHE_NOHASHDIR=1 CCACHE_COMPILERCHECK=content \
    CCACHE_SLOPPINESS=include_file_mtime,include_file_ctime,time_macros,pch_defines CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-4G}"
  launcher=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
            -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache)
  ccache --zero-stats >/dev/null
fi

rm -rf "$out.tmp"
mkdir -p "$out.tmp/engine" "$out.tmp/strata" "$out.tmp/lib" "$build"

# what ROCm's CMake packages read from the environment (a system ROCm guesses them, a TheRock wheel does not)
export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr ROCM_PATH="$rocm" HIP_PATH="$rocm"
# the device libraries of the ROCm the compiler belongs to (a TheRock wheel keeps them under lib/llvm)
bitcode="$rocm/amdgcn/bitcode"
[[ -d "$bitcode" ]] || bitcode="$rocm/lib/llvm/amdgcn/bitcode"

echo "== engine: ROCm $(cat "$rocm/.info/version" 2>/dev/null || echo '?'), archs $archs"
cmake -G Ninja "${launcher[@]}" -S "$src" -B "$build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON \
  -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_Q6K_EXPERTS=ON \
  -DSTRATA_MMQ_KQUANTS=ON \
  -DCMAKE_HIP_ARCHITECTURES="$archs" \
  -DCMAKE_HIP_COMPILER="$rocm/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$rocm" \
  -DCMAKE_HIP_FLAGS="--rocm-path=$rocm --rocm-device-lib-path=$bitcode" \
  -DCMAKE_PREFIX_PATH="$rocm" \
  -DSTRATA_GGML_DIR="$src/third_party/llama.cpp"
cmake --build "$build" --target strata -j "$jobs"
install -m 0755 "$build/strata" "$out.tmp/engine/strata"

# THE ROCm LIBRARIES THE ENGINE LOADS: every shared library ldd resolves, and
# the ones HIP opens itself (the code object manager, libamd_comgr), with all
# they need in turn - but the C and C++ runtime the node image already has (the
# same Ubuntu). One flat lib/: the engine finds them through
# LD_LIBRARY_PATH=<runtime>/lib (the plugin sets it), which comes before the
# libraries' own RUNPATHs.
echo "== ROCm libraries"
skip='^(linux-vdso|ld-linux|libc\.|libm\.|libdl\.|libpthread\.|librt\.|libstdc\+\+\.|libgcc_s\.)'
search="$rocm/lib:$rocm/lib/llvm/lib:$rocm/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
copy_deps() {   # the libraries $1 needs that lib/ does not have yet; prints the new ones
  LD_LIBRARY_PATH="$search" ldd "$1" | awk '/=> \//{print $1, $3}' | while read -r name path; do
    if [[ ! "$name" =~ $skip && ! -e "$out.tmp/lib/$name" ]]; then
      cp -L "$path" "$out.tmp/lib/$name"
      echo "$out.tmp/lib/$name"
    fi
  done
}
pending="$(copy_deps "$out.tmp/engine/strata")"
for opened in libamd_comgr.so.3 libamd_comgr.so.2; do
  found="$(find -L "$rocm/lib" -maxdepth 1 -name "$opened" | head -1)"
  if [[ -n "$found" ]]; then
    cp -L "$found" "$out.tmp/lib/$opened"
    pending+=$'\n'"$out.tmp/lib/$opened"
    break
  fi
done
while [[ -n "${pending//[[:space:]]/}" ]]; do
  next=""
  while read -r lib; do
    [[ -n "$lib" ]] && next+="$(copy_deps "$lib")"$'\n'
  done <<< "$pending"
  pending="$next"
done
# rocBLAS and hipBLASLt load their GEMM kernels from a library directory next
# to their own .so: only this build's architectures (the rest is gigabytes).
keep_arch() {
  local file="$1"
  [[ "$file" =~ gfx[0-9a-f]+ ]] || return 0
  local arch
  for arch in ${archs//;/ }; do
    [[ "$file" == *"$arch"* ]] && return 0
  done
  return 1
}
for lib in rocblas hipblaslt; do
  [[ -d "$rocm/lib/$lib/library" ]] || continue
  mkdir -p "$out.tmp/lib/$lib/library"
  find -L "$rocm/lib/$lib/library" -maxdepth 1 -type f | while read -r file; do
    if keep_arch "$(basename "$file")"; then
      cp -L "$file" "$out.tmp/lib/$lib/library/"
    fi
  done
done
if LD_LIBRARY_PATH="$out.tmp/lib" ldd "$out.tmp/engine/strata" | grep 'not found'; then
  echo "the engine has unresolved libraries" >&2
  exit 1
fi
echo "   $(du -sh "$out.tmp/lib" | cut -f1) in lib/"

# The vision encoder (llama.cpp's mtmd at the engine's pinned commit) runs on the CPU: the plugin reads images there,
# so the model keeps the whole card (engine.rs, the config's vision section). Portable: AVX2 baseline, no OpenMP.
echo "== vision encoder"
cmake -G Ninja "${launcher[@]}" -S /src/tools/vision -B /work/build-vision \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR=/src/third_party/llama.cpp \
  -DSTRATA_PORTABLE=ON \
  -DSTRATA_VISION_CUDA=OFF
cmake --build /work/build-vision --target strata-vision -j "$jobs"
install -m 0755 /work/build-vision/bin/strata-vision "$out.tmp/engine/strata-vision"
if ldd "$out.tmp/engine/strata-vision" | grep 'not found' ; then
  echo "the vision encoder has unresolved libraries" >&2
  exit 1
fi

echo "== server and tools"
cp -a "$src/serve" "$src/tools" "$src/data" "$src/LICENSE" "$out.tmp/strata/"
# the packages the server imports from the source root (strata_control, strata_cli), where the source has them
for pkg in strata_control strata_cli; do [[ ! -d "$src/$pkg" ]] || cp -a "$src/$pkg" "$out.tmp/strata/"; done
mkdir -p "$out.tmp/strata/third_party/llama.cpp"
cp -a "$src/third_party/llama.cpp/gguf-py" "$out.tmp/strata/third_party/llama.cpp/"
find "$out.tmp/strata" -name '__pycache__' -prune -exec rm -rf {} +
find "$out.tmp/strata/serve" "$out.tmp/strata/strata_control" "$out.tmp/strata/strata_cli" -name 'test_*.py' -delete 2>/dev/null || true
install -m 0644 "$plugin/runtime/pom_strata_server.py" "$out.tmp/pom_strata_server.py"
install -m 0644 "$plugin/runtime/pom_sessions.py" "$out.tmp/pom_sessions.py"

echo "== python packages (for $(python3 --version))"
grep -vE '^\s*(#|$)|^(cmake|ninja)==' "$src/requirements.txt" > "$work/requirements.txt"
"${PYTHON:-python3}" -m pip install --quiet --no-cache-dir --break-system-packages \
  --only-binary=:all: --target "$out.tmp/pylib" -r "$work/requirements.txt"

python3 - "$out.tmp" "$rocm" <<'PY'
import json, os, sys
root, rocm = sys.argv[1], sys.argv[2]
try:
    version = open(os.path.join(rocm, ".info/version")).read().strip()
except OSError:
    version = ""
meta = {
    "backend": "hip",
    "strata_commit": os.environ.get("STRATA_COMMIT", ""),
    "llama_cpp_commit": os.environ.get("LLAMA_CPP_COMMIT", ""),
    "rocm": version,
    "hip_architectures": os.environ["HIP_ARCHITECTURES"].split(";"),
    "python": "%d.%d" % sys.version_info[:2],
    "runtime_version": os.environ.get("RUNTIME_VERSION", ""),
}
with open(os.path.join(root, "RUNTIME.json"), "w") as f:
    json.dump(meta, f, indent=1)
print(json.dumps(meta))
PY

rm -rf "$out"
mv "$out.tmp" "$out"
[[ ${#launcher[@]} -eq 0 ]] || ccache --show-stats | sed -n '1,12p'
echo "== runtime ready: $out"
