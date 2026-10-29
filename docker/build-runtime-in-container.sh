#!/usr/bin/env bash
# Runs inside docker/runtime.Dockerfile. Inputs: /src (the Strata source at the
# locked commit, with third_party/llama.cpp), /plugin (this repository, read
# only). Output: /out/runtime, the directory the plugin runs Strata from.
set -euo pipefail

archs="${CUDA_ARCHITECTURES:?CUDA_ARCHITECTURES is required}"
jobs="${BUILD_JOBS:-$(nproc)}"
build=/work/build
out=/out/runtime

rm -rf "$out.tmp"
mkdir -p "$out.tmp/engine" "$out.tmp/strata" "$build"

echo "== engine: CUDA $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'), archs $archs"
cmake -G Ninja -S /src -B "$build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_PORTABLE=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$archs" \
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Static \
  -DCMAKE_PROJECT_INCLUDE=/plugin/cmake/pom-static-cuda.cmake \
  -DSTRATA_GGML_DIR=/src/third_party/llama.cpp
cmake --build "$build" --target strata -j "$jobs"
install -m 0755 "$build/strata" "$out.tmp/engine/strata"

# The engine may need the driver and the C/C++ runtime, and nothing of CUDA:
# a POM node image has no CUDA shared library to give it.
if ldd "$out.tmp/engine/strata" | grep -E 'libcu(dart|blas|blasLt)' ; then
  echo "the engine links a CUDA shared library; it must use the static ones" >&2
  exit 1
fi
if ldd "$out.tmp/engine/strata" | grep 'not found' | grep -v 'libcuda.so.1' ; then
  echo "the engine has unresolved libraries" >&2
  exit 1
fi

# The vision encoder (llama.cpp's mtmd at the engine's pinned commit) runs on the CPU: the plugin reads images there,
# so the model keeps the whole card (engine.rs, the config's vision section). Portable: AVX2 baseline, no OpenMP.
echo "== vision encoder"
cmake -G Ninja -S /src/tools/vision -B /work/build-vision \
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
cp -a /src/serve /src/tools /src/data /src/LICENSE "$out.tmp/strata/"
# the packages the server imports from the source root (strata_control, strata_cli), where the source has them
for pkg in strata_control strata_cli; do [[ ! -d /src/$pkg ]] || cp -a /src/$pkg "$out.tmp/strata/"; done
mkdir -p "$out.tmp/strata/third_party/llama.cpp"
cp -a /src/third_party/llama.cpp/gguf-py "$out.tmp/strata/third_party/llama.cpp/"
find "$out.tmp/strata" -name '__pycache__' -prune -exec rm -rf {} +
find "$out.tmp/strata/serve" "$out.tmp/strata/strata_control" "$out.tmp/strata/strata_cli" -name 'test_*.py' -delete 2>/dev/null || true
install -m 0644 /plugin/runtime/pom_strata_server.py "$out.tmp/pom_strata_server.py"
install -m 0644 /plugin/runtime/pom_sessions.py "$out.tmp/pom_sessions.py"

echo "== python packages (for $(python3 --version))"
grep -vE '^\s*(#|$)|^(cmake|ninja)==' /src/requirements.txt > /work/requirements.txt
python3 -m pip install --quiet --no-cache-dir --break-system-packages \
  --only-binary=:all: --target "$out.tmp/pylib" -r /work/requirements.txt

python3 - "$out.tmp" <<'PY'
import json, os, subprocess, sys
root = sys.argv[1]
meta = {
    "strata_commit": os.environ.get("STRATA_COMMIT", ""),
    "llama_cpp_commit": os.environ.get("LLAMA_CPP_COMMIT", ""),
    "backend": "cuda",
    "cuda": subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.split("release ")[-1].split(",")[0],
    "cuda_architectures": os.environ["CUDA_ARCHITECTURES"].split(";"),
    "python": "%d.%d" % sys.version_info[:2],
    "runtime_version": os.environ.get("RUNTIME_VERSION", ""),
}
with open(os.path.join(root, "RUNTIME.json"), "w") as f:
    json.dump(meta, f, indent=1)
print(json.dumps(meta))
PY

rm -rf "$out"
mv "$out.tmp" "$out"
echo "== runtime ready: $out"
