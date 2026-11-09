# pom-plugin-strata

POM plugin that serves **Qwen3.8-Flash-Next** (`qwen4exp`),
**Qwen3.6-35B-A3B** (`qwen35moe`) and **DeepSeek-V4.1-Flash** (`deepseek41`)
with the [Strata](https://github.com/Niko1221/Strata) engine (the fork in
`strata.lock`, which adds the stage split) instead of llama.cpp. A machine
with the plugin publishes the `inference` service "Strata"; a deployment that
picks that service runs the model on that machine's NVIDIA or AMD cards,
and the POM pipeline sends each prompt to Strata, which streams the tokens back.
The plugin screen embeds Strata's own web app with its live GPU, speed and
cache monitor.

How the POM uses it is in [docs/README.md](docs/README.md) (also shown in the
POM documentation screen). The ISTA-DAS Coder DeepSeek set keeps its Engram
file beside its quant folder; put all five model shards in that folder and
`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` beside it. Strata's newer Qwen4
model shapes need separately compiled engines; this runtime only carries the
Flash-Next geometry and does not run those shapes.

## Layout

| Path | What it is |
|---|---|
| `src/lib.rs` | plugin ABI v1 entry point (`pom_strata_plugin_v1`) and the `query` operations |
| `src/engine.rs` | manager thread: prepares the model (Strata's pack tool), starts and stops Strata's server, follows its readiness |
| `src/runtime.rs` | where the Strata runtime lives: `STRATA_RUNTIME_DIR`, `build/runtime` of the source, or the pinned release asset |
| `src/gguf.rs` | GGUF header reader: architecture, layer count, shards, the n-gram table's shard |
| `src/proxy.rs` | loopback server behind the POM plugin proxy: Strata's web app and monitor with the API key added |
| `runtime/pom_strata_server.py` | Strata's own server plus `POST /pom/generate`, the route the POM gateway uses |
| `cmake/pom-static-cuda.cmake` | links Strata with the static cudart/cuBLAS/cuBLASLt, as the POM links llama.cpp |
| `docker/` | the runtime build image (the POM's `nvidia/cuda:13.4.1-devel-ubuntu24.04`) and its script |
| `strata.lock` | the Strata repository and commit, and the llama.cpp commit Strata pins |
| `ui/`, `i18n/`, `docs/` | `pom-plugin-ui/v1` manifest, the Strata screen (status, and Strata's app while a model is served), catalogs and the guide |
| `scripts/` | runtime build, UI build, library build and release packaging |

## Operations

| Operation | Who | What |
|---|---|---|
| `host.services` | POM | the `inference` service, its state, `architectures: ["qwen4exp", "qwen35moe", "deepseek41"]` and the engine `settings` (`src/settings.rs`) |
| `engine.load` | the local POM node | loads a model: `model_path`, `model_label`, `n_ctx`, `max_sessions`, `kv`, `gpus`, `settings` (applied with Strata's `serve/runconfig.py` rules by the bridge) |
| `engine.unload` | the local POM node | stops Strata and gives the GPU back |
| `engine.status` | the local POM node | `idle`, `installing`, `preparing`, `loading`, `ready`, `stopping` or `error`, with `detail` and `n_ctx` |
| `engine.upstream` | the local POM node | port and token of Strata's server on loopback, while ready |
| `ui.upstream` | POM | port and token of the interface server, for the plugin proxy |
| `rpc.status` | any profile | the engine state; the log tail for the administrator |
| `ui.manifest`, `ui.asset` | POM | the embedded screens, catalogs and guide |

`engine.*` are neither `peer.*` nor `rpc.*`: no other machine and no browser
reaches them. The POM calls them from the runtime of a deployment
(`crates/node/src/inference/plugin_engine.rs` in the POM repository).

## Build

The plugin library is a small Rust `cdylib`. The Strata runtime is built
separately, in the same CUDA image the POM is compiled in, and links the same
static CUDA libraries the POM does; the POM node image needs no CUDA library
for it (only the driver), and the Python packages the server needs are built
for the node image's Python 3.12 and kept in the runtime's `pylib/`.

```bash
STRATA_SOURCE_DIR=../Strata scripts/build-runtime.sh   # build/runtime (Docker, ~15 min)
scripts/build-ui.sh                                     # ui/dist
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
scripts/package.sh --version 0.1.0                      # release files in dist-release/
```

`CUDA_ARCHITECTURES` (default `86;120`) and `CUDA_VERSION` (default `13.4.1`)
follow the POM's defaults. Without `STRATA_SOURCE_DIR` the script clones
`STRATA_REPOSITORY` from `strata.lock`.

**Development on a POM node.** The POM's local rebuild (Plugins screen,
`POM_DESENV=1`) compiles the mounted source with Cargo only. Run
`scripts/build-runtime.sh` and `scripts/build-ui.sh` on the host first: the
library finds `build/runtime` through the source mount, so nothing has to be
downloaded.

**AMD cards** run the HIP runtime: `scripts/build-runtime.sh --hip` builds it
into `build/runtime-hip`, carrying the ROCm libraries the engine needs. CI
builds it in ROCm's image (tens of GB). On a development machine, run
`scripts/setup-host-rocm.sh` once: it installs AMD's TheRock ROCm SDK for one
GPU family (`ROCM_GPU_FAMILY`, default `gfx110X-all`) into
`~/.local/share/pom-strata-rocm`, and `--hip` then builds natively with it, no
image (`--image` forces the image; `--native` with `ROCM_PATH` takes another
host ROCm). Keep `HIP_ARCHITECTURES` within that family.

**macOS (Apple Silicon) and Windows** build their runtime natively with the
same command: `scripts/build-runtime.sh` runs `scripts/build-runtime-macos.sh`
(the Metal engine; Command Line Tools, cmake, ninja) or, under Git Bash in the
MSVC environment with `CUDA_PATH`, `scripts/build-runtime-windows.sh` (CUDA,
with cuBLAS's DLLs beside the engine). Both carry a standalone CPython 3.12 in
`python/`, since those nodes have no Python of the POM image.

**Runtimes** are built by the `Build Strata runtime` workflow
(`.github/workflows/runtime.yml`): one per backend - Linux CUDA and HIP, macOS
Metal, Windows CUDA - each on its own runner, from the head of the Strata
branch `strata.lock` names, published to the release `strata-runtime` (its
`RUNTIME.txt` names their Strata commit). Nothing is compiled twice
(`scripts/runtime-key.sh`): a runtime whose key (Strata commit, build files,
bridge) matches the `<asset>.key` already in `strata-runtime` is skipped
(`force` rebuilds it); an engine whose key matches one in the Actions cache is
reused, so a change to the bridge only reassembles the runtime around it; and
otherwise ccache compiles only what changed in Strata. On Windows nvcc also goes
through ccache. `strata_commits` (e.g. `cuda=<sha>,cuda-windows=<sha>`) keeps a
backend on a Strata commit whose engine is cached, when a new Strata commit
only matters to another backend.

**Releases** are made by the `Publish release` workflow
(`.github/workflows/release.yml`, a few minutes): for each selected platform
(Linux x86_64, macOS aarch64, Windows x86_64) it builds the library on that
platform's runner, pinned to that platform's runtimes from `strata-runtime`
(with `build_runtime`, it first brings them up to date through the runtime
workflow, which builds only what changed). The release `v<version>` gets, per
platform, the library (`pom-plugin-strata-<platform>.so|dylib|dll`),
`pom-plugin-<platform>.json` and the runtime archives
(`strata-runtime-<platform>.tar.gz`, plus `strata-runtime-hip-linux-x86_64.tar.gz`
on Linux), whose URLs, sizes and SHA-256 the library pins; a node downloads and
verifies the runtime its GPUs need on first use.

## Known limits

- A whole model on one machine, or layer slices on several machines (the
  stage link of the Strata fork pinned in `strata.lock`, carried by the POM
  with its pieces in flight and hidden-state precision); a machine's slice
  runs on all its GPUs of one vendor.
- **DeepSeek-V4.1-Flash** (`deepseek41`) runs on `engine/strata-dsv41`, from a
  pack of its own (`tools/dsv41/pack_gguf.py`), whole or as POM slices. The only
  valid cut of the published model is layer 20. It runs text only, without saved
  sessions or the prefill/decode split. A DSpark drafter chosen on the deploy
  screen is packed and turned on for the stage with the head; with no choice it
  stays off. A prompt chunk of a split is
  at most 512 rows, one at a time. **The split's output has not been compared
  with the whole model's yet.**
- Linux x86_64 with NVIDIA cards of compute capability 7.5 or newer (CUDA) or
  wave32 AMD cards (HIP); Windows x86_64 with NVIDIA cards (CUDA); an Apple
  Silicon Mac (M1 or newer, macOS 15 or newer, Metal). The macOS and Windows
  builds are new and not yet validated on a node: the layer-slice stage link
  on Metal and AMD cards on Windows are untested or absent.
- The model is prepared once per file into `<POM database dir>/strata/packs/`
  (about 1.5 GB for the IQ3_XXS file). The MTP draft layer is used only when a
  prepared one is found (see the guide).
- Images and typed decisions are not served through Strata.
