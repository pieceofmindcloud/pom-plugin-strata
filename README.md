# pom-plugin-strata

POM plugin that serves a whole **Qwen3.8-Flash-Next** model (GGUF
architecture `qwen4exp`) with the [Strata](https://github.com/Niko1221/Strata)
engine (the fork in `strata.lock`, which adds the stage split) instead of
llama.cpp. A machine with the plugin publishes the
`inference` service "Strata"; a deployment that picks that service runs the
model on that machine's NVIDIA cards (one or more: Strata splits the layers),
and the POM pipeline sends each prompt to Strata, which streams the tokens back.
The plugin screen embeds Strata's own web app with its live GPU, speed and
cache monitor.

How the POM uses it is in [docs/README.md](docs/README.md) (also shown in the
POM documentation screen).

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
| `host.services` | POM | the `inference` service, its state, `architectures: ["qwen4exp"]` and the engine `settings` (`src/settings.rs`) |
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
in ROCm's image into `build/runtime-hip` (with `--native` and `ROCM_PATH`, on
the host, e.g. with AMD's TheRock wheels), carrying the ROCm libraries the
engine needs.

**Releases** are made by the `Publish release` workflow
(`.github/workflows/release.yml`): it builds both runtimes for the NVIDIA and
AMD families on their own runners and attaches four files to the GitHub
release `v<version>`: the library, `pom-plugin-linux-x86_64.json`,
`strata-runtime-linux-x86_64.tar.gz` and
`strata-runtime-hip-linux-x86_64.tar.gz`, whose URLs, sizes and SHA-256 the
library pins; a node downloads and verifies the runtime its GPUs need on first
use.

## Known limits

- A whole model on one machine, or layer slices on several machines (the
  stage link of the Strata fork pinned in `strata.lock`, carried by the POM
  with its pieces in flight and hidden-state precision); a machine's slice
  runs on all its GPUs of one vendor.
- Linux x86_64 with NVIDIA cards of compute capability 7.5 or newer (CUDA) or
  wave32 AMD cards (HIP).
- The model is prepared once per file into `<POM database dir>/strata/packs/`
  (about 1.5 GB for the IQ3_XXS file). The MTP draft layer is used only when a
  prepared one is found (see the guide).
- Images and typed decisions are not served through Strata.
